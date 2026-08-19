# QASPER long-context working-set sweep

## Purpose

The synthetic hot/cold workload isolates the intended cache-capacity effect,
but it deliberately creates frequent transfer overlap. This supplemental sweep
checks whether the policy helps a real long-context multi-user workload and how
the result changes as the active KV working set crosses the L1 capacity.

No production source was changed. The workload used the repository's existing
`benchmarks/multi_round_qa/multi-round-qa.py` from a RAID-backed runtime copy,
with reproduction-only changes to stop replacement users after one fixed cohort
and to hold the revisit gap constant across cohort sizes.

## Dataset and workload

The source is the original paper-centric AllenAI QASPER v0.3 train JSON:

- pinned source: `ag2435/qasper@a8de10174c66470ee25cd1a5af9f34a494b60ab6`;
- source SHA-256: `9458bfe76074a8fa8d1685af02bcc73537aa6d338ad20591dfaff1946bc88bf4`;
- selected-conversation SHA-256: `12408dd492f77a43d1de800e598d4b3618af17c2528bc04e66f4fe280b7a5362`.

Each user owns one real research paper and QASPER's human-written questions.
Round 1 sends the full paper and its first question. The generated answer is
appended to chat history; round 2 asks a real follow-up question about the same
paper. Selected papers contain 8K--16K Qwen tokens (median about 9.4K). There is
no synthetic history filler.

Fixed settings:

- production commit `8e4e851f91316bb7994be3d096966f0d1ef0b52b`;
- Qwen3-8B, TP=4 on NVIDIA H200;
- 20 GiB GPU KV and 40 GiB CPU L1;
- QPS 2, two rounds, 16-second per-user revisit gap;
- at most 64 generated tokens;
- fixed dataset order and fixed cohort;
- two repetitions at every point, with policy order reversed in repetition 2.

The coverage column below is aggregate over both rounds. Because round 1 is
cold, approximately 0.5 is the maximum possible value.

## Results

> These are the original pre-change runs (reps 720/721, production commit
> `8e4e851f`, before the free-queue read was decoupled from the drain
> budget, eager/eviction-aware only). The post-change resweep with a
> no-connector baseline supersedes them; see
> "Post-change resweep" below.

Ranges are the two same-machine repetitions. Positive latency percentages mean
eviction-aware is faster.

| users | estimated KV working set | eager coverage | eviction-aware coverage | round-2 TTFT p50 improvement | round-2 TTFT p90 improvement | round-2 E2E p50 improvement | round-2 E2E p90 improvement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 23.0 GiB | 0.492 | 0.492 | 5.9%--24.5% | 1.5%--6.6% | **-7.4% to -5.3%** | 1.6%--6.4% |
| 24 | 34.6 GiB | 0.001 | 0.461 | **29.1%--32.9%** | **32.7%--35.2%** | **28.2%--28.3%** | **13.6%--14.3%** |
| 32 | 45.5 GiB | 0.001 | 0.200--0.268 | **14.3%--16.0%** | 3.7%--6.8% | **14.3%--20.9%** | **11.2%--14.9%** |
| 40 | 55.7 GiB | 0.001 | 0.152--0.172 | 7.1%--17.4% | -3.4%--11.9% | **12.5%--15.8%** | 5.9%--16.3% |
| 48 | 67.0 GiB | 0.001 | 0.057--0.119 | 0.7%--11.9% | 2.5%--6.2% | 5.6%--7.3% | 7.0%--9.1% |

All 20 runs completed the exact expected request count with zero vLLM
preemptions. There were no failed-store or request-drop losses in the lazy
counter ledgers.

The MP server emitted the same rate-limited `has no lookup ipc key, skipping
touch` warning ten times in every eager and eviction-aware run. It did not
correlate with request failure or the policy under test, but these runs must not
be described as warning-free. Server logs also end with the expected
`CancelledError`/`KeyboardInterrupt` traceback because the harness stops the
HTTP server with SIGINT after collecting results; there was no runtime
traceback before teardown.

## Interpretation

The result is capacity-regime dependent rather than a blanket speedup:

1. **Working set well below L1 (23.0 GiB):** both policies retain the returning
   prefixes. Eviction-aware moves time between TTFT and generation, but round-2
   E2E p50 is 5%--7% worse, so there is no overall win to claim.
2. **Unique working set near L1 (34.6 GiB):** eager's repeated incremental
   snapshots churn the 40 GiB cache and leave effectively no returning
   coverage. Eviction-aware keeps almost every reusable paper, cutting round-2
   E2E p50 by 28% in both policy orders.
3. **Working set moderately above L1 (45.5--55.7 GiB):** only part of the cohort
   survives, but the retained 8K--16K prefixes still reduce E2E p50 by
   12%--21%. TTFT p90 is less stable because lower-tier transfer can overlap
   foreground work.
4. **Working set far above L1 (67.0 GiB):** retained coverage falls below 0.12.
   Typical TTFT benefit becomes small/variable, while E2E p90 remains 7%--9%
   lower.

This explains why the shorter 4.3K-token ShareGPT trial raised cache coverage
without improving typical latency: TP=4 recomputation was already cheap. The
policy's production benefit is concentrated in long reusable prefixes whose
working set is near, or moderately above, lower-tier capacity. The hot/cold
stress-test TTFT regression remains relevant and is not hidden by this result.

## Post-change resweep with a no-connector baseline (reps 900/901)

After the per-step free-queue read was decoupled from the drain budget
(see `PR_INFO.md`), the full sweep was rerun on the current production
tree with a third configuration added: `off`, the same engine with no KV
connector at all. Same dataset, cohorts, QPS, revisit gap, L1 = 40 GiB and
20 GiB GPU pool; GPUs 1,2,4,5; repetition 901 reverses the configuration
order. The request stream is byte-identical across configurations and
repetitions (`apc_queries` matches exactly), so round-2 latencies pair
per user. `qasper_panel.py` reproduces every number below from the
archived `QP_*_900/901` files.

Median per-user round-2 deltas against `off`, in ms (rep 900 / rep 901;
negative is faster than no connector):

| users | KV working set | eager coverage | ev-aware coverage | eager dE2E p50 | ev-aware dE2E p50 | ev-aware vs eager dE2E |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 16 | 23.2 GiB | 0.492 | 0.441--0.465 | -24.6 / -28.4 | **-67.9 / -71.8** | -38.5 / -36.4 |
| 24 | 34.8 GiB | 0.000 | 0.460 | +61.2 / +46.6 | **-37.8 / -57.1** | -96.5 / -98.8 |
| 32 | 45.8 GiB | 0.000 | 0.231--0.287 | +36.4 / +34.0 | **-43.7 / -6.7** | -68.4 / -44.7 |
| 40 | 56.0 GiB | 0.000 | 0.167--0.177 | +35.4 / +39.8 | +12.7 / +7.9 | -31.1 / -37.1 |
| 48 | 67.5 GiB | 0.000 | 0.066--0.112 | +40.6 / +42.4 | +11.2 / -5.4 | -25.3 / -52.9 |

What changed against the pre-change sweep, and what the `off` anchor adds:

1. **Eviction-aware beats eager at every point in both repetitions**
   (last column, paired per user), by 25--99 ms of round-2 E2E p50.
   Pre-change, the 23 GiB point was a 5--7% *loss* to eager; that loss was
   the prepaid free-queue read, and it is gone.
2. **Eager at or above L1 capacity is worse than having no connector**:
   +34 to +61 ms per returning request with 0.000 coverage -- its own
   incremental snapshots churn the 40 GiB L1, so it pays every store and
   recovers nothing. This is the same failure mode the agentic workload
   shows at its 20 and 10 GiB budgets.
3. **The envelope's far end is a bounded, honest loss**: at 56--67 GiB the
   retained 0.07--0.18 coverage no longer pays for retrieval against `off`
   (-5 to +13 ms), while still beating eager by 25--53 ms.
4. **The 23 GiB point's coverage split is a tier upgrade, not lost reuse.**
   Eviction-aware's external coverage reads lower than eager's
   (0.44--0.47 vs 0.492), but total reuse is identical in every
   repetition: 167024 hit tokens each, with 17k--31k of eviction-aware's
   served by vLLM's GPU prefix cache instead of external retrieval
   (`apc_hits` 17104/30928 vs eager's 240). The mechanism is in vLLM's
   block pool: pinning for the drain's D2H copy removes blocks from the
   free queue and unpinning re-appends them at the tail -- the youngest
   eviction position -- so exactly the prefixes the policy judged
   reuse-worthy also survive longer on the GPU. Zero stores were lost at
   this point (`dropped_evicted=0`, the 4 still-pending operations at
   shutdown are stores that never became necessary).

All 30 runs returned rc=0 with zero vLLM preemptions. Eviction-aware
ledgers close with 0--7 operations dropped to eviction per run, all at the
over-capacity points. The two known mode-independent caveats from the
original sweep reproduce unchanged: the rate-limited `has no lookup ipc
key, skipping touch` server warning, and the expected SIGINT teardown
traceback pair (present in `off` runs too).

## Raw data

Machine-readable aggregate counters and all per-request CSV/JSON files are
under [`results/qasper_working_set/`](results/qasper_working_set/): reps
720/721 are the pre-change eager/eviction-aware runs, reps 900/901 the
post-change off/eager/eviction-aware resweep. `qasper_panel.py` (one level
up) tabulates a resweep repetition from those files. The original 31 MiB
QASPER source and model files remain on RAID and are not committed.
