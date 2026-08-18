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

## Raw data

Machine-readable aggregate counters and all 20 per-request CSV/JSON files are
under [`results/qasper_working_set/`](results/qasper_working_set/). The original
31 MiB QASPER source and model files remain on RAID and are not committed.
