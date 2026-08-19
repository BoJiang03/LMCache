# PR title

```text
[MP] Add an eviction-aware policy for lazy offload
```

# PR description

## Summary

This PR adds an eviction-aware drain policy for MP lazy offload and makes it
the default when lazy offload is enabled.

The existing FIFO policy submits buffered stores after a fixed number of
requests finish. The new policy waits until the GPU blocks holding an operation
approach eviction, avoiding stores that vLLM's GPU prefix cache can still serve.
This reduces host-cache writes and prevents useful cold prefixes from being
displaced by hot KV that did not need a lower-tier copy.

Eviction-aware draining is the default when lazy offload is enabled:

```text
lmcache.mp.lazy_offload = true
lmcache.mp.lazy_offload_horizon_steps = 2.5
```

Set `lmcache.mp.lazy_offload_policy = FIFO` explicitly to retain the legacy
count-triggered fallback.

## Design

Scheduler-side lazy offload is isolated behind `LazyOffloadManager`; the MP
connector only forwards lifecycle events and applies explicit store/session
actions.

- `LazyOffloadRequestRegistry` owns request phase, request/store epochs, and
  submitted batches.
- Reset, request-ID reuse, and stale receipts are handled as epoch transitions.
  An old receipt still releases its pins but cannot break a successor's prefix
  or end its session.
- FIFO and eviction-aware backends return the same policy-neutral drain plan.
- The controller is the only owner of block pinning, receipt interpretation,
  and session teardown.
- Worker-side failures are reported with completion receipts. A failed current-
  epoch store breaks the prefix chain, preventing unreachable suffix stores.

The eviction-aware backend provides:

- pressure-triggered draining from vLLM free-queue ranks;
- admission-time block snapshots and prefix-integrity validation;
- prefix-closed, eviction-imminence-ordered drains;
- content deduplication across request IDs;
- a configurable minimum-prefix economy heuristic;
- bounded scheduler-step candidate discovery using allocation deltas and a
  block-to-request reverse index.

This phase implements eviction timing and the current economy heuristic. Reuse
prediction is intentionally left for a later phase; unknown reuse preserves the
current eviction-only behavior.

## Compatibility and behavior changes

- `EVICTION_AWARE` is the default policy; `FIFO` remains an explicit legacy
  fallback.
- Lazy offload requires vLLM prefix caching because eviction validation depends
  on block hashes.
- Requests shorter than one LMCache chunk can finish without a pending store and
  release their session immediately.
- Store completion is aggregated across worker ranks before blocks are unpinned.
- The connector now records aligned vLLM APC hits even when LMCache misses. This
  fixes eager under-store: GPU-resident APC KV can be copied to LMCache before
  eviction instead of being silently omitted from store coverage. Existing
  LMCache-covered ranges are not stored twice.

## Performance

Reported TP=1 runs used one NVIDIA H200; tensor-parallel runs used the
corresponding number of H200 GPUs. All runs used the LMCache MP connector and
CPU L1.

### Hot/cold long documents

The workload has three hot documents that remain GPU-resident and eleven cold
documents whose reuse requires a lower-tier copy: 38.5 GiB of distinct KV,
20 GiB GPU KV pool, 40 GiB L1, 14 warmup requests, and 120 measured requests.

Representative final-tree run:

| policy | external hit | total coverage | wall time | L1 eviction cycles |
| --- | ---: | ---: | ---: | ---: |
| eager | 0.000 | 0.725 | 43.1s | 14 |
| eviction-aware | 0.677 | 0.911 | 30.3s | 5 |

Across repeated runs, eager took 41--43 seconds with 14--15 L1 eviction cycles;
eviction-aware took 27--31 seconds with 3--6 cycles. The improvement comes from
not writing the GPU-resident hot set into L1, preserving capacity for cold
prefixes that actually need retrieval.

For comparison, legacy FIFO with its default `threshold=100/select_count=10`
took 41.9--42.0 seconds and left total coverage at 0.725. Snapshot validation
rejected 15 stale request batches per run after FIFO waited past their GPU block
lifetime. A tuned 10/10 FIFO reduced median time to 33.1 seconds but still had
lower coverage (0.715) than eviction-aware (0.933). At TP=4, eviction-aware was
11.9% faster than default FIFO with coverage 0.948 versus 0.725.

The trade-off is foreground interference: at TP=1, hot TTFT p50 was 130ms for
eager/default FIFO, 157ms for tuned FIFO, and 160ms for eviction-aware; at TP=4
it was 118ms, 114ms, and 161ms, respectively. Controlled TP=4 attribution shows
this is not decision-loop overhead: eviction-aware with all stores suppressed
measured 120ms, serialized real retrieval/stores measured 125ms, and an
unconstrained 64 GiB L1 measured 136ms. The 161ms result requires concurrent
cold retrieval/drain work under the original 40 GiB L1 pressure; non-adjacent
hot requests remain at baseline. That interference cuts cold p50 from 369ms to
204ms versus FIFO and still reduces total wall time by 18.4% versus eager.

### Real long-context paper QA

A supplemental TP=4 sweep used the original AllenAI QASPER v0.3 data: each
session sends a real 8K--16K-token research paper and question, then returns 16
seconds later with a human-written follow-up question. The fixed workload used
QPS 2, 20 GiB GPU KV, 40 GiB L1, and two repetitions per point with policy
order reversed.

| distinct KV working set | eager coverage | eviction-aware coverage | returning E2E p50 improvement | returning E2E p90 improvement |
| ---: | ---: | ---: | ---: | ---: |
| 23.0 GiB | 0.492 | 0.492 | -7.4% to -5.3% | 1.6%--6.4% |
| 34.6 GiB | 0.001 | 0.461 | 28.2%--28.3% | 13.6%--14.3% |
| 45.5 GiB | 0.001 | 0.200--0.268 | 14.3%--20.9% | 11.2%--14.9% |
| 55.7 GiB | 0.001 | 0.152--0.172 | 12.5%--15.8% | 5.9%--16.3% |
| 67.0 GiB | 0.001 | 0.057--0.119 | 5.6%--7.3% | 7.0%--9.1% |

The operating envelope is explicit: there is no E2E p50 win when both policies
fit comfortably in L1; the largest gain occurs when the unique reusable set is
near L1 capacity but eager's repeated incremental snapshots churn it. Benefits
decline as the reusable set grows far beyond L1. Returning TTFT p90 is less
stable than E2E because lower-tier work can overlap foreground execution. A
separate 4.3K-token ShareGPT multi-round trial similarly raised coverage without
improving typical latency, confirming that short TP=4 recomputation can already
be cheaper than retrieval.

### Real agentic sessions

A third supplemental TP=4 workload replays published `nebius/SWE-agent-trajectories`
runs: real SWE-agent sessions against real GitHub issues, where a prompt grows one
recorded action and one tool observation at a time and the session idles while the
tool runs. The engine's answer is discarded and the recorded action appended, so both
policies replay byte-identical request streams. Trajectories are replayed whole -- 4
to 158 steps, final prompts up to 34591 tokens -- as 14 concurrent slots each issuing
158 steps from a queue of whole sessions, which holds concurrency and step rate
constant across a forty-fold spread in session length. 36.7 GiB is live at once and
182.6 GiB of distinct KV passes through the cache per run; pressure is swept with the
L1 budget at a fixed 20 GiB GPU KV pool.

| L1 budget | live/L1 | eager coverage | eviction-aware coverage | eager external hit | TTFT p90 improvement | E2E p90 improvement |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 GiB | 0.9x | 0.763 | 0.895 | 0.658 | 22.1% | 21.0% |
| 20 GiB | 1.8x | 0.310 | 0.626 | 0.004 | 18.2% | 16.3% |
| 10 GiB | 3.7x | 0.308 | 0.398 | 0.000 | 1.8% | 1.0% |

Improvements above are against eager. Measured instead against an engine with no KV
connector at all, eager is a net loss at the 20 GiB budget and eviction-aware is
roughly neutral to positive; the section below reports that comparison, which is the
more informative one. Two repetitions with reversed variant order agree to within
0.013 coverage and 2.5 ms of mean paired E2E at the 20 and 40 GiB budgets. At 20 GiB the mean end-to-end
latency over 2141 token-identical request pairs improves by 39.4 ms, and L1 eviction
cycles fall from 714 to 322. Eager's external hit rate of 0.004 at 20 GiB and 0.000
at 10 GiB is the failure mode this policy exists to prevent: eager writes every step
of a working set the budget cannot hold, and its own next writes evict what it just
stored.

### Why eviction-aware can lose to eager, and what that costs

Seven full runs of the same 2212-request workload at the 20 GiB budget separate the
causes, each varying one thing. `off` runs the engine with no KV connector and is the
baseline; paired means are over the 2141 requests common to every run.

| variant | TTFT | decode | E2E | coverage | throttled | dropped_evicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | +27.2 ms | +30.5 ms | **+57.7 ms** | 0.310 | -- | -- |
| eviction-aware, `max_drain_per_step=64` | -6.2 | +40.6 | +34.4 | 0.623 | 0 | 138 |
| eviction-aware, `=4` | -7.7 | +26.0 | +18.3 | 0.626 | 134 | 350 |
| eviction-aware, `=1` | -2.1 | **+2.8** | **+0.7** | 0.600 | 731 | 1017 |
| `=4` with `min_prefix_tokens=12000` | -8.8 | +25.5 | +16.8 | 0.612 | 95 | 242 |

The first row is the important one: **at this budget eager is worse than having no
connector at all, at every prompt length.** Its TTFT cost against `off` rises from
6 ms at 2760-token prompts to 66 ms at 26231-token ones -- it writes in proportion to
what it reads in -- while its external hit rate is 0.004, so none of it comes back.
Eviction-aware turns that 57.7 ms per-request loss into a 0.7 ms one at
`max_drain_per_step=1`, with coverage 0.600 against 0.310.

**The residual cost is one bounded read, not the deferral.** `collect_due` reads the
free queue to `danger_depth + max_drain_per_step x largest live operation` on every
scheduler step. Split into octiles, eviction-aware decodes *faster* than no connector
for three quarters of a run and then doubles -- 89 ms rising to 202 ms at `=4` --
while `off` and eager stay flat. Dropping the multiplier to 1 flattens it completely
(92 ms in the final octile, +2.8 ms of decode over the whole run). Pending depth is
not the cause (`=1` ends with 81 pending against `=4`'s 101), and neither is context
length (the final octile's prompts are the run's shortest).

**That parameter is doing two unrelated jobs, and the two regressions are the
result.** It sets the drain rate, which decides how much KV survives to be written,
and it sets the depth of a per-step read. Tuning it down removes the decode cost and
takes the retrieval benefit with it: at `=1` the 26231-token TTFT gain falls from
93 ms to 25 ms, 28% of admitted operations are lost to eviction rather than merely
delayed, and E2E p99 rises to 1031 ms against `=4`'s 905 ms. Tuning it up restores
the gain and the cost together. Bounding the read independently of the drain
budget -- by the pending operations' actual block count rather than
`budget x largest op`, or by a separate cap -- should yield the 100 ms long-prompt
gain of `=64` together with the flat decode of `=1`. That is the actionable finding
here; it is a production change and was not attempted in this measurement.

**Gate 3 does not address the short-prompt regression, and does something else worth
knowing.** `lmcache.mp.lazy_offload_min_prefix_tokens` defaults to 0, and setting it
to 12000 makes the policy refuse 1108 operations, so a shorter request provably has
nothing of its own session in L1 to retrieve -- yet its penalty against `off` is
unchanged at 16 ms. Short requests are not losing by retrieving what they could have
recomputed. What the gate does instead is stop spending L1 on short prefixes, which
grows the 26231-token TTFT gain from 93 ms to 135 ms for 0.014 of coverage.

Finally, three of the nine first-repetition runs recorded vLLM preemptions -- 12, 2
and 2 -- all under eviction-aware and all under budget pressure, where all nine eager
runs and both repetitions of the reported 20 and 40 GiB eviction-aware points
recorded none. The 10 GiB row carries the largest count and should be read as
unconfirmed until that is explained.

### GSM8K correctness

The correctness workload runs 120 questions twice (cold then cached),
concurrency four, with approximately 51 GiB of KV against a 68 GiB L1.
Strict scores stayed within the existing 0.900--0.925 run-to-run range. A
representative cached run produced:

| policy | strict score | external coverage | cached wall time |
| --- | ---: | ---: | ---: |
| eager | 0.908 | 0.961 | 21.7s |
| eviction-aware | 0.908 | 0.961 | 22.3s |

This workload is intentionally unfavorable to deferral because reuse distances
exceed GPU residency; it verifies retrieval correctness and bounds the cost when
the eviction gate has little work to eliminate.

### Eager APC backfill

An isolated hardware A/B used identical production code except for disabling
eager APC-hit accounting in the baseline. After clearing L1, replaying a
2565-token APC-resident prompt, displacing it from GPU, and requesting it again:

- the PR rebuilt 10 L1 objects and retrieved 2560 tokens in every repetition;
- the one-line baseline rebuilt no objects and recomputed the prefix;
- all generated outputs matched, with no warnings or tracebacks;
- two five-repeat runs reduced median third-request latency by 16--26%
  (approximately 1.19--1.35x speedup).

The extra store is intentional eager behavior. Eviction-aware lazy offload is
where future Reuse and Economy gates can avoid paying that cost for dead KV.

## Validation

- 178 relevant unit/contract tests passed.
- The suite covers policy invariants, epoch transitions, stale failures,
  request-ID reuse, FIFO compatibility, connector delegation, worker receipt
  aggregation, and eager APC backfill behavior.
- Ruff check, Ruff format, compileall, and `git diff --check` pass.
- GSM8K, hot/cold performance, and eager APC-backfill A/B were rerun on the
  final production tree with no warnings or tracebacks.
- A supplemental architecture matrix covers Gemma 3 12B hybrid attention and
  DeepSeek V2 Lite MLA/MoE. Two-run hot/cold medians improved by 33.1% and
  13.9%, respectively; byte-level eager/lazy KV comparisons also matched.
- TP=2 and TP=4 validation registered every worker adapter and GPU cache.
  GSM8K cached score/coverage matched eager at both sizes; two-run hot/cold
  medians improved by 24.3% and 18.4% with no failed store or request-drop
  loss.
- Eager/default-FIFO/tuned-FIFO/eviction-aware A/B confirms that FIFO's request
  count is not a proxy for GPU block lifetime: default FIFO produced no usable
  GSM8K stores, while eviction-aware remained faster at TP=1 and TP=4.
- A real QASPER TP=4 working-set sweep completed 20 fixed-cohort runs with
  reversed policy order. It reproduced a 28% returning-session E2E p50 gain
  near L1 capacity and bounded the no-gain and over-capacity regimes. The MP
  server emitted a mode-independent missing-touch-key warning, so these
  supplemental runs are not described as warning-free.
- A real SWE-agent TP=4 replay completed 30 capped-cohort runs and 13
  whole-trajectory runs. Every run's engine-reported prompt-token count matched
  the cohort's tokenizer count for every step, so the replay sent exactly the
  prompts the cohort selection reasoned about, and schedule lag p90 was 0.0 ms
  throughout. All 30 capped runs and 10 of the 13 whole-trajectory runs pass
  every harness guard; the three exceptions recorded vLLM preemptions and are
  named in the report rather than averaged in.

## Reproduction

The hardware harness is intentionally kept outside the merge diff because it is
one-off experiment infrastructure.

- Production code: [`8e4e851f`](https://github.com/BoJiang03/LMCache/commit/8e4e851f91316bb7994be3d096966f0d1ef0b52b)
- Immutable reproduction package: [`5476816a`](https://github.com/BoJiang03/LMCache/tree/5476816ae7f1ae72a9d5af88bfd109a91acd877b/repro/pr4499)
- Reproduction guide: [`repro/pr4499/README.md`](https://github.com/BoJiang03/LMCache/blob/5476816ae7f1ae72a9d5af88bfd109a91acd877b/repro/pr4499/README.md)
- Raw JSON from the reported runs is included in the package.
- Additional model matrix: [`COMPLEX_MODELS.md`](https://github.com/BoJiang03/LMCache/blob/47d40c49afe7e806c2f580b94427c4975de56fb6/repro/pr4499/COMPLEX_MODELS.md)
- TP=2 report and raw results: [`TP2.md`](https://github.com/BoJiang03/LMCache/blob/0c7d26db0d9d7ac46b068208095c13f67726c446/repro/pr4499/TP2.md)
- TP=4 report and raw results: [`TP4.md`](https://github.com/BoJiang03/LMCache/blob/bd543fe03736f0f6a629afda1803b3881d19844c/repro/pr4499/TP4.md)
- Policy A/B report: [`POLICY_COMPARISON.md`](https://github.com/BoJiang03/LMCache/blob/c28dd7761239848fde601e39d6e6cd81c0295377/repro/pr4499/POLICY_COMPARISON.md)
- Hot-TTFT attribution controls: [`HOT_TTFT_ATTRIBUTION.md`](https://github.com/BoJiang03/LMCache/blob/8df519590b31715d2eab420e1b9ba81c435aed23/repro/pr4499/HOT_TTFT_ATTRIBUTION.md)
- Real long-context working-set sweep: [`QASPER_WORKING_SET.md`](https://github.com/BoJiang03/LMCache/blob/7e1c4ed57e0c131624d0c78f64f64f7c59828a8e/repro/pr4499/QASPER_WORKING_SET.md)
- Real agentic session replay: [`AGENTIC_WORKLOAD.md`](https://github.com/BoJiang03/LMCache/blob/5a76005c52abf25789d6a3fdee3551084ba66b65/repro/pr4499/AGENTIC_WORKLOAD.md)

Exact hot/cold comparison:

```bash
export SMOKE_GPU=0
export SMOKE_MODEL=Qwen/Qwen3-8B
export SMOKE_HORIZON=2.5
./repro/pr4499/run_hot_cold.sh
```

Eager APC-backfill isolated A/B:

```bash
export SMOKE_GPU=0
export SMOKE_MODEL=Qwen/Qwen3-0.6B
./repro/pr4499/run_apc_backfill_ab.sh
```

The scripts record source/environment identity, verify the selected mode and
workload guards, reject warnings or tracebacks, and emit machine-readable JSON.

## Documentation

- `docs/design/integration/vllm/lazy_offload_decision_model.md`
- `docs/design/integration/vllm/lazy_offload_policy/eviction_aware.md`
- `docs/design/integration/vllm/lazy_offload.md`
