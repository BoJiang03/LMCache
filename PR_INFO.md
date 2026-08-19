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

All performance workloads compare three configurations of the same engine on
byte-identical request streams: `off` (no KV connector at all), `eager` (the MP
connector's immediate-offload behavior, today's default), and eviction-aware
lazy offload with this PR's defaults. Every run uses Qwen/Qwen3-8B on NVIDIA
H200 at TP=4, a fixed 20 GiB GPU KV pool, and the LMCache MP connector with CPU
L1. Everything below was measured on the final production tree. Latency
comparisons are paired per request -- same request, same position in the same
stream -- because the effects are tens of milliseconds on populations whose own
spread is hundreds.

### Long-context QA: sweeping the working set across L1 capacity

Real AllenAI QASPER v0.3 research papers (8K--16K tokens each). Each user sends
one paper and a real question, then returns 16 seconds later with a
human-written follow-up; QPS 2, L1 fixed at 40 GiB, and the cohort size swept
so the distinct KV working set crosses L1 capacity. Two repetitions with
configuration order reversed. Values are median per-user round-2 E2E deltas in
ms against `off` (rep 1 / rep 2; negative is faster):

| KV working set | vs L1 | eager coverage | eviction-aware coverage | eager vs off | eviction-aware vs off | eviction-aware vs eager |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 23.2 GiB | 0.6x | 0.492 | 0.44--0.47 | -24.6 / -28.4 | **-67.9 / -71.8** | -38.5 / -36.4 |
| 34.8 GiB | 0.9x | 0.000 | 0.460 | +61.2 / +46.6 | **-37.8 / -57.1** | -96.5 / -98.8 |
| 45.8 GiB | 1.1x | 0.000 | 0.23--0.29 | +36.4 / +34.0 | **-43.7 / -6.7** | -68.4 / -44.7 |
| 56.0 GiB | 1.4x | 0.000 | 0.17--0.18 | +35.4 / +39.8 | +12.7 / +7.9 | -31.1 / -37.1 |
| 67.5 GiB | 1.7x | 0.000 | 0.07--0.11 | +40.6 / +42.4 | +11.2 / -5.4 | -25.3 / -52.9 |

- Eviction-aware is faster than eager at every point in both repetitions.
- **At or above L1 capacity, eager is worse than having no connector**
  (+34 to +61 ms per returning request at 0.000 coverage): it writes every
  incremental snapshot of a working set the cache cannot hold, and its own
  next writes evict what it just stored. That is the failure mode this policy
  exists to prevent; at 0.9x--1.1x, eviction-aware retains 0.23--0.46
  coverage and stays 7--57 ms faster than no connector.
- The 23.2 GiB coverage split is a tier upgrade, not lost reuse: total hit
  tokens are identical in every repetition (167,024 each), with 17k--31k of
  eviction-aware's served by vLLM's GPU prefix cache instead of external
  retrieval, because the drain's D2H pin re-ranks its blocks to the free
  queue's youngest eviction position. No stores were lost
  (`dropped_evicted=0`).
- Far above capacity the loss against `off` is bounded (-5 to +13 ms) while
  eager's is +40; the policy degrades gracefully.

Zero vLLM preemptions in all 30 runs. Per-request data, ledgers and caveats:
`QASPER_WORKING_SET.md`.

### Real agentic sessions: sweeping L1 pressure at a fixed working set

Replays of published `nebius/SWE-agent-trajectories` -- real SWE-agent sessions
against real GitHub issues, where a prompt grows one recorded action and one
tool observation at a time. Trajectories are replayed whole (4--158 steps,
final prompts up to 34.6K tokens) as 14 concurrent slots at a constant step
rate, which holds concurrency fixed across a forty-fold spread in session
length. 36.7 GiB of KV is live at once and 182.6 GiB of distinct KV passes
through the cache per run; the L1 budget is swept with everything else fixed.
Paired means over the same 2212 requests, in ms against `off`:

| L1 budget | live/L1 | config | coverage | external hit | TTFT | decode | E2E | preempt |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 GiB | 0.9x | eager | 0.776 | 0.677 | -33.3 | +2.1 | -31.2 | 0 |
| | | eviction-aware | 0.896 | 0.852 | **-44.2** | **-11.7** | **-55.8** | 0 |
| 20 GiB | 1.8x | eager | 0.310 | 0.003 | +27.8 | +29.6 | +57.4 | 0 |
| | | eviction-aware | 0.614 | 0.450 | **-5.9** | **-1.9** | **-7.8** | 1 |
| 10 GiB | 3.7x | eager | 0.308 | 0.000 | +23.6 | +22.4 | +46.0 | 0 |
| | | eviction-aware | 0.429 | 0.121 | +13.3 | +3.6 | +16.8 | 11 |

(`off` reaches 0.308 coverage from the GPU prefix cache alone; coverage counts
any cache tier, external hit only LMCache.)

**Where both policies work (40 GiB), eviction-aware nearly doubles eager's
win** and takes the tail with it: E2E p99 is 556 ms against eager's 1133 and
off's 1049, and decode octile means run flat at 89--91 ms while off and eager
spike to 120--131 mid-run. Storing only what eviction threatens cuts L1
eviction cycles from 123 to 44 and raises the external hit rate from 0.677 to
0.852.

**Under pressure (20 GiB) eager is worse than no connector at every prompt
length** while eviction-aware stays faster than no connector. At 10 GiB the
0.121 external hit a 0.27x budget can retain no longer repays retrieval, so
eviction-aware is net cost against `off` (+16.8 ms) -- but still 29.2 ms faster
than eager, paired. The policy has a working range; below it, it loses least.

**The decision loop's cost is bounded by its own emissions.** The policy's
free-queue read runs on the scheduler's critical path every step; it opens at
the eviction danger depth and widens only by the blocks a drain's emission has
already pinned out of the queue (an O(1) `ref_cnt` check per block). Across
these budgets it reads 67--95 blocks per step, and the paired decode delta
against `off` stays between -11.7 and +3.6 ms. Four ledger counters
(`drain_steps`, `free_queue_blocks_read`, `requests_validated`,
`blocks_validated`) make that cost readable from the ledger, and the layer-1
scenarios pin the semantics: the policy ledger is identical counter for
counter to the previous, prepaid read that this bound replaced.

**The eviction-aware-only preemptions are the drain's transient pin, and
`max_drain_per_step` bounds them.** A pending operation's blocks sit in the
free queue and cost nothing to hold; emission pins them out for the duration
of the D2H copy, and under budget pressure those bursts coincide with the
allocation spikes that pressed the queue (sampled pool peaks reach 99.1% where
the offered load alone never exceeds 38.2%). Capping the drain at 4 cuts the
peaks to 65.4% and the 10 GiB count from 11 to 1, at the cost of 186 more
operations lost to eviction and ~10 ms of the E2E win. Every victim resumed
and finished (worst observed +34 ms of TTFT, ~0.15 ms amortized per request),
so the default cap stays 64; an operator who cannot tolerate preemptions knows
which knob bounds them and what it costs.

**What a short prompt still pays** (+7 to +15 ms TTFT below 8K tokens)
decomposes into a per-request connector overhead that eager pays too, plus
retrieval of hits too small to repay the ~9 ms transfer floor; the measured
break-even is near 8K tokens. The fix -- a minimum-hit gate in the connector's
lookup path -- is common MP-connector code, outside this PR.

Full panels, the short-prompt decomposition, and the preemption census:
`AGENTIC_WORKLOAD.md` (§5--§8).

### GSM8K correctness across configurations

120 questions, 20-shot, greedy, run twice (cold pass then cached pass) at
concurrency 4 against a 68 GiB L1 (~51 GiB of KV); three repetitions per
configuration:

| config | strict score, cold | strict score, cached | external coverage | cached wall time |
| --- | ---: | ---: | ---: | ---: |
| off | 0.917--0.925 | 0.917--0.933 | -- | 15.0--15.1 s |
| eager | 0.925--0.933 | 0.917--0.925 | 0.961 | 13.5--14.6 s |
| eviction-aware | 0.917--0.933 | 0.925 | 0.961 | 13.3--14.4 s |

Every cross-configuration score delta is within one question -- the same band
as `off` against itself (two `off` runs agree on 118/120 answers; eager and
eviction-aware agree with `off` on 116--118). Eviction-aware reaches eager's
exact external coverage while storing fewer objects (5536 against 5792 on the
cold pass). All nine runs pass every guard and every ledger closes. Cached
per-question TTFT (eviction-aware 101 ms mean, eager 108, off 86) sits above
`off` because a ~3.1K-token prompt is below the ~8K retrieval break-even
measured on the agentic workload -- two workloads, one curve -- while cached
wall time still improves about 10% over `off` because prefill savings compound
at concurrency.

## Validation

- 178 relevant unit/contract tests pass, covering policy invariants, epoch
  transitions, stale failures, request-ID reuse, FIFO compatibility, connector
  delegation, worker receipt aggregation, and eager APC backfill. Ruff check,
  Ruff format, compileall, and `git diff --check` pass.
- The free-queue read change is pinned by a layer-1 ledger A/B: on the
  scenarios that exercise pressure and a capped drain, the policy ledger is
  identical counter for counter, down to per-step store submission sizes.
- Hot/cold stress test (3 hot + 11 cold documents, 38.5 GiB of KV through a
  40 GiB L1): eviction-aware finished in 27--31 s against eager's 41--43 s with
  a third of the L1 eviction cycles, at TP=1 and TP=4. Legacy FIFO's request
  count is not a proxy for GPU block lifetime: default FIFO produced no usable
  GSM8K stores and snapshot validation rejected its stale batches
  ([`POLICY_COMPARISON.md`](https://github.com/BoJiang03/LMCache/blob/c16726a0842fac51dfe3a398b0c9de1f2de5339e/repro/pr4499/POLICY_COMPARISON.md),
  [`HOT_TTFT_ATTRIBUTION.md`](https://github.com/BoJiang03/LMCache/blob/bafbb2c80a806a072609aafd52b1c1003672ee3a/repro/pr4499/HOT_TTFT_ATTRIBUTION.md)
  for the foreground-interference controls).
- Eager APC-backfill isolated A/B: identical code except the one-line
  accounting change; the PR rebuilt the displaced prefix's 10 L1 objects and
  cut median third-request latency by 16--26% with byte-identical outputs.
- Architecture matrix: Gemma 3 12B (hybrid attention) and DeepSeek V2 Lite
  (MLA/MoE); two-run hot/cold medians improved 33.1% and 13.9%, and byte-level
  eager/lazy KV comparisons matched
  ([`COMPLEX_MODELS.md`](https://github.com/BoJiang03/LMCache/blob/517db749ea3ec549fb11c63cd06e2bb7636c581f/repro/pr4499/COMPLEX_MODELS.md)).
- TP=2 and TP=4 registered every worker adapter and GPU cache; GSM8K cached
  score and coverage matched eager at both sizes
  ([`TP2.md`](https://github.com/BoJiang03/LMCache/blob/93d300ae56cd095ba2eae95925d11c064de3ef37/repro/pr4499/TP2.md),
  [`TP4.md`](https://github.com/BoJiang03/LMCache/blob/46634b7ed0dceef511aad55273ecec81cd9a6dcc/repro/pr4499/TP4.md)).
- QASPER resweep: 30 of 30 runs rc=0 with zero preemptions; the two known
  mode-independent caveats (a rate-limited missing-touch-key server warning
  and the SIGINT teardown traceback) reproduce in `off` runs too, so they are
  workload artifacts, not policy behavior.
- Agentic replay: the engine-reported prompt-token count matched the cohort
  tokenizer for every step of every run, and schedule lag p90 was 0.0 ms
  throughout, so the replay sent exactly the prompts the cohort selection
  reasoned about. All 30 capped-cohort runs and 18 of 24 whole-trajectory runs
  pass every harness guard; every exception is a vLLM preemption count, named
  in the report rather than averaged in, with the mechanism measured there.

## Reproduction

The hardware harness is intentionally kept outside the merge diff because it is
one-off experiment infrastructure.

- Production code: [`df199979`](https://github.com/BoJiang03/LMCache/commit/df199979ae1a70f60d76003beb40b8cec46affc3)
- Reproduction package (harness, raw results, analysis tools): [`10eafd5a`](https://github.com/BoJiang03/LMCache/tree/10eafd5a2d1d5ca5703bff7defa7aa76a5547da5/repro/pr4499)
- Reproduction guide: [`repro/pr4499/README.md`](https://github.com/BoJiang03/LMCache/blob/10eafd5a2d1d5ca5703bff7defa7aa76a5547da5/repro/pr4499/README.md)
- Long-context QA report: [`QASPER_WORKING_SET.md`](https://github.com/BoJiang03/LMCache/blob/10eafd5a2d1d5ca5703bff7defa7aa76a5547da5/repro/pr4499/QASPER_WORKING_SET.md)
  (`qasper_panel.py` tabulates the archived per-request data)
- Agentic replay report: [`AGENTIC_WORKLOAD.md`](https://github.com/BoJiang03/LMCache/blob/10eafd5a2d1d5ca5703bff7defa7aa76a5547da5/repro/pr4499/AGENTIC_WORKLOAD.md)
  (`agentic/analyze_panel.py` renders the paired panels)
- GSM8K sweep: `run_gsm8k.sh` (all three configurations, three repetitions,
  results under `results/`)

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
