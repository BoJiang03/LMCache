# Lazy Offload: The Performance Model

Companion to [lazy_offload_decision_model.md](lazy_offload_decision_model.md)
(the per-chunk store criterion) and [lazy_offload.md](lazy_offload.md) (the
mechanism). This document answers a different question: **under which operating
conditions does deferred emission beat eager emission, by how much, and where
must it be exactly neutral**. It is an empirical model, calibrated on measured
runs (setup in section 6); the numbers are workload-dependent, the structure is
not.

## 1. What deferral actually changes

Store cost is not it. On the calibration workload, D2H store time is 0.65% -
2.7% of wall clock in every configuration; there is no transfer cost worth
hiding. What deferral changes is **when in a request's lifetime the write
lands**:

- **Eager** emits chunk-by-chunk during prefill, i.e. near **turn start**.
- **Lazy** emits when the GPU copy approaches eviction or the request
  finishes, i.e. near **turn end** -- for an ~85 s turn, ~85 s later.

A stored prefix is useful only if it is still resident in L1 when the reuse
arrives. The two strategies therefore face different survival requirements:

```
eager:  must survive  (rest of own turn) + (inter-turn gap)
lazy:   must survive  (inter-turn gap) only
```

## 2. The two clocks: L1 residence and the inter-turn gap

**L1 residence** is how long a written byte lives before eviction, in steady
state approximately `occupancy / store byte rate`. Store rate itself falls as
the hit rate rises (a resident prefix is deduplicated instead of re-stored),
so residence grows superlinearly with L1 capacity.

**The inter-turn gap** is a property of the workload. Agentic multi-turn
traffic is extreme: the next turn follows the previous response within seconds
(measured p50 2 s, p75 16 s), while a turn itself runs ~85 s under load. The
survival requirement is therefore wildly asymmetric: lazy needs seconds of
residence, eager needs a turn plus the gap (measured p50 requirement: 90 s).

Coverage of reuse-opportunity tokens vs residence, from the measured gap
distribution (L1 sizes map to residences via the calibration store rates):

| residence (L1) | lazy can cover | eager can cover |
|---|---|---|
| 20 s (30 GB) | 76% | 2% |
| 46 s (60 GB) | 80% | 7% |
| 105 s (~90 GB) | 87% | 63% |
| 450 s (180 GB) | 98% | 97% |

## 3. The scenario map

The controlling variable is **residence relative to the two workload clocks**:

- **Residence < gap** (L1 far too small): nothing survives either way. Both
  strategies retrieve ~nothing; deferral must be neutral. Measured: -0.05%
  paired TTFT delta at 30 GB. **Neutral, achieved.**
- **gap < residence < turn + gap** (the favorable band): lazy's turn-end
  writes survive to the next turn, eager's turn-start writes die inside their
  own turn. On the calibration workload this band is roughly L1 20 - 130 GB.
  Measured at 60 GB: lazy retrieves 5.1x the tokens off 25% less stored
  volume, -4.3% whole-run paired TTFT, -6.5% on reuse-opportunity turns.
  **This is where the feature earns its keep.**
- **Residence > turn + gap** (L1 fits the working set): both strategies'
  writes survive; timing is irrelevant and only deferral's own costs remain.
  Deferral must be neutral. Measured at 180 GB: **+16.8% against lazy -- not
  achieved**; the entire loss traces to the admission-burst drop defect
  (section 5), not to anything inherent in deferral.
- **No-reuse traffic** (first turns, single-shot requests): nothing to
  retrieve for either strategy. Measured on first-turn pairs: -57 ms median,
  i.e. **neutral, achieved** -- the deferral machinery itself is free when
  there is nothing to win.

The favorable band self-narrows from above: when recall works, dedup cuts the
store rate, residence stretches, and eager's writes start surviving too. The
band's peak location is therefore an empirical question per workload, not a
constant.

## 4. The realized win is a fraction of coverage

Timing coverage is an upper bound; the measured conversion into actual
retrieves is well below it. At 60 GB, lazy converts 16% of reuse-opportunity
tokens (coverage: 80%); eager converts 3% (coverage: 7%). The lazy/eager
*ratio* matches the coverage model, but both leak ~4/5 of covered tokens.
Splitting paired deltas by gap length shows no difference, so the leak is not
eviction-before-reuse. Known and suspected channels:

1. **Admission-burst drops** (known, section 5): dropped chunks punch holes in
   the stored prefix chain, and retrieval truncates at the first hole.
2. **Watermark batch eviction** (suspected): L1 evicts in bursts (one purge
   per ~12 s under load), so effective survival for a large share of bytes is
   far below mean residence. The 30 GB point is the extreme case: 20 s mean
   residence against 2 s gaps should cover 76%, measured retrieves were ~zero.
3. **Lookup gating** (unconfirmed): a lookup racing a just-deferred store.

Closing this conversion gap is worth more than any tuning inside the current
policy: full coverage at 60 GB is ~5x the current retrieved volume, an
estimated -15% to -30% TTFT against the measured -4.3% (a retrieve is ~25x
cheaper than the prefill it replaces, before queue amplification).

## 5. Where the implementation violates the model

The decision model assumes the scheduler's token budget upper-bounds next-step
block allocation ("one-step allocation feedforward",
[lazy_offload_decision_model.md](lazy_offload_decision_model.md)). An external
KV connector breaks this: vLLM allocates blocks for the full matched prefix
(`num_external_computed_tokens`) while excluding those tokens from
`num_scheduled_tokens`. The danger-window forecast under-reads real allocation
by exactly the L1 hit ratio -- it is blindest precisely where the cache
succeeds. Measured: window 103 blocks at ~0% hit rate, 50 blocks at 77% hit
rate, against admission bursts of ~2800 blocks; `dropped_evicted` rises from
9% to 44% of admitted stores.

The fix direction is feedforward, not feedback: admission bursts are announced
(`on_new_request` fires when a request enters the waiting queue, steps before
allocation, and the request length bounds the blocks its admission will
consume). A control loop on the drop counter would be slower than the
disturbance and would destroy the counter's value as a quality signal.

## 6. Calibration setup

Qwen/Qwen3-Coder-30B-A3B-Instruct on a single GPU, GPU KV pool 24 GiB (16384
blocks, 16-token blocks, 96 KiB/token). Agentic multi-turn replay
(`inferencex-agentx-mvp`, 42-trace pool), concurrency 32, 1800 s rounds,
fixed seed. Effects are median paired TTFT deltas keyed on (conversation,
turn) against a same-config control band of ~650 ms; per-arm counters from
the lazy offload ledger and LMCache store/retrieve totals. Calibration date
2026-08. Re-derive the map before porting conclusions to workloads with
different turn lengths, gap distributions, or context sizes.
