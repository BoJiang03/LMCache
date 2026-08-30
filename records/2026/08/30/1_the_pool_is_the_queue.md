# The pool is the queue

Record 8 left two arms in flight and a matrix planned on the L1 axis. Both arms
landed negative, the matrix was never run, and the reason is the same one that
now closes several other doors: on this corpus the external hit rate and the
admission queue are not two variables that happen to move together, they are
two names for one event, the GPU KV pool reaching 100 percent. Everything below
either measures that identity or fails to escape it.

## 1. Speculative decoding is dead at this working point

`l72b64L192d0spec` is `l72b64L192d0` plus ngram drafting, four draft tokens.

| | d0 | d0 + spec |
|---|---|---|
| acceptance length | -- | 3.38 |
| tpot p50 | 87.1 ms | 112.4 ms |
| decode_tps p50 | 11.5 | 8.9 |
| lat p50 | 55.63 s | 59.48 s |
| running / waiting | 24.33 / 3.49 | 25.76 / 2.62 |
| TTFT p50 | 9.35 s | 7.59 s |
| ttft_shape | NON-MONOTONE | NON-MONOTONE |

G1 was wrong in the favourable direction: the prediction was acceptance 1.4-1.8
and the measurement is 3.38. G2 was wrong. G3 is unresolvable, 7.59 s sits
inside the TTFT p50 noise floor of plus or minus 30 percent. G4 was the kill
condition and it tripped: tpot above 80 ms, measured 112.4. The drafting worked
and the arithmetic still lost, which is the strongest form the kill can take.

The engine's own counter agrees with the client: aggregate generation
throughput 316.4 to 295.7 tok/s, per request 10.96 to 9.73. Two independent
sensors, same sign.

Correction to an intermediate claim made while the arm was still draining. The
engine log showed `Waiting: 0-8` in the steady window and this was reported as
the queue collapsing. It was not. running_mean plus waiting_mean is 27.8 before
and 28.4 after. Requests moved from the waiting bucket to the running bucket
and nothing drained, while end to end latency got worse.

### The probe already contained the kill signal

`dp_ngram.log` and `dp_base.log` were run before the arm, at batch 1, same
model, same tp=2, 101,883 token prompt:

| batch 1 | decode | tpot | acceptance |
|---|---|---|---|
| no spec | 177.5 tok/s | 5.6 ms | -- |
| ngram, 4 draft | 66-90 tok/s | 11-15 ms | 1.97 |
| ngram_gpu, 4 draft | 59-83 tok/s | 12-17 ms | 1.41 |

Both variants are 2 to 3 times slower than no spec at batch 1. Record 7 quoted
the acceptance length from this file and did not read the throughput lines in
the same file, so the arm should never have been launched.

### The mechanism, revised

Record 8 attributed the loss to MoE expert fanout growing with batch size. The
batch-1 data fits that (one token routes to 8 of 128 experts, five positions
route to as many as 40, so the weight read multiplies while acceptance is 1.97).
The batch-30 data does not: the loss there is only 1.1x, and section 6 below
shows the weight term is a minority of the traffic at 107k contexts. The
defensible statement is narrower than record 8's. Speculative decoding
amortises the weight read. At 107k contexts the weight read is not where the
time goes, so spec is the wrong tool for long-context serving irrespective of
dense or sparse. MoE fanout makes it worse at small batch; it is not why it
cannot win at ours.

There is no fallback: fewer draft tokens reduces fanout and acceptance
together, ngram_gpu was already measured worse, and a real draft model adds a
second model.

## 2. CACHE_WARM is a load knob, not a cache knob

`l72b64L512d0cw` was meant to test L1 = 512 GiB. It changed two things and the
second one invalidated it.

| CONC=72 | d0 (cw=0) | 512 (cw=900) |
|---|---|---|
| inflight mean / peak | 30.85 / 50 | 100.59 / 122 |
| running / waiting | 24.33 / 3.49 | 44.63 / 70.01 |
| oversub | 0.81 | 2.20 |
| TTFT p50 | 9.35 s | 194.63 s |
| gpu prefix hit max | 65.8% | 19.4% |
| ext hit mean | 26.05% | 56.39% |
| compute | 19.1% | 24.1% |
| l1_gib | 134.51 | 361.10 |

In a closed loop in-flight cannot exceed the offered concurrency. It is 100.59
against CONC=72, which is direct evidence that the harness issued more than
CONC. The archive confirms it is systematic: `k20_384cw_s2` and `k20_96cw_s1b`
ran CONC=20 with `cachewarm=600` and reported inflight 25.73 and 22.96, also
above their CONC, with TTFT 308.77 s and 629.51 s. No arm using CACHE_WARM can
be compared on latency to one without it. The knob is retired.

What the arm does establish, because capacity and coverage are not latency
figures, is that the retention failure of record 8 section 6 is a capacity
failure and L1 = 512 fixes it:

| reuse interval | L1=192 coverage | L1=512 coverage |
|---|---|---|
| 120-300 s | 50% | 93% |
| 300-600 s | 54% | 98% |
| 600-1200 s | 14% | 100% |
| 1200+ s | 0% | 100% |

G5 landed (l1_gib 361.10 inside the 350-410 band, watermark events 8 against a
prediction of below 5). G6 was wrong in the wrong direction, compute rose from
19.1 to 24.1 percent. G7 half landed: ext above 48 percent, but local collapsed
from 65.8 to 19.4, which is the identity in section 7 doing exactly what it
says. G8 was wrong by a factor of 23.

### A second harness bug

`phase.py` compares wall-clock times as strings. This arm ran 23:31:02 to
00:11:12, so the profiling window is empty under a lexical comparison and every
`phase:` line in its snapshot reads zero. The whole-run figures are the usable
ones for that arm. Any arm crossing midnight has the same defect.

## 3. Ninety percent of reuse is short-gap, and it is the corpus

Distribution of reuse intervals, from `miss.py`, counts not coverage:

| interval | CONC=48 | CONC=60 | CONC=72 |
|---|---|---|---|
| 0-120 s | 570 | 597 | 558 |
| 120-300 s | 32 | 41 | 38 |
| 300-600 s | 5 | 10 | 13 |
| 600-1200 s | 9 | 11 | 14 |
| 1200+ s | 1 | 1 | 1 |
| share over 120 s | 7.6% | 9.5% | 10.6% |

The shape does not move with load, so it is not produced by queueing. For d0
the same split by tokens is 6,760,344 of 62,129,244 reusable tokens over 120 s,
10.9 percent, matching the event share.

## 4. The scenario does not compress think time, and NO_SCENARIO is a null experiment

`ScenarioSpec` for `inferencex-agentx-mvp` forbids compressing trace delays in
three separate places: `forbid_ignore_trace_delays`, `forbid_trace_idle_gap_cap`,
`forbid_inter_turn_delay_cap`. It also pins the loader to the semianalysis
corpora, forbids input truncation, requires `ignore_eos` and streaming, requires
a 900 s minimum, and requires first-turn cache bust. The certification is an
anti-gaming contract about the workload, not a claim that the workload is
representative.

The single permitted acceleration is `system_idle_gap_cap_seconds = 10.0`, whose
implementation fires only when `in_flight == 0` and `scheduler.running_count == 0`
and whose docstring is "Bound true system-idle time without changing an
individual trace". Every arm logs a summary per phase:

| arm | warmup | profiling |
|---|---|---|
| f8k256c48 | jumps=27, skipped=20,552 s | jumps=0, skipped=0.000 s |
| f8k256c60 | jumps=33, skipped=78,583 s | jumps=0, skipped=0.000 s |
| f8k256c72 | jumps=32, skipped=78,507 s | jumps=0, skipped=0.000 s |
| l72b64L192d0 | jumps=32, skipped=78,509 s | jumps=0, skipped=0.000 s |
| e72b64L192 | jumps=33, skipped=78,505 s | jumps=0, skipped=0.000 s |

Every reported number comes from the profiling window and the cap compressed
0.000 seconds there. All the skipping is in warmup, which runs single-lane with
in_flight 0 to 1 and is genuinely idle.

The comment in `arm.sh` claiming the scenario compresses think time from about
400 s to about 89 s and removes the long-gap tail is wrong and needs deleting.
`NO_SCENARIO=1` would change nothing in the measured window. This is the second
retraction of that proposal; the first was argued, this one is measured.

## 5. Corpus survey

Same statistic on every candidate, prefix matched through `hash_ids`, bytes per
token taken as this model's 49,152 at fp8.

| corpus | isl mean / p50 | over 120 s events | over 120 s reused tokens | working set | scenario legal |
|---|---|---|---|---|---|
| CC 062126-256k | 107,094 / 101,568 | 10.7% | 10.9% | 406 GiB | yes |
| Mooncake conversation | 12,035 / 6,909 | 19.7% | 50.8% | 4,284 GiB | no |
| Mooncake toolagent | 8,596 / 6,346 | 10.0% | 21.8% | 4,296 GiB | no |
| Mooncake arxiv | 8,590 / -- | 10.0% | 21.8% | 4,293 GiB | no |
| Bailian traceA | 2,331 / 1,046 | 29.3% | 48.1% | 1,853 GiB | no |

Every agent-driven corpus has a thin tail, including Mooncake's own toolagent
variant, which is the same 10 percent as ours. Agent loops are machine paced.
The other CC date variants are the same workload and will not differ.

Mooncake `conversation_trace` is the candidate. Its long-gap share is 4.7 times
ours by token, its working set is 23 times L0 against our 2.2, so L0 cannot hold
it and any external hit is structural rather than overflow, and its p50 input of
6,909 tokens is comfortably above the `FLOOR=2048` store threshold. It replays
`fixed_schedule` from recorded timestamps, so there is no closed-loop CONC and
the decode queue no longer gates admission. Risks: offered prefill at 1x is
40,937 tok/s against our measured 5,400-7,300 of compute, so it needs
`--replay-speedup` or a windowed offset; offered decode is 1,165 tok/s against
316 measured at 107k contexts, unmeasured at 12k; and our BLOCK, FLOOR and
HORIZON were fitted to 100k prompts.

Bailian has the best gap distribution but a p50 input of 1,046 tokens, below
the store floor, so it would not exercise the policy.

The three trace files are in the session scratchpad. No code change is needed:
`mooncake_trace` is a built-in loader and `hash_ids_synthesis.py` synthesises
prompts that share real prefixes.

## 6. What holds L0, and why batching does not help

Decode-phase KV, estimated as `running_mean` times the decode share of admitted
time, each request charged `isl + osl/2`:

| arm | CONC | running | decode share | decode-held | of pool | of allocated |
|---|---|---|---|---|---|---|
| f8k256c48 | 48 | 18.2 | 98% | 88.9 GiB | 48% | 81% |
| f8k256c60 | 60 | 21.4 | 98% | 99.2 GiB | 53% | 84% |
| f8k256c72 | 72 | 24.8 | 96% | 118.2 GiB | 63% | 93% |
| e72b64L192 | 72 | 24.9 | 96% | 116.9 GiB | 63% | 92% |
| l72b64L192d0 | 72 | 24.3 | 94% | 112.8 GiB | 60% | 91% |

Pool is 186.7 GiB. Eager and lazy are identical here, which is the same finding
as record 8 section 3: hit rate does not reduce block occupancy.

Aggregate decode throughput against batch, from the engine counter over the
steady window:

| arm | running | aggregate decode | per request | aggregate prefill |
|---|---|---|---|---|
| f8k256c48 | 20.2 | 360 tok/s | 17.83 | 2,530 |
| f8k256c60 | 22.8 | 388 tok/s | 17.03 | 3,047 |
| f8k256c72 | 30.7 | 363 tok/s | 11.82 | 4,480 |
| f8k256c84 | 33.5 | 344 tok/s | 10.28 | 5,855 |
| l72b64L192d0 | 30.8 | 319 tok/s | 10.34 | 6,540 |

Batch grows 66 percent and aggregate decode does not rise. Per request it falls
42 percent. `dsweep` measured `tpot(L) = 4.247 ms + 0.0136 ms per 1k tok` at
batch 1; the fixed 4.25 ms is all that batching can amortise, and it is
amortised by batch 8 to 10. Beyond that the KV read and the expert weight read
both scale with batch.

Traffic at batch 24, 32k context, weights bf16 with 79 percent expert fanout:
KV 37.7 GB at 3.93 ms of roof, weights 47 GB at 4.92 ms, MoE and attention FLOPs
together under 0.1 ms, roof 8.85 ms against a measured 34.15 ms, so 26 percent
of roof. The unexplained 25 ms is most consistent with per-layer launch and sync
cost: attention is in `splitting_ops` and so runs outside the cudagraph, 48
times per step.

## 7. Engine configuration has no headroom

`bsweep.py` fires B concurrent requests with mutually distinct prefixes at fixed
context, no aiperf, no scenario, no LMCache. B=24, 32k, tpot p50. Baseline has
four replicates at 34.18, 33.65, 34.65, 34.15, spread 1.5 percent.

| configuration | tpot | against baseline |
|---|---|---|
| FLASH_ATTN, fp8 KV, BLOCK 64 (current) | 34.16 ms | -- |
| BLOCK=128 | 33.90 ms | within noise |
| weights fp8 | 34.24 ms | within noise |
| KV bf16 | 37.68 ms | 10% worse |
| MoE flashinfer_cutlass | 39.37 ms | 15% worse |
| attention FLASHINFER | 60.61 ms | 77% worse |
| attention TRITON_ATTN | 111.20 ms | 226% worse |
| MoE flashinfer_trtllm | bringup failed | -- |

Seven alternatives, none better. The configuration every arm has been running is
already the best of them, and `KV_DTYPE=fp8` is worth 10 percent on speed on top
of the capacity it buys.

Two artefacts found while running this. `VLLM_ATTENTION_BACKEND` was removed from
`vllm/envs.py` in 0.23; the selector is `--attention-backend`, and the first pair
of points ran with the env var and both landed on FLASH_ATTN, so they are two
replicates of the default rather than a comparison. And a fresh server captures
cudagraphs for unseen batch shapes inside the first measurement: identical
configurations differed 36 percent at 8k and 1.6 percent at 32k until a
discarded warmup pass was added, after which the three 8k replicates are 13.55,
13.62 and 13.60.

`up.sh` gained `QUANT` and `ATTN_BACKEND`. Neither is worth keeping in a run
config on this evidence; `QUANT=fp8` frees about 28 GiB of HBM and buys no
speed.

## 8. The queue is the pool hitting 100 percent, and the mean says nothing

| arm | kv_mean | kv_p50 | kv_p90 | kv_max | p90 absolute | headroom | Wq |
|---|---|---|---|---|---|---|---|
| f8k256c48 | 59.0% | 66.5% | 77.5% | 89.5% | 144.7 GiB | 1.29x | 0.3 s |
| f8k256c60 | 63.4% | 68.4% | 84.2% | 93.8% | 157.2 GiB | 1.19x | 0.2 s |
| f8k256c72 | 67.9% | 88.8% | 99.3% | 100.0% | 185.4 GiB | 1.01x | 11.3 s |
| f8k256c84 | 78.2% | 96.2% | 99.5% | 100.0% | 185.6 GiB | 1.01x | 26.5 s |
| e72b64L192 | 67.7% | 90.6% | 98.8% | 100.0% | 184.6 GiB | 1.01x | 15.5 s |
| l72b64L192d0 | 66.7% | 86.0% | 99.0% | 100.0% | 184.9 GiB | 1.01x | 13.2 s |

Mean occupancy rises 32 percent across this table while Wq rises 88 fold. The
discriminator is `kv_max`. The two arms that never touch the ceiling have no
queue; every arm that reaches 100.0 percent has one.

This kills a plan agreed earlier in the session. The proposal was to hold
CONC=60 and shrink the pool with `GPU_UTIL` until decode-held reached 80 percent
of it, on the reasoning that decode holds only 53 percent at CONC=60 so there is
room. That 53 percent is a mean. The peak demand at CONC=60 is 93.8 percent of
186.7, or 175.1 GiB. A pool of 124 GiB would sit below even the p90 demand of
157.2 GiB and would bind a larger fraction of the time than CONC=72 does. The
largest shrink that keeps `kv_max` under 100 percent is to about 175 GiB,
`GPU_UTIL` about 0.86, which moves spare cache from 68 to 57 GiB and will not
move the external hit rate.

It also tightens record 8 section 7. External hit and queue are not correlated
variables, they are the same event. The external hit rate is positive only when
L0 misses; on this corpus, where 90 percent of reuse is inside 120 s, L0 misses
only when the pool is full; the pool being full is what requests wait for.

## 9. Corrections made this session

1. Reported that spec collapsed the queue, from a mid-run engine log. It did
   not; running plus waiting is flat and end to end latency got worse.
2. Record 8 attributed the spec loss to MoE fanout as a batch-size effect. The
   dependence is the other way round, the penalty is largest at batch 1, and at
   our batch the weight term is too small for spec to win regardless of
   architecture.
3. Claimed the dsweep MoE backend sweep had been run. `ds_cutlass.log` and
   `ds_trtllm.log` contain only a header; it produced no measurements.
4. Proposed `TP=4` as the decode lever on the grounds that the weight read
   dominates. It does not at 107k contexts; TP=4 would help the KV term, but at
   26 percent of roof bandwidth is not the constraint.
5. Proposed `NO_SCENARIO=1` a second time, on the reuse histogram. Withdrawn on
   the idle-cap measurement in section 4.
6. Proposed `GPU_UTIL=0.68`, withdrawn in section 8.
7. Wrote `VLLM_ATTENTION_BACKEND` into the sweep; the variable no longer exists.
8. Patched `LENS` in `bsweep.py` while `bs.sh` still passed the lengths
   explicitly, so a round meant to run 32k and 100k ran 8k and 32k.
9. Quoted the whole base string as one argument in `bsw2.sh`, so the BLOCK=128
   point got `MAX_MODEL_LEN` and `BLOCK` as part of a single KEY=VALUE and
   failed to start.

## 10. In flight

`l64b64L192d0` and `l68b64L192d0`, started 01:50, current base, lazy,
`DEFER_SECS=0`, `L1_GB=192`, `FLOOR=2048`, `BLOCK=64`, CONC 64 and 68. There is
no measured point between CONC=60 and CONC=72 and Wq spans 0.2 s to 11.3 s
across that gap. The question is whether the pool can bind for a small fraction
of the time, which evicts prefix blocks and gives L1 customers, without
accumulating much queue.

Written before they land:

- H1: CONC=64 keeps `kv_max` under 100 percent, ext below 15 percent, Wq under
  1 s. Still on the free side.
- H2: CONC=68 reaches `kv_max` 100 percent with ext 20-35 percent, Wq 2-5 s and
  TTFT 3-6 s. That is a usable working point and eager/lazy runs there next.
- H3: the transition is a step, 64 looks like 60 and 68 looks like 72. Then this
  corpus has no usable middle and the corpus has to change.

Near the knee the closed loop has two self-consistent branches, so a point with
unusually wide variance in `waiting` needs a replicate before it is believed.

## 11. Open

- `max_deferral_seconds` PR on a `_pr` branch. This session adds nothing against
  it; record 8's result still argues for shipping the default as 0.
- The L0 and L1 point-in-time intersection probe, still a prerequisite for
  move-not-copy and still unwritten.
- The exclusive move-not-copy retrieve path, untouched.
- `arm.sh` comment about the scenario compressing think time: delete.
- `phase.py` midnight comparison: fix or note on every affected arm.
- Nothing has been pushed to any remote.
