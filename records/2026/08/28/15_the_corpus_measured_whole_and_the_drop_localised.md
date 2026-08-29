# The corpus measured whole, and the drop localised

Continues record 14. Four of that record's conclusions are corrected here, three
of them from measurements taken today rather than from re-derivation.

## 1. Where the day started

Record 14 closed with a prediction for an N=60 arm: TTFT p50 a few seconds, p90
tens of seconds. The prediction failed twice, and the two failures pointed at
different things, neither of which was the one predicted.

```
arm        conc  pc   L1    ttft_p50   ttft_p90   inflight  oversub  waiting
n60          60  --   576    283.98s    799.51s     28.03     2.35     23.88
n60pc12      60  12   576    158.73s    439.15s     19.07     1.63      9.73
```

`--prefill-concurrency` is aiperf's client-side admission cap on requests in the
prefill stage. The scenario spec does not constrain it (`config.py:519` requires
only `--streaming`), so setting it keeps the run a valid submission. It cut the
queue by 59 % and TTFT by 44 %, and moved throughput 4 %: `running_mean` went
9.60 to 9.14 and `kv_mean` 81.3 % to 80.3 %. It shortens the queue without
making the engine do more work.

It was dropped after that one arm. The user's framing settles it: `N` is the
working set and may not be changed, and neither may the number of requests in
process. A client-side cap pins `Effective Prefill Concurrency` at 12, which is
the second of those. Both arms are kept as measurements, not as candidates.

## 2. Neither compute resource is saturated

Record 14 section 7 concluded prefill was the bottleneck. A later message in the
session extended that to "prefill and decode together saturate the GPU". Both
are wrong. Computed from the 121 per-request records of n60pc12:

```
decode HBM   813 GB/s KV + 324 GB/s weights = 23.7 % of 4.8 TB/s
prefill      8,058 tok/s of 13,000-21,700   = 37-62 %
KV pool      decode residency 70.7 % of pool token-seconds
             kv_mean 80.3 %  kv_p90 98.4 %  kv_max 100.0 %
```

The pool is what is full. `running_mean` 9.14 conversations at 178,830 tokens is
1.63M of a 2.03M pool. Requests are not waiting for arithmetic; they are waiting
for blocks. This is the user's own opening statement of the session, now
measured: the in-flight KV must fit in GPU memory or decode cannot start, and
that is the only real queue.

ITL is a second-order symptom of the same thing:

```
ITL p50 = 51.4 ms   (min 13.7, p90 168.3)
batch 8.59 @ 178,830 tok:  KV 73.7 GiB/GPU -> 16.5 ms
                           weights 28.45 GiB -> 6.4 ms
                           floor    22.9 ms at 100 % bandwidth
```

45 % of the bandwidth floor. `max_num_batched_tokens=8192` with a decode batch
of 8.6 means nearly every step also carries a full prefill chunk. Leading
explanation, not proven; testing it means counting step composition in the
engine log or moving `max_num_batched_tokens`.

## 3. The corpus, measured from the arrow files

393 traces, 98,827 leaf requests (subagent chains flattened), read directly from
the HuggingFace cache rather than through aiperf.

```
requests per trace:  p10=27  p50=86  p90=618  max=4,114

in-tokens by turn index
  turn   0:    448        turn  20:   93,120
  turn   1: 50,880        turn  40:  131,008
  turn   5: 65,344        turn  80:  198,528
  turn  10: 75,008        turn 320:  239,808

consecutive-turn delta:   p50 = 1,408   mean = 1,008   16 % negative
reusable prefix / prompt: p50 = 98.7 %
genuinely new tokens/turn: p50 = 2,304  mean = 22,244
corpus recorded TTFT:      p50 = 2.64s  p90 = 6.98s  p99 = 22.90s

ALL requests in-tokens:   p50 = 142,016  mean = 218,922
our arms sent:            p50 = 106,581  mean = 174,166
```

Two results. The corpus is exactly the shape a coding agent should have: a
450-token first turn, 98.7 % prefix reuse thereafter, context accumulated over
86 median turns. And the ISL we send is *below* the corpus's own, so the long
contexts are not a harness artefact -- they are the 80th turn of a real session.

Record 14 recorded a 2.3x gap between the corpus's recorded ISL (245,248) and
ours (184,157) and left it uninvestigated. Measured whole, the corpus mean is
218,922 and the gap is 1.26x, explained by our window sampling earlier turns.

## 4. What the replay does with it

`agentic_replay.py` samples each trajectory at an instant t\* (the scenario
widens the default 25-75 % to 0-100 %), splits the trace there, and:

- WARMUP replays turn `n-1`, the last request before t\*, with the full prefix
  back-seeded under the weka `DELTAS_WITH_RESPONSES` context mode, expressly to
  "prime the server cache to the stream's state at t\*".
- PROFILING resumes at turn `n`.

So lanes do parachute into mid-conversation, and the priming pass exists to pay
for it. Turn `n` shares 98.7 % of its prefix with turn `n-1`, so the design
intends the first profiled turn to hit.

The per-turn prefill cost that follows from the sampling:

```
real steady state:  (165k entry + 147 x 1.5k) / 148 turns =  2,642 tok/turn
our 30-min window:  (165k entry +   1.4 x 1.5k) / 2.4 turns = 68,700 tok/turn
```

The entry cost is real but in production it amortises over 148 turns. In a
window that sees 2.4 turns per conversation it is the whole bill. This is a
property of the observation window, not of the workload.

`trajectory_start_{min,max}_ratio` are `default_*` fields, and
`validator.py:610` is titled "explicit honored" -- overriding them does not
raise a violation. Narrowing them would shrink entry contexts and make the
numbers move. Not done: 0-100 % uniform join is what a real fleet looks like,
and narrowing it softens the workload rather than removing an artefact.

## 5. Three hit-rate numbers, and which one is real

They disagreed by 2.6x. All three are now accounted for.

```
arm.sh ext_hit_mean        21.33 %   averages over warmup, where it is
                                     structurally 0. Broken metric.
vLLM external hit gauge    32-36 %   rolling window (294 samples, 19 of them
                                     decreasing), not cumulative, not interval
LMCache token ratio        54.8 %    11,544,832/rank retrieved of 21,074,117
                                     profiling input tokens
```

Split at the profiling boundary:

```
             stores                    retrieves
warmup    142 / 11,029,248 tok       2 /     27,648
profiling  92 /  4,656,384 tok     158 / 11,544,832
```

Warmup stored 11.03M tokens per rank; profiling retrieved 11.54M. **The priming
pass was consumed in full.** L1 reached 411 of 576 GiB and never filled, and
`l1_objects=38,182`. Capacity and lookup are both fine.

An earlier draft of this analysis concluded L1 had overflowed during priming and
proposed L1=896 GiB, derived from 65 trajectories x 165k tokens = 983 GiB
against a 761 GiB tier. That was wrong -- L1 never filled -- and the arm was not
run. The error came from reading `ext_hit_mean`, a metric this record retires.

`phase.py` now splits stores, retrieves, the lazy ledger, and both prefix-hit
gauges at the boundary read from `aiperf.log`, and `arm.sh` calls it. Every
whole-run counter in `snapshot.txt` is contaminated by warmup; the `phase:`
lines are the ones to read. `ttft.py`, `inflight.py` and `decode.py` already
filtered on `benchmark_phase`.

## 6. Where the missing content goes

Profiling input 21.07M tokens, 11.54M served from L1, 9.53M recomputed. The
recomputed part was not in L0 either -- `gpu_prefix_hit` ran 0-3.5 % all window.
Neither tier had it. The ledger, split to the profiling window:

```
admitted=1413  emitted=784 (55.5 %)  dropped_evicted=421 (29.8 %) / 1,571,584 tok
covered_prefix_tokens_skipped=932,096   danger_floor_raises=0   throttled_drains=3
```

Warmup's drop rate over the same counters is 5 %. Under pool pressure it is
29.8 %. Nearly a third of what the lazy queue buffered was destroyed by eviction
before its drain came due, and dropped rather than stored stale.

`eviction_aware.py` gives the mechanism exactly:

```python
horizon_blocks = max(blocks_per_step_ema, next_step_estimate) * horizon_steps
```

`horizon_steps` is not a survival time. It is how deep into the free queue to
treat as at-risk. With `max_num_batched_tokens=8192` and block size 16, one step
allocates at most 512 blocks, so at `horizon_steps=2.5` the danger window is
about 1,280 blocks. Admitting one 174k-token request takes 10,875 blocks -- 8.5x
the window. The class is named in the `danger_floor_max_blocks` docstring:

> the rate model forecasts the *mean* consumption, so an allocation burst larger
> than the danger window destroys waiting operations without ever being seen as
> due

`danger_floor_max_blocks=0` disables the mitigation written for it, which is why
`danger_floor_raises=0`: it has never fired in any arm to date.

## 7. The n60floor arm

One switch, the targeted one. `horizon_steps` is left at 2.5 because raising it
moves the policy toward eager and gives back the store reduction; the floor is
reactive and only rises when loss is measured.

```
CONC=60  L1_GB=576  HORIZON=2.5  FLOOR=0 -> 8192  DUR=1800
```

8192 is a non-binding cap. The floor rises to the recent peak *step* allocation,
about 512-1024 blocks; the cap exists to switch the mechanism on, not to set a
level.

Three criteria, from the standing standard (少存保持 / 不丢 block / 时延优势不回吐):

```
不丢 block   drop_rate 29.8 %  -> should collapse
少存保持     emit_rate 55.5 %  -> must not approach 100 % (that is eager)
时延不回吐   retrieve/input 54.8 % up, ttft_p50 158.73s down
```

Result: negative on the first criterion, which is the one the arm was for.

```
                          n60      n60pc12   n60floor
drop_rate (profiling)       -       29.8 %    32.0 %
emit_rate (profiling)       -       55.5 %    60.5 %
danger_floor_raises         0            0         1
free_queue_blocks_read   6.03M      5.43M   131.34M
retrieve/input (prof)       -       54.8 %    55.4 %
ttft_p50                284.0s     158.7s    272.7s
throughput              0.0563     0.0588    0.0615 req/s
```

The mechanism engaged: the free-queue scan went 24x deeper and `emit_rate` rose
five points. `drop_rate` did not fall. The floor jumps to the recent peak *step*
allocation -- 512-1024 blocks -- while one request admission takes 10,875, and
it is reactive with decay, so it raised once in the whole window. It covers
step-scale bursts, not request-scale ones. The section 6 localisation is wrong
as a cause: the floor was the mitigation written for that failure class, and
switching it on does not move the failure.

## 7a. What the three arms have in common

```
              ttft_shape        waiting_mean   oversub
n60          NON-MONOTONE          23.88        2.35
n60pc12      NON-MONOTONE           9.73        1.63
n60floor     NON-MONOTONE          21.26        2.46
```

All three are queue-bound. Under a thrashing pool the lazy queue's drops are a
symptom, not a cause, and three arms were spent treating the symptom.

The closed loop has two self-consistent solutions and both are feasible:

```
congested:  R = 556s -> demand 60/956 = 0.063 req/s   measured 0.0615
uncongested: R = 40s -> demand 60/440 = 0.136 req/s
             pool capacity 11.4 conversations / 40s = 0.285 req/s, 2.1x margin
```

Every arm lands in the congested basin because the profiling phase releases all
60 lanes at once. n60pc12's 159s -- the best of the three -- came from
`--prefill-concurrency` incidentally ramping that handoff, not from the cap
itself.

`agentic_replay.py` documents the intended ramp:

> Accelerated cache warmup synthesizes a new replay boundary at the warmup
> handoff and carries each live stream's residual next-turn delay into
> profiling, so the handoff ramps instead of firing every live stream at once

`--agentic-cache-warmup-duration` does not cap the number of requests in
process, so it is admissible under the standing constraint that neither the
working set nor the in-process count may be changed. A synchronised cold start
is an artefact of the harness; 60 users do not arrive in the same second.

Next arm proposed: `CONC=60 L1_GB=576 FLOOR=0 CACHE_WARM=600 DUR=1800`, on the
theory that the synchronised handoff is what pins the congested basin.

**Retracted before running.** The premise is false; see record 2026/08/29/1
section 1. Profiling dispatch is already spread (first-request offset p50 115 s,
max 1363 s, 16 of 51 conversations inside the first 30 s), and the congested
branch is the only feasible one once decode residency is taken from the measured
mean rather than the p50. No `CACHE_WARM` arm was run.

## 8. Corrections to earlier records

- Record 14 section 7, "prefill is the bottleneck": wrong at this operating
  point. Decode uses 23.7 % of HBM bandwidth and prefill 37-62 % of compute;
  the KV pool is what is full. Section 2 above.
- Record 14 section 3, the uninvestigated 2.3x ISL gap: measured whole, it is
  1.26x and is window sampling. Section 3 above.
- A mid-session claim that the GPU is compute-saturated at 90-107 %: wrong,
  derived from a model rather than measured. Section 2.
- A mid-session claim that the profiling window's entries are cache-busted
  turn-0 requests: wrong. `turn_index` p50 is 14 and only 11 of 121 are turn 0.
  Section 4.
- A mid-session claim that L1 overflowed during priming and needs 896 GiB:
  wrong. Warmup's 11.03M/rank was retrieved in full and L1 never filled.
  Section 5.
- `ext_hit_mean` as reported by `arm.sh` in every arm to date is not a hit rate.
  Prior records quoting it (the "never above 5 %" line in record 14 section 5
  among them) understate L1's contribution by an unknown factor.
