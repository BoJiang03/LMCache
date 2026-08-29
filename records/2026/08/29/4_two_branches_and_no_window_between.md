# Two branches and, so far, no window between them

Record 3 fixed the store timing and left TTFT at 44 s against the corpus's
recorded 2.64 s. Chasing that produced the sizing arithmetic, a concurrency
sweep, and a structural reason why the thing this project is trying to
demonstrate may not be demonstrable on this box.

## 1. TTFT was queueing, and the queue was for GPU blocks

`n24defer30`: TTFT p50 44 s decomposes into about 39 s of queueing and 4.6 s of
work. `ttft_by_isl` is nearly flat over a 20x ISL range (23.0 s at under 50k,
56.2 s at 400k-1M), so the dominant term is a constant. Little on the waiting
queue gives 5.82 / 0.0947 = 61 s, and
`vllm:num_requests_waiting_by_reason` attributes 5.58 of the 5.82 to
`capacity`, 0.24 to `deferred`. Requests wait for GPU KV blocks, not for KV
transfer.

## 2. Two averages, two tiers

Sizing L0 and L1 needs different averages and conflating them cost a wrong
recommendation mid-session.

L0 occupancy is time-weighted, because a long request holds the pool longer.
`sum(isl_i * lat_i) / sum(lat_i)` = 248,698 tokens against an arrival mean of
197,772, a 1.26x size bias. Effective capacity is 2,035,056 / 248,698 = **8.2
slots**, so `n24defer30`'s 11.67 in flight is 143 % of the pool, not the 13 %
overshoot the arrival mean suggests.

L1 residence is per conversation and unweighted: each live conversation costs
its latest context, measured 184,783 (`n24floor`) and 209,820 (`n24defer30`).
The 0.80 watermark of 576 GiB is 5,033,165 tokens, so L1 holds **25 live
conversations**. Retired conversations' entries are dead weight LRU may drop
for free, so eviction count alone is not a failure signal; degradation in the
long reuse-clock buckets is.

Empirically the crossover sits where that predicts: `n24floor` at 26
conversations was 0.95x the watermark with 1 profiling eviction, `n24defer30`
at 29 was 1.21x with 3, and those 3 did no damage (the >=1200 s bucket hit 1 of
1).

So N came down for two different reasons. 60 to 24 because L1 was 1.81-2.83x
over, which stands. 24 to about 16 because L0 is 43 % over, which is a
different tier. L1 at 24 lanes is fine at 95 %.

Per-conversation context grows as the replay advances, so L1's limit is 24-27
lanes rather than a constant.

## 3. The sweep

All arms `DEFER_SECS=30`, `HORIZON=2.5`, `FLOOR=8192`, `ANNOUNCE=false`,
`DUR=1800`.

```
CONC  L1    in-flight  kv     TTFT p50   local   external  recompute  X req/s
 14   576      2.64    37 %     1.34 s   94.8 %    2.5 %     2.7 %     0.115
 14   256      3.06    43 %     1.45 s   94.6 %    0.6 %     4.7 %     0.101
 16   320      9.76    70 %    61.6  s   21.5 %   52.3 %    26.1 %     0.066
 18   320     12.13    72 %    54.9  s   27.8 %   48.9 %    23.3 %     0.070
 20   320     14.36    73 %   160.3  s   28.0 %   40.6 %    31.4 %     0.061
 24   576     11.67    76 %    44.2  s   14.3 %   73.8 %    11.9 %     0.095
```

CONC=14 beats `n24defer30` on every axis at once: TTFT 33x better, throughput
21 % higher, recompute a quarter. Past the knee the system is in congestion
collapse: from 14 to 20 lanes TTFT rises 120x and throughput falls 47 %.

The knee is between 14 and 16, and it is not L1 size: `n14L256` has a smaller
L1 than the 320 GiB arms and stays on the good branch. **CONC=15 was not
measured.**

## 4. Why the two things move together

The same physical quantity sets both. A local prefix hit reuses cached blocks
and allocates nothing; an L1 load allocates the full prefix. So:

```
arm          alloc blk/s   free blks   block life   local hit
n14L576            81        80,385       997 s      94.8 %
n14L256            77        73,008       949 s      94.6 %
n60floor          607        21,622        36 s      13.6 %
n24defer30      1,012        30,780        30 s      14.3 %
```

Both directions are self-reinforcing. Long block life gives local hits, which
allocate nothing, which keeps block life long. Short block life sends traffic
to L1, whose loads allocate, which shortens block life further.

TTFT and L1 utilisation are therefore not two independent quantities that
happened to flip together. They are one quantity read twice, and on a single
node where an L1 load consumes GPU blocks they are anti-correlated by
construction.

## 5. It is bistable, and the branch is chosen in five minutes

Local prefix hit rate across each profiling window:

```
n14L576   53 -> 75 -> 83 -> 85 -> 86 -> 88 %    waiting 0 throughout
n16L320   39 -> 4.6 -> 4.9 -> 1.2 -> 1.6 -> 0.7 %   waiting 3 -> 6 -> 7 -> 5
n18L320   47 -> 19 -> 1.4 -> 0.6 -> 0.5 -> 0.2 %    waiting 2 -> 8 -> 8 -> 9
```

All three start in the same middle state and diverge. So the branch depends on
the initial condition as well as on the offered load, and every arm starts
profiling from an empty GPU pool: the opening wave misses locally, loads from
L1, and each load allocates. `n18ramp` (CONC=18, `PREFILL_CONC=4`,
`PREFILL_RAMP=600`) tests whether starving that opening wave reaches the good
branch at 18 lanes. Predictions in `ramp_predictions.md`. **Result not yet in.**

## 6. The scenario is not distorting anything

`env.sh` claims the scenario compresses corpus think time from ~400 s to ~89 s
and "removes the long-gap tail that is L1's entire customer base", and a
`NO_SCENARIO=1` round was proposed on that basis. The claim is false and the
proposal is withdrawn.

`inferencex-agentx-mvp` is almost entirely prohibitions: agentic-replay timing,
required streaming and ignore_eos, forbidden trace-delay skipping, forbidden
input truncation, forbidden per-trace idle-gap and inter-turn-delay caps, a
fixed loader list, 900 s minimum, first-turn prefix cache-bust. The one thing
it adds is `system_idle_gap_cap_seconds=10.0`, and
`agentic_replay.py:433` only acts when the whole benchmark is idle (returns
immediately if `in_flight > 0` or `scheduler.running_count > 0`) and then
shifts every pending timer by the same amount, preserving relative spacing.

aiperf prints what it did:

```
n14L576     jumps=2   skipped=84.061 s
n16L320     jumps=0   skipped=0
n18L320     jumps=0   skipped=0
n24defer30  jumps=0   skipped=0
```

Turning the scenario off would change the timing by nothing and would cost
`submission_valid`.

## 7. The corpus is agentic, and that is the real constraint

Z is short because the trace is short. Inter-turn gap on `n14L576`, faithful
replay, uncongested:

```
p10 1.1 s   p50 2.4 s   p75 10.0 s   p90 98.8 s   p99 697 s   mean 42.3 s
```

The "user" in this corpus is a tool loop, not a person typing. Weighted by
prompt tokens at stake:

```
gap >=   turns          tokens
  10 s   25 %    16,984,215  41 %
  30 s   21 %    14,820,824  36 %
  60 s   12 %     6,016,309  15 %
 120 s    9 %     4,004,413  10 %
 300 s    3 %     1,335,606   3 %
 600 s    1 %       255,953   1 %
```

Against a GPU block life of 997 s at CONC=14, 99 % of the reuse token volume
returns inside the GPU cache's own memory. That is why L1 carries 2.5 % there.

In the tier-ratio notation, `k_need = Z/R` = 42/23 = 1.8: L1 need only be 1.8x
L0 for this workload. It is 3.1x. The hierarchy is over-provisioned, not short.

## 8. Neither branch is a normal deployment

Raised by the user and it reframes section 7. The middle state is not merely
where lazy would be measurable, it is what a real deployment looks like. No
production serving stack runs at an external hit rate of zero, and none runs at
a TTFT of 44 s. The corpus's own recorded figures are TTFT p50 2.64 s and p90
6.98 s, on a fleet that presumably had a working external cache. Both of our
branches are abnormal:

```
CONC=14   TTFT 1.34 s  (better than the corpus)   external 2.5 %   kv 37 %
CONC=16+  external 52 %                           TTFT 61.6 s      kv 70 %
```

So the finding is not "this box has no window for lazy". It is that **we have
not yet found any workload configuration in which the parameters are normal**,
and every conclusion about the policy is downstream of that.

The structural candidate: the GPU pool holds 8.2 average contexts and a single
p90 request is 2.6 of them. A pool that shallow has no stable operating point
at a realistic utilisation, because the arrival of one long request moves
occupancy by a third. `n14L576` avoids the instability by running the machine
at a quarter of its capacity (in flight 2.64 of 8.2, kv 37 %), which is itself
not a deployment anyone would run.

Scaling the pool does not change the ratio of alive-conversation working set to
pool -- both are proportional to the lane count at a fixed duty cycle -- but it
does change the variance. At 25 slots a p90 request is 1.3 slots instead of
2.6, and the machine can be run at 60 % occupancy without one arrival tipping
it. That is the untested lever: **TP=4 rather than TP=2**, which amortises the
56.9 GiB of weights over four GPUs and should take the pool from 2.03M tokens
to roughly 5M, i.e. 8.2 slots to about 25.

## 9. Where this leaves the project

Shrinking L1 does not create a job for it (`n14L256` is the control). Shrinking
L0 cannot either: block life would have to fall from 997 s to about 120 s,
needing a pool of roughly 23 GiB, which cannot hold two contexts.

The condition for lazy offload to have anything to buy is `Z/R >> 1`, a
workload whose conversations idle for minutes. This corpus does not have it.
That is a statement about the workload and the L0/L1 ratio, not about the
policy.

Open, in order:

0. A TP=4 arm on GPUs 4-7, to get a pool deep enough that a normal
   utilisation is stable. Until the parameters look like a deployment,
   nothing measured about the policy generalises.
1. `n18ramp`. If it lands on the good branch the "no window" conclusion is an
   artifact of cold starts and the sweep should continue upward. If it lands
   congested, the conclusion stands and, per the pre-stated prediction, do not
   reach for a third knob.
2. CONC=15, to pin the knee.
3. `max_deferral_seconds` is worth its own PR regardless of how this resolves:
   on the congested branch it took drop rate 45.2 % to 12.8 % and recompute
   23.4 % to 11.9 %.
