# How much is lazy worth at a realistic latency?

> **Superseded by record 10.** Two things here are wrong. (1) Section 2 treats
> "working set > pool" as a latency constraint; it is not -- it is a cache-miss
> constraint, which is what an L1 tier exists to absorb, so the "no comfortable
> margin" conclusion denies L1 any reason to exist. (2) Option (b) below claims
> wall time grows the distinct-session count; measured, it does not (32 lanes
> over 1 492 s touched 27 sessions, over 2 389 s touched 28). The estimate in
> section 3 is also superseded: the payoff is session capacity, not a few
> percent of TTFT. Record 10 has the corrected causal chain, the closed-form
> lane/in-flight relationship, and the measured knee.

Record 8 established that R1-R4 ran past congestion collapse and that the
corpus's own recorded production TTFT (p50 2.64 s, p90 6.98 s) is the target.
The obvious next question is what the sweep will show once it is re-run there.
The answer turns on one number, and it is not encouraging.

## 1. Does L1 have anything to do?

Working set = the sum, over distinct conversations a run touched, of that
conversation's peak input length -- what would be resident if every session's
context were held at once. Against the 186.6 GiB GPU KV pool:

| arm | lanes | reqs | sessions | per session | working set | vs pool | L1 retrieves |
|---|---|---|---|---|---|---|---|
| r0_lazy | **10** | 94 | **11** | 13.0 G | 142 G | **0.76x** | **0** |
| cal_c32_s1 | 32 | 67 | 27 | 15.9 G | 428 G | 2.29x | 18 |
| r1_lazy_s2 | 32 | 92 | 28 | 15.7 G | 439 G | 2.35x | 10 |
| r2_lazy_s2 | 32 | 91 | 28 | 15.7 G | 439 G | 2.35x | 22 |
| r3_lazy_s2 | 32 | 88 | 28 | 15.6 G | 438 G | 2.35x | 34 |
| r4_lazy_s2 | 32 | 86 | 27 | 15.7 G | 423 G | 2.27x | 14 |
| cal_c64_s2 | 64 | 56 | 40 | 14.2 G | 568 G | 3.04x | 0 |

**At the operating point that reproduces production latency, the entire
working set fits inside the GPU prefix cache.** 0.76x. A session that comes
back hits GPU and never reaches L1. R0's zero retrieves in 900 s was not bad
luck; it is what 0.76x means.

Every round that showed L1 doing useful work was at 2.3x, and every one of
those was in the collapsed regime.

## 2. The break-even, and how tight it is

A session's working set is stable at **~14-16 GiB** across every arm. Against
a 186.6 GiB pool that puts the crossover at **~13 concurrent sessions**.

Realistic latency needs roughly **<= 14 lanes** (10 lanes gives TTFT p50
1.15 s against production's 2.64 s, so there is some headroom; 32 gives
154.7 s).

The two conditions land on the same point. There is no comfortable margin
between "L1 has a job" and "latency is real" on this node.

## 3. The estimate

For the sweep re-run at ~10-14 lanes, before it lands:

| quantity | expected | why |
|---|---|---|
| tokens written to L1 | **10-30x less than eager** | emission rate falls with GPU pressure: 2019/2679 = 78 % of admitted at 32 lanes, 215/873 = 25 % at 10. Less eviction, more deferral. |
| L1 hit rate | **~0 % for both arms** | working set 0.76x pool; nothing reaches L1 |
| TTFT, end-to-end | **no measurable difference, +-2 %** | 1.15 s is prefill-bound and neither policy touches prefill |

So the expected headline is: **lazy avoids writing ~1.5-1.8 TiB per 30 minutes
into a tier that, at this load on this node, nobody reads.** That is a real
saving in host memory bandwidth, L1 capacity and CPU, and it is a weaker claim
than improving service. The thing saved had no benefit to either arm.

The no-give-back standard would be met, but met trivially -- doing nothing at
all would also meet it.

## 4. Three ways the picture changes

**(a) 20 lanes still serves production latency.** Then the working set is
~1.5x the pool and both conditions hold at once. This is the good case and
`calib2` at 14 vs 20 is testing exactly it.

**(b) Longer runs.** *(Measured and false -- see record 10.)* The idea was
that wall time would rotate more distinct sessions through at a fixed lane
count. It does not: 32 lanes over 1 492 s touched 27 sessions, and over
2 389 s touched 28. A trace averages 145 turns and an arm serves ~90
requests, so lanes never finish a session and recycle. **Distinct sessions
~= lane count**, and wall time buys only statistics.

**(c) A smaller pool.** If neither of the above works, the GPU cache on this
node is simply larger than its own workload needs, and studying an L1 tier
requires shrinking it.

Which reframes something record 4 dismissed. **The 24 GiB pool of every
earlier round was not artificial in the way record 4 argued.** It made the
working set far exceed the GPU cache, which is the precondition for an L1 tier
to exist at all. What was artificial was the *preemption* it caused -- the
pool was too small to hold the in-flight set, not just too small to hold the
working set. Those are different failures and record 4 collapsed them into
one.

The principled pool is one that comfortably holds the **in-flight** set while
being smaller than the **working** set. At 10 lanes the in-flight KV demand is
~1.7 requests x 165 k tokens = ~26 GiB, and the working set is 142 GiB. A pool
somewhere in the 60-90 GiB range satisfies both -- but a single 1 M-token
request needs 92 GiB, and the scenario forbids input truncation, so the floor
is set by the largest request rather than by the average. That tension is the
next thing to resolve if (a) and (b) both fail.

## 5. Tooling

`workingset.py` added: distinct sessions, per-session working set, total, and
the ratio to the pool. It belongs beside `ttft.py` as a gate -- a round whose
working set is below 1.0x cannot say anything about an L1 policy, however
clean its other numbers look.

## 6. State

`calib2` running: 14 vs 20 lanes, 900 s, lazy, L1=96 G, started 09:54.
Repo clean at `f010e44f`, nothing pushed.
