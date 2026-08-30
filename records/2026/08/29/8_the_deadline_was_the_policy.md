# The deadline was the policy

Record 7 designed an eager/lazy comparison on the L1 axis and asked two
questions before running it. Both were answered: this round is a conclusion for
the lazy-offload half, move-not-copy comes after it, and `off` does not earn a
slot because eager beats it by construction. The matrix started, and was
stopped one round in, because the arm it was measuring turned out not to be
the policy.

## 1. Ninety-five percent of stores fired on a clock, not on eviction

Every arm on this base ran `DEFER_SECS=30`. The ledger has a counter for
exactly this question. `emitted_overdue` counts operations released because
they hit `max_deferral_seconds` rather than because a block came due, and
`eviction_aware.py:527` says how to read it: near zero means the danger window
fires inside the deadline and the bound costs nothing, a majority means the
free queue drains far slower than the workload reuses and the spatial signal
would have been late on most of the traffic.

| arm | emitted | emitted_overdue | ratio |
|---|---|---|---|
| l72b64L192 (DEFER=30) | 5,223 | 4,969 | 95.1% |
| f8k256c72b64 (DEFER=30, L1=320) | 4,630 | 4,423 | 95.5% |

The two clocks are two orders of magnitude apart. The danger window looks
`horizon_steps` 2.5 ahead, which at 87.5 ms per step is 0.22 s; the deadline is
30 s, 137 times longer, so the deadline always fires first. Worse, 30 s lands
inside the dominant reuse band: 497 of 561 requests with a predecessor reuse
within 120 s, so the deadline was copying blocks to L1 while L0 still held them
and was about to serve them locally.

Every result through record 7 therefore measured "eager, delayed 30 s".

## 2. The DEFER axis

One variable, everything else identical to the round-1 lazy arm. L1=192 chosen
because eviction pressure is highest there and round 1 already supplies the
paired eager and DEFER=30 cells.

| | eager | d30 | d0 | d120 |
|---|---|---|---|---|
| emitted_overdue/emitted | -- | 95.1% | 0% | 84.3% |
| compute | 23.1% | 22.6% | 19.1% | 20.2% |
| local | 40.7% | 40.3% | 42.8% | 44.9% |
| ext | 36.1% | 37.1% | 38.1% | 34.8% |
| total hit | 76.8% | 77.4% | 80.9% | 79.7% |
| profiling write tokens | 16,064,768 | 14,638,848 | 10,975,488 | 13,737,984 |
| store events | 9,152 | 2,586 | 796 | 1,310 |
| l1_gib | 149.37 | 145.11 | 134.51 | 149.95 |
| write amp (written/retrieved) | 0.720 | 0.633 | 0.440 | 0.623 |
| ext per GiB resident vs eager | -- | +5.8% | +17.2% | -4.0% |
| drop_rate | -- | 3.0% | 20.4% | 5.7% |
| waiting_mean | 3.92 | 3.87 | 3.49 | 3.87 |
| tpot p50 | 89.1 ms | 87.5 ms | 87.1 ms | 81.2 ms |
| TTFT p50 | 10.53 s | 9.86 s | 9.35 s | 9.13 s |
| preempt | 6 | 3 | 11 | 7 |

d0 against eager: writes -31.7%, store events -91.3%, L1 residency -9.9%,
compute -4.0 pt, waiting_mean -11.0%. No axis moved the wrong way.

A 20.4% drop rate did not cost hit rate because the drops are not the point.
With the deadline gone the policy writes once, late, large and complete: mean
store grew from 16,306 to 34,972 tokens while store events fell from 2,586 to
796, and `covered_prefix_advances` went 100 to 214. A late store of a longer
prefix subsumes the early fragments, so losing the fragments costs nothing.

### Predictions F1-F5

- F1 half wrong. d0's `emitted_overdue` is 0 as required, but d120 still sits
  at 84.3%. Even a 120 s net is inside the reuse band. Intermediate deadlines
  are not a compromise, they are the same failure at lower amplitude.
- F2 confirmed: -25.0% writes against d30, predicted at least 20%.
- F3 numerically right, threshold wrong. drop_rate did rise, to 20.4%. The
  pre-stated reading, that above 15% the spatial signal is unusable and the fix
  is code, rested on drops costing hit rate. Hit rate rose 3.5 pt. The
  threshold is withdrawn.
- F4 confirmed: d120 sits between d0 and d30 on both write reduction and drop
  rate.
- F5 wrong. d120's ext fell to 34.8%, outside 37% +/- 1.5. It moved hits from
  L1 to L0 rather than losing them (local 44.9%, the highest of the four).

### A decomposition that only works at short deferral

Record 7 section 5 said to split the write reduction into
`covered_prefix_tokens_skipped` (real dedup) and `dropped_*` (writes that never
happened). At d30 those two summed to 1,455,616 against a measured 1,425,920
difference and the split looked exact. At d0 `covered_prefix_tokens_skipped` is
16,530,432, larger than the arm's entire write volume. The docstring already
said why: it is a weight, not an outcome, and stands outside the admission
ledger's arithmetic. It double counts a long prefix probed repeatedly. The only
defensible statement at d0 is that of the 5,089,280 tokens not written relative
to eager, 2,602,752 (51%) are `dropped_evicted` and the remainder is genuine
skipping.

## 3. Where lazy's TTFT gain actually comes from

TTFT p50 10.53 -> 9.35 s.

| | eager | d0 | |
|---|---|---|---|
| compute tokens per request | 23,371 | 19,399 | -17.0% |
| compute tokens in window | 14,290,677 | 12,525,766 | -12.4% |
| completed throughput | 0.2513 | 0.2635 | +4.8% |
| waiting_mean | 3.92 | 3.49 | -11.0% |
| mean queue wait (Little) | 15.60 s | 13.25 s | -15.1% |

The saving is 1,764,911 tokens of recompute, about 294 s of engine prefill time
returned over a 2,308 s window, 12.7% of it. Roughly half goes into each
request's own prefill and half into draining the queue, which matches the
measured 1.18 s.

What did not move is what makes the attribution stand: inflight_mean 30.81 to
30.85, kv_mean 67.7% to 66.7%, tpot 89.1 to 87.1 ms. Block occupancy and decode
speed are unchanged, so the gain cannot come from holding fewer blocks or
decoding faster. The store path is not the source either: d0's individual
stores are slower (p99 0.232 -> 1.442 s) and are async.

Hit rate does not reduce block occupancy. A request's KV must sit in GPU blocks
during decode whatever tier it came from. That is the whole reason the TTFT
effect is second order at this working point.

## 4. TTFT is 7 parts queue, 3 parts work

Three independent lines agree.

`ttft_by_isl` is non-monotone in every arm. For d0, requests under 50k wait
15.4 s while 50-150k wait 4.6 s. Work cannot produce that ordering.

Little's law gives 13.2 s of mean admission wait for d0, larger than the 9.35 s
median TTFT, against a p90 of 54.31 s.

Scaling the uncongested branch: c60 had waiting_mean 0.09 and TTFT p50 1.04 s at
compute 7.7%, essentially pure work. At d0's 19.1% that is about 2.6 s of work,
leaving about 6.8 s of queue, 72% of the median.

Blocks are released only at completion, and decode holds them for about 83% of
a request's life (osl p50 389 x tpot 87.1 ms = 33.9 s, tail to 46 s, against
2.6-3.6 s of prefill). So the queue is waiting for in-flight requests to finish
decoding, and no storage-side lever reaches it.

## 5. Three quarters of the prefill compute is retention failure

`irr.py` (new, in this session's scratchpad) splits compute by asking what an
earlier turn already materialised. For turn N with predecessor P,
`reusable = min(isl[N], isl[P] + osl[P])` and the rest is new by construction.

| arm | compute | irreducible | retention failure |
|---|---|---|---|
| eager | 23.1% | 5.2% | 17.9 pt |
| d30 | 22.6% | 5.2% | 17.4 pt |
| d0 | 19.1% | 5.1% | 14.0 pt |
| d120 | 20.2% | 4.9% | 15.3 pt |

For d0 that is 9,220,751 of 12,525,766 computed tokens, 73.6%, recomputing
something that existed before. The 5.1% floor is 1.7% cold first turns (50
requests with no predecessor anywhere in the run) plus 3.3% per-turn increment.

Consistency check: 94.9% of presented tokens are theoretically reusable and the
two tiers served 80.9%, so they already capture 85% of what is available.

Caveat: the split assumes turn N's prompt begins with turn P's prompt plus
completion. `rejected_prefix_broken` is 0, which supports it, but truncation or
editing in the corpus would overstate reusable.

## 6. Two causes, both capacity, neither the policy

`where.py` (new) buckets reusable tokens and L1 supply by reuse interval. d0:

| interval | requests | reusable tok | L1 supplied | coverage |
|---|---|---|---|---|
| 0-120 s | 501 | 55,368,900 | 20,845,056 | 37.6% |
| 120-300 s | 33 | 3,507,496 | 1,302,528 | 37.1% |
| 300-600 s | 12 | 1,563,217 | 668,160 | 42.7% |
| 600-1200 s | 14 | 1,570,236 | 198,912 | 12.7% |
| 1200+ s | 1 | 119,395 | 0 | 0% |

Cause A, the evicted tail. Beyond 120 s, 60 requests carrying 11% of the
reusable volume get 32% L1 coverage and essentially nothing from L0, so almost
all of it is recomputed: 40-50% of the run's total recompute from 11% of the
traffic. The working set at 406 GiB is 2.6x the 154 GiB watermark, so anything
untouched for two minutes is gone. The 12.7% coverage at 600-1200 s is where
the eviction frontier sits.

Cause B, instantaneous capacity. The 0-120 s band carries 89% of the reusable
volume and still recomputes 11.6% of it after L1's 37.6% and L0's share. The
two tiers cannot hold enough at once: L0's pool is 4.08M tokens of which 66.7%
is live blocks, leaving about 64 GiB for caching, plus L1's 134.5 GiB resident,
against a 406 GiB working set.

Neither is the policy dropping data. eager drops nothing and recomputes 12.6%
in the same band against d0's 11.6%.

### L0 is the more efficient tier per byte

| tier | cache capacity | share of working set | supplied |
|---|---|---|---|
| L0 | ~64 GiB | 15.5% | 42.8% |
| L1 | 134.5 GiB | 33.1% | 38.1% |

L0 supplies 2.4x more per byte because 89% of reuse happens within 120 s and L0
holds the most recent blocks by construction.

This was first written up as an argument against move-not-copy, on the grounds
that moving content out of L0 weakens the most efficient tier. That is wrong
and is retracted. The policy stores at the danger window, which is the moment
the block is already in the free queue and about to be recycled, so the move
takes only what L0 has already given up. `emitted_overdue=0` on the d0 arm is
the direct evidence that every store now happens at that moment.

The cost of move-not-copy is elsewhere and stands: today a block retrieved from
L1 stays in L1, so when L0 later evicts it `covered_prefix` skips the write. If
retrieval deletes from L1, that write has to happen. Upper bound on the extra
traffic is the retrieved volume, 24,939,968 tokens against a current total
write volume of 10,975,488. The gain is the duplicated stock, roughly 40-100
GiB against L1's 134.5 GiB resident, which is 30-75%. That interval is too wide
to design against, and narrowing it needs the point-in-time L0-intersect-L1
probe that was deferred at the start of this session. It is now a prerequisite
for move-not-copy, not an option.

## 7. ext hit and queue are the same variable

Lookup prefers L0, so ext hit is only positive when L0 misses, and L0 misses in
proportion to how little of the pool is spare. Spare fraction is 1 - oversub,
and oversub is what produces the queue. So on a single node with an inclusive
tier:

    ext hit up  <=>  L0 spare down  <=>  oversub up  <=>  queue up

c60 demonstrated it: waiting_mean 0.09 and TTFT p50 1.04 s with ext at 5.6%.
No configuration knob separates the two, because lowering CONC, shrinking the
pool with `--num-gpu-blocks-override` and adding memory all only move oversub.
The only escape is a working set that grows while in-flight count does not,
which is a property of the corpus.

`NO_SCENARIO=1` was proposed as that escape and then withdrawn. The claim in
`arm.sh` that the scenario compresses think time from ~400 s to ~89 s is not
supported at these concurrencies. The cap fires only when all sessions are
globally idle, which at CONC=72 has probability around 1e-18, and the measured
duty cycles show implied idle time rising with load (43 s at n14, 46 s at c60,
74 s at c72) rather than falling. `seq60.sh`'s comment is the correct one.
Dropping the scenario would also drop agentic_replay, ignore_eos, the
first-turn cache bust and no-truncation, and would make the run
non-certifiable. It is more fake and buys nothing. No arm has ever run with
`scenario=OFF`.

The consequence is that c72 is the only working point this corpus offers where
a second tier is load-bearing, and the demonstration should say so rather than
look for a friendlier setting.

## 8. Corrections made this session

- `miss.py` hardcoded `B = 98304`, the bf16 bytes per token, on runs whose KV
  is fp8 at 49,152. Every GiB figure and every oversubscription ratio it
  printed for the fp8 arms was 2x. The working set is 401 GiB, not 802, and
  oversubscription at L1=192 is 2.61x, not 5.22x. `l1_gib` comes from the
  server's `memory_used_bytes` and was never affected. The constant now reads
  from `KVB` with the old default kept for the bf16 archive.
- The claim that `CACHE_WARM` had never been used is wrong. Three `k20_*cw_*`
  arms ran `cachewarm=600`, and the L1=384 one reached 277 GiB.
- The pessimism in section 6 about moving content out of L0 is retracted, see
  above.
- The estimate that speculative decoding would land TTFT near 4.7 s treated all
  of TTFT as queue. With the work term separated the projection is about 6 s.

## 9. Two arms in flight

Started 22:58. TTFT decomposes into a queue term and a work term, so one arm
attacks each; they are run apart because the coefficients have to be measured
before they can be added.

- `l72b64L192d0spec`, slot 1. Only `SPEC_CFG` added, ngram with 4 draft tokens,
  the same config the `dp_ngram` probe used. Note its KV pool is 3,980,736
  tokens against the control's 4,081,024, 2.5% smaller, because drafting
  reserves memory.
- `l72b64L512d0cw`, slot 2. `L1_GB=512` with `CACHE_WARM=900`. 512 puts the
  0.80 watermark at 410 GiB against a 406 GiB working set, so the tier should
  stop evicting. L1/L0 = 2.74.

Predictions G1-G8 are in `f8k256_predictions.md`, written before either arm
started. G4 is a kill condition: if `tpot_p50` stays above 80 ms the draft
compute is eating the bandwidth saving at this batch size and the lever is dead
at CONC=72. G1 already looks wrong in the favourable direction, the engine log
reporting acceptance length 3.39 against a predicted 1.4-1.8.

The L1 arm moves two knobs and a compute drop is not cleanly attributable. The
confound is bounded and its direction known, since at L1=192 the tier is
already at its watermark, so the control to run next if G6 lands is L1=192 with
`CACHE_WARM=900`, not a repeat of 512.

## 10. The matrix that follows

The eager/lazy comparison is worth running, on `DEFER=0` rather than the
crippled 30 s configuration, over L1/L0 in [1,2]. That band is exactly the
region where L1 evicts:

| L1/L0 | L1 | 0.80 watermark | oversubscription vs 406 GiB |
|---|---|---|---|
| 1.03 | 192 GiB | 154 GiB | 2.64x |
| 1.37 | 256 GiB | 205 GiB | 1.98x |
| 1.71 | 320 GiB | 256 GiB | 1.59x |
| 2.0 | 373 GiB | 299 GiB | 1.36x |
| 2.72 | 508 GiB | 406 GiB | eviction stops |

The eviction-free boundary is 2.72, just above the band, so selectivity is
meaningful across all of it and no cell degenerates. 192 already has both
policies; 256 and 320 need four arms in two rounds. Three points are enough for
the iso-hit read, which is the presentation that says the result in GiB: the L1
eager would need to match lazy's hit rate. A fourth point at 2.0 is not worth a
round, since oversubscription only moves from 1.59 to 1.36 across it.

This separates cleanly from the TTFT story, which lives at L1=512 and ratio
2.74. The policy is worth most where L1 is tight; TTFT is worth most where L1
is generous. Two conclusions, stated apart.

## 11. Open

- G1-G8 to score when the two arms land.
- The eager/lazy matrix at 256 and 320, on DEFER=0.
- Whether the working point moves off the congested branch if speculative
  decoding succeeds. If `ttft_by_isl` turns monotone the system is prefill
  dominated, which is where lazy's effect stops being diluted: at fixed L1 the
  compute gap of 23.1% to 19.1% would map to TTFT nearly one for one, -17.3%
  against the measured -11.2%. The correct response then is to raise CONC until
  the working point sits back at the knee, not to celebrate the low TTFT.
- The L0-intersect-L1 probe, now a prerequisite for move-not-copy.
- `max_deferral_seconds` still deserves its own PR on a `_pr` branch. Note that
  this session's result argues for shipping the default as 0.
