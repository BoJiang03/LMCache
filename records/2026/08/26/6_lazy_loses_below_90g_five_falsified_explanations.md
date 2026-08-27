# Lazy loses below L1=90G, and five explanations are dead

Follow-on to record 4 (which concluded "lazy wins on this workload" from 90G
data alone) and record 1 (whose L1-sweep latency cells were invalidated by
measurement drift). This record re-measures those cells in the parallel harness
and reverses record 4's scope.

## 1. The result: 90G is not representative

In-round paired comparison, agentx compliant workload, 4 arms simultaneously,
slots rotated. `sumTTFT` per arm in seconds.

| L1 | eager arms | lazy arms | lazy - eager | within-config spread | lazy medD |
|---|---|---|---|---|---|
| 90G | 285.8, 282.4, 283.5 | 243.5, 253.2 | **-35.5s (lazy wins)** | eager 3.4 | +9, +2 ms |
| **60G** | 254.2, 261.7 | 270.6, 267.8 | **+11.2s (lazy loses)** | **eager 7.5, lazy 2.8** | **+18, +19 ms** |
| 30G | 326.4, 347.7 | 392.1, 360.2 | +39.1s (lazy loses) | eager 21.3, lazy 31.9 | +3, +4 ms |

**The trustworthy negative is 60G, not 30G.** 30G's +39.1s sits inside a 21-32s
within-config spread and cannot be resolved 2v2. 60G's +11.2s sits against
spreads of 2.8 and 7.5s, and the *median* moves with it: +18 and +19 ms against
the eager-vs-eager control's +4 ms. Half of the 60G deficit (270 requests x 19ms
= 5.1s) is a flat per-request tax; the rest is tail.

Correction to the reply I gave mid-investigation: I quoted the 60G deficit as
+15.0s. The arm numbers give +11.2s.

## 2. The reframing this forces

L1 watermark events -- how often L1 had to run its evictor:

| L1 | eager | lazy | lazy verdict |
|---|---|---|---|
| 90G | 6 | 6 | wins |
| 60G | 14, 14 | 13, 13 | loses |
| 30G | 52, 53 | 38, 36 | loses |

**Lazy wins only in the regime where L1 is large enough that it barely evicts.
It loses in every regime where L1 actually has to make eviction decisions.** An
eviction-aware policy failing exactly where eviction matters is the opposite of
the intended shape, and it is a material limitation on record 4's conclusion.
Record 4 should be read as "lazy wins at L1=90G on this workload", not "on this
workload".

Record 1's *volume* rows survive and are consistent: lazy's L1 hygiene is better
at every L1 size (at 30G it retrieves 1.02 Mtok mean against eager's 0.44 and
stores 210 ops against 1100). Better hygiene is simply not buying latency below
90G.

## 3. Five explanations, all falsified

Each was a hypothesis I stated before measuring, and each died on a number.

| Hypothesis | What killed it |
|---|---|
| Emission pins blocks out of the free queue -> allocation starves -> queue backs up | `Waiting` queue is 0 at p50 **and** p90 in every arm at every L1, max 1-2. Nothing ever queues. |
| Preemption pays the bill (lazy 3-5 events vs eager 1-2) | Log timestamps give 0.5-0.9s per preempted->resumed. Five events is ~4s, not 39s. A marker, not the bill. |
| L1 store/retrieve got slow under pressure | Server-side totals: retrieve 0.76-1.60s, store 4.4-5.7s for the whole run -- and **eager's store total is higher** (8.3-10.4s). |
| `prepend=True` converts free GPU hits into paid L1 hits | At 60G lazy is higher on **both** axes: APC +9.9 points, EXT +7.2. Not a swap. |
| Lazy computes more prefill | At 60G lazy computes the same (1.83 vs 1.81 Mtok) and still loses. At 30G it computes 0.19 Mtok **less** (~40s of compute) and loses anyway. The sign is backwards: the L1 where lazy saves the most prefill is where it loses the most. |
| Policy Python on the scheduler thread (`blocks_validated` rises 738k -> 826k -> 1.27M as L1 shrinks) | Per-request decode rate, n=44-70 intervals per config: 95-99 tok/s/req everywhere, spread 0.2-4.0% with fully overlapping IQRs. Decode rate is pure step-rate, so **scheduler step time is unaffected by lazy.** |

## 4. Instrument integrity problem, recorded because it invalidates sizing

Two whole-run aggregates disagree by 2.3x on a fixed replay:

- vLLM's `Prefix cache hit rate` + `External prefix cache hit rate` say lazy at
  60G covers 17 points more of the prompt than eager.
- The integral of `Avg prompt throughput` says lazy computes the same number of
  prompt tokens as eager (1.83 vs 1.81 Mtok).

With identical ISL per request (`dIsl% = 0.0` in every pairing) both cannot be
true. Either the throughput counter is not "tokens actually forwarded", or the
hit rates are not token-weighted the way I read them. **Four of the five
hypotheses above were built on these aggregates.** No magnitude claim derived
from them is safe until this is resolved -- resolving it is a prerequisite, not
a footnote.

## 5. The puzzle, stated as tightly as the data allows

At L1=60G:

- 53 of ~270 requests retrieve from L1. **80% of requests never touch LMCache's
  data path** -- yet the median request is 18-19 ms slower. A cost borne by 20%
  of requests cannot move the median.
- The scheduler step rate is identical, so it is not policy overhead.
- Nothing queues, so it is not admission or free-block starvation.
- Prefill volume is identical, so it is not lost cache coverage.

So the tax is paid by requests that neither retrieve nor store, in a path whose
step rate did not change, without queueing, at equal compute. That is a narrow
box and none of the six mechanisms above fits inside it.

One family not yet examined: lazy changes the *free-queue order* (blocks come
back with `prepend=True`), so the identity of the blocks every request is
allocated differs, which changes GPU prefix-cache content for all 270 requests
including the ones that never talk to LMCache. That is a state difference rather
than an overhead, and it is the only lazy-vs-eager difference found so far that
touches every request. It predicts the median shift; it does not obviously
predict the *sign*, since APC measured higher for lazy.

## 6. The decisive isolation test (queued, prediction stated in advance)

Lazy differs from eager in exactly two ways:

- **(a)** stores are deferred and emitted in batches under pool pressure
- **(b)** a completed store's blocks are returned with `prepend=True`, donating
  them to the eviction head

`StoreReleasePlacement.LRU_TAIL` turns off (b) and leaves (a) intact, so it
splits the difference cleanly.

`r60_*` round, **L1=60G** where the spread is 2.8-7.5s rather than 30G's 21-32s:
2 plain lazy vs 2 `lru_tail`, one round, no cross-round comparison needed. Plain
lazy sits on slots 0 and 3, the same slots it had in the `n60_` round, which
buys a free cross-round consistency check.

- If `lru_tail` recovers the ~11s, the cost is the block-release placement.
- If `lru_tail` matches plain lazy, placement is innocent and the cost is in the
  deferral itself.

Note this reverses the 90G finding, where `eviction_head` beat `lru_tail` by
14.5s (record 4). If placement is the cost at 60G, then the placement default is
L1-size dependent and cannot be a fixed default.

Also in flight: `q30_*` at 30G (`lru_tail`, `max_pending_ops=64`, plain lazy,
eager), launched 12:11 before the 60G reasoning above was complete. It lands on
the noisy L1 point, so it is a reference reading, not a judgement.

## 7. Housekeeping

- No commit. Working tree clean at `1c43ca02`; no code changed in this
  investigation, which was entirely measurement. `records/` stays untracked per
  `/home/bo/LMCache/.git/info/exclude:19`.
- Discipline from record 5 held: no working-tree edits while a round was in
  flight.
- GSM8K correctness sweep (record 5 section 3) finished clean before the 60G and
  30G analysis began, so no code-integrity doubt hangs over these readings.
