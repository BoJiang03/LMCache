# The L1 sweep: R1, R2, R3

> **Superseded in part by record 8.** Every round here ran at `CONC=32`,
> which is past congestion collapse for this workload: throughput 6.4 k
> input tok/s against 15.7 k at 10 lanes, and TTFT p50 265-304 s against the
> corpus's recorded production 2.64 s. **Section 2 (serving performance) is
> void** -- a 2-5 % TTFT delta on a queue-dominated 300 s baseline is not
> interpretable. The store-side results in sections 1, 3, 5 and 6 rest on L1
> churn, which is driven by request volume rather than by the queue, so they
> are expected to hold, but they are re-measured at a faithful operating
> point before they are relied on.

The sweep record 4 designed, on the workload record 5
calibrated. Paired arms, same node, same seed, `CONC=32 DUR=1800 GRACE=600`.
Eager on GPU4+5, lazy on GPU6+7, both rounds.

## 1. The rounds

| | R1 eager | R1 lazy | R2 eager | R2 lazy |
|---|---|---|---|---|
| L1 | 32 G | 32 G | 96 G | 96 G |
| requests | 89 | 92 | 91 | 91 |
| served input tokens | 15.38 M | 15.50 M | 15.41 M | 15.70 M |
| running_mean | 7.95 | 7.78 | 7.73 | 7.73 |
| kv_mean | 75.2% | 76.5% | 76.1% | 77.6% |
| **store ops** | 5587 | **149** | 5647 | **217** |
| **tokens stored** | 19.65 M | **3.37 M** | 20.03 M | **10.65 M** |
| **stored / served** | 1.28x | **0.218x** | 1.30x | **0.678x** |
| store time | 234.5 s | 17.4 s | 250.2 s | 79.1 s |
| store p50 tokens | 8 192 | 20 224 | 8 192 | 75 776 |
| L1 watermark crossings | 344 | 57 | 116 | 65 |
| l1_gib at end | 23.87 | 23.58 | 63.36 | 73.68 |
| **retrieves** | **0** | **10** | **0** | **22** |
| tokens retrieved | 0 | 163 072 | 0 | 743 168 |
| external hit rate (mean/max) | 0.00% / 0.00% | 0.40% / 1.10% | 0.00% / 0.00% | **4.05% / 5.50%** |
| gpu prefix hit (mean) | 4.1% | 3.5% | 3.4% | 2.6% |
| preempted / resumed | 0 / 0 | 16 / 16 | 1 / 1 | 19 / 19 |

Errors on all four arms: `cudaMemcpy failed`, `AcceleratorError`,
`cudaErrorInvalidValue`, `Traceback` -- zero.

Ledgers balance. R1 lazy: `2189 + 106 + 59 + 462 = 2816 = admitted`.
R2 lazy: `2019 + 182 + 11 + 467 = 2679 = admitted`.

The two arms of a round are matched: request counts within 3, served input
tokens within 2%, `running_mean` within 0.2, `kv_mean` within 1.5 points. The
policy is the only thing that differs.

## 2. Serving performance

medD in milliseconds is not readable on its own here, because **TTFT p50 is
265-304 seconds**. At 32 lanes the scheduler is running ~8 and holding ~10
waiting, and a prompt is 107 k tokens at p50 and 325 k at p90, so TTFT is
mostly queue plus prefill. A 1 400 ms medD against a 300 s baseline is half a
percent; the figure has to be relative to mean anything.

**Throughput is identical between arms.** This is the first thing to say,
because it is what makes the store reduction free:

| round | arm | req/min | input tok/s | output tok/s | span |
|---|---|---|---|---|---|
| R1 | eager | 2.24 | 6 440 | 48.1 | 2 388 s |
| R1 | lazy | 2.30 | 6 469 | 53.2 | 2 396 s |
| R2 | eager | 2.29 | 6 462 | 37.0 | 2 385 s |
| R2 | lazy | 2.28 | 6 558 | 52.6 | 2 394 s |
| R3 | eager | 2.25 | 6 283 | 35.2 | 2 321 s |
| R3 | lazy | 2.34 | 6 698 | 36.5 | 2 255 s |

Per-arm latency distributions, profiling phase only:

| round | arm | TTFT p50 | p90 | p99 | latency p50 | p90 | TPOT p50 |
|---|---|---|---|---|---|---|---|
| R1 | eager | 304.3 s | 675.8 s | 746.2 s | 529.7 s | 922.0 s | 78.2 ms |
| R1 | lazy | 292.1 s | 668.7 s | 740.1 s | 435.4 s | 871.9 s | 45.8 ms |
| R2 | eager | 304.3 s | 579.7 s | 751.6 s | 450.6 s | 811.9 s | 86.5 ms |
| R2 | lazy | 271.8 s | 613.0 s | 722.6 s | 516.2 s | 885.4 s | 81.3 ms |
| R3 | eager | 265.7 s | 641.4 s | 717.4 s | 397.7 s | 806.9 s | 55.2 ms |
| R3 | lazy | 255.3 s | 574.8 s | 738.5 s | 434.1 s | 753.3 s | 61.1 ms |

The replay is closed-loop, so the arms do not serve identical request sets and
these are descriptive. The controlled view is paired on
`(conversation_id, turn_index)`, lazy minus eager, negative = lazy faster:

| round | matched | med dTTFT | rel | p25 rel | p75 rel | lazy faster | med dLatency | rel |
|---|---|---|---|---|---|---|---|---|
| R1 | 88 | -1 707 ms | **-1.94%** | -8.2% | +0.4% | **73%** | -92 ms | -0.75% |
| R2 | 87 | -1 331 ms | **-5.02%** | -13.3% | +6.1% | **69%** | +263 ms | +0.77% |
| R3 | 85 | -2 964 ms | **-4.53%** | -11.7% | +3.7% | **71%** | -2 222 ms | -5.07% |

Read together:

- **TTFT: a consistent 2-5% median improvement**, in the same direction in all
  three rounds, on 69-73% of matched turns. The distribution is skewed -- the
  better quartile is 8-13% faster, the worse quartile 0.4-6% slower.
- **End-to-end latency: a wash** at 32 G and 96 G (median within +-0.8%, win
  rate 48-51%), -5.07% at 160 G. The TTFT gain does not reliably carry through
  the decode.
- **TPOT: noise.** R1 favours lazy 78 -> 46 ms, R3 favours eager 55 -> 61 ms.
  Nothing here separates the policies.

So the standard was *no give-back*, and that is what the data supports:
throughput identical, TTFT slightly better, full latency neutral. The result
of this work is the store-side reduction in section 3, not a latency win --
latency is the constraint it clears, not the prize.

## 3. What the two points say

**Eager's write volume does not depend on L1.** 19.65 M tokens at a 32 G
budget, 20.03 M at 96 G -- it stores everything either way. **Lazy's does**:
3.37 M at 32 G, 10.65 M at 96 G.

Converted to bytes (98 304 B/token across both ranks), against the budget:

| round | L1 | eager written | x capacity | lazy written | x capacity |
|---|---|---|---|---|---|
| R1 | 32 GiB | 1 799 GiB | 56x | 309 GiB | **9.7x** |
| R2 | 96 GiB | 1 834 GiB | 19x | 975 GiB | **10.2x** |

Lazy lands on ~10x its L1 capacity in both rounds while eager's multiple falls
only because the denominator grew. That is the adaptive behaviour the policy
was for, measured: it writes what the tier can hold a working set of, not what
the engine happens to evict.

**Eager stored 1.8 TiB into L1 across each round and served zero retrieves.**
Lazy stored a fraction and served 10 and 22. Both arms run the same connector
and the same lookup path -- `_process_retrieve_requests` is called on every
request regardless of policy -- so eager is querying L1 and missing every
time. The residency arithmetic explains it without needing anything else: at
19-56x capacity nothing eager writes is still there when a session comes back.
Lazy writes fewer, larger, later-evicted objects (store p50 75 776 tokens at
96 G versus eager's 8 192) and they survive.

This is the L0/L1 duplication half of the goal, and it shows up as the
external hit rate: 0.00% eager, 4.05% lazy at 96 G.

## 4. The cost side, stated plainly

**Lazy is preempted and eager is not**: 16 vs 0 at 32 G, 19 vs 1 at 96 G.
`_has_preemption_reqs` is called unconditionally in `build_connector_meta`
(`lmcache/integration/vllm/lmcache_mp_connector.py:982`), so this is not a
lazy-only log line -- vLLM genuinely preempted under lazy and not under eager.
The mechanism is the obvious one: a deferred store keeps GPU blocks pinned, so
there is less free space to schedule into. `lazy_offload_danger_floor_max_blocks`
was 0 in both rounds, i.e. the guard built for exactly this was off.

The paired TTFT delta says the preemptions did not eat the gain in any
round. It is still a real cost, and the floor is worth a round of its own.

## 5. R3, and the prediction it broke

Record 4 put the top of the sweep at 256 G per arm. That did not fit: another
user holds ~420 GB of node 1, leaving 383 GB free, and 2 x 256 G = 512 GB
would push part of one arm's L1 onto node 0. A cross-socket asymmetry between
paired arms is the one thing the within-round pairing exists to prevent.
**R3 ran at 160 G per arm** (320 GB, 63 GB of headroom), making the sweep
32 / 96 / 160 -- 1 : 3 : 5.

"L1 large enough that eager stops churning" is not reachable on this box.
Eager writes ~1.8 TiB per 30-minute run; even 256 G would be 7x capacity.

| | R3 eager | R3 lazy |
|---|---|---|
| requests / served input | 87 / 14.58 M | 88 / 15.11 M |
| running_mean / kv_mean | 7.95 / 77.7% | 7.05 / 74.7% |
| store ops | 5580 | **232** |
| tokens stored | 19.71 M | **12.79 M** |
| stored / served | 1.35x | **0.847x** |
| store time | 256.1 s | 110.3 s |
| L1 watermark crossings | 66 | 45 |
| **retrieves** | **4** | **34** |
| tokens retrieved | 348 416 | 1 467 904 |
| external hit (mean / max) | 2.11% / 3.80% | **7.38% / 11.30%** |
| preempted / resumed | 0 / 0 | 10 / 10 |

Ledger balances: `2073 + 275 + 22 + 233 = 2603 = admitted`. Errors zero.

Paired TTFT, lazy minus eager: median -2 964 ms = -4.53%, lazy faster on
71% of 85 matched turns; end-to-end latency -5.07%.

**The ~10x-capacity rule does not survive the third point**, and it was
written down here before the round landed, so: predicted ~17.4 M tokens for
lazy at 160 G, measured 12.79 M -- 7.3x capacity, not 10x. The predicted
convergence to a 1.15x gap did not happen either; the gap is 1.54x.

| L1 | eager written | x cap | lazy written | x cap | eager/lazy |
|---|---|---|---|---|---|
| 32 GiB | 1 799 GiB | 56.2x | 309 GiB | 9.6x | **5.83x** |
| 96 GiB | 1 834 GiB | 19.1x | 975 GiB | 10.2x | **1.88x** |
| 160 GiB | 1 805 GiB | 11.3x | 1 171 GiB | 7.3x | **1.54x** |

Lazy converges toward eager as L1 grows, but more slowly than proportional
writing would give, and it is still writing a third less at 160 G.

## 6. The sweep, read across the three rounds

| L1 | eager stored | lazy stored | ratio | eager retr | lazy retr | eager ext% | lazy ext% | med dTTFT | lazy preempts |
|---|---|---|---|---|---|---|---|---|---|
| 32 G | 19.65 M | 3.37 M | 5.83x | 0 | 10 | 0.00 | 0.40 | -1.94% | 16 |
| 96 G | 20.03 M | 10.65 M | 1.88x | 0 | 22 | 0.00 | 4.05 | **-5.02%** | 19 |
| 160 G | 19.71 M | 12.79 M | 1.54x | 4 | 34 | 2.11 | 7.38 | -4.53% | 10 |

Four things the shape says:

1. **Store reduction is largest where the tier is tightest** and decays as L1
   grows -- 5.83x, 1.88x, 1.54x. That is the pressure-relief case, and it is
   strongest exactly where relief matters.
2. **Retrieval goes the other way.** Lazy's retrieves rise 10 -> 22 -> 34 and
   its external hit rate 0.40% -> 4.05% -> 7.38%. Storing less does not cost
   hits; it buys them, because what is stored survives.
3. **Eager needs 160 G before it retrieves anything at all** (4 retrieves,
   2.11%). Below that it writes 1.8 TiB per round into L1 for nothing. Lazy is
   already retrieving at 32 G.
4. **The latency effect is small and flat**: median paired TTFT -1.94%,
   -5.02%, -4.53% across the three L1 points, on 69-73% of turns, with
   throughput identical. Section 2 has the full picture.

Lazy's preemptions fall as L1 grows (16 / 19 / 10), consistent with deferred
stores draining faster when the tier has room to take them.

## 7. R4

`off` (no connector) against lazy at **96 G** -- the L1 where lazy's paired
TTFT gain was largest across R1-R3. 96 G also repeats R2's point on the same
slot, so R4's lazy arm doubles as the cross-round comparability check record 4
asked for: R2 lazy s2 and R4 lazy s2 differ only in which round they ran.

Still unverified: the slot-equivalence assumption. Every policy comparison
here puts eager/off on GPU4+5 and lazy on GPU6+7, resting on i60L's 92 ms
spread between identical arms -- measured under the old configuration, not
this one. A lazy-vs-lazy round at 96 G would settle it and re-use the same
cross-round point.
