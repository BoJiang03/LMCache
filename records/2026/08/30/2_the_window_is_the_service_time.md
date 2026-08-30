# The L0 cache window is the service time, and that is why nothing moved

Record 1 closed the decode and config levers and concluded that on this corpus
the external hit rate and the admission queue are one event. That conclusion
survives, but almost every number supporting it in records 6 through 8 was
wrong, and the reason they were wrong is the same in each case: a statistic
read at the wrong weighting, or two arms from different rounds spliced into a
comparison. This record corrects them, derives the identity that explains why
every capacity lever we tried returned zero, and scores the first eager/lazy
pair that was actually run as a pair.

## 1. The 10 percent TTFT gain was splicing, not lazy

The figure quoted since record 6 was eager 10.53 s against lazy 9.35 s at
CONC=72. Those two arms are not a pair.

| arm | config | slot | landed |
| --- | --- | --- | --- |
| e72b64L192 | eager | 1 (GPU 4,5) | 14:54:23 |
| l72b64L192 | lazy, DEFER=30 | 2 (GPU 6,7) | 14:54:21 |
| l72b64L192d0 | lazy, DEFER=0 | 1 (GPU 4,5) | 16:25:51 |

The first two ran simultaneously. The third is a separate round an hour and a
half later and differs from the eager arm in two variables at once. Every
figure quoted as "eager versus lazy" was the eager arm against the third.

The only simultaneous pair is e72 against l72:

| | eager | lazy (D30) | delta | noise floor |
| --- | --- | --- | --- | --- |
| ttft p50 | 10.53 s | 9.86 s | -6.4% | +-30% |
| lat p50 | 59.86 s | 57.74 s | -3.5% | |
| waiting_mean | 3.92 | 3.87 | -1.3% | +-5% |
| tpot p50 | 89.1 ms | 87.5 ms | -1.8% | +-2% |
| completed | 580 | 586 | +1.0% | |
| stores | 11202 | 2712 | -75.8% | deterministic |
| stored/prefill | 39.04% | 35.52% | -9.0% | deterministic |
| ext hit (token) | 36.19% | 37.14% | +0.95 pt | deterministic |

The queue delta drops from -11% to -1.3% and the store volume delta from -40%
to -9.0%. Both of the larger figures were borrowed from the DEFER=0 arm.

The by-ISL bucket counts confirm which comparison is clean. The pair has
102/354/124 and 102/359/124 requests in the three buckets, nearly the same
population, and all three buckets improve. The spliced comparison has
110/366/135 against 102/354/124 and the 150-400k bucket gets worse, 14.9 s to
17.1 s.

The comparison that matters, eager against lazy with DEFER=0 run
simultaneously, had never been run. Section 8 runs it.

## 2. The total hit rate is 85 percent, not 49

Record 1 section 6 and the session that followed used a "local hit mean" taken
as the arithmetic mean of the engine's per-interval `Prefix cache hit rate`
lines. That estimator weights an idle interval the same as a saturated one and
it is not usable. It gave 39.8% at c64 and 13.2% at c72.

The token-weighted figure comes from the engine's own prefill work rate:

| CONC | prompt tok/s | computed | prefill total | computed share | total hit |
| --- | --- | --- | --- | --- | --- |
| 64 | 5,548 | 11.11M | 74.6M | 14.9% | 85.1% |
| 68 | 6,199 | 12.42M | 63.8M | 19.5% | 80.5% |
| 72 | 6,585 | 13.19M | 65.4M | 20.2% | 79.8% |

At c64 that splits as L0 75.6% and L1 9.5%. The 75.6% agrees with
`gpu_prefix_hit_max=77.6%`, the statistic that had been discarded as noise.

Cross-check by predicting TTFT. At ISL p50 101,568 and 14.9% computed, 15,134
tokens at the engine's own peak prompt throughput of 38,298 tok/s is 0.40 s;
plus Wq 0.65 s gives 1.05 s against a measured ttft p50 of 1.07 s. The 49%
version predicts 2.0 s and does not fit.

## 3. The corpus is not the problem it was said to be

Block-level decomposition over the whole trace, 338.1M block references:

```
new  (first appearance anywhere)          5.7M    1.7%
hot  (in the immediately preceding
      request of the same conversation) 323.6M   95.7%
warm (seen earlier, not adjacent)         8.7M    2.6%

conversations 2090, requests 98827
turns per conversation p50 21, mean 47.3, max 2977
single-turn conversations 0.1%
first-turn requests 2.1% of requests, 0.3% of blocks
```

The ceiling is 98.3% and 95.7% of it is the easiest possible case. Record 1
section 3 said the corpus's thin long-gap tail was the binding problem. The
tail is thin, but the corpus is close to ideal for caching, and calling it
"the problem" was wrong.

## 4. The window identity

L0's prefix cache is not a capacity, it is a residency time. Derive it:

```
window        = cache contexts / arrival rate of new contexts
cache contexts = pool * (1 - f) / ctx        f = in-flight fraction
arrival rate   = running / T = [pool * f / ctx] / T
=> window      = T * (1 - f) / f
```

and f is the inverse of the load's peak-to-mean ratio, because the deployment
point is defined by peak in-flight KV reaching the pool. So

```
window = (burstiness - 1) * T = (burstiness - 1) * OSL * tpot
```

Pool size and context size cancel. Measured: (1.77 - 1) * 67.1 s = 51.7 s
against about 47 s computed from the occupancy directly.

The condition for L1 to have any work is therefore

```
reuse gap > (burstiness - 1) * OSL * tpot
```

Every term on the right is the corpus (burstiness, OSL) or the engine (tpot).
Pool size, context length, KV bytes per token, L1 capacity and the store policy
do not appear. This is why shrinking `gpu_memory_utilization`, enlarging L1,
changing block size, and the de-duplication ideas all returned zero: none of
them is in the identity.

On this corpus the inter-turn gap has p50 1.85 s (256k-filtered event set;
3.29 s over the unfiltered trace) against a window near 50 s. A request holds
the pool for 67 s while its reuse comes back in under 2 s. The next turn
arrives while the previous one is still decoding, so L0 has it by construction.

## 5. Reuse against capacity, and what a corpus needs

Capacity is expressed as a multiple of the in-flight decode footprint so the
number is comparable across corpora.

CC 062126-256k, conversation-level, in-flight 118 GiB at running 24:

| residency | capacity / in-flight | reuse caught |
| --- | --- | --- |
| 10 s | 0.99x | 71.7% |
| 30 s | 1.76x | 81.0% |
| 48 s | 2.30x | 84.8% |
| 120 s | 3.88x | 90.2% |
| 300 s | 6.05x | 96.0% |
| 600 s | 8.36x | 97.2% |

We hold 81 GiB of L0 cache plus 145 GiB of L1, 2.17x, and measure 85.1%. The
curve gives 84.8% at 2.30x, so the model tracks the measurement.

The first 1x of cache buys 71.7%, and 1x is about what the in-flight set itself
occupies. That is the quantitative form of the observation that on this corpus
you barely need to store anything beyond what is already in flight. L1's
marginal rate is 15 GiB per point now and 46 GiB per point for the next
232 GiB.

Mooncake, block-level exact:

| residency | conversation | toolagent |
| --- | --- | --- |
| 10 s | 0.24x -> 0.9% | 0.24x -> 19.3% |
| 60 s | 1.34x -> 23.7% | 1.25x -> 39.4% |
| 120 s | 2.25x -> 49.4% | 2.06x -> 61.9% |
| 300 s | 3.79x -> 76.4% | 3.30x -> 85.2% |
| 600 s | 4.83x -> 93.3% | 3.92x -> 98.2% |

Record 1 section 5 said Mooncake's toolagent was "no better than ours" on the
basis of a single 120 s statistic. On the whole curve it is clearly better.
That judgement was wrong.

## 6. What the Mooncake traces actually are

| | conversation | toolagent | arxiv |
| --- | --- | --- | --- |
| requests / rate | 12,031 / 3.40 s-1 | 23,608 / 6.67 s-1 | 23,608 / 6.56 s-1 |
| ISL p50 | 6,909 | 6,346 | 6,345 |
| OSL p50 | 350 | 30 | 30 |
| turn gap p50 | 123.0 s | 0.0 s | 0.0 s |
| turn gap mean | 214.1 s | 45.6 s | 46.4 s |

`conversation` is human chat, mostly single-turn, and its curve favours L1
precisely because a person takes 123 s to reply. Switching to it would win by
changing the workload class, not by being a fairer test of agent serving.
`toolagent` is the agent-shaped one, 0 s median gap and 30-token outputs, but
its ISL p50 of 6,346 is fourteen times shorter than CC's and cannot carry a
long-context claim.

The session-chaining used to get turn counts keys on `hash_ids[1]` and produces
a 9,203-turn chain, so turns-per-session from these traces is unreliable. The
ISL, OSL and gap distributions are not affected. `toolagent` and `arxiv` have
different checksums but statistically indistinguishable distributions; treat
them as one workload.

## 7. The trilemma

```
long context  -> pool holds 38 -> few sessions -> window 50 s -> gap 1.85 s
                 is caught by L0 -> L1 has no work
short context -> pool holds 465 -> many sessions -> short window
                 -> L1 has work, but it is not long-context serving
human in loop -> gap 123 s > window -> L1 has work, but it is not an agent
```

Two of three. CC takes long context and agent. Mooncake conversation takes
long gap and L1. Mooncake toolagent takes agent and L1. None of the four
corpora surveyed takes all three, and the corpus that would have to exist is
specific: ISL above 50k, agent-paced, with tens of seconds of external wait
between turns.

## 8. The first real pair, round 1

CONC=64, both arms simultaneous, identical except eager/lazy, lazy on
DEFER_SECS=0. Round 2 with the slots swapped was still running when this record
was written.

| | e64r1 (eager, slot1) | l64r1 (lazy, slot2) | delta |
| --- | --- | --- | --- |
| completed | 673 | 727 | +8.0% |
| ttft p50 | 1.16 s | 1.06 s | -8.6% |
| ttft p90 | 9.97 s | 5.62 s | -43.6% |
| ttft shape | monotone | monotone | both healthy |
| lat p50 | 33.43 s | 29.26 s | -12.5% |
| waiting_mean | 0.45 | 0.23 | -49% |
| tpot p50 | 65.8 ms | 61.0 ms | -7.3% |
| kv_max | 99.3% | 99.8% | neither saturates |
| stores | 10168 | 1346 | -86.8% |
| stored/prefill | 34.19% | 23.37% | -31.6% |
| ext hit (token) | 8.14% | 10.02% | +1.9 pt |
| total hit | 81.3% | 85.2% | +3.9 pt |
| prompt tok/s | 6550 | 5564 | -15.1% |
| watermark events | 31 | 19 | -39% |

Eager does not tip over the knee at CONC=64: kv_max 99.3% and the shape stays
monotone. The hypothesis that lazy admits a concurrency eager cannot is not
supported.

What is supported, subject to round 2: lazy stores 31.6% fewer tokens and 7.6x
fewer operations and gets a higher hit rate, not merely an equal one. tpot is
7.3% lower, outside the 2% floor, and completed requests are 8.0% higher. The
tpot channel is the per-store-operation cost, not bandwidth: 10,168 operations
against 1,346, while the byte rates differ by only a few hundred MB/s against a
PCIe budget in the tens of GB/s.

Both arms have eager on slot 1 and lazy on slot 2, so arm and GPU pair are
collinear. Round 2 swaps them. Nothing here is a result until both rounds agree.

## 9. Choosing CONC without a sweep does not work

The rule "set CONC so peak in-flight KV is about the pool" reproduces the
empirical choice. Its a priori half is exact:

```
capacity = pool / context = 4,081,024 / 104,300 = 39.1 contexts
measured running_max over the saturated arms: 35 / 45 / 39, mean 39.7
```

Applied as a test it selects CONC=64: inflight_peak 37 against 39.1, kv_max
99.5%, monotone. At CONC=68 the peak is 54, 38% over.

Applied as a formula it fails. CONC = capacity / (burstiness * duty) needs
duty, and duty resists every summary:

| think time treatment | mean | duty | predicted CONC |
| --- | --- | --- | --- |
| raw | 251.8 s | 0.210 | 127 |
| truncated to the 1800 s window | 46.7 s | 0.590 | 45 |
| truncated to 600 s | 30.8 s | 0.685 | 39 |
| measured | | 0.391 | 64 |

A 2.8x swing with the measurement in the middle. The distribution is the
reason: p50 1.85 s against mean 265.7 s.

A discrete-event replay simulator was written to avoid the mean-field problem
(`sim.py`, `mkevents.py`, both in the harness scratchpad). It does not
reproduce the measurements either. At CONC=48 it gives inflight_mean 36.6
against a measured 18.67, and saturates the pool at every CONC while the
measured c48 has kv_max 89.5%. Analytic 8.1, measured 18.67, simulated 36.6:
three answers spanning 4.5x from the same inputs. The gap is in the session
pacing model, which we do not have; compressing think time in the direction
`arm.sh` describes moves the simulator further from the measurement, not
closer, so that comment's status is unresolved and record 1 section 4 should
not have declared it wrong.

Conclusion: CONC has to be measured. It does not need a sweep. Compute the
capacity a priori, start below saturation, walk up two or three points, stop at
the last arm with inflight_peak under capacity, kv_max under 100% and a
monotone shape. Approach only from below: between CONC 64 and 68 the peak jumps
37 to 54, a 46% rise for a 6% change, so extrapolation across the knee is void.

To cut the cost, keep one server up across several CONC segments rather than
rebuilding it per point; bringup and teardown are 10 to 16 minutes of each
56-minute arm. Calibrating with `config=off` is not safe on a corpus where
prefill is a first-order load, which is every corpus except this one.

## 10. Why a faster model is the only lever left

From section 4 the only term a model change moves is tpot. KV bytes per token
cancels out of the window: a smaller KV lets the cache hold more contexts and
lets the pool run more requests, and the two effects divide out.

Lower tpot shortens the window because the window is a residency time, and
residency ends when new contexts overwrite the cache. Faster decode retires
requests faster, admits new ones faster, and churns the cache faster.

Candidates in the local cache, KV bytes per token at fp8 counting only
full-attention layers:

| model | type | full layers | KV B/token | vs current | max len |
| --- | --- | --- | --- | --- | --- |
| Qwen3-Coder-30B-A3B | MoE | 48/48 | 49,152 | 1.0x | 262k |
| Qwen3.8-Flash-Next-FP8 | hybrid | 12/48 | 12,288 | 0.25x | 262k |
| gpt-oss-20b | hybrid+SWA | 12/24 | 12,288 | 0.25x | 131k |
| Qwen3.6-27B | hybrid | 16/64 | 32,768 | 0.67x | 262k |
| Qwen3-8B | dense | 36/36 | 73,728 | 1.5x | 40k |
| MiniMax-M2.5 | dense | 62/62 | 126,976 | 2.6x | 196k |

gpt-oss-20b tops out at 131k and CC's ISL p90 is 209,408, so it would truncate
a tenth of the requests. Qwen3-8B is worse on both axes. The candidate is
Qwen3.8-Flash-Next-FP8, whose `model_type` is `qwen4_exp` and whose support in
vLLM 0.23.0 is unverified; the fallback is Qwen3.6-27B.

The first-order effect is bounded. At an optimistic tpot of 18 ms the window
falls from 50 s to 16 s, L0 catches about 74% instead of 85%, and L1's share
goes from 9.5 points to about 20. A doubling, not a reversal, because no
achievable tpot brings a 50 s window near a 1.85 s median gap.

The second-order effect is larger and is the actual argument. A quarter of the
KV per token means the pool holds 152 contexts instead of 38, and with the
lower tpot the request rate rises about twelvefold, from 0.36 to roughly
4.3 per second. Prefill demand scales with it:

| | prefill demand | against an assumed 100k tok/s |
| --- | --- | --- |
| no cache | 445,000 tok/s | 4.5x, saturated |
| L0 only (75.6%) | 108,600 tok/s | at the edge |
| L0 + L1 (85.1%) | 66,750 tok/s | comfortable |
| current engine, for scale | 5,548 tok/s | 14% |

Today prefill is 14% of capacity, so the compute a cache saves has nowhere to
be spent. At twelve times the request rate prefill becomes the first-order
bottleneck and every saved token converts to throughput. That is the regime in
which a cache tier is worth multiples rather than percentage points, and it is
reachable on this corpus without changing the corpus.

What does not improve: the total hit rate stays near 85%, because reuse is
caught either by L0 or by L1 and L1's 375 s window is long enough for the
remainder. The claim is that L1 carries a larger share and that the system
enters a prefill-bound regime, not that the cache hits more.

Both projections rest on an assumed tpot of 18 ms and an assumed prefill
capacity near 100k tok/s. Neither is measured. Section 12 measures them.

## 11. Corrections in this record

1. The 10% TTFT figure came from splicing two rounds. The paired figure is
   -6.4% on ttft p50, inside the +-30% floor.
2. The -40% store volume figure came from the same splice. The paired figure at
   DEFER=30 is -9.0%; -31.6% is the paired DEFER=0 figure from section 8.
3. The total hit rate is 85.1% at c64, not 49.3%. The per-interval arithmetic
   mean is not a usable estimator.
4. "Half the prefill on this corpus never hits anything" is wrong. The corpus
   is 95.7% hot with a 98.3% ceiling.
5. "The corpus is the binding problem" is too strong. The binding quantity is
   the window identity, in which the corpus appears only through burstiness and
   OSL.
6. Mooncake toolagent is materially better than CC for L1 across the whole
   capacity curve. Record 1 section 5 judged it on one point.
7. `covered_prefix_tokens_skipped` is not the L0/L1 intersection. It counts
   store ranges shortened because a still-buffered operation already covers the
   prefix, which is the deferral's own de-duplication. The point-in-time
   intersection remains unmeasured.
8. Store-side move-not-copy does not exist as an option: gate 1 already fires
   the store at the eviction point, so there is nothing earlier to move.
   `EVICTION_HEAD` is a placement choice after that store, worth about 19 GiB
   of re-leased capacity, and it has never been run.
9. `cmp.py`'s greedy regex was reporting `tpot_p50` in ms under a "decode p50
   tok/s" label. Fixed; the row is now split in two.
10. Record 1 section 4 declared the `arm.sh` scenario comment wrong. That is
    not established. Leave the comment.
11. Shrinking `gpu_memory_utilization` does not raise L1's share materially and
    costs about half the throughput, because 37.7 ms of the 61 ms tpot is a
    fixed per-step cost that does not amortise.

## 12. In flight

Round 2 of the CONC=64 pair, slots swapped, `l64r2` on slot 1 and `e64r2` on
slot 2, due about 05:25. Scoring is the mean of the two paired differences so
the GPU pair cancels.

Chained behind it, a model characterisation: `mdl.sh` runs
Qwen3.8-Flash-Next-FP8 on slot 1 and the incumbent on slot 2, B=24, at 8k, 32k
and 100k, `config=off`. The 100k point is direct rather than extrapolated; the
existing two-point fit predicted 98.6 ms at 107k where the live arm reads
61 ms, and that 38% discrepancy is unexplained. Both models are measured in the
same round on the same machine.

`env.sh:14` had `MODEL` hardcoded and now reads `${MODEL:-...}`; backup at
`env.sh.bak_model`.

## 13. Open

- The L0/L1 point-in-time intersection is still unmeasured.
- `phase.py` compares wall-clock times as strings and returns an empty window
  for any arm crossing midnight.
- The `max_deferral_seconds` default-to-zero change still has no PR.
- The retrieve path is untouched and out of scope for this line.
- Nothing has been pushed to any remote.
