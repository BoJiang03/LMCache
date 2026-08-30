# The paired result, and what it costs to state honestly

Record 2 left the first genuine eager/lazy pair half run: round 1 landed, round 2
was still in flight, and nothing from round 1 counted as a result until the two
agreed. Round 2 landed at 05:24 and it agrees. This record scores the pair,
decomposes the headline number so the decomposition travels with it, and
withdraws four claims made earlier in the same session.

## 1. The pair

Four arms, CONC=64, `fp8 / maxlen=262144 / block=64 / FLOOR=2048 / DEFER_SECS=0
/ L1=192G`, CC 256k corpus, DUR=1800. Slots swapped between rounds so the GPU
pair cannot be confounded with the arm.

```
round 1  03:33:01 - 04:28:47   slot1(GPU4,5)=eager e64r1   slot2(GPU6,7)=lazy l64r1
round 2  04:28:47 - 05:24:28   slot1(GPU4,5)=lazy  l64r2   slot2(GPU6,7)=eager e64r2
```

| | e64r1 | l64r1 | l64r2 | e64r2 |
|---|---|---|---|---|
| requests | 673 | 727 | 696 | 666 |
| ttft p50 | 1.16s | 1.06s | 1.20s | 1.16s |
| ttft p90 | 9.97s | 5.62s | 7.88s | 9.48s |
| lat p50 | 33.43s | 29.26s | 31.94s | 32.92s |
| in-flight mean | 26.55 | 25.05 | 25.23 | 26.57 |
| running / waiting | 25.49 / 0.45 | 24.02 / 0.23 | 24.02 / 0.30 | 25.22 / 0.43 |
| kv_mean | 73.0% | 69.9% | 69.7% | 72.5% |
| gpu prefix hit max | 77.6% | 77.4% | 77.6% | 77.6% |
| ext hit, token weighted | 8.14% | 10.02% | 10.56% | 8.68% |
| stores | 10168 | 1346 | 1338 | 10039 |
| stored/isl | 34.19% | 23.37% | 24.11% | 34.35% |
| watermark events | 31 | 19 | 19 | 31 |
| tpot p50 | 65.8 ms | 61.0 ms | 61.1 ms | 65.5 ms |
| isl mean | 104287 | 103274 | 104427 | 104322 |

The slot effect is not systematic. Eager ran 673 in slot 1 and 666 in slot 2,
lazy ran 727 in slot 2 and 696 in slot 1. The two arms prefer opposite slots, so
what looks like a slot difference is round to round variation. The paired design
did its job.

### What replicates

Both rounds, same sign, outside the noise floor.

| | round 1 | round 2 | mean | floor |
|---|---|---|---|---|
| stores | -86.8% | -86.7% | -86.8% | -- |
| stored/isl | -31.6% | -29.8% | -30.7% | -- |
| ext hit, token weighted | +1.88pt | +1.88pt | +1.88pt | -- |
| watermark events | 31 to 19 | 31 to 19 | -- | -- |
| tpot p50 | -7.3% | -6.7% | -7.0% | 2% |
| decode tok/s p50 | +7.9% | +7.2% | +7.6% | 2% |
| waiting mean | -49% | -30% | -40% | 5% |
| kv_mean | -3.1pt | -2.8pt | -3.0pt | -- |

### What is directional only

Same sign, wide spread between rounds.

| | round 1 | round 2 | mean |
|---|---|---|---|
| goodput, e2e>=15 and ttft<=10s | +35.2% | +25.2% | +30.2% |
| completed requests | +8.0% | +4.5% | +6.3% |
| lat p50 | -12.5% | -3.0% | -7.7% |
| ttft p90 | -43.6% | -16.9% | -30% |

### What does not hold

TTFT p50 went -8.6% in round 1 and +3.4% in round 2. The sign flips and both
values sit inside the plus or minus 30 percent floor. Every earlier claim that
rested on TTFT p50, including record 2 section 1's already reduced -6.4%, is now
without support. The metric is too noisy at this sample size to carry anything.

## 2. Goodput, and how the 30 percent is built

Goodput here is the count of requests that met both an end to end token rate and
a TTFT bound, over the same wall clock. It does not require the two arms to have
served the same requests, which matters because a closed loop lets the faster arm
pull more work.

| SLO | eager r1 / r2 | lazy r1 / r2 | mean gain |
|---|---|---|---|
| e2e>=10 tok/s and ttft<=10s | 475 / 480 | 591 / 522 | +16.6% |
| e2e>=15 tok/s and ttft<=10s | 256 / 274 | 346 / 343 | +30.2% |
| e2e>=20 tok/s and ttft<=10s | 120 / 136 | 167 / 157 | +27.3% |

The 30 percent is the largest of the three bars, and the bar is ours to choose.
Reporting it without this table is selection. It factors as

```
goodput ratio = completed count ratio x pass rate ratio
round 1   1.353 = 1.080 x 1.253
round 2   1.254 = 1.045 x 1.200
mean      1.302 = 1.063 x 1.226
```

so the completed count contributes 6.3 percent and the pass rate contributes
22.6 percent relative. The pass rate moves that much because the threshold sits
just above both medians, where the density is highest:

```
e2e tok/s     p25    p50    p75
eager         9.9   13.4   18.5
lazy         11.1   14.7   19.6
```

p25 to p75 holds half the mass across 8.6 tok/s, so the local density is about
5.8 points per tok/s. A shift of 1.3 tok/s predicts 7.5 points crossing the bar
and 9.2 are measured. The distribution moved 9.7 percent and the crossing count
moved 25 percent. That is leverage, not a second effect, and the leverage is
what makes the number bar sensitive.

The safer statement for external use is the pair of factors, 6.3 percent more
requests completed and 22.6 percent relative more of them meeting the bar, since
neither is amplified.

### The OSL mix is not the cause

A longer output amortises TTFT and raises e2e, so a lazy arm with longer outputs
would pass more often for free. It does not. Round 2's OSL means are 984.9 and
983.7, and lazy leads in every stratum.

| OSL bin | e64r1 | l64r1 | l64r2 | e64r2 |
|---|---|---|---|---|
| 64-256 | 40.7% | 53.6% | 54.4% | 46.4% |
| 256-1024 | 43.3% | 49.2% | 54.0% | 45.0% |
| 1024-4096 | 22.4% | 37.4% | 33.6% | 26.1% |
| 4096+ | 40.0% | 60.6% | 56.2% | 41.9% |

The one bin lazy loses is OSL under 64, 31 to 36 requests per arm.

## 3. The e2e gain is the tpot gain

Four independent measurements, no shared estimator, all landing at 7 to 9
percent.

| | round 1 | round 2 |
|---|---|---|
| e2e tok/s median | 13.40 to 14.66, +9.4% | 13.72 to 14.86, +8.3% |
| decode_duration median at matched OSL | -- | 29.82s to 27.41s, -8.1% |
| engine tpot p50 | -7.3% | -6.7% |
| output tokens in the same 2400s | 649596 to 736025, +13.3% | 655136 to 685489, +4.6% |

Stratified by OSL the median e2e gain is 3 to 13 percent with no stratum
negative. TTFT contributes almost nothing: it is 4.6 to 5.2 percent of request
duration, a 1.1 to 1.2 s head on a 30 s decode tail. e2e is the reciprocal of
tpot and nothing else.

Total request seconds is fixed by the closed loop at about 50,000 s for all four
arms. Within that fixed budget lazy fits more requests, each shorter: mean 72.0 s
against 76.5 s in round 2.

## 4. The chain, and what lazy did not improve

An eager store pins its GPU blocks for the duration of the copy. Ten thousand
stores against thirteen hundred is the whole difference in pool occupancy:

```
stores -86.8%  ->  kv_mean -3.0pt, waiting -40%, watermark 31 to 19
               ->  in-flight 26.6 to 25.1
               ->  tpot -7.0%  ->  e2e +8.9%
               ->  pass rate +22.6% rel  x  completed +6.3%  =  goodput +30%
```

Hit rate is not in the chain. L0 is flat at 77.6 percent across all four arms and
ext gains 1.88 points. This line's value is store volume and pool pressure, not
cache coverage, and record 2 section 4's window identity says why: at a 51.7 s
window L1's entire customer base is 23 percent of blocks.

## 5. Speculative decoding: record 1 killed it without the KV term

Record 1 section 1 concluded that spec is dead at this working point and that
the reason is that spec amortises the weight read while at 107k the weight read
is not where the time goes. The measurements it rests on stand. The reasoning
has a hole: the KV read is also per step, not per token, so a verify step that
accepts k tokens reads the KV cache once for k tokens. Spec amortises the KV
read too, and at 107k the KV read is the larger of the two.

Costed at our operating point, the amortisable share is not 9 percent but about
77 percent:

| | 32k measured | 107k scaled | share of a 56 ms step |
|---|---|---|---|
| KV read | 37.7 GB, 3.93 ms | 126 GB, 13.1 ms | 23% |
| weight read | 47 GB, 4.92 ms | 47 GB, 4.92 ms | 9% |
| MoE and attention FLOPs | <0.1 ms | <0.1 ms | ~0 |
| per layer launch and sync | ~25 ms | ~25 ms | 45% |

Only the FLOPs scale with query positions. The measurement says the opposite
happened:

| | query positions per step | step cost | ratio |
|---|---|---|---|
| batch 1, no spec | 1 | 5.6 ms | -- |
| batch 1, ngram 4 draft | 5 | 21.7-29.7 ms | 3.9-5.3x |
| batch ~25, no spec | 25 | 87.1 ms | -- |
| batch ~25, ngram 4 draft | 125 | 380 ms | 4.4x |

The batch 25 figure is `112.4 ms x 3.38` accepted tokens. Nearly nothing was
amortised at either batch size.

Splitting the batch 1 case against its roof locates the growth. KV is 5.2 GB at
0.54 ms and weights are 8 of 128 experts, 3.75 GB at 0.39 ms, a roof of 0.93 ms
against 5.6 ms measured. With four draft tokens the fanout reaches 40 of 128
experts, 18.75 GB at 1.95 ms, KV unchanged, roof 2.5 ms against about 25 ms
measured. The roof grew 2.7x and the unexplained residual grew from 4.7 ms to
22 ms, which is 4.7x. What scales with query positions is the residual that
record 1 section 6 attributed to per layer launch and sync, with attention in
`splitting_ops` and therefore outside the cudagraph, 48 launches per step.

This is a reading of two numbers, not an experiment. It matters because it
changes what kind of dead spec is. If the loss is physics then no configuration
recovers it. If the verify path simply leaves the decode fast path when q>1 per
sequence, it is a vLLM path question and the bandwidth arithmetic above says a
recovered spec should win. Record 1's conclusion is narrowed to: measured, spec
loses at both batch sizes tried; the mechanism at our batch is not established.

The cheap discriminator is a `bsweep.py` style isolated run with spec on and off,
reading attention time per step against q. Not run.

## 6. Two withdrawn claims about the decode ceiling

Both were made earlier in this session and both are wrong.

**The 37.7 ms fixed cost.** `tpot ~ 37.7 ms + 0.963 x running` was fitted over
live arms with running between 19 and 31. I read the intercept as a physical per
step cost and extrapolated it to running 1 to claim a 26.5 tok/s ceiling. The fit
has no validity below its range. Record 1 section 6 measures the real batch 1
behaviour, `tpot(L) = 4.247 ms + 0.0136 ms per 1k tok`, a length independent term
of 4.25 ms that batch 8 to 10 already amortises.

**"Lowering concurrency cannot reach 50 tok/s."** It can. Per request rate is
aggregate decode divided by running, and aggregate decode is roughly flat:

| arm | running | aggregate decode | per request |
|---|---|---|---|
| f8k256c48 | 20.2 | 360 tok/s | 17.83 |
| f8k256c60 | 22.8 | 388 tok/s | 17.03 |
| f8k256c72 | 30.7 | 363 tok/s | 11.82 |
| f8k256c84 | 33.5 | 344 tok/s | 10.28 |

50 tok/s per request needs running near 7. That is reachable, and it is also the
end of this line's story: at running 7 the pool holds 39 contexts and 7 are in
flight, a capacity multiple near 5.6x, where section 8's curve puts L0 above 95
percent and leaves L1 nothing to do. On this machine 50 tok/s per user and an L1
bottleneck are mutually exclusive, because the same 360 tok/s sets both.

**A third, smaller one.** I reported aggregate decode as up only 1.8 percent for
lazy, computed as `mean(running) x median(decode_tps)`, a mean times a median.
The client's own token totals give +13.3 and +4.6 percent. The gain is not mostly
a division effect.

## 7. Per request speed against the target

The user's bar for a healthy deployment is around 50 tok/s per request. Neither
arm is close.

```
                        p10    p25    p50    p75    p90
e64r1  decode           9.1   10.8   15.2   19.9   26.7
l64r1  decode           9.9   11.9   16.2   21.3   29.0
e64r1  e2e              7.3    9.9   13.4   18.5   23.4
l64r1  e2e              8.6   11.1   14.7   19.6   25.7
```

At CONC=64 the median request gets 15 tok/s. The demo operating point is one
where the service is already outside any interactive SLO, which is worth stating
alongside the goodput gain rather than after it.

## 8. What a faster model would buy, priced from the trace

The window identity from record 2 section 4 says L1 only sees blocks whose reuse
gap exceeds `window = (burstiness - 1) x OSL x tpot`. The gap distribution is a
property of the corpus and can be measured without running anything. Block
weighted over 98,827 requests and 338 million blocks:

```
p10   6.6s    p25  10.5s    p50  18.2s    p75  43.3s    p90 135.1s    p95 243.5s
```

The model is validated at the current operating point. At window 51.7 s the CDF
puts 20.7 percent of all blocks beyond the window, predicting L0 = 95.7 - 20.7 =
75.0 percent against 75.6 measured. L1's customer base is 20.7 plus 2.6 warm =
23.3 percent, and ext measures 10.0, so L1 catches 43 percent of what reaches it.

Holding OSL and the catch rate:

| tpot | window | blocks beyond | L0 | L1 customers | ext |
|---|---|---|---|---|---|
| 61 ms, now | 51.7s | 20.7% | 75.0% | 23.3% | 10.0% measured |
| 30 ms | 25.4s | 37% | 59% | 40% | 17% |
| 25 ms | 21.2s | 44% | 52% | 47% | 20% |
| 18 ms | 15.2s | 56% | 40% | 59% | 25% |

Two things follow. The benefit is insensitive to where tpot lands, so the lever
does not depend on hitting a particular number. And total hit rate falls, from 85
to about 65, because L1 catches less than half of what it is handed. That is the
point rather than a problem: it moves L1 from comfortable to binding, which is
the regime where the catch rate, and therefore this line, is worth points.

## 9. The model characterisation half failed

`mdl.sh` ran both slots with `config=off`, B=24, lengths 8000/32000/100000.

Qwen3.8-Flash-Next-FP8 does not load. The checkpoint declares model type
`qwen4_exp` and the installed transformers does not recognise it. Fixing it means
upgrading transformers in a shared venv, which is out of bounds. The candidate is
unavailable, not rejected. Qwen3.6-27B at 32,768 bytes per token remains the
fallback and has not been run.

The incumbent measured:

```
m_coder30 B=24 ctx~  8000  ttft p50=  1.8s  tpot p50= 13.54 ms  agg 1773 tok/s  per-req 73.87
m_coder30 B=24 ctx~ 32000  ttft p50=  8.0s  tpot p50= 35.58 ms  agg  675 tok/s  per-req 28.10
m_coder30 B=24 ctx~100000  ttft p50= 42.7s  tpot p50=155.71 ms  agg  154 tok/s  per-req  6.42
fit: tpot(L) = 1.175 ms + 1.5453 ms per 1k tok  -> at 107k: 166.5 ms, agg 144 tok/s
```

This does not reproduce the live arms. At nominally the same batch and length the
isolated harness gives tpot 166.5 ms where the live arm gives 61 to 66 ms, a
factor of 2.6. An earlier isolated run in this session extrapolated 98.6 ms, so
the two isolated measurements also disagree with each other by 1.7x. Something
about firing 24 mutually distinct 100k prefixes differs from the live mixture,
and until that is understood no isolated number should be used to predict a live
one. Candidates not tested: cascade attention over shared prefixes in the live
arm, and the live batch's length distribution differing from a uniform 100k.

## 10. Corrections in this record

1. Record 2 section 1's TTFT p50 result is withdrawn. Round 2 flips its sign.
2. Record 1 section 1's kill of speculative decoding is narrowed. The measurement
   stands, the mechanism does not, and the KV read term was missing from the
   argument.
3. The 37.7 ms fixed cost and the 26.5 tok/s ceiling derived from it are
   withdrawn. Invalid extrapolation below the fitted range.
4. "Lowering concurrency cannot reach 50 tok/s" is withdrawn.
5. "Aggregate decode rose only 1.8 percent for lazy" is withdrawn. A mean times a
   median.
6. The +30% goodput headline is kept but never travels without its bar, its
   sensitivity table, and its two factors.

## 11. In flight and open

- Nothing is running. `pair64.sh` and `mdl.sh` have both exited.
- Nothing has been pushed to any remote.
- The PR for this line remains the `max_deferral_seconds` default to zero change,
  on a `_pr` branch, without `records/`. Not started.
- Qwen3.6-27B fallback not run. Any new arm needs its design agreed first.
- The spec verify path discriminator not run.
- The isolated versus live tpot discrepancy in section 9 is unexplained.
- L0 intersect L1 point in time overlap still unmeasured.
- `phase.py` compares wall clock times as strings, so an arm crossing midnight
  gets an empty window.
