# Where the three requirements collide

The goal for the experiment is one configuration that is simultaneously
reasonable, realistic, and favourable to lazy offloading. Written out:

```
reasonable   k = L1/L0 >= 2                     no inverted hierarchy
             L0 not oversized (186.6 GiB of KV for a 30B model is not a
             provisioning anyone does)
realistic    TTFT p50 <= 2.64 s, no --unsafe-override, submission_valid true,
             comparable with the existing reference arms
lazy-favoured
             N*ISL_med*(1-cut) < L1_tok < N*ISL_med
             L1 small enough that eager thrashes, large enough that lazy does not
```

This record is about the fact that the three do not currently intersect, why,
and what the measurements say about each factor.

## 1. The oversubscription round

`ov16_s1` and `ov18_s2`, matched to `c2_14` in every respect except
concurrency: L1 = 96 G, lazy, `DUR=1800`, `GRACE=600`, no cache warmup, no
ramp, no scenario override, L0 not overridden.

| conc | oversub | inflight_mean | waiting_mean | ttft_p50 | ext_hit | branch |
|---|---|---|---|---|---|---|
| 14 | 0.31 | 3.60 | <1 | | | uncongested |
| 16 | **0.97** | **10.73** | 4.68 | 234.2 s | 2.02 % | congested |
| 18 | 1.14 | 12.00 | 5.54 | 239.2 s | 4.80 % | congested |

In-flight goes 3.60 -> 10.73 for a concurrency change of 2. The branch change
sits between 14 and 16, not somewhere in 14-20 as assumed when the round was
designed. `oversub_max` is therefore bracketed as `>= 0.31` and `< 0.97`, and
a conc=15 arm could only push the lower bound to about 0.4.

## 2. The condition the three requirements reduce to

Chaining the three:

```
in-flight fits GPU     N*d*ISL_mean < oversub_max * L0_tok
lazy window            L1_tok < N*ISL_med
normal hierarchy       L1_tok >= K_MIN * L0_tok

K_MIN*L0_tok <= L1_tok < N*ISL_med < oversub_max*L0_tok*ISL_med/(d*ISL_mean)
```

`L0_tok` appears on both ends and cancels:

```
k_max = oversub_max * (ISL_med/ISL_mean) / d
      = 0.31 * 0.58 / 0.228
      = 0.78            needs >= 2
```

So shrinking L0 is not a lever. It lowers the hierarchy floor and lowers the
session ceiling by the same factor. Checked numerically: L0 186.6 -> 100 GiB at
the same oversubscription gives 2.05 in-flight requests, 9 sessions, a window
ceiling of 88 GiB, k = 0.88 against 0.79 before. Unchanged.

Physically: a session occupies GPU memory for a fraction `d` of the time and
occupies the tier all of the time. Each GPU slot therefore carries `1/d = 4.4`
sessions. A tier three times the pool that is still undersized for the workload
needs about 11 sessions per slot.

L0 should still come down, but for two other reasons: the configuration is not
a realistic provisioning, and a large L0 is what makes L1 redundant (section 4).

## 3. What each factor can do

| factor | value | movable |
|---|---|---|
| `d` | 0.228 | see section 6 |
| `oversub_max` | in [0.31, 0.97) | maybe; the threshold 0.78 is inside the bracket |
| `ISL_med/ISL_mean` | 0.58 | no, corpus tail |
| `L0` | 186.6 GiB | cancels |

## 4. L1 = 96 GiB fails at both ends, for different reasons

96 GiB is 1 048 576 tokens at 96 KiB/token, which is 6.4 requests at the mean
input length. Pulling `gpu_prefix_hit_max` alongside the retrieval counters:

| arm | conc | L1 | stored_M | retr_M | retr/sto | gpu_hit_max |
|---|---|---|---|---|---|---|
| c2_14_s1 | 14 | 96 | 4.75 | 0.20 | 0.042 | **81.0 %** |
| c2_20_s2 | 20 | 96 | 5.69 | 0.69 | 0.122 | 31.7 % |
| cal_c32_s1 | 32 | 96 | 7.90 | 0.61 | 0.077 | 0.0 % |
| r2_lazy_s2 | 32 | 96 | 10.65 | 0.74 | 0.070 | 29.0 % |
| cal_c64_s2 | 64 | 96 | 14.37 | 0.00 | 0.000 | 0.0 % |
| k20_96cw_s1b | 20 | 96 | 16.03 | 0.12 | 0.008 | 14.5 % |
| k20_384cw_s2 | 20 | 384 | 11.92 | **18.77** | **1.575** | 15.5 % |

At light load L0 serves 81 % on its own and L1 holds a copy of the same
content, so it is never asked: 0.20 M retrieved. The failure there is
duplication, not capacity, and lazy cannot reach it -- lazy governs how much is
written, not whether what is written is already in L0.

At heavy load in-flight KV eats L0, `gpu_hit` falls to zero, L1 has to carry
everything, and 1.05 M tokens thrashes. There is no load at which 96 GiB is
useful. 384 G is the only arm with `retr/sto > 1`, i.e. the only tier that ever
returned more than it was fed.

## 5. The residency criterion, and the reuse intervals

A tier is useful only if its contents survive to be re-read:

```
period = L1_tok / store_rate  >  reuse interval
```

With `store_rate = N*ISL_med / RI` the interval cancels and the criterion
becomes a pure capacity statement:

```
N*ISL_med*(1-cut) < L1_tok < N*ISL_med
```

Lower bound: lazy's write cut has to be enough. Upper bound: if L1 exceeds one
round of everyone's input, eager already works and lazy adds nothing.

Checked against every arm that has the counters:

| arm | sess | RI_med | L1_tok M | N*ISL M | predicted | retrieved |
|---|---|---|---|---|---|---|
| c2_14_s1 | 15 | 15 s | 1.05 | 1.66 | lazy passes | 0.20 M |
| c2_20_s2 | 20 | 63 s | 1.05 | 2.17 | marginal | 0.69 M |
| r2_lazy_s2 | 28 | 380 s | 1.05 | 3.00 | fails | 0.74 M |
| k20_96cw_s1b | 33 | 688 s | 1.05 | 2.52 | fails | 0.12 M |
| k20_384cw_s2 | 35 | 369 s | 4.19 | 2.62 | oversized | 18.77 M |

All five land where the criterion says. The 384 G arm is above `N*ISL`, so its
result is credited to size, not to lazy.

The reuse interval was measured, not assumed: median gap between consecutive
turns of the same conversation, from the exports. It runs 15 s to 688 s. The
15 s at `c2_14` is the scenario's idle cap (`system_idle_gap_cap_seconds=10.0`)
advancing timers whenever nothing is in flight, against a corpus think time
whose mean is 251.8 s. An earlier version of this analysis assumed a 115 s
interval and produced an L1 window table that was wrong throughout; the
analytic form above does not depend on the interval, which is why it survives.

## 6. Decode is not the bottleneck it looked like

`d = R/(R+T)`, and `R` is 90 % decode, so decode was the obvious lever. A
single-stream probe on the same stack (LMCache connector in the path, ~100 k
tokens of real repository source as the prompt) says otherwise:

```
dp_base  echo   ttft= 4.70s  out=395  decode=176.9 tok/s  tpot=5.7 ms
dp_base  echo   ttft= 0.60s  out=385  decode=177.4 tok/s  tpot=5.6 ms
dp_base  novel  ttft= 0.54s  out=383  decode=177.9 tok/s  tpot=5.6 ms
dp_base  novel  ttft= 0.54s  out=385  decode=177.5 tok/s  tpot=5.6 ms
```

177 tok/s single stream, above the production reference of 161.7. The 57.6
tok/s measured at `c2_14` is the per-user rate under load, not a hardware
ceiling. Record 9's reading of it as a hardware or model property is wrong, and
so is the conclusion that end-to-end throughput can never approach production.

Per-user decode against in-flight count:

```
in-flight 1.00 (probe)    177 tok/s   aggregate 177
in-flight 3.60 (c2_14)     57.6       aggregate 207
in-flight 10.73 (ov16)     30.8       aggregate 330
```

Aggregate rises sublinearly, per-user falls roughly as 1/L. Decode is bound by
the KV bytes the batch reads per step, so `R` inflates with concurrency. That
makes `d` a function of load rather than of hardware, which changes what
"raising decode" can mean: there is little single-stream headroom left, and the
speculative-decoding arm is a test of whether drafting several tokens per step
amortises the batch KV read, not a test of a slow model.

Three arms were run to try to raise it. All three failed, in two different
ways.

```
dp_ngram  echo   decode= 66.3 tok/s  tpot=15.1 ms      acceptance length 1.97
dp_ngram  echo   decode= 90.2 tok/s  tpot=11.1 ms      draft acceptance 24.3 %
dp_ngram  novel  decode= 69.4 tok/s  tpot=14.4 ms
dp_ngram  novel  decode= 90.8 tok/s  tpot=11.0 ms

dp_ngpu   echo   decode= 58.8 tok/s  tpot=17.0 ms      acceptance length 1.41
dp_ngpu   echo   decode= 78.3 tok/s  tpot=12.8 ms      draft acceptance 10.2 %
dp_ngpu   novel  decode= 82.7 tok/s  tpot=12.1 ms
dp_ngpu   novel  decode= 75.8 tok/s  tpot=13.2 ms

dp_fp8    echo   decode=161.2 tok/s  tpot= 6.2 ms      pool 4 076 944 tokens
dp_fp8    echo   decode=167.5 tok/s  tpot= 6.0 ms
dp_fp8    novel  decode=168.1 tok/s  tpot= 5.9 ms
dp_fp8    novel  decode=167.6 tok/s  tpot= 6.0 ms
```

### Speculative decoding loses because the step cost is linear in tokens

Both `ngram` variants cost roughly half the baseline rate. Moving the draft
from CPU to GPU did not recover it -- `ngram_gpu` is slower than `ngram` and
its acceptance is worse (1.41 against 1.97, 10.2 % against 24.3 %), so the
drafting overhead was never the main term.

Converting TPOT per accepted token back to cost per engine step, with
`num_speculative_tokens=4` (a 5-token verify batch):

| arm | tokens/step | ms per accepted tok | ms per step | vs baseline step |
|---|---|---|---|---|
| baseline | 1.00 | 5.6 | 5.6 | 1.0x |
| ngram | 1.97 | 11.0-15.1 | 21.7-29.7 | 3.9-5.3x |
| ngram_gpu | 1.41 | 12.1-17.0 | 17.1-24.0 | 3.1-4.3x |

A 5-token step costs about 4-5x a 1-token step. Step cost is close to linear
in the number of tokens, so there is nothing for speculation to amortise and
it can only lose. The reading is MoE expert fan-out: 128 experts, top-8 per
token, so at batch 1 each additional token in the verify batch pulls in a
nearly disjoint set of 8 experts and the weight traffic scales with the token
count rather than staying fixed. On a dense model the same arm would likely
win; on this one speculative decoding is the wrong lever, and no acceptance
rate reachable by an n-gram drafter would change that.

### fp8 KV doubles the pool and is a no-op on the identity

LMCache follows `cache_config.cache_dtype`, so `--kv-cache-dtype fp8` carries
through to the tier with no other change. The pool went 2 038 560 -> 4 076 944
tokens, exactly 2.00x, which confirms both halves of the path.

Decode did not improve: 177.5 -> 167.6 tok/s, 5.6 -> 6.0 ms, about 5 % worse.
At batch 1 the KV read is a small part of the step and dequantisation is not
free. The earlier expectation that fp8 would lower `d` through faster decode
is therefore wrong.

That leaves fp8 as a scale-invariant transform of the whole problem, for the
same reason shrinking L0 was:

```
in-flight    N*d*ISL_mean*b < oversub*L0_bytes    b halves -> N doubles
lazy window  L1_bytes < N*ISL_med*b               b halves, N doubles -> unchanged
hierarchy    L1_bytes >= K_MIN*L0_bytes           unchanged
k_max = oversub*(ISL_med/ISL_mean)/d              b cancels, d did not move
```

Section 2 already showed `k_max` is invariant under L0. It is invariant under
bytes-per-token as well. Both of the obvious escapes are no-ops.

### What this settles

`d = R/(R+T)`. `T` is the corpus think time and `OSL` is the corpus output
length; neither is ours to set under the fidelity standard. `R`'s remaining
term is decode, and decode has now been probed three ways and cannot be
raised -- the baseline already sits above the corpus's own 161.7 tok/s p50.
So `d` is fixed by the corpus, not tunable.

The only quantity in `k_max` still holding an untested range is `oversub`.
Measured: conc=14 gives 0.31 and passes, conc=16 gives 0.97 and is congested.
`k = 2` needs `oversub = 2*0.228/0.58 = 0.786`, inside that bracket. But two
concurrency steps span 0.31 to 0.97, which is a cliff rather than a dial:
once queueing starts, in-flight inflates and carries oversub past the target
in one step. Whether a steady state exists at 0.79 is the one open question
that decides whether the three requirements can hold together at all.

## 7. eager vs lazy: no valid comparison exists yet

`r1`/`r2`/`r3` are matched eager/lazy pairs at conc=32, `DUR=1800`, identical
ISL p50 of 107 k.

| arm | L1 | stored_M | retr_M | wmark | ttft_p50 | e2e_p50 | dec_tps |
|---|---|---|---|---|---|---|---|
| r1_eager | 32 | 19.65 | 0.00 | 344 | 304.3 | 529.7 | 14.1 |
| r1_lazy | 32 | 3.37 | 0.16 | 57 | 292.1 | 435.4 | 21.8 |
| r2_eager | 96 | 20.03 | 0.00 | 116 | 304.3 | 450.6 | 13.0 |
| r2_lazy | 96 | 10.65 | 0.74 | 65 | 271.8 | 516.2 | 14.2 |
| r3_eager | 160 | 19.71 | 0.35 | 66 | 265.7 | 397.7 | 18.3 |
| r3_lazy | 160 | 12.79 | 1.47 | 45 | 255.3 | 434.1 | 16.4 |

Store side, lazy wins all three: writes down 83 / 47 / 35 %, retrieves 0 -> 10,
0 -> 22, 4 -> 34, watermark crossings down 83 / 44 / 32 %.

Latency side it is a wash: TTFT p50 favours lazy in all three by 4-11 %, e2e
p50 favours eager in two of three, request counts are within 3 %. With 87-92
samples and every arm at 300 s TTFT, none of these separations mean anything.
All six ran in the congested branch, so the pairs are void as evidence for
lazy's value. There is no eager arm on the uncongested branch at all, and no
uncongested arm with `ext_hit` instrumented -- that field was only added to
`arm.sh` today.

## 8. Deduplication, and how much it is worth

The exclusive-tier design (retrieve moves rather than copies) raises effective
capacity from `max(L0_cache, L1)` to `L0_cache + L1`. Since the hit-rate
surface's knee is linear in N at about 6 GiB/session, session capacity scales
one-for-one with effective tier size, so the gain is `1 + 1/k`: +208 % at
k=0.48, +52 % at 1.92, +40 % at 2.5, +25 % at 4.0. The deduplicable part is
`L0 - L*I`, not L0, because in-flight KV is 50-70 % of the pool, which takes
+40 % down to +20-27 % at k=2.5.

Where it pays depends on where the tier sits relative to the knee. At the small
N this node can run, a normal-k tier is already past the knee and dedup is
worth 0.1-0.5 pp of hit rate. At N in the hundreds it is worth 3-5 pp. That is
the regime frontier systems run in, so "more memory stops helping" is not a
general statement -- at N=256 this corpus still gains 40 pp going from 187 GiB
to 1024 GiB.

Two design notes. Strict move-on-retrieve turns every L0 eviction into a
mandatory L1 write, and with 95.7 % of references hot the ping-pong rate is
high, which works against the write-economy half of the goal. Marking the L1
copy clean instead, and evicting clean entries first, gets exclusive capacity
under pressure while keeping one write per block. It also keeps the redundant
copy that a preemption would otherwise have to recompute; preempts were 3-9
across these arms, not zero.

## 9. Corrections

- `Surface.at()` applied `SURFACE_CAL` only on the interpolation path, so a
  lookup that landed exactly on a grid point came back un-discounted. Any
  comparison with one side on a grid point was wrong by 1/0.85. Fixed.
- The claim that lazy at 32 GiB matches eager at 160 GiB was based on residency
  period (186 s vs 160 s). Retrieved tokens say the opposite (0.16 M vs
  0.35 M). Residency is a proxy; retrieved tokens are the result. The
  defensible statement is lazy at 96 GiB (0.74 M) against eager at 160 GiB
  (0.35 M), so lazy is worth more than 1.67x its memory, lower bound only.
- The L1 window table computed from `r`-series store rates is void with the
  `r`-series itself, since those rates were measured in the congested branch.
- "Our decode is 2.8x slower than production, a hardware property" is refuted
  by section 6.

## 10. Model portability

Covered in record 12 section 9 and unchanged: bytes per token is config
arithmetic, weights and overhead come from one boot log, the two rates need one
low-load arm, and the corpus constants plus the reuse surface carry over free.
Section 6 adds a caveat: the decode rate that matters is the per-user rate at
the operating point, not the single-stream rate, so it has to be measured under
load and re-measured whenever the load changes.

## 11. Tooling

```
sens.py      feasibility vs bytes-per-token and decode rate
accel.py     the two ceilings on N as decode is accelerated
l0free.py    the k_max identity with L0 free, showing it cancels
favor.py     L1 window where lazy flips the residency threshold (rates void)
dprobe.py    single-stream decode probe, echo and novel request shapes
dprobe.sh    brings up one slot with the arm's stack, probes, tears down
up.sh        + SPEC_CFG hook for --speculative-config (empty = every arm
             through ov18); backup at up.sh.bak
configure.py at() haircut fix
```

vLLM 0.23.0 offers `ngram`, `ngram_gpu` and `suffix` without a draft model;
this model has no MTP head. `lmcache/integration/vllm/utils.py:412` adds draft
KV layers only for `deepseek_mtp` and the eagle family, so ngram and suffix
leave the KV layout untouched and need no connector change.

## 12. Where this leaves the experiment

The uncongested branch ends at conc=14, about 15 sessions. At that session
count the lazy window tops out at 147 GiB, so k <= 0.79 with the current L0 and
k <= 1.47 even at the L0 floor of 100 GiB, which is set by the corpus's largest
request of 766 265 tokens.

Reaching k >= 2 needs `oversub_max * 0.58 / d >= 2`. Neither factor is settled:
the oversubscription ceiling is bracketed but not measured, and `d` turns out to
be load-dependent rather than hardware-limited. The decode probe decides how
much of `d` is recoverable.

If it is not recoverable, the two honest options are to report lazy's value at
an inverted hierarchy, stating plainly that this is the memory-constrained
regime and noting that every shipped LMCache example is inverted
(`max_local_cpu_size` defaults to 5.0), or to report that this node cannot
satisfy all three and say what would.

Repo clean at `9907627c` before this record; nothing pushed.
