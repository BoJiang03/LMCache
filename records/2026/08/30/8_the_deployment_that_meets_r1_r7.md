# A deployment for R1..R7, rebuilt from the engine logs

R2 now carries a number (L1/L0 in [1,3]) and R7 is new (TTFT < 10 s, better
< 5 s). Reworked the deployment question against those. This record supersedes
the constants and most of the arithmetic in record 7, keeps its model pick, and
corrects records 2 and 6 as well.

Nothing was launched. Section 10 is the probe to run first.

## 1. The instrument I should have been reading

Every arm archived its engine log as `<arm>/vllm.log.gz`. One line per ten
seconds carries everything the last three records were trying to infer:

    Avg prompt throughput: X tokens/s, Avg generation throughput: Y tokens/s,
    Running: R reqs, Waiting: Q reqs, GPU KV cache usage: U%,
    Prefix cache hit rate: H%, External prefix cache hit rate: E%

`GPU KV cache usage` is f. vLLM keeps completed-but-cached blocks in the free
queue, so the gauge counts only blocks pinned by running requests, which is
exactly `in-flight KV / pool`. `Y/R` is per request decode speed. `H` is L0,
`E` is L1 on the residual after L0
(`v1/core/sched/scheduler.py:633`: `connector_prefix_cache_queries =
request.num_tokens - num_new_local_computed_tokens`), so on a common
denominator L1's share is `E x (1-H)` and

    L1/L0 = E x (1-H) / H

No fitting required. The tpot law in record 6 was fitted from client side
Little's law in-flight; the engine had been reporting the real thing all along.

Coder30, TP=2, pure decode samples only (prompt throughput 0, nothing waiting):

| Running | f | tok/s/req | tpot ms | KV tokens/req |
|---|---|---|---|---|
| 4 | 0.170 | 95.2 | 10.5 | 173,444 |
| 8 | 0.287 | 61.0 | 16.4 | 146,407 |
| 10 | 0.345 | 50.5 | 19.8 | 140,795 |
| 16 | 0.480 | 30.1 | 33.2 | 122,431 |
| 22 | 0.715 | 22.3 | 44.9 | 132,541 |
| 26 | 0.753 | 21.0 | 47.6 | 118,193 |

50 tok/s is Running 10 at f = 0.345. That is the whole R1 operating point,
measured, and it agrees with the cost model in section 3 to 5%.

## 2. Two corrections to record 7

**The window identity.** Record 7 section 1 used
`window = ((1-f)/f) x (ISL/A) x T_turn` with `ISL/A = 59` taken from the
corpus's new-block fraction (5,717,185 of 338,052,834 blocks are new, 1.7%),
and concluded the coder30 window is about 40 minutes. That 1.7% is what the
trace would allocate against an infinite cache. What a real engine allocates is
every block not resident in L0, which is `1 - L0`, and L0 is measured, not
assumed. The identity is

    W = D (1 - f) / (f (1 - L0)),   D = request latency, L0 = f_L0(W)

solved as a fixed point against the block-weighted gap CDF. Checked against the
archived arms:

| arm | f | L0 gauge | D p50 | W implied | L0 from CDF at W |
|---|---|---|---|---|---|
| f8k256c48 | 0.674 | 0.833 | 21.65 | 62.7 s | 0.780 |
| f8k256c60 | 0.710 | 0.789 | 26.19 | 50.7 s | 0.746 |
| l64r1 | 0.804 | 0.471 | 29.26 | 13.5 s | 0.350 |
| l64r2 | 0.811 | 0.462 | 31.94 | 13.8 s | 0.362 |
| e64r1 | 0.862 | 0.546 | 33.43 | 11.8 s | 0.287 |

Within 6% at f = 0.7, under-predicting L0 by 20 to 50% at f > 0.8. The residual
is in the right direction for a known reason: past f = 0.8 the engine starts
preempting, and the L0 gauge is a rolling window over the last 1,000 requests
that swings from 1% to 77% inside a single arm. It is an indicative
instrument, not a precise one. The token ratios in `snapshot.txt` are the
precise ones.

Record 7's other claim from that section, "the measured L1 share is 2.6%", is
the corpus's `warm` column, a property of the trace file. Measured serving, from
l64r1's profiling window: L1 returned 7,526,144 tokens against 75,080,439 input
tokens served, 10.0%. Eager on the paired slot returned 5,716,480 of
70,185,149, 8.1%.

**The bandwidth.** Record 7 used b = 1,483 GB/s per GPU, 31% of peak, from the
20-arm tpot slope. That fit had no weight-read term, so the expert read was
absorbed into the KV slope. Refitting the engine's own pure-decode samples
against

    tpot x b x N = c(B) x P_exp + P_dense + f x pool_bytes

with `c(B) = 1 - (1 - k/E)^B` gives b = 2.1 to 2.6 GB/ms per GPU, 44 to 53% of
H200 peak. Record 7 states "every number below scales with b". Every f in its
section 4 table is low by about 1.6x.

Two smaller ones. Record 7's sweep used ISL 313,543, the main-turn mean of the
raw corpus; the scenario actually serves ISL mean 103,274 (l64r1), so its
in-flight counts are about 3x low (it predicts 3.0 in flight for the coder30
baseline where the engine measures Running 24 to 30). And its overhead was
fitted at 5 GB/GPU from a 30B model; section 9 keeps that but shows the answer
under 10 GB/GPU as well, because it moves the pick's operating point.

## 3. The cost model

    step bytes = c(B) x P_exp + P_dense + f x pool_bytes
    tpot       = step bytes / (b x N)
    pool       = N x (HBM x util - overhead) - P
    f          = B x ISL x kv_per_token / pool
    c(B)       = 1 - (1 - k/E)^B          distinct experts touched at batch B

HBM 150.75 GB, util 0.9, overhead 5 GB/GPU, b 2.4 GB/ms/GPU. Validated on
coder30 at E/k = 16: predicts f = 0.31 at B = 11.8 for 20 ms against the
engine's f = 0.345 at Running 10.

Two consequences fall straight out.

**f has a ceiling at 50 tok/s.** As P goes to zero,

    f_max = tpot x b / (HBM x util - overhead) = 20 x 2.4 / 130.67 = 0.367

Any model that is small next to HBM sits at f ~= 0.3 at 20 ms no matter how many
GPUs it gets. Coder30 at TP=2, TP=4 and TP=8 all land at f = 0.305 to 0.33.
That is the mechanism behind record 6's result, and it is why R1 and R2 cannot
both hold on coder30.

**Raising f means raising P.** `df/dP > 0` iff `c(B) < tpot x b / (HBM x util -
overhead)`, and getting to f = 0.6 needs `P ~= 65 GB per GPU` with
`c(B) <~ 0.2`, which needs `E/k >= 32`. KV bytes per token cancels out of f
entirely; it only sets how many requests that byte budget buys, and therefore B,
and therefore c(B). Larger KV per token is mildly helpful, by keeping B and c
down.

## 4. What R2 asks for, in seconds

L1/L0 = P(gap > W) / P(gap < W) on the block-weighted reuse-gap distribution.
Inverting the corpus CDF:

| L1/L0 | needs P(gap > W) | W |
|---|---|---|
| 1.0 | 0.500 | 18.2 s |
| 2.0 | 0.667 | 12.6 s |
| 3.0 | 0.750 | 10.5 s |

So R2 is the statement `W in [10.5 s, 18.2 s]`. Today's coder30 operating point
at 50 tok/s gives W = 272 s and L1/L0 = 0.05.

That is a ceiling, not a prediction. L1 realises part of it: the window says
which blocks L0 no longer has, and the offload policy plus L1 capacity decide
how many of those L1 can actually return. Measured realisation in the archived
arms runs 20 to 75%. Aim the ceiling at 3 to 6 to land realised in [1, 3].

## 5. R7 is a queueing condition, not a prefill condition

TTFT p50 across the archived arms is bimodal, and the split is exactly
`waiting_mean`:

| arm | waiting_mean | f | TTFT p50 | TTFT p90 |
|---|---|---|---|---|
| r0_lazy | 0 | 0.10 | 1.15 s | 4.18 s |
| f8k256c48 | ~0 | 0.674 | 1.02 s | 2.73 s |
| l64r1 | 0.23 | 0.804 | 1.06 s | 5.62 s |
| e64r1 | 0.45 | 0.862 | 1.16 s | 9.97 s |
| l72b64L512d0cw | 70.0 | 0.965 | 194.63 s | 226.77 s |

Prefill itself is never the problem at these context lengths; l64r1 serves
102k-token prompts at TTFT p50 1.06 s against the corpus's own production
reference of 2.64 s. R7 holds as long as the free pool can admit an arrival's
prefill, which means keeping f off the 0.85+ shelf. The pick in section 9 runs
at f = 0.55 to 0.62 with 11 to 20 full contexts of free pool, so R7 is not the
binding constraint.

## 6. The catalogue

Everything on the box, at ISL 107k, tpot held at 20 ms, TP chosen per row.
`ctx/pool` is how many 107k contexts the pool holds. KV per token accounts for
vLLM replicating KV heads when `num_kv_heads < TP`.

| model | TP | kv B/tok | pool GB | B | f | tok/s | W s | L1/L0 ceil | ctx/pool |
|---|---|---|---|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B bf16 | 2 | 49,152 | 200 | 11.8 | 0.310 | 50.0 | 272 | 0.05 | 38 |
| Qwen3-Coder-30B-A3B bf16 | 4 | 49,152 | 462 | 26.8 | 0.305 | 50.1 | 280 | 0.05 | 88 |
| Qwen3.5-122B-A10B fp8 | 4 | 24,576 | 396 | 38.3 | 0.255 | 50.0 | 402 | 0.04 | 151 |
| Qwen3.8-Flash-Next fp8 | 2 | 12,288 | 75 | 21.3 | 0.372 | 50.1 | 118 | 0.13 | 57 |
| MiniMax-M2.7 fp8 | 4 | 126,976 | 293 | 9.4 | 0.436 | 50.3 | 61 | 0.24 | 22 |
| **Qwen3.5-397B-A17B fp8** | **4** | **30,720** | **117** | **18.4** | **0.518** | **50.1** | **21** | **0.83** | **36** |
| Qwen3-Coder-480B-A35B fp8 | 4 | 126,976 | 41 | 2.7 | 0.901 | 87.7 | 0.8 | L0 gone | 3 |
| Devstral-2-123B dense fp8 | 4 | 180,224 | 395 | 3.3 | 0.161 | 50.1 | 851 | 0.02 | 21 |
| GLM-5.3-Flash fp8 DSA2048 | 4 | 5,632 | 195 | 30.2 | 0.093 | 50.1 | 1921 | 0.01 | 323 |
| DeepSeek-V4-Flash DSA512 | 2 | 24,768 | 101 | 35.1 | 0.918 | 50.9 | 0.9 | L0 gone | 38 |

Readings:

- Dense loses on c. Devstral-2-123B reads all 128 GB of itself every step, which
  is two thirds of the TP=4 budget, so it holds 3.3 requests.
- Sparse attention is not the lever record 7 hoped for. It does break
  `read = stored`, and DeepSeek-V4-Flash does reach f = 0.92 with 35 in flight on
  two cards at 50 tok/s, which is the ideal shape. But GLM-5.3-Flash stores only
  5.6 KB per token, so its 195 GB pool holds 323 contexts and no achievable
  batch fills it: f = 0.09. Small stored KV and a useful L1 are opposed.
- DeepSeek-V4-Flash is still dead on Hopper. `expert_dtype = fp4` in the local
  snapshot's config, and the gate is present in both builds:
  `models/deepseek_v4/nvidia/model.py:285` (vllm-lazy) and `:335` (vllm-main),
  `NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")`.

Going to Hugging Face does not change this. The spec that section 3 derives is
`P ~= 65 GB per GPU, E/k >= 32, KV per token large enough that B stays near
15 to 20`; Qwen3.5-397B-A17B-FP8 satisfies it and is already on disk at
`/raid/data/hub/models--Qwen--Qwen3.5-397B-A17B-FP8`. The one model that beats
it is blocked by the cards, not by availability.

## 7. The pick

**Qwen3.5-397B-A17B-FP8, TP=4, fp8 KV, `gpu_memory_utilization` 0.9, no block
override, `max_model_len` 262144, LMCache L1 on CPU.**

406 GB of weights, 60 layers, `full_attention_interval` 4 so 15 full-attention
layers and 45 gated-linear, 2 KV heads at head_dim 256, 512 experts top-10.
E/k = 51.2 is the highest on the box, which is what keeps the expert read to
about 118 GB of the 192 GB step budget at B = 18. The pool is 117 GB because
the weights are large, not because a knob was turned: R4 and R5 are untouched.

Same pick as record 7 section 6, different numbers behind it: f = 0.518 rather
than 0.34, 18.4 in flight rather than 8.1, and an L1/L0 ceiling that crosses
R2's band instead of staying an order of magnitude below it.

## 8. Where the requirements meet, and how narrow it is

Concurrency is the run-time knob. ISL 107k, overhead 5 GB/GPU:

| B | f | tpot ms | tok/s | W s | L1/L0 ceil | free contexts |
|---|---|---|---|---|---|---|
| 14 | 0.394 | 15.9 | 63.0 | 59.8 | 0.25 | 21.5 |
| 16 | 0.451 | 17.8 | 56.3 | 38.2 | 0.40 | 19.5 |
| 18 | 0.507 | 19.6 | 51.0 | 23.7 | 0.70 | 17.5 |
| 20 | 0.563 | 21.4 | 46.7 | 13.1 | 1.94 | 15.5 |
| 22 | 0.620 | 23.2 | 43.2 | 9.0 | 4.48 | 13.5 |

R1 wants B <= 18, R2 wants B >= 20. They miss by two requests, which is inside
the uncertainty of every input. Two of those inputs move it the right way and
neither is a choice being made to get the answer:

- Overhead. 5 GB/GPU is fitted from a 30B model at TP=2. At 10 GB/GPU the pool
  is 97 GB and B = 16 gives 56.3 tok/s at ceiling 2.94.
- ISL. 107k is the arms' measured mean. At the corpus's own 142k, B = 14 gives
  57.4 tok/s at ceiling 2.02.

So the honest statement is that this configuration puts R1 and R2 within about
two requests of each other, on either side depending on inputs the model cannot
resolve, where coder30 misses by a factor of twenty (B = 12 for R1, B >= 28 for
R2). Whether they actually meet is a measurement, and it is a cheap one.

## 9. What still has to be true

1. That Qwen3.5-397B-A17B-FP8 loads at TP=4 at all, and what the pool is after
   activations and CUDA graphs. 406 + 117 leaves 20 GB across four cards. This
   is the single input the answer is most sensitive to and the cheapest to get.
2. That `c(B) = 1 - (1-k/E)^B` holds at E/k = 51.2. It was validated at
   E/k = 16 on coder30. Real routing is skewed, which makes c smaller than
   uniform, which helps.
3. That b = 2.4 GB/ms/GPU carries to 2 KV heads at head_dim 256 with 45 linear
   layers. It was fitted on coder30's 4 KV heads at head_dim 128.
4. That hybrid prefix caching works end to end. Record 7 section 5 checked the
   pieces: `mamba_cache_mode` defaults to `align` under prefix caching
   (`vllm/config/cache.py:191`), `lmcache/integration/vllm/kv_cache_groups.py:31`
   treats an align-mode Mamba spec as a one-block sliding window, and
   `kv_cache_group_edits.py` names Qwen3.5. Not exercised.
5. That the realisation fraction (section 4) is 20 to 75% here too. It is a
   property of the offload policy, which is this project's subject.

## 10. The probe

One server, four cards, no pair. Cards 0, 5, 6 and 7 are free; 1 through 4 are
running this account's own vLLM and LMCache processes (pid 3530903 and its
workers) and are not to be touched.

Two questions, one run each:

1. Does it load, and what is the pool? Start the server, read
   `GPU KV cache size: N tokens` from the log, stop. Minutes.
2. Where does the curve actually sit? Sweep CONC to place Running at roughly
   10 / 14 / 18 / 22, healthy region only, and at each point record from the
   engine log: `Running`, `GPU KV cache usage`, `Y/R`, `Waiting`, and both hit
   rates, plus `tokens_retrieved / isl_sum` from the snapshot. That is R1, R2,
   R7 and f, all on the same samples.

Not launched. Design goes to the user first.

## 11. Open

1. Section 8 is decided by the pool, and the pool is decided by an overhead I
   guessed. Everything else waits on probe 1.
2. The realisation fraction is measured at CONC=64 on coder30 only, where L1
   returned 10.0% of input tokens under lazy and 8.1% under eager. Whether that
   ratio survives at Running 18 on a different model is untested, and it is the
   same open item record `deployment_requirements.md` carries.
3. `Prefix cache hit rate` is a rolling gauge over the last 1,000 requests and
   swings by a factor of 70 inside one arm. Every L0 number in this record that
   comes from it should be read as indicative. The token ratios are not
   affected.
