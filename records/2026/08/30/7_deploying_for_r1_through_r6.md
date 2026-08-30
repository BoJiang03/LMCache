# Deploying for R1..R6: what the hardware allows

Rebuilt the deployment question from scratch against the requirements in
`records/deployment_requirements.md` (R1 50 tok/s per request, R2 L1 must carry
more reuse, R4 no pool shrink, R5 no gpu_memory_utilization reduction, R6 the
real agentx workload). Nothing here inherits the model choices or the harness
conventions of records 1..6.

## 1. The window formula in records 2 and 6 was missing a factor

Records 2 and 6 used `window = T(1-f)/f` with `T = OSL x tpot`, which assumes
each turn allocates its whole prompt into the pool. With prefix caching on it
does not. Measured on the corpus: of 338,052,834 blocks requested, 5,717,185
are new (1.7%), 323,588,385 are L0 hits (95.7%), 8,747,264 are L1 hits (2.6%).
A turn allocates 3,702 tokens on average, not 218,922.

The corrected identity is

    window = ((1-f)/f) x (ISL / A) x T_turn

where A is tokens allocated per turn (new prompt tokens plus blocks re-admitted
from L1) and ISL/A = 59 at the current operating point. That factor of 59 is
why record 6 predicted a 84 s window and 14.7% L1 share while the measured L1
share is 2.6%. The real window at the coder30 operating point is about 40
minutes.

A is not a constant: every block L1 serves is re-inserted into the pool, so A
grows with L1's own share. Solving the fixed point reproduces the measurement:
at f = 0.233 and T_turn = 17 s the model returns 1.1%, at T_turn = 8 s it
returns 3.4%, against 2.6% measured. That agreement is the reason to trust the
rest of the table below.

## 2. Corpus, measured this session

`reuse.tsv`, 98,827 turns, 2,090 conversations, 393 traces, 254.8 h of wall
time, block size 64 tokens.

| | all turns | main turns only (R6) |
|---|---|---|
| turns | 98,827 | 56,798 |
| ISL mean | 218,922 | 313,543 |
| new tokens per turn | 3,702 | 3,792 |
| ISL / new | 59.1 | 82.7 |

Block-weighted reuse gap, main turns: p50 21 s, p75 51 s, p90 166 s.
P(gap > 20 s) = 0.46, P(gap > 60 s) = 0.19, P(gap > 300 s) = 0.04.

Output length is not in `reuse.tsv`. From the dataset card of
`semianalysisai/cc-traces-weka-062126-256k`: 68,266 requests, 58,728,807 output
tokens, so OSL mean is 860. At 50 tok/s that is T_turn = 17.2 s.

Concurrent live conversations over the trace: mean 26, p50 10, p90 68, max 393.
Working set p50 4.14 M tokens, p90 22.49 M.

## 3. What actually sets L1's share

For full attention, read equals stored, so the in-flight KV in bytes is capped
by the latency budget and

    f = (tau x G x b - W) / (G x h x u - overhead - P)

with tau the target tpot, G GPUs, b the achieved marginal decode bandwidth,
W the weight bytes read per step, P the weight bytes resident, h the HBM per
GPU, u gpu_memory_utilization. KV bytes per token and ISL both cancel: they
set how many requests that budget buys, not what fraction of the pool is live.

So under R1 (tau fixed), R4 and R5 (u and the pool untouched), the only two
levers left are P up and W down. Bigger model, sparser activation.

For sparse attention the identity breaks, because read no longer equals stored:
f = B x ISL x kv / pool with B set by the weight read alone. That is the only
mechanism that can push f past what bandwidth allows, and it is what makes the
DSA models worth checking.

b = 1483 GB/s per GPU, from the slope of the 20-arm tpot fit (16.5692 ms per
1e6 in-flight-token-units at 49,152 B/token = 2,966 GB/s over two GPUs), 31% of
H200 peak. Every number below scales with b: if the achieved bandwidth is
twice this, every f doubles.

## 4. Sweep over the models on this box

ISL 313,543 (main turns), OSL 860, tau 20 ms, u 0.9, h 143,771 MiB, overhead
fitted to the measured 187 GiB pool at G=2. Weight bytes from checkpoint size
on disk. L1 columns are the fixed point of section 1 at T_turn = 17.2 s and at
8 s.

| model | G | pool GB | in flight | f | L1 @17s | L1 @8s |
|---|---|---|---|---|---|---|
| Qwen3-Coder-30B-A3B bf16 (baseline) | 2 | 201 | 3.0 | 0.233 | 0.011 | 0.034 |
| Qwen3-Coder-30B-A3B bf16 | 4 | 468 | 6.3 | 0.207 | 0.009 | 0.028 |
| Qwen3.5-122B-A10B fp8 | 2 | 134 | 6.6 | 0.201 | 0.009 | 0.028 |
| MiniMax-M2.5 fp8 | 4 | 298 | 2.2 | 0.298 | 0.019 | 0.047 |
| Qwen3.5-397B-A17B fp8 | 4 | 122 | 8.1 | 0.340 | 0.025 | 0.163 |
| Qwen3.5-397B-A17B fp8 | 8 | 656 | 18.9 | 0.148 | 0.006 | 0.013 |
| Qwen3-Coder-480B-A35B fp8 | 8 | 580 | 3.5 | 0.238 | 0.011 | 0.035 |
| GLM-5.1 fp8 (DSA topk 2048) | 8 | 306 | 6.9 | 0.316 | 0.022 | 0.056 |
| gpt-oss-120b mxfp4 | 4 | 463 | 15.6 | 0.194 | 0.008 | 0.025 |

Three configurations reach f = 0.9 and L1 near 1.0: MiniMax-M2.5 at G=2 (pool
31 GB), Qwen3-235B-A22B bf16 at G=4 (pool 59 GB), Qwen3-Coder-480B at G=4 (pool
46 GB). All three hold one to two conversations. They are not deployments and
are excluded.

Two structural readings of the table:

- More GPUs always lowers f for a fixed model. The pool grows linearly with G
  while the latency budget grows linearly too but the weight read is amortised
  across the extra cards only if the batch stays fixed, which it does not.
  Record 6 section 5 had the direction right.
- f = B / capacity, where capacity is the pool measured in conversations. With
  ISL 313k, no full-attention model on this hardware buys more than a handful of
  in-flight requests at 50 tok/s, so raising f means cutting capacity to the
  same handful.

## 5. Support checks

- DeepSeek-V4-Flash (149 GiB, DSA topk 512) is the ideal shape and is dead on
  H200. `vllm/models/deepseek_v4/nvidia/model.py:334` still raises
  `NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")` in vllm-main
  (0.28.1rc1), and the checkpoint is fp4-packed, which Hopper cannot consume.
- GLM-5.3-Flash (`Glm5NextForConditionalGeneration`) and Qwen3.8-Flash-Next
  (`Qwen4ExpForConditionalGeneration`) are not in any local vLLM registry.
- GLM-5.1-FP8 (`GlmMoeDsaForCausalLM`, DSA topk 2048, 757 GB) is registered in
  all four builds and is fully downloaded. It reads 237 GB of weights per step
  at its operating batch, which is the whole latency budget, so it lands at
  f = 0.316 with 6.9 requests in flight on eight cards.
- Qwen3.5-397B-A17B-FP8 (`Qwen3_5MoeForConditionalGeneration`) is registered.
  It is a hybrid: 45 gated-linear layers and 15 full-attention layers, 512
  experts top-10, 262,144 context.
- Hybrid prefix caching works. vLLM has `mamba_cache_mode` with an `align` mode
  that is the default when prefix caching is enabled
  (`vllm/config/cache.py:191`). LMCache handles the hybrid case explicitly:
  `lmcache/integration/vllm/kv_cache_groups.py:31` treats an align-mode Mamba
  spec as a one-block sliding window, and
  `lmcache/integration/vllm/kv_cache_group_edits.py` carries a Mamba page-view
  edit registry that names Qwen3.5 in its comments.

## 6. The pick

Qwen3.5-397B-A17B-FP8, TP=4, fp8 KV, gpu_memory_utilization left at its
default, no block override, max_model_len 262144. The paired arm runs on the
other four cards.

Why it is the pick and not something newer or larger:

- It is the only supported model on this box whose f exceeds the coder30
  baseline while still holding more than a handful of conversations. 407 GB of
  weights inside a 543 GB budget leaves 122 GB of pool, and the pool shrank
  because the weights are large, not because a knob was turned. R4 and R5 are
  untouched.
- 512 experts top-10 keeps the weight read at 77 GB per step at batch 8, which
  is what lets it hold 8 requests in flight at 20 ms rather than the 2 to 3 a
  dense-activation MoE of the same size would.
- 15 full-attention layers of 60 at 2 KV heads x 256 gives 15,360 B/token, so
  L1 carries 3.2x fewer bytes per cached token than coder30. That helps the
  write path independently of the read share.

Numbers to expect: f = 0.34 against 0.23 for the baseline, in-flight 8.1
requests, pool capacity about 24 conversations at 256k, tpot 20 ms, aggregate
about 400 tok/s on four cards.

Note on TP=4 with 2 KV heads: vLLM replicates KV heads up to the TP width, so
stored KV per token is 30,720 B, not 15,360. f is unchanged by this (it is a
byte ratio) but the in-flight count halves to about 4.

## 7. What this does not achieve

R2 is improved, not satisfied. L1's modelled share goes from 2.6% to between
2.5% and 16%, the range set by T_turn, which is the least certain input. The
fixed point is near critical: it is flat below f = 0.4 and runs away above
f = 0.55, and 0.34 sits on the flat part.

Reaching the runaway side needs f >= 0.5, which on this hardware needs either

- a pool of one to two conversations, which is not a deployment, or
- sparse attention with a model small enough to read inside the latency budget.
  The second exists (DeepSeek-V4-Flash) and is blocked by SM100.

So the honest claim from a Qwen3.5-397B deployment is a 1.5x improvement in f
and a several-fold improvement in L1's read share, on a curve where several-fold
is still a single-digit-to-teens percentage.

## 8. Open

1. T_turn. OSL 860 comes from the dataset card of the 256k build, which is 58%
   subagent requests. R6 says main turns only, and main-turn OSL is not
   separately recorded. This is the single input that moves the answer most.
2. Whether the reuse gaps are independent of serving speed. In a closed-loop
   agentic replay the gap is mostly the previous turn's response time, so both
   the window and the gaps scale with tpot and the ratio is invariant. If that
   is right, tpot drops out of the L1 share entirely and only f matters. Not
   verified against the harness.
3. Whether Qwen3.5-397B-A17B-FP8 loads at TP=4 at all, and what the real pool
   is after activations and CUDA graphs. Estimated overhead is fitted from a
   30B model at G=2 and is likely low for a 60-layer 397B model.
4. b = 1483 GB/s per GPU is a marginal slope fitted across 20 arms with 20%
   mean residual. Everything in section 4 scales with it.
