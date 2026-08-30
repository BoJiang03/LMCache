# Deployment requirements

Standing requirements for the lazy-offload demonstration, set by Bo Jiang.
This is the living reference; dated records under `records/YYYY/MM/DD/` are
session logs and do not supersede it. Status lines are evidence, not
requirements, and are the only part that should change without the requirement
owner saying so.

Last updated 2026-08-30. R2 gained a numeric band and R7 was added on
2026-08-30; see those sections.

## R1. Per request decode must reach about 50 tok/s

> 正常情况下50左右才比较好
>
> 我们必须要把 per request tok/s 搞到50左右

Operationally: median per request decode throughput, measured client side over
the profiling phase, on the CC agentic workload at its native context length.
Not aggregate throughput, not a short context number.

50 tok/s is the market rate for this model. OpenRouter lists Qwen3-Coder-30B-A3B
at Alibaba Cloud 53 tok/s, NovitaAI 47, SiliconFlow 21.

**Status: reachable, operating point not yet pinned.** The tpot law fitted over
twenty existing arms, `tpot_ms = 8.03 + 16.5692 x (in-flight x ISL / 1e6)`,
puts 50 tok/s at in-flight 6.8 for ISL 107k. Four arms already on disk deliver
57 to 77 tok/s at in-flight 1.7 to 3.6, at ISL 165k to 204k, which is heavier
than the target. The CONC that produces in-flight 7 is unmeasured: in-flight
over CONC is 0.19 to 0.26 in the low load arms and 0.39 at CONC=64, and the
middle has never been run.

## R2. L1 must carry a larger share of reuse, and L1/L0 must land in [1, 3]

> 还要把l1重要性提升
>
> L1 must carry a larger share of reuse，并且 l1 / l0 in [1,3]

Background: the standing project goal is an adaptive policy that decides per
step the minimum number of blocks to store to L1, to reduce store pressure on
L1 and reduce duplication between L0 and L1, held to 少存保持、不丢 block、
时延优势不回吐.

Operationally: L1 tokens returned over L0 tokens hit, on a common denominator.
From the engine log that is `E x (1-H) / H` with `H` the local prefix cache hit
rate and `E` the external one, which vLLM measures on the residual after L0.

**Status: not reachable with Qwen3-Coder-30B. A configuration that reaches it
is identified but unmeasured.** L1's share is `P(gap > W)` with
`W = D(1-f)/(f(1-L0))`, so the band [1, 3] is the statement
`W in [10.5 s, 18.2 s]` on this corpus. Coder30 at 50 tok/s runs at f = 0.345
(engine measured), giving W = 272 s and L1/L0 = 0.05; reaching 1.0 needs
Running 28, which is 24 tok/s. Qwen3.5-397B-A17B-FP8 at TP=4 puts R1 and R2
within about two requests of each other. See
`records/2026/08/30/8_the_deployment_that_meets_r1_r7.md`.

## R4. The KV pool may not be shrunk

> 实际部署不可能缩池子啊

No `--num-gpu-blocks-override`, no artificially reduced pool. A real deployment
takes all the KV it can get.

**Status: binding, accepted.** Proposed three times in different forms and
withdrawn each time. Recorded in `records/2026/08/30/6_the_slo_pins_the_window.md`
section 7.

## R5. `gpu_memory_utilization` may not be lowered

> 难不成他们 也缩 gpu util？那不是疯了吗

Stays at 0.9. This is the same constraint as R4 by a different route and is
listed separately because it was raised separately.

**Status: binding, accepted.**

## R7. TTFT must be under 10 s, preferably under 5 s

> ttft 得< 10s 最好<5s

Operationally: client-side TTFT p50 under 5 s and p90 under 10 s, on the agentic
workload at its native context length. The corpus's own recorded production
figures are p50 2.64 s and p90 6.98 s, so this is the market rate, not a
stretch.

**Status: held whenever the engine is not queueing.** TTFT p50 across the
archived arms is bimodal and the split is `waiting_mean`: arms with
`waiting_mean` under 0.5 give TTFT p50 1.02 to 1.16 s at ISL ~102k, arms that
queue give 190 to 300 s. Prefill is not the binding cost at these context
lengths. R7 reduces to keeping the free pool able to admit an arrival, which
means keeping f off the 0.85+ shelf.

## R6. The corpus must reflect real usage

> 我自己用了这么久的claude code，基本没怎么启动过subagent

The workload must not be chosen, filtered, or weighted to favour the result.

**Status: known bias, quantified, not corrected.** The corpus in use,
`semianalysis_cc_traces_weka_with_subagents_062126_256k`, carries 4.3 subagents
per main session (1,697 subagent conversations against 393 main, 42.5 percent of
turns). Measured against 135 local Claude Code sessions from 2026-08-05 to
08-30: 34 `Agent` invocations total, 0.25 per session, 3.0 percent of turns.
**The corpus over-weights subagent fan-out by about 17x.**

Not corrected, for two reasons. Subagents carry only 17.7 percent of reused
blocks, so stripping them entirely moves L1's share from 14.7 to 17.1 percent,
2.4 points. And switching to `semianalysis_cc_traces_weka_no_subagents` to
collect those 2.4 points would be choosing a corpus for its answer, which R6
forbids. The bias is recorded instead.

If a second corpus is added later it must be as corroboration, both corpora
reported, never as a replacement.

## Constraints these operate under

Not requirements about the deployment, but they bound what can be run.

- At most 4 GPUs. Paired A/B uses 2 per arm, so each arm is TP=2.
- Go easy on host memory. Only touch processes owned by uid 1016.
- No shared environment mutation: no rebuilding `lmcache/*.so`, venvs, `/raid`,
  or `/usr/local`.
- Paired comparisons run two rounds with the slots swapped, or slot effects
  contaminate the result.
- No experiment is launched before its design has been discussed.

## Joint feasibility

R1 and R2 cannot both be satisfied with Qwen3-Coder-30B, for a reason that also
says which model would satisfy them.

```
W    = D (1-f) / (f (1-L0))            D = request latency, L0 = f_L0(W)
tpot = (c(B) P_exp + P_dense + f pool) / (b N)
f    = B ISL kv / pool,  pool = N (HBM u - overhead) - P
```

R1 fixes tpot, which caps the bytes read per step. R4 and R5 fix the pool at
whatever the model leaves. So as the weights go to zero,

```
f_max = tpot x b / (HBM x u - overhead) = 20 x 2.4 / 130.67 = 0.367
```

and any model small next to HBM sits at f ~= 0.3 at 50 tok/s regardless of GPU
count. Coder30 lands at 0.305 to 0.33 at TP=2, 4 and 8.

Raising f means raising P: `df/dP > 0` iff `c(B) < tpot b / (HBM u - overhead)`,
and f = 0.6 needs about 65 GB of weights per GPU with `c(B) <~ 0.2`, hence
`E/k >= 32`. KV bytes per token cancels out of f; it only sets B, and through B
it sets c. This is R4-compatible: the pool shrinks because the model is large,
not because a knob was turned.

The escape routes, rechecked:

- **More GPUs.** Does not help. f is flat in N for a fixed model, because the
  pool and the byte budget both scale with N.
- **A different model.** This is the lever. Qwen3.5-397B-A17B-FP8 at TP=4
  (406 GB of weights, 512 experts top-10, 117 GB pool) reaches f = 0.518 at
  50 tok/s with 18 in flight, against coder30's 0.31. R1 and R2 land within two
  requests of each other. Unmeasured.
- **Sparse attention.** Breaks `read = stored` and is the only thing that can
  push f past what bandwidth allows. DeepSeek-V4-Flash has exactly the right
  shape and is blocked on Hopper (`expert_dtype = fp4`, and
  `models/deepseek_v4/nvidia/model.py` raises on non-SM100). GLM-5.3-Flash runs
  but stores 5.6 KB per token, so no achievable batch fills its pool: f = 0.09.
- **A different corpus.** Worth 2.4 points and forbidden by R6.

## Consequence for what may be claimed

Superseded in part. The earlier reading, that R2 cannot be delivered as an L1
read share number and the claim has to move to the write path, was based on f
being pinned near 0.19 by the latency SLO. It is not pinned; it is set by how
much of HBM the weights occupy, and a large enough sparse MoE moves it. R2 is
now a model selection question, not an impossibility.

The write path claim stays as the fallback and stays true either way: lazy's
effect on stores is independent of f, because every block is stored whether or
not it is ever read.

Whichever way the probe lands, report stores, stored/isl, L1 watermark, tpot,
waiting, TTFT and the two hit rates, with cross-replica reuse stated as
unmeasurable on a single replica harness rather than omitted.

## Undetermined

1. Whether Qwen3.5-397B-A17B-FP8 loads at TP=4 and what pool is left after
   activations and CUDA graphs. Everything in the R2 status hangs on it.
2. Where the concurrency curve actually crosses, i.e. whether R1 and R2 meet.
3. Whether lazy's write path advantage survives at Running 18 on that model.
   Measured only at CONC=64 on coder30, where L1 returned 10.0 percent of input
   tokens under lazy against 8.1 percent under eager.
