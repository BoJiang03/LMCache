# Deployment requirements

Standing requirements for the lazy-offload demonstration, set by Bo Jiang.
This is the living reference; dated records under `records/YYYY/MM/DD/` are
session logs and do not supersede it. Status lines are evidence, not
requirements, and are the only part that should change without the requirement
owner saying so.

Last updated 2026-08-30.

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

## R2. L1 must carry a larger share of reuse

> 还要把l1重要性提升

Background: the standing project goal is an adaptive policy that decides per
step the minimum number of blocks to store to L1, to reduce store pressure on
L1 and reduce duplication between L0 and L1, held to 少存保持、不丢 block、
时延优势不回吐.

**Status: not reachable jointly with R1. See "Joint feasibility" below.** L1's
read share is `P(gap > T(1-f)/f)` with `f = in-flight KV / pool`. At 50 tok/s on
2xH200 with the full pool this is pinned at about 15 percent, currently measured
at 14.7 percent (all turns) or 17.1 percent (main agent turns only).

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

R1 and R2 cannot both be satisfied here, and R4 and R5 close the only
mathematical exit.

```
window   = T (1-f) / f          T = OSL x tpot,  f = in-flight KV / pool
```

A per token latency SLO fixes tpot. That fixes T, and it fixes the in-flight KV
budget, which is `(tpot - fixed term) x bandwidth`. With the pool held constant
by R4 and R5, f is constant, the window is constant, and L1's read share is
constant. Concurrency and context length trade freely inside the budget and
nothing moves. Raising f requires shrinking the pool, which R4 and R5 forbid.

Any two of the three are already in hand:

| | held | evidence |
|---|---|---|
| R1 + long context | yes | the low in-flight arms, 57 to 77 tok/s |
| long context + R2 | yes | the CONC=64 pair, L1 share 20.7 percent |
| R1 + R2 | **no solution** | this document |

Three escape routes were checked and closed:

- **More GPUs.** f falls, not rises: 0.190 at TP=2 to 0.148 at TP=16, because
  weights are sharded once while HBM scales linearly.
- **A different model.** `df/dP > 0` only for `c < 0.234`, where P is resident
  weight bytes and c the fraction read per step. KV bytes per token cancels
  entirely. Nothing on the box qualifies: the 122B has four times smaller KV per
  token so its pool holds more contexts, the 480B does not fit, and
  DeepSeek-V4-Flash has the right profile but FP4 experts that vLLM only runs on
  SM100, while these cards are SM90.
- **A different corpus.** Worth 2.4 points and forbidden by R6.

## Consequence for what may be claimed

R2 cannot be delivered as an L1 read share number. What lazy offload actually
does is on the write path, which is independent of f, because every block is
stored whether or not it is ever read. Store traffic falls only about a quarter
going from CONC=64 to in-flight 7, roughly 42k to 32k new prefix tokens per
second.

So the claim is stores, stored/isl, L1 watermark, tpot, waiting, with
cross-replica reuse stated as unmeasurable on a single replica harness rather
than omitted.

**This is a proposal, not an agreed requirement change.** R2 stands as written
until its owner changes it.

## Undetermined

1. Which CONC yields in-flight 7.
2. Whether lazy's write path advantage survives at in-flight 7. Untested, and
   the only thing that would invalidate the consequence above.
