# AgentX workload selection and benchmark environment setup

Date: 2026-08-25
Branch: lazy-offload-publish (code), lazy-offload-policy-repro (repro package)
Code state: clean, no commits this session. All work was investigation and
environment setup outside the repo tree.

## Goal

Two open problems after lazy-offload dev completed:

1. A better workload. The four workloads in the repro package (hot/cold,
   GSM8K, QASPER sweep, SWE-agent replay) sit at the edges of the policy's
   envelope. The SWE-agent cohort's median continuation prompt was 5635
   tokens against a measured break-even near 6000, so its median request
   could not win.
2. Understanding the implementation. Requested as an interactive chat
   walkthrough; started, not finished.

## Workload decision

Rejected: replaying local Claude Code transcripts under ~/.claude/projects.
66 sessions from one developer on one repo is a sample, not a serving
workload.

Compared three public options:

| | AgentX cc-traces-weka | TraceLab | Inferact codex_swebenchpro |
|---|---|---|---|
| source | 393 Claude Code sessions, proxy-captured | 43 developers, Claude + Codex | 610 Codex runs on SWE-bench Pro |
| requests | 98,827 | 357,161 rounds | 20,230 |
| subagents | 1,697 groups | none | none |
| prefix truth | 64-token chained block hashes | prefix/appended token split | real text |
| license | Apache 2.0 | GitHub release | MIT |

Chose AgentX. Reasons specific to this PR: median input is ~20x the
break-even so gate 3 stops being marginal; subagent fan-out is a KV
structure the others lack and is the sharpest test of "do not copy what
the GPU can still serve"; inter-turn tool time is preserved in the replay
DAG and that gap is gate 1; block hashes make reuse ground truth rather
than a counter we emit. LMCache also already has an AgentX industry-impact
page, so results land on an existing vendor-neutral scoreboard.

Content is deterministic synthetic tokens reconstructed from hashes. Fine
for cache-structure claims, useless for anything content-dependent, so the
GSM8K correctness run stays.

## Corpus characterization (measured from traces.jsonl, not the blog)

393 traces, 98,827 requests, 42,029 of them subagent-inner.

- input tokens: p50 142,016  p90 549,504  p99 863,424  max 989,824
- output tokens: p50 444, so 203:1 input:output
- turns per trace: p50 67, max 3052
- think_time between turns: p50 2.3 s, p90 37.5 s, p99 1392 s (23 min)
- subagent groups: 1,697, p50 16 inner requests, max 789

Schema: {id, models[], block_size:64, hash_id_scope, requests[]} where a
request is {t, model, in, out, hash_ids[], api_time, think_time, ttft,
type}. type is "s" (streaming main turn), "n" (non-streaming), or
"subagent" (a group carrying agent_id, duration_ms, status and its own
nested requests list).

The 256k variant is not a tail clip. Against the full corpus it drops main
turns 56,798 -> 28,444 and total input tokens 21.6B -> 6.9B, a 68% cut.
Over 10% of requests exceed 256k.

## Model selection

Constraint from the user: use at most 4 GPUs, not the whole box.

KV per token computed from each model's config.json, bf16 KV:

| model | weights | KV/token | ctx | TP=4 fit |
|---|---|---|---|---|
| Kimi K3 | 1561 GB | 105 KiB MLA | 1M | no, exceeds 8x H200 |
| DeepSeek-V4-Pro-0813 | 893 GB | ~122 KiB | 1M | no |
| GLM-5.2-FP8 | 756 GB | 88 KiB MLA | 1M | no, needs TP=8 |
| MiniMax-M3 | 854 GB | 120 KiB | 1M | no |
| Qwen3.5-397B-A17B-FP8 | 406 GB | 30 KiB (45 linear + 15 full) | 256k | yes, ~124 GB KV |
| DeepSeek-V4-Flash | 160 GB | ~86 KiB | 1M | yes, ~370 GB KV |
| Qwen3.5-122B-A10B-FP8 | 127 GB | 24 KiB (36 linear + 12 full) | 256k | yes at TP=2 |
| Qwen3-Coder-480B-FP8 | 482 GB | 248 KiB | 256k | no useful headroom |
| MiniMax-M2.7 | 230 GB | 248 KiB | 205k | fits but only ~9 median reqs |

Two findings:

1. Every model actually served now has compressed or hybrid KV: MLA (Kimi,
   GLM, DeepSeek) or linear-attention hybrids (Qwen3.5 runs 45 of 60 layers
   linear). Dense GQA at 248 KiB/token is the previous generation. So
   "avoid LMCache's hybrid code path" is unsatisfiable; picking plain GQA
   means picking an unrepresentative model. Run the hybrid path
   deliberately and validate it.
2. The policy's constants were calibrated against the wrong economics.
   min_prefix_tokens and horizon_steps=2.5 were tuned on Qwen3-8B at 144
   KiB/token. Real served models are 24-122 KiB/token, so fetch is up to 6x
   cheaper and gate 3's break-even prefix length drops proportionally. This
   probably needs a recalibration sweep inside the AgentX work.

NVFP4 is not an escape hatch on H200. GLM-5.2-NVFP4 is 465 GB and would fit
TP=4, but on Hopper vLLM falls back to Marlin (memory savings, no throughput
gain), and vllm issue #49070 reports NVFP4-MoE on sm90 producing garbage
output plus CUDA illegal memory access, specifically for MiniMax-M3-NVFP4 on
H200.

Final roles, after the 256k truncation finding:

- DeepSeek-V4-Flash, TP=4: headline. Only way to run the full corpus at <=4
  GPUs. TP=2 is not enough: a 990k-token request is ~81 GiB of KV and TP=2's
  ~106 GB headroom leaves no room for concurrency.
- Qwen3.5-397B-A17B-FP8, TP=4: board-model cross-check on the 256k corpus.
- Qwen3.5-122B-A10B-FP8, TP=2: adversarial cheap point, 24 KiB/token, where
  offload has least to offer.

## Environment set up

- aiperf 0.12.0 in /home/bo/venvs/aiperf. Required python3.12; the system
  python is 3.10 and silently resolved to 0.11.0, which does not carry the
  SemiAnalysis loader. Verify with:
  aiperf plugins public_dataset_loader | grep semianalysis
- Date-pinned loaders available: semianalysis_cc_traces_weka_062126 and
  semianalysis_cc_traces_weka_062126_256k. Pin these, do not use the rolling
  with_subagents alias.
- Corpus downloaded to /raid/data/hub (HF_HUB_CACHE), both variants.
- Models downloaded to /raid/data/hub: Qwen3.5-122B-A10B-FP8 (119 GB, done),
  Qwen3.5-397B-A17B-FP8 (406 GB).
- DeepSeek-V4-Flash already local, 160 GB, 92 shards.
- Measured HF throughput on this box: 36 MB/s single stream, 142 MB/s across
  6 streams.

## Sizing and schedule

Machine: 8x H200 141 GB, 2 TB DRAM, 160 cores, 6.4 TB free on /raid. Shared
with another user, so plan for 4 GPUs.

For 12 concurrent sessions at ~200k final context:

| | working set | pool sweep | L1 |
|---|---|---|---|
| DeepSeek-V4-Flash TP=4 | ~197 GiB | 60 -> 200 GiB | ~400 GB |
| Qwen3.5-397B TP=4 | ~69 GiB | 20 -> 80 GiB | ~150 GB |

At cheap KV the working set can be smaller than the pool, so no natural
eviction pressure forms. Pool size has to be set explicitly with
--kv-cache-memory, the same lever the QASPER and agentic sweeps used. State
that as an assumption in any writeup rather than burying it in a flag.

Estimated: ~23 GPU-hours for a 12-run matrix (3 configs x 2 pool points x 2
reps with reversed order) at ~1.3 h per run following AgentX's one-hour
profiling window. Integration is the schedule risk, not GPU time.

## Open items

1. Point aiperf at a vLLM + LMCache MP connector server on DeepSeek-V4-Flash
   and get one short replay to complete. Not yet attempted.
2. Confirm DeepSeek V4 KV width against vLLM's actual allocation. Its config
   exposes num_key_value_heads=1, head_dim=512, qk_rope_head_dim=64 with no
   kv_lora_rank, so the cached width may be 576/layer rather than 1024 and
   the 86 KiB figure could be ~1.8x high.
3. Confirm vLLM 0.23.0 in venvs/vllm-lazy serves these models at the needed
   context.
4. Recalibrate min_prefix_tokens and horizon_steps for 24-122 KiB/token.
5. Finish the code walkthrough. Menu offered: (a) drain loop and danger
   depth, (b) epochs and receipt window, (c) one request end-to-end, (d)
   connector/manager boundary. None started.
6. Budget the pilot for finding a bug rather than producing a number. AgentX
   already surfaced LMCache #3382 (KV pool deadlock at 100k+ contexts) and
   #4524 (shared-chunk lock accounting under DRAM offload).

## References

- https://inferencex.semianalysis.com/agentx
- https://inferencex.semianalysis.com/agentx/optimizations/lmcache
- https://huggingface.co/datasets/semianalysisai/cc-traces-weka-062126
- https://github.com/SemiAnalysisAI/InferenceX
- https://github.com/uw-syfi/TraceLab
- https://arxiv.org/abs/2608.00101 (GitHub Copilot production characterization)
