# Session log: the hybrid falls to correctness, and the re-pick that followed

Conversation record for 2026-08-30, continuing from the handoff in
`temp_ctx.md`. Outcome: the Qwen3.5 hybrid line is abandoned on Bo's call
after the L1 round trip failed correctness; the replacement pick is
Trinity-Large-Thinking-FP8-Block, chosen from a catalogue-wide sweep after
Bo directed the search off the local disk.

## 1. Blocker 2 was fixed, verified, and then reverted

The handoff's blocker 2 was the Mamba unified-view reshape crash:
`shape '[5542, 1072, 1, -1]' is invalid for input of size 6015508480` in
`_MambaUnifiedViewEdit.apply()`. Root cause: on vLLM 0.28 the registered
Mamba tensor's last dim is `state_content_size_bytes` (unpadded, 1085440),
which does not divide by block_size, while the block stride is the padded
page (`page_size_bytes` 1097728). The fix re-strided the view over
`spec.page_size_bytes` instead of trusting the tensor extent.

Two corrections to the handoff's proposed approach, kept here because they
survive the revert:

- The proposed `storage_offset == 0` check is wrong. Layer views are cut
  from one flat buffer, so layer 1 already sits at offset 3293184. The fix
  must bound-check against the underlying storage instead.
- The pass-through alternative (leave the tensor alone) is not viable.
  LMCache's rank-4 detector would read the raw shape as BS=1, so the edit
  is load-bearing.

Verified with 12 unit tests on real Qwen3.5-2B geometry plus a live engine
start (`mamba-unified-view: 18` applied). All of it reverted at Bo's request
("你把修复的代码还原。我换个模型"); the test file was deleted. If the hybrid
line is ever resumed, this section is the recipe.

## 2. Blocker 3, found and worked around: torch IPC across venvs

New, not in the handoff. vllm-lazy carries torch 2.11.0+cu130, vllm-main
carries 2.13.0+cu130, and CUDA IPC sharable handles are torch-version
specific: `received sharable handle from a future version of torch`. The
handoff's arrangement (MP server on vllm-lazy for its deps, engine on
vllm-main) can never register KV. Workaround: run the MP server on
vllm-main and `uv pip install --target` the missing packages into the
scratchpad (`opentelemetry-exporter-prometheus`, `cupy-cuda13x`, and
friends). No shared environment was mutated.

## 3. The verification verdict: retrieval is consistent but not faithful

Layered check on Qwen3.5-2B, MP mode, align mamba cache, needle probe with
880 unique log lines and a planted fact.

- vLLM's own local prefix cache reproduces the cold decode 5/5.
- LMCache L1 reproduces it 0/5 under NHD and 0/5 under HND.
- L1 retrieval is deterministic and self-consistent (3/3) and the needle
  is recalled, so the state is usable, just not equal.
- Same top-1 token and top-5 set; max logprob delta 0.34 nats, far above
  fp8 rounding.

A separate real bug surfaced on the way: on vLLM 0.28 the FULL_ATTENTION
group registers as logical (B,H,N,C) = (5542, 2, 1072, 512) with stride
(1097728, 512, 1024, 1), and LMCache's rank-4 NHD detector reads BS=2,
NH=1072 where the truth is BS=1072, NH=2. That affects every model on
0.28, not just hybrids, and cannot be fixed inside
`apply_kv_cache_group_edits` because the registry is gated on
`has_mamba_layers`. Forcing HND makes the registered shape match and the
round trip still fails, so this bug is real but not the sole cause. The
leading suspect for the residual is align-mode state snapshot timing: the
per-block mamba state LMCache stores may be the state after the whole
prompt rather than at the block boundary. Unproven; the line was stopped
here.

## 4. The re-pick, round one: the disk

With hybrids off the table the local catalogue reduces to record 8's own
table. MiniMax-M2.7 fp8 TP=4 is the only runnable non-hybrid: R1 and R7
hold, L1/L0 ceiling 0.24 with fp8 KV or 0.72 with bf16 KV, both short of
the [1, 3] band. Everything else fails harder: GLM-5.1 (705G) does not fit
four cards, GLM-5.2-FP8 is a config-only download whose expert count alone
computes to ~727B params, GLM-5.3-Flash and Qwen3.8-Flash-Next are
linear-attention hybrids, DeepSeek-V4-Flash's local snapshot is fp4-expert
and SM100-gated, Qwen3-Coder-480B leaves a 41 GB pool, Qwen3-235B at
E/k=16 spends the step budget on experts.

## 5. Bo's correction, and round two: the world

> btw，我没说一定要是盘上的模型吧？你要充分考虑各种模型

Same correction as record 10 section 4, and it changed the answer this
time. The search spec, from record 8: P in 300-430 GB fp8, E/k >= 32,
long-context KV carried by few layers, no mamba/GDN. Swept and rejected:

- MiniMax-M3 (428B/A23B, GQA + blockwise sparse attention, E/k=32):
  vllm-main only (`MiniMaxM3Sparse*` absent from vllm-lazy 0.23), which
  re-enters the 0.28 wall; pool ~82 GB is R7-marginal anyway.
- Kimi K2, DeepSeek-V3.2, DeepSeek-V4: over 560 GB, do not fit.
- Llama-4 Maverick (400B, E/k=128): numbers close, but April 2025 vintage
  and weak at agentic coding; fails "选较新的".
- DeepSeek-V4-Flash-FP8 (sgl-project, 291B/A13B, MLA + DSA, E/k=42.7):
  viable fallback. The fp8 repack dodges the SM100 gate and vLLM issue
  #47648 says H200 works without DSpark. At B=25: ~71 tok/s, f=0.64,
  window ~7 s, ceiling 4-6, 16 free contexts. Cost: MLA and DSA are both
  unexercised paths in LMCache, two new risks instead of one.

## 6. The pick: Trinity-Large-Thinking-FP8-Block

arcee-ai, 398B total, 13B active, Apache-2.0, tech report
arxiv.org/html/2602.17004v1. Architecture `afmoe`: 60 layers of which 15
are full attention and 45 sliding-window 4096 (`global_attn_every_n_layers`
4), GQA 8 KV heads at head_dim 128, 256 experts top-4 so E/k=64.

It is the 397B pick's winning shape without the mamba: long-context KV
lives in 15 layers at 30,720 B/token, the same figure as Qwen3.5-397B, and
the sparsity is higher (64 vs 51.2). Estimated operating point with record
8's constants: weights ~400-420 GB, pool ~100 GB holding ~27 contexts of
107k, B~18 at ~53 tok/s, f~0.6, ~9 free contexts, window ~7 s, L1/L0
ceiling in the 3-6 aim band. R1, R2, R7 all hold on paper.

Why it dodges all three blockers:

1. No mamba, so no state snapshot semantics and no unified-view edit.
2. `AfmoeForCausalLM` is in vllm-lazy 0.23's registry with attn_type
   `decoder`, so prefix caching is allowed and the whole verified stack
   applies: no vllm-main, no 0.28 rank-4 layout bug, no torch IPC split.
3. No forced attention block size, so chunk 256 divides cleanly.

Remaining risk, one item: LMCache offload semantics for the sliding-window
layers next to full-attention layers have not been exercised on this
stack. The gemma line's SWA support is circumstantial evidence, not proof.
Weights are ~400 GB to pull from HF.

Bo asked for the single recommendation and it is this model. M2.7 remains
the zero-risk floor at the price of giving up R2.

## 7. State at close

- `kv_cache_group_edits.py` back at HEAD; my test file deleted.
- Blocker 1's kv_layout fix (utils, mp_connector, service factory, layout
  hints test) still uncommitted from the prior session; committed with
  this record as part of the snapshot.
- The 2B probe server was left running on GPU 7 (ports 8973/8971/8972) at
  Bo's instruction earlier in the session.
- Next step, pending design discussion: pull Trinity, start it on
  vllm-lazy at TP=4, and rerun the layered correctness ladder, needle
  probe included, before any sweep.
