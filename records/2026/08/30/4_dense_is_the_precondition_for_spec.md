# Dense is the precondition for spec

Record 3 closed the CONC=64 pair and left two things open: why speculative
decoding loses on this model, and whether any model on this box reaches the
50 tok/s per request the user treats as the healthy bar. They turn out to be the
same question. This record settles the mechanism, picks a model, and launches
the measurement.

## 1. The mechanism, third and final attempt

Three records have now attributed the fixed part of the decode step to three
different things.

| record | attribution | status |
|---|---|---|
| 13 | per layer launch overhead | withdrawn by record 14 |
| 14 section 1 | MoE expert GEMMs at M=1, cudagraphs on | stands |
| 1 section 6, repeated in 3 section 5 | per layer launch and sync | withdrawn here |

Record 14 section 1 measured it directly and said so in as many words: cudagraphs
are on, `enforce_eager=False` and `cudagraph_mode=FULL_AND_PIECEWISE` with a
capture size of 1 present, so the 75 percent fixed term is kernel execution at
M=1 and not launch overhead. Record 1 section 6 re-made the withdrawn attribution
at batch 24, and record 3 section 5 built on it to argue that spec might be
recoverable as a vLLM path question. That argument is withdrawn. Record 3 section 5
now carries a pointer here.

With the right mechanism the spec result is not a puzzle. Qwen3-Coder-30B-A3B is
128 experts, top-8. Count token to expert assignments per step:

| | token positions | assignments | active experts | M per expert |
|---|---|---|---|---|
| batch 24, no spec | 24 | 192 | ~101 (79%) | 1.9 |
| batch 24, 4 draft | 120 | 960 | ~128 | 7.5 |

Both are far below the M at which a grouped GEMM becomes compute bound. Cost is
therefore close to proportional to the number of assignments, which is
proportional to token positions, which is what a verify batch multiplies. The
measured step cost ratios are 3.9 to 5.3x at batch 1 and 4.4x at batch 25 for a
5x increase in positions. The arithmetic and the measurement agree.

The conclusion is now stated positively rather than as a failure to explain:
**on an A3B style MoE the per step cost is not fixed with respect to token count,
so speculative decoding has nothing to amortise.** This is a property of sparse
MoE at small M, not of long context, not of this corpus, and not of vLLM's verify
path. It is physics for this model.

The corollary is the useful half. On a dense model the weight read is one pass
regardless of how many token positions are in the step, and at 107k the KV read
is also one pass per step. Both are amortisable and together they are the
majority of the step. Spec should behave normally on dense.

## 2. Withdrawing the capacity half of the model argument

Earlier in this session I argued that a dense model with large KV per token would
help twice: spec would work, and the larger KV would shrink the pool in contexts,
pushing the capacity multiple toward 1.0 where the reuse curve puts L0 at 71.7
percent instead of 85, handing L1 the difference.

The second half double counts. Record 2 section 4 derives

```
window = cache_contexts / arrival_rate
       = [pool x (1-f) / ctx] / [pool x f / ctx / T]
       = T x (1-f)/f = (burstiness - 1) x OSL x tpot
```

and pool cancels. It cancels because `f`, the in-flight fraction, is fixed by the
rule the user uses to choose CONC, namely that peak in-flight KV should be about
the pool size. Under that rule a smaller pool gets a proportionally smaller CONC
and the capacity multiple is unchanged by construction. KV bytes per token does
not appear in the window and therefore does not move L1's share.

**The model lever is tpot and nothing else.** That is what record 2 section 4
said and what record 3 section 8 priced. The pick should be made on tpot, and KV
per token matters only for how much load the box can carry, not for how much work
L1 gets.

## 3. What is on the box

Enumerated from `/raid/data/hub`, no downloads. Requirement: dense MLP, long
context, and an architecture vLLM 0.23.0 recognises.

| model | dense MLP | context | attention | KV bytes/token fp8 |
|---|---|---|---|---|
| Qwen3.8-27B | yes | 262,144 | 16 full + 48 linear | 32,768 |
| Qwen3.6-27B | yes | 262,144 | 16 full + 48 linear | 32,768 |
| gemma-4-31B-it | yes | 262,144 | 10 full + 50 sliding(1024) | 81,920 |
| Mistral-Small-3.1-24B | yes | 131,072 | 40 full | 81,920 |
| Qwen2.5-14B-Instruct | yes | 32,768 | 48 full | 98,304 |
| Qwen3-Coder-30B-A3B (incumbent) | no, 128 experts | 262,144 | 48 full | 49,152 |

There is no pure full attention model with 262k on this box. Every modern long
context model here is hybrid, either linear attention or sliding window. That is
a finding rather than an inconvenience: the pure dense long context option does
not exist and the comparison has to be made against a hybrid.

Mistral-Small is the only pure full attention candidate and its 131k ceiling
truncates the corpus, whose ISL p90 is 209,408.

## 4. The pick, verified three ways

`Qwen/Qwen3.8-27B`, with `Qwen/Qwen3.6-27B` as an identical-architecture
fallback. 55.6 GB bf16, 64 layers, `full_attention_interval=4`, dense MLP with
`intermediate_size=17408` and no expert keys anywhere.

**Architecture is supported.** `Qwen3_5ForConditionalGeneration` and `Qwen3_5MTP`
are both in vLLM 0.23.0's registry. Contrast Qwen3.8-Flash-Next-FP8, which record
3 section 9 found declares `qwen4_exp` and does not load.

**MTP weights ship in the checkpoint.** Fifteen `mtp.*` tensors in
`model-00018-of-00018.safetensors`, verified by opening the shard rather than
trusting the index: `mtp.fc.weight` is (5120, 10240), taking concatenated
embedding and hidden back to hidden, plus one full transformer layer and three
norms. `vllm/model_executor/models/qwen3_5_mtp.py` defines `fc`,
`pre_fc_norm_hidden` and `pre_fc_norm_embedding`, and its `remap_weight_names`
rewrites `mtp.` to `model.`. The names line up.

**The config override resolves.** Run directly, without starting an engine:

```
model_type    qwen3_5                            -> qwen3_5_mtp
architectures ['Qwen3_5ForConditionalGeneration'] -> ['Qwen3_5MTP']
n_predict     None
```

`qwen3_5_mtp` at `vllm/config/speculative.py:46` is the architecture family name,
not a version. Both 27B models declare `model_type: qwen3_5` at the top level,
which is what line 460 matches.

`n_predict` resolving to None is a real vLLM inconsistency and harmless here.
Line 463 reads `mtp_num_hidden_layers` off the top level config while this
checkpoint is a multimodal wrapper that keeps it in `text_config`; line 474, the
`intern_s2_preview` branch two blocks down, reads the same key off `text_config`
and gets it right. `qwen3_5_mtp.py` never reads `n_predict`, using
`hf_text_config` throughout, so the None does not propagate.

Sizing on 2 x H200, 282 GB, util 0.90:

```
weights            55.6 GB
KV pool            ~190 GB / 32,768 B per token  =  5.8M tokens   (now 4.08M)
                   = 54 contexts of 107k                          (now 39)
per step, run 24   KV  24 x 107k x 32,768 = 84 GB -> 8.8 ms
                   weights          55.6 GB       -> 5.8 ms       fixed per step
                   roof                            ~15 ms
```

Two risks carried into the run. The MTP head is one layer, so acceptance is
capped at 2 and spec alone cannot take 15 tok/s to 50; the dense model has to be
faster on its own as well. And 48 of 64 layers are linear attention, whose
recurrent state is not prefix cacheable the way KV is, so the cache value
proposition for this line has to be re-established on a hybrid before it can
carry a demo. This repo has a `hybrid-benchmarking` skill for exactly that
ladder. Neither risk is in scope for the run below, which is `config=off`.

## 5. What is running

`mdl2.sh`, driver pid 3346892, started 05:49:24. Isolated: `config=off`, no
LMCache, no scenario, no aiperf. B=24, lengths 8000/32000/100000,
`KV_DTYPE=fp8 MAX_MODEL_LEN=262144 BLOCK=64`.

```
round A  05:49 - ~06:05   slot1 q38_nospec              slot2 q38_mtp   k=1
round B  ~06:05 - ~06:20  slot1 c30_ngram  4 draft      slot2 q38_mtp2  k=2
```

Four questions, one pass. Is the dense model faster at 107k, against the
incumbent's measured 13.54 / 35.58 / 155.71 ms. Does spec win once the model is
dense. What acceptance length does a one layer MTP head reach on this corpus.
And does ngram still lose on the MoE at batch 24, which record 13 only ever
measured at batch 1 and which section 1 above reaches by arithmetic rather than
by measurement.

Round B's `c30_ngram` is the control that makes section 1 a measurement instead
of a reading. If it comes back within noise of the incumbent baseline, the M
argument is wrong and section 1 has to be reopened.

## 6. Corrections in this record

1. Record 1 section 6 and record 3 section 5 attribute the fixed step cost to per
   layer launch and sync. Withdrawn. Record 14 section 1 had already measured
   cudagraphs on and identified MoE expert GEMMs at small M. Record 3 section 5
   now carries a pointer here.
2. Record 3 section 5's suggestion that spec might be recoverable as a vLLM
   verify path issue is withdrawn. On a sparse MoE at small M the step cost is
   proportional to token positions and there is nothing to amortise.
3. The claim made in conversation that a dense model with larger KV per token
   would help L1 twice, once through spec and once through a smaller pool, is
   withdrawn. Pool cancels out of the window identity under the CONC rule in use.
   The model lever is tpot alone.

## 7. Open

- `mdl2.sh` in flight. Nothing else running.
- Nothing pushed to any remote.
- The `max_deferral_seconds` default to zero PR, on a `_pr` branch without
  `records/`, not started.
- Hybrid prefix caching and offload for linear attention layers, unexamined.
- The isolated versus live tpot discrepancy from record 3 section 9, a factor of
  2.6 at nominally matched batch and length, still unexplained. It applies to
  everything measured by `bsweep.py` including the run above, so these numbers
  compare models against each other and must not be read as live predictions.
- L0 intersect L1 point in time overlap still unmeasured.
