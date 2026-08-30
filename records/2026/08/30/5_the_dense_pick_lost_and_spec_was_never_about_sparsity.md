# The dense pick lost, and spec was never about sparsity

Round A and round B of `mdl2.sh` landed at 06:07. Four arms, one isolated
harness, `config=off`, B=24, no LMCache. The run was designed in record 4
section 5 to answer four questions. It answered all four, and two of the
answers overturn record 4 section 1.

## 1. The measurement

Client tpot, milliseconds per output token, `bs.sh` at B=24.

| arm | model | spec | 8k | 32k | 100k | fit at 107k |
|---|---|---|---|---|---|---|
| m_coder30 | Qwen3-Coder-30B-A3B | none | 13.54 | 35.58 | 155.71 | 166.5 |
| c30_ngram | Qwen3-Coder-30B-A3B | ngram, 4 draft | 38.36 | 57.70 | 201.95 | 214.4 |
| q38_nospec | Qwen3.8-27B | none | 30.76 | 96.15 | 330.06 | 352.8 |
| q38_mtp | Qwen3.8-27B | MTP k=1 | 49.44 | 158.14 | 578.05 | 618.3 |
| q38_mtp2 | Qwen3.8-27B | MTP k=2 | 62.14 | 201.88 | 610.36 | 652.1 |

Acceptance, token weighted over the whole run from the server metrics:

| arm | drafts per step | accepted | drafted | acceptance length | per position |
|---|---|---|---|---|---|
| c30_ngram | 4 | 3230 | 20159 | 1.641 | .295 .174 .090 .054 |
| q38_mtp2 | 2 | 15879 | 22236 | 2.428 | .82 .65 |

The k=1 window was not captured. `off_vllm.log` is truncated per server start
and round B overwrote slot 2 before the watcher read it. The 1.80 used below is
inferred from the k=2 position 0 rate and is the one soft number in this record.

Cudagraphs were on for every arm. `FULL_AND_PIECEWISE`, 48 or 49 sizes captured,
`enforce_eager=False`, capture completed. The MTP arm loaded a real drafter
(`Loading drafter model`, then `Detected MTP model. Sharing target model
embedding weights` and the same for `lm_head`), so it is a one layer head and
not a second copy of the target. Record 4 section 4 called the `n_predict=None`
vLLM inconsistency harmless. That still holds.

## 2. The dense model is slower, and the loss is not where I said it would be

`q38 / c30` no spec, per token: 2.27 at 8k, 2.70 at 32k, 2.12 at 100k.

Record 4 section 2 established that the only model lever on L1's share is tpot.
This swap moves tpot the wrong way by a factor of two at every length. **The
dense pick is withdrawn. The incumbent stays.**

The interesting part is where the two seconds go. Fixed terms are 1.175 ms
(c30) and 4.734 ms (q38), a 3.5 ms difference, which is the dense weight read:
27B bf16 over two GPUs is 27 GB per GPU, 5.6 ms at H200 peak, and the measured
4.7 ms is that number. Dense costs exactly what dense should cost, and it is
nothing at 107k.

The gap is almost entirely slope: 3.2533 vs 1.5453 ms per 1k tokens. That is
backwards. Qwen3.8-27B is 16 full attention layers and 48 linear attention
(`layer_types` counts 48 `linear_attention`, 16 `full_attention`,
`full_attention_interval` 4). Its KV is 32,768 B/token against the MoE's
49,152, and only a quarter of its layers read KV at all. Its context slope
should be far smaller. It is 2.1x larger.

Converting the slopes to effective KV bandwidth per GPU at TP=2:

| model | slope ms/1k | KV B/token | implied | share of H200 peak |
|---|---|---|---|---|
| c30, 48 full layers | 1.5453 | 49,152 | 382 GB/s | 8.0% |
| q38, 16 full layers | 3.2533 | 32,768 | 121 GB/s | 2.5% |

Neither arm is anywhere near bandwidth bound in this harness. This is record 3
section 9's unexplained isolated-versus-live 2.6x, seen from the other side, and
it is worse on q38 than on c30. Section 5 below is about what that does to the
rest of this record.

## 3. Spec: the amortisation is real, and it is on the wrong model

Client tpot hides the step. Multiply by acceptance length to recover it.

Step time, ms:

| arm | positions per request | 8k | 32k | 100k |
|---|---|---|---|---|
| c30_base | 1 | 13.5 | 35.6 | 155.7 |
| c30_ngram | 5 | 62.9 | 94.7 | 331.4 |
| q38_base | 1 | 30.8 | 96.2 | 330.1 |
| q38_mtp1 | 2 | 89.0 | 284.7 | 1040.5 |
| q38_mtp2 | 3 | 150.9 | 490.2 | 1482.0 |

Now price one extra draft position in units of one no spec step. This is the
whole question. Spec wins when a position costs less than a step.

| model | 8k | 32k | 100k |
|---|---|---|---|
| c30, per position | 0.91 | 0.42 | 0.28 |
| q38, per position | 1.89 / 2.01 | 1.96 / 2.14 | 2.15 / 1.34 |

The MoE amortises, and it amortises more the longer the context gets. At 8k a
draft position costs 0.91 of a full step, which is the proportional behaviour
record 4 section 1 predicted from small M expert GEMMs. At 100k it costs 0.28.
The KV read has become the step, the KV read is shared across positions, and
there is now three quarters of a step to amortise.

The hybrid does not amortise at all. Every extra draft position costs about two
full steps, at every length, with no trend. This is worse than serial decoding.

## 4. What that does to record 4 section 1

Section 1 claimed that an A3B MoE runs expert GEMMs at M around 2, so step cost
is proportional to token positions, so spec has nothing to amortise, so the
failure is a property of sparsity and a dense model should fix it. Two halves,
and they come apart.

**The mechanism half is confirmed, at short context only.** 0.91 per position at
8k is proportionality, measured, at our batch, which is what round B was for.
Record 13 only ever had batch 1.

**The scope half is wrong.** It is not true at the length we operate at. At 100k
the MoE amortises fine. Section 1 generalised an 8k mechanism to a 107k
workload without checking that the KV term reverses it.

**The forward claim is falsified.** Dense was supposed to let spec win. The
drafter was good, 2.428 acceptance with 0.82 at position 0, so drafter quality
was never the obstacle, and it lost anyway by 1.85 to 2.10x per token. Whatever
kills spec on Qwen3.8-27B is not sparsity, because there is no sparsity.

I picked the model knowing it was hybrid. Record 4 section 3 says in as many
words that there is no pure full attention 262k model on the box. I recorded
that as an availability fact and did not carry it into the hypothesis, where it
belonged: 48 recurrent layers break "extra positions are nearly free" for their
own reasons, exactly as small M does. The premise needed a full attention model
and no such model was available at this context length. That should have been
stated before the run, not after it.

I am not naming the hybrid mechanism. Three records have now attributed this
class of cost to launch overhead, then to expert GEMMs, then to sparsity, and
two of those were withdrawn. What is measured is that a draft position on
Qwen3.8-27B costs about two no spec steps and that the cost scales with context.
A recurrent state that is O(1) in length cannot by itself produce a cost that
scales with length, so the leading guess, a per position gated delta net state
snapshot and rollback, does not fit the trend either. It is open.

## 5. Breakeven, and why ngram was the wrong test all along

Acceptance length needed to break even is exactly the step ratio.

| arm | 8k | 32k | 100k | achieved |
|---|---|---|---|---|
| c30_ngram, 4 drafts | 4.65 | 2.66 | 2.13 | 1.64 |
| q38_mtp1, 1 draft | 2.89 | 2.96 | 3.15 | 1.80 |
| q38_mtp2, 2 drafts | 4.90 | 5.10 | 4.49 | 2.43 |

q38 is hopeless: the bar rises with more drafts because positions are not free,
so there is no k that helps.

c30 at 100k needs 2.13 with four drafts. ngram delivers 1.64 on this corpus and
loses by 1.30x. But 2.13 is not a high bar. The MTP head on q38 reached 2.428
on the same corpus with only two draft slots. **A drafter of that class on
Qwen3-Coder-30B-A3B would clear 100k breakeven**, by about 1.14x at 2.43 and
about 1.41x at 3.0.

So spec is 0 for 4 across records 1, 13, and this one, but the reason has moved.
It was never that the MoE cannot amortise at our length. It is that the only
drafter we ever gave the MoE was ngram, and ngram accepts 1.64 on this corpus.
Record 1 section 1 and record 3 section 5 read a drafter failure as a model
failure. Qwen3-Coder-30B-A3B ships no MTP head, which is why ngram was reached
for, and that is an availability problem, not a physics one.

## 6. The caveat that qualifies section 3 and section 5

Section 2 measured both arms reading KV at 2.5 and 8.0 percent of H200 peak.
The amortisation argument in section 3 rests entirely on the context term being
a genuinely shared KV read. If the harness inflates that term with something
that is not shared across positions, the 0.28 per position at 100k is too good
and the 2.13 breakeven is too low.

The error is in the optimistic direction. Do not treat "an MTP class drafter
would win at 100k on the MoE" as established. It is a projection from a harness
whose absolute numbers are already known to disagree with the live arms by 2.6x
(record 3 section 9) and to disagree with themselves across runs by 1.7x. It is
enough to justify designing a test. It is not a result.

## 7. Corrections

1. Record 4 section 1, forward claim, that a dense model would let spec win.
   Falsified by q38_nospec against q38_mtp with 2.428 acceptance. Withdrawn.
2. Record 4 section 1, scope, that an A3B MoE has nothing to amortise. True at
   8k, false at 100k, where a position costs 0.28 of a step. Narrowed to short
   context.
3. Record 1 section 1 and record 3 section 5, that spec does not work on this
   model. Restated: ngram does not work on this corpus. The model at 100k needs
   only 2.13 acceptance from four drafts.
4. Record 4 section 3 recorded that no pure full attention 262k model exists on
   the box, and section 4 picked a hybrid anyway without flagging that hybrid
   breaks the same premise. The risk existed before the run and was not stated.

## 8. Where this leaves the goal

The window identity, record 2 section 4, leaves tpot as the only model lever.
Three model configurations have now been tried against the incumbent and all
three are worse: dense 2.1x, dense with MTP 3.7x, MoE with ngram 1.29x. The
incumbent is the right model and the model search is closed.

The lever for per request tok/s is CONC, not the model. Record 3 established
that aggregate decode is flat near 360 tok/s, so per request is 360 divided by
running and running near 7 reaches 50. That is a scheduling decision, and it is
the user's to make against throughput.

The deliverable is unchanged: the CONC=64 pair from record 3. Stores down 86.8
percent, external hit up 1.88 points, tpot down 7.0 percent, waiting down 40
percent, goodput up 30.2 percent at a 15 tok/s and 10 s bar.

## 9. Open

- The 2.5 to 8.0 percent of KV peak in the isolated harness. This is now
  blocking: it is the same wound as record 3 section 9, it qualifies section 3
  and section 5 of this record, and every isolated number produced so far
  inherits it. It should be settled before the harness is used to decide
  anything else.
- Whether a real drafter clears 2.13 on Qwen3-Coder-30B-A3B at 100k. Options
  are a small dense Qwen3 draft model with matching vocabulary via
  `method: draft_model`, or EAGLE. Not launched. Design first.
- Why a draft position on a hybrid costs two steps and why that cost scales
  with context. Named as unattributed on purpose.
- The k=1 acceptance for q38_mtp was lost to log truncation. `bs.sh` should
  archive `off_vllm.log` per arm rather than let the next server truncate it.
- L0 intersect L1 point in time overlap, still unmeasured.
- `phase.py` compares wall clock times as strings, so an arm crossing midnight
  gets an empty window.

## 10. Addendum: expert coverage saturates, so the MoE penalty is a batch problem

Raised in review: verify activates more experts, so the extra positions cost
extra expert weight reads. True, and it is self limiting. Qwen3-Coder-30B-A3B
is 128 experts at top 8, so the distinct experts touched in one step follows a
coupon collector curve and saturates.

| batch | positions, 0 then 4 drafts | assignments | M | distinct experts | expert weight read |
|---|---|---|---|---|---|
| 1 | 1 to 5 | 8 to 40 | 1.03 to 1.16 | 7.8 to 34.5 | 4.43x |
| 8 | 8 to 40 | 64 to 320 | 1.27 to 2.72 | 50.5 to 117.6 | 2.33x |
| 24 | 24 to 120 | 192 to 960 | 1.93 to 7.50 | 99.6 to 127.9 | 1.28x |
| 64 | 64 to 320 | 512 to 2560 | 4.07 to 20.0 | 125.7 to 128.0 | 1.02x |

99 percent of the 128 experts are touched by 74 positions. At batch 24 with no
spec, 99.6 are already touched, so five times the positions buys only 28 more
experts.

This retrodicts record 13. That spec probe ran at batch 1, where the model
predicts a 4.43x expert weight read, and it measured 3.9 to 5.3x. The agreement
is close enough to take seriously. So the extra expert cost of speculation is a
small batch effect, not a MoE effect, and it is gone by batch 64.

It also opens a hole in this record's own account. At batch 24 and 8k the
measured step ratio is 4.65x where coverage explains 1.28x, and ngram's drafter
runs no model, so all of it is verify. Expert activation is not what is
expensive at batch 24. The measurement sits at the "cost proportional to
assignments" end of the range rather than the "cost proportional to distinct
expert weight read" end, which would mean the Triton MoE backend does not reuse
an expert's weights across its token group. That is a hypothesis. It is not
measured, it is the fourth candidate mechanism in this line, and it is probably
the same thing as the 8 percent of peak bandwidth in section 2.

None of this changes the operating conclusion. The 0.28 per position at 100k in
section 3 is an end to end measurement with expert activation already inside it.
