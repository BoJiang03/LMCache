# A window opens once loop gain drops to five percent

Record 4 closed on two self-reinforcing branches with nothing between them,
and framed that as possibly structural. It is not structural. It is a
consequence of one ratio, and lowering that ratio opens the window.

## 1. The reframing that unblocked it

The target `local 40-60%, ext >25%, TTFT <10 s` is not a lazy-offloading
convenience band, it is what a normal deployment looks like. So the question
stopped being "which knob rescues the middle" and became "which of our own
settings is abnormal". Two were:

- `MAX_MODEL_LEN=1048576` via YaRN factor 4 on a model whose native
  `max_position_embeddings` is 262,144. Nobody deploys 4x YaRN to serve a
  corpus in order to make it harder to serve.
- bf16 KV on H200 for long context. fp8 KV is the standard production
  setting there, not a concession.

TP=4 was considered and dropped: 3B active parameters over 4 GPUs makes
all-reduce dominate and is itself an abnormal deployment.

## 2. The ratio that governs the branches

Block life is free blocks over allocation rate. A local hit allocates almost
nothing and lengthens block life; an L1 load allocates a whole prefix and
shortens it. The loop closes on itself in both directions, so what decides
whether the middle is a stable point or a tipping point is how much one
arrival moves the pool:

    loop gain = p90 turn / pool

Measured on the parent corpus at bf16: 652,928 / 2,038,560 = **32%**. One
arrival flips the regime, which is exactly the bistability record 4 measured.

## 3. Two normal levers, multiplied

| config | pool tok | slots | p90 turn / pool |
|---|---|---|---|
| bf16 + parent corpus | 2,038,560 | 8.2 | 32% |
| fp8 + parent corpus | 4,077,024 | 16.4 | 16% |
| fp8 + 256k corpus @ native 262,144 | 4,077,024 | 40.4 | **5.0%** |

fp8 doubles the denominator; the 256k corpus cuts the numerator. Pool tokens
were confirmed on the box, not estimated: engine reported 4,079,248 and
4,077,968 on the two slots.

L1 needs no retuning. `integration/vllm/utils.py:268` derives `kv_dtype` from
vLLM's `cache_cfg.cache_dtype` and `get_size_bytes` multiplies by
`kv_dtype.itemsize`, so L1 bytes per token halves with L0 and the tier ratio
k = L1/L0 = 3.1 is unchanged.

## 4. The 256k corpus, checked before use

`semianalysisai/cc-traces-weka-062126-256k` drops any request whose in+out
exceeds 256,000 proxy-tokenizer tokens. Reapplying that filter offline to the
parent reproduces the published stats exactly (68,266 requests, total input
6,891,228,864, total output 58,728,807), which validates the filter reading.

    in  mean 100,946   p50 88,768   p90 204,288   max 255,808
    dropped: main 28,354 of 56,798;  subagent 2,207 of 42,029

The cut falls almost entirely on the main-chain long tail. Subagent fan-out,
which is what makes this corpus bursty, survives at 95%.

The proxy-tokenizer risk was retired rather than assumed away: prompts are
synthesised as token sequences against the tokenizer we pass
(`hash_ids_synthesis.py`), and 256,000 < 262,144. Measured: `errors=0` and
zero length rejections in the engine log.

## 5. Results

All arms: fp8, native 262,144 context, 256k corpus, L1 320 GB, DEFER 30,
FLOOR 8192, HORIZON 2.5, TP=2, 1800 s window, no `--unsafe-override`.

| arm | x slots | compute | local | ext | TTFT p50 | TTFT p90 | shape |
|---|---|---|---|---|---|---|---|
| f8k256c48 | 1.19 | 6.3% | 90.7% | 3.0% | 1.02 s | 2.73 s | monotone |
| **f8k256c72** | **1.78** | 11.8% | **40.6%** | **47.6%** | **9.23 s** | 36.86 s | non-monotone |
| n16L320 (bf16, parent) | 1.95 | 26.1% | 21.5% | 52.3% | 61.6 s | -- | non-monotone |

c72 meets all three targets. The comparison that matters is c72 against
n16L320: near-identical load in units of pool capacity, and the difference
between a 9 s TTFT and a 62 s one is the loop gain, nothing else.

c48 is also worth stating on its own. TTFT p50 1.02 s and p90 2.73 s beat the
corpus's own recorded production numbers (2.64 / 6.98), with `waiting_mean`
0.09, zero preemptions and zero eviction drops, and `ttft_by_isl` monotone in
ISL, meaning prefill-bound rather than queue-bound. Throughput 0.341 req/s
against 0.115 for the best parent-corpus arm.

## 6. What it costs

- c72 sits on the knee. `ttft_shape` is non-monotone and p90 is 36.9 s. The
  p50 target is met without margin.
- `decode_tps` p50 12.3 tok/s, tpot 81.6 ms, against the corpus's recorded
  161.7 tok/s and 6.2 ms. This is the batch effect of ~30 in-flight requests
  sharing decode, not fp8: the single-stream fp8 probe on this box measured
  167 tok/s and 5.98 ms.
- c72 shows 7 preemptions, `kv_max` 100%, and 1.9% eviction drop. Close to
  the capacity edge without crossing it.

## 7. Standing wrong claim, corrected

Record 4 section 8 concluded the middle may not exist on this box. It exists.
What did not exist was a configuration with small enough loop gain to hold it,
and both changes needed to get there are ordinary production settings rather
than benchmark contrivances.

`n18ramp` is scored and negative: `--prefill-concurrency 4` with a 600 s ramp
at CONC=18 on bf16/parent still gave TTFT p50 17.79 s and a non-monotone
by-ISL profile. Its pre-stated P5 held: the congested branch is a property of
offered load against pool capacity, not of the starting condition, and
admission control at the client does not substitute for pool depth.

## 8. Open

`f8k256c60` (1.49x) and `f8k256c84` (2.08x) are running with predictions
pre-stated in the harness. c60 asks whether the working point can be moved off
the knee; c84 asks whether the transition is soft across the whole range or
whether the window is a bounded band between roughly 1.2x and 1.8x.

A harness bug was fixed in passing: `env.sh` exported `MAX_MODEL_LEN` and
`ROPE_OVERRIDE` unconditionally, which silently clobbered per-arm overrides
passed as trailing arguments. Now conditional.
