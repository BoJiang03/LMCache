# Deployment requirements and the candidate that may meet them

Two things in one place: the standing requirements set by Bo Jiang, and the one
configuration on this hardware that the evidence so far says may satisfy them
jointly.

This is a living reference, not a session log. Companion to
`records/deployment_requirements.md`, which carries the full history and
rationale of each requirement; this document restates them compactly and adds
the candidate.

Last updated 2026-08-30 (second revision). Status: the previous candidate,
Qwen3.5-397B-A17B-FP8, fell when the L1 round trip on the hybrid path proved
unfaithful (0/5 under both layouts on the 2B probe; full account in
`records/2026/08/30/11_the_hybrid_falls_and_the_repick.md`). The replacement
candidate below was picked from a catalogue-wide sweep. **Not yet started. No
request has been served. Weights are not yet on disk.**

## Part 1. The requirements

### R1. Per request decode about 50 tok/s

> 我们必须要把 per request tok/s 搞到50左右

Median per request decode throughput, client side, over the profiling phase, on
the CC agentic workload at its native context length. Not aggregate throughput,
not a short context number. 50 tok/s is the market rate: OpenRouter lists
Qwen3-Coder-30B-A3B at Alibaba Cloud 53, NovitaAI 47, SiliconFlow 21.

### R2. L1 must carry a larger share of reuse, and L1/L0 in [1, 3]

> 还要把l1重要性提升
>
> L1 must carry a larger share of reuse，并且 l1 / l0 in [1,3]

L1 tokens returned over L0 tokens hit, on a common denominator. From the engine
log that is `E x (1-H) / H`, with `H` the local prefix cache hit rate and `E`
the external one, which vLLM measures on the residual after L0.

Equivalent statement in seconds: L1's share is `P(gap > W)` on the corpus
reuse-gap distribution, so the band [1, 3] means the L0 eviction window must sit
in **[10.5 s, 18.2 s]**.

### R4. The KV pool may not be shrunk

> 实际部署不可能缩池子啊

No `--num-gpu-blocks-override`, no artificially reduced pool. A real deployment
takes all the KV it can get. A pool that is small because the model is large is
not a violation; a pool made small by a knob is.

### R5. `gpu_memory_utilization` may not be lowered

> 难不成他们 也缩 gpu util？那不是疯了吗

Stays at 0.9.

### R6. The corpus must reflect real usage

> 我自己用了这么久的claude code，基本没怎么启动过subagent

The workload must not be chosen, filtered, or weighted to favour the result.
The corpus in use over-weights subagent fan-out by about 17x against 135 local
Claude Code sessions; this is recorded rather than corrected, because switching
corpora to collect the 2.4 points it is worth would itself violate R6.

### R7. TTFT under 10 s, preferably under 5 s

> ttft 得< 10s 最好<5s

Client side p50 under 5 s, p90 under 10 s, at native context length. The
corpus's own recorded production figures are p50 2.64 s and p90 6.98 s, so this
is the market rate, not a stretch.

### Constraints these operate under

- At most 4 GPUs. The standing budget is 4; GPU 0 carries other users' small
  allocations and GPU 7 holds the running 2B probe server.
- Go easy on host memory. Only touch processes owned by uid 1016.
- No shared environment mutation: no rebuilding `lmcache/*.so`, venvs, `/raid`,
  or `/usr/local`.
- Paired comparisons run two rounds with the slots swapped.
- No experiment is launched before its design has been discussed.

## Part 2. Why the current baseline cannot satisfy them

With Qwen3-Coder-30B-A3B, R1 and R2 have no common operating point, and the
reason generalises.

```
tpot = (c(B) P_exp + P_dense + f pool) / (b N)
f    = B ISL kv / pool
pool = N (HBM u - overhead) - P
W    = D (1-f) / (f (1-L0))
```

R1 fixes tpot, which caps bytes read per step. R4 and R5 fix the pool at
whatever the model leaves. So as the weights go to zero,

```
f_max = tpot b / (HBM u - overhead) = 20 x 2.4 / 130.67 = 0.367
```

Any model small next to HBM sits at f ~= 0.3 at 50 tok/s no matter how many
GPUs it gets: coder30 measures 0.305 to 0.33 at TP=2, 4 and 8. At f = 0.31 the
window is 272 s and L1/L0 is 0.05. Reaching 1.0 needs Running 28, which is
24 tok/s, and fails R1.

Raising f means raising P. `df/dP > 0` iff `c(B) < tpot b / (HBM u -
overhead)`, so f = 0.6 needs about 65 GB of weights per GPU with `c(B) <~ 0.2`,
hence `E/k >= 32`. KV bytes per token cancels out of f entirely; it only sets B,
and through B it sets c.

**The model is the lever, and it is R4-compatible**: the pool shrinks because
the model is large, not because a knob was turned.

## Part 3. The candidate

**Trinity-Large-Thinking-FP8-Block, TP=4, on vllm-lazy (vLLM 0.23.0), with the
LMCache MP connector.**

Not on disk. `arcee-ai/Trinity-Large-Thinking-FP8-Block` on Hugging Face,
81 safetensors shards, 403.8 GB; `/raid` has 5.4 TB free. Tech report
arxiv.org/html/2602.17004v1. License tagged "other" on the FP8 repo (LICENSE
file present; the base bf16 repo is Apache-2.0); fine for an internal
experiment, check before anything external.

| property | value |
|---|---|
| total / active | 398B / A13B, fp8 (compressed-tensors, block quantized), 403.8 GB on disk |
| layers | 60 (`afmoe`): 15 full attention, 45 sliding window 4096 (`global_attn_every_n_layers` 4) |
| experts | 256, top-4, so E/k = 64, above the 397B pick's 51.2 |
| attention | GQA, 8 KV heads at head_dim 128, hidden 3072 |
| long-context KV | 30,720 B/token (fp8 KV), carried by the 15 full layers only |
| SWA KV | fixed 0.38 GB/request (45 layers x 4096 tokens x 2048 B) |
| context | 262,144 |

This is the 397B pick's winning shape without the mamba: long-context KV
concentrated in 15 of 60 layers at exactly the same 30,720 B/token, higher
sparsity, and a plain decoder attention type. It dodges all three blockers that
killed the hybrid line:

1. No mamba, so no state snapshot semantics and no unified-view edit.
2. `AfmoeForCausalLM` is in vllm-lazy 0.23's registry with attn_type `decoder`
   (`models/afmoe.py:178`), so the 0.23 prefix-caching refusal for hybrids does
   not trigger and the whole verified harness stack applies: no vllm-main, no
   0.28 rank-4 layout bug, no torch IPC split between venvs.
3. No forced attention block size, so the default chunk 256 divides cleanly.

### Launch configuration

Both processes on the vllm-lazy venv (`/home/bo/venvs/vllm-lazy`), which has
LMCache's deps and the engine in one interpreter version.

MP server:

```
python -m lmcache.v1.multiprocess.http_server \
  --host 127.0.0.1 --port <MP> --http-host 127.0.0.1 --http-port <HTTP> \
  --l1-size-gb 250 --chunk-size 256 --eviction-policy LRU \
  --separate-object-groups \
  --script-allowed-imports hashlib --max-workers 4
```

vLLM:

```
vllm serve arcee-ai/Trinity-Large-Thinking-FP8-Block --tensor-parallel-size 4 \
  --max-model-len 262144 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":<MP>}}'
```

Required environment (both bit us on the first two bare starts, 08-30):

- `PATH` must include `/usr/local/cuda/bin`. flashinfer's
  `fp8_blockscale_gemm_sm90` JIT-compiles its kernels with nvcc at first
  use; without it the cubin comes back empty and cudagraph capture dies
  with `Assertion failed: !cubin.empty() || isPathValid(path_)`. Compiled
  kernels are cached under `~/.tensorrt_llm/cache`, so only the first
  start pays.
- `CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:/raid/data/hub/pr4499_agentic/pydev/usr/include`
  -- no `python3.12-dev` on this box, so Triton launcher builds fail
  without it. Both entries are required: the second resolves
  `x86_64-linux-gnu/python3.12/pyconfig.h`. Same as the harness
  `env.sh:13`.
- `PYTHONPATH` must resolve `lmcache` to this repo.
- `CUDA_VISIBLE_DEVICES` avoiding GPU 0 (other users) and GPU 7 (probe
  server), e.g. `1,2,3,4`.

Do **not** set `--block-size`, `--num-gpu-blocks-override`, or lower
`gpu_memory_utilization` (R4, R5). `--kv-cache-dtype fp8` is load-bearing:
every figure in Part 4 assumes 1 B/element KV.

## Part 4. The predicted operating point

Nothing here is measured on this model. Constants from record 8's cost model:
b = 2.4 GB/ms/GPU (fitted on coder30), c(B) = 1-(1-4/256)^B, usable HBM
510 GB at TP=4. Pool = 510 - resident weights; disk size 403.8 GB puts the
pool at roughly 100-106 GB. Per context at ISL 107k: 3.29 GB long-context KV
plus 0.38 GB SWA, 3.66 GB, so the pool holds 27 to 29 contexts.

With pool 106 GB (P_exp ~385, P_dense ~20):

| B | f | tpot ms | tok/s | free contexts |
|---|---|---|---|---|
| 15 | 0.52 | 16.2 | 62 | 14 |
| 16 | 0.55 | 17.1 | 58 | 13 |
| 18 | 0.62 | 18.8 | 53 | 11 |
| 20 | 0.69 | 20.5 | 49 | 9 |

Window and ceiling, by interpolation on record 8's own table rather than the
closed form: at B = 18 the window sits near 7 s and the L1/L0 ceiling lands in
the 3 to 6 aim band, which is where measured realisation (20 to 75 percent on
coder30) puts realised L1/L0 inside [1, 3]. B in 16 to 19 satisfies R1 and R2
together on paper; R7 should follow from 9 to 13 free contexts, as in every
archived arm where the engine was not queueing.

**The absolute L1 number.** R2 is a ratio; the absolute figure it implies at
this operating point is the headline. L0 = P(gap < 7 s) ~= 0.2 of input
tokens, so realised L1/L0 in [1, 3] means L1 returns **20 to 60 percent of
input tokens**, target around 0.4. The best figure ever measured on this
harness is 10.0 percent (coder30, l64r1); the same metric is already in the
sweep recording list as `tokens_retrieved / isl_sum`. Equivalent statement:
at 0.4 the prefill recompute per request is roughly halved against serving
with no L1 at all.

## Part 5. What is measured, what is not

**Measured on the hardware (2026-08-30, steps 0-3 of Part 8):**

| | |
|---|---|
| weights pulled | 81 shards, 377 GiB, ~25 min |
| fp8 checkpoint loads on vllm-lazy 0.23 | 81/81 shards in 90 s, TP=4 |
| pool | 26.73 GiB/GPU x 4 = 106.9 GiB, 3,275,586 tokens |
| max concurrency at 262k | 12.50x (engine log), ~31 contexts at ISL 107k |
| kernel groups | 4 x 15 layers: three SWA (`sw_size_tokens=4096`), one full (`-1`); tokens_per_block 16; fp8, nh=2/rank, hs=128 |
| connector | "Using external LMCacheMPConnector" on all ranks |
| stored bytes per token | 122,880 = all 60 layers dense (SWA groups stored full-length); 8 objects per chunk |
| L1 round trip, repeat-paragraph x5 | byte-identical across an engine restart, 872 chunks read each |
| L1 round trip, needle x3 | needle recalled 3/3, answer tokens reproduced; tail flips at near-tie punctuation only |
| L1-path perturbation | max 0.16-0.24 nats, BELOW the same-engine L0 partial-hit control at 0.32 nats |

The control matters: a plain L0 partial-hit rerun on one engine, no LMCache
in the loop, already moves logprobs by 0.32 nats and reshuffles top-5 (fp8 KV
plus recompute-tail kernel shapes). Byte-exact 400-token reproduction is not
achievable on this stack even engine-to-itself; the correct acceptance test
is L1-path perturbation <= L0-control perturbation, and it passes. This also
retro-softens the hybrid line's 0.34 nats verdict (record 11 section 3): that
number now looks like the same stack noise, not proof of corruption; the
hybrid's hard blocker remains the 0.28 rank-4 layout bug.

Note: this 0.23 build has no `/reset_prefix_cache` route, so L0/L1 isolation
is done by restarting the engine (L1 lives in the MP server); `cached_tokens`
is not reported either, judge hits by the MP metrics and the engine log.

**Computed, not measured.** The Part 4 table (tok/s, f, window, ceiling), on
constants calibrated on coder30. No multi-request serving yet: no measured
tok/s, TTFT, or hit rate under load.

## Part 6. What would disqualify the candidate

1. ~~The fp8 checkpoint failing to load on 0.23.~~ **Cleared 08-30**: loads
   and serves. Two environment requirements surfaced on the way (nvcc PATH,
   full CPATH), both in Part 3.
2. **SWA is supported by design; sub-risk (b) resolved, (a) still open.**
   Measured 08-30: the correctness ladder passes across engine restarts
   (Part 5), so the mixed SWA-plus-full transfer path is sound under the
   default (eager-style) store. Sub-risk (b) is answered: storage IS dense,
   122,880 B/token, so the Part 8 L1 sizing scales 4x (~250 GB holds ~2.2M
   tokens, about 25-40 s of worst-case stream at load; raise toward 500 GB
   for the sweep if eviction age drops below the window target). Sub-risk
   (a), lazy x SWA unhashed-block rejection, is untested until the sweep
   runs the lazy policy: watch `rejected_unhashed` there. Original analysis
   kept below.
   The
   support itself is real, not circumstantial: the connector reads vLLM's
   per-layer `SlidingWindowSpec` and tags each kernel group with its window
   (`kv_cache_groups.py:76`), group validation accepts SWA, and
   `--separate-object-groups` splits object groups by window size so
   retrieve loads only each SWA group's 16-chunk suffix
   (`kv_layer_groups.py`); gemma's SUPPORTED verdict on this stack exercised
   the mixed SWA-plus-full path in MP mode. What is unmeasured:
   (a) **lazy x SWA**: lazy stores snapshot block hashes at eviction time
   across all groups' blocks, and out-of-window SWA blocks may by then be
   hash-less null blocks, which rejects the store and breaks the request's
   chain (`REJECTED_UNHASHED_BLOCK`, anticipated in
   `lazy_offload_pending_store.py:321`); the fork's
   `VLLM_PREFIX_CACHE_RETENTION_INTERVAL` machinery exists precisely to
   manage SWA checkpoint density, so watch `rejected_unhashed` in step 3.
   (b) **stored volume**: if every SWA chunk is stored densely, stored
   bytes per token is 122,880, not 30,720, which quadruples the L1 stream
   and the sizing in Part 8.
3. **The bandwidth constant not carrying.** If b is materially below 2.4 on
   this attention shape the Part 4 table shifts left and B = 18 may miss
   50 tok/s.
4. **Weight residency well above disk size.** Pool = 510 - resident; every
   10 GB of unexpected residency costs about 3 contexts and compresses the
   feasible B band.

## Part 7. The fallbacks

Two tiers, from `records/2026/08/30/11_the_hybrid_falls_and_the_repick.md`
section 5.

**DeepSeek-V4-Flash-FP8 (sgl-project repack), 291B/A13B, MLA + DSA,
E/k = 42.7.** The fp8 repack dodges the SM100 gate and vLLM issue #47648 says
H200 works without DSpark. At B = 25: about 71 tok/s, f = 0.64, window ~7 s,
ceiling 4 to 6. Cost: MLA and DSA are both unexercised LMCache paths, two new
risks instead of one.

**MiniMax-M2.7 fp8, TP=4, already on disk.** 230 GB, 62 layers, GQA 8 x 128,
256 experts top-8: the ordinary full-attention path every arm on disk has
exercised. Zero stack risk; the cost is R2. At 50 tok/s the L1/L0 ceiling is
0.24 with fp8 KV or 0.72 with bf16 KV, against the [1, 3] band.

## Part 8. Verification plan

Design discussed and approved 08-30; steps 0-3 executed the same day, all
passed (results in Part 5). Step 4 not started.

0. Pull the weights: 403.8 GB to `/raid/data/hub` (5.4 TB free). Hours at
   typical HF throughput; run in the background and verify shard count.
1. Bare engine start, no connector: vllm-lazy, TP=4 on GPUs 1-4,
   `--kv-cache-dtype fp8`, util 0.9. Confirms the fp8 checkpoint loads
   (disqualifier 1), records resident weights, pool bytes and tokens, block
   size, prefix caching on. Replaces the top rows of Part 5.
2. MP server at chunk 256 plus connector: check "Using external
   LMCacheMPConnector" in the log, then one long request twice and confirm
   Stored and Retrieved both move. Also measure stored bytes per token:
   30,720 B/token means only the full-attention groups carry long-context
   KV; near 122,880 means the SWA groups are stored densely too, which is
   not wrong (retrieve only reads their suffix) but quadruples the L1
   stream, so the 250 GB in the sizing below scales to ~950 GB for r = 3
   worst case and the size needs revisiting (host holds it, 1.4 TB free).
3. The correctness ladder from the 2B probe, unchanged: (a) vLLM local prefix
   cache reproduces the cold decode 5/5; (b) `correctness2.py` exact-match
   round trip through L1, 5/5, with `/reset_prefix_cache` between runs;
   (c) `needle.py` non-repetitive prompt with a planted fact, 5/5. This is
   the test that killed the hybrid line and it gates everything after it
   (disqualifier 2). Under the lazy policy, also check `rejected_unhashed`
   stays at zero; a nonzero count is sub-risk (a) firing.
4. Only after 3 passes: the CONC sweep placing Running at about 15 / 18 / 21,
   healthy region only, recording from the same samples Running, GPU KV cache
   usage, generation throughput over Running, Waiting, both hit rates, and
   tokens_retrieved / isl_sum. That is R1, R2, R7 and f together, and it
   replaces every computed number in Part 4.

**L1 size: `--l1-size-gb 250`.** Sized so that capacity is never what caps
realised L1/L0; whatever the sweep then measures is the offload policy, not
the pool. Reaching realised ratio r needs L1 to retain the offload stream for
T(r) from the gap CDF: r = 1 needs T ~= 15 s, r = 2 ~= 30 s, r = 3 ~= 60 s
(anchors: P(gap < 18.2 s) = 0.5, P(gap < 50.7 s) = 0.746). Worst-case store
rate with no dedup is the input KV rate, 0.7 to 1.2 req/s x 107k tokens x
30,720 B = 2.3 to 3.9 GB/s, so r = 3 needs at most 60 x 3.9 ~= 235 GB; with
dedup the true need is well under that. Host memory allows it: 2 TB total,
1.4 TB available, and prior arms ran up to 576 GB. Verify in the sweep that
L1 eviction age stays above 60 s; if it does not, the no-dedup bound was
optimistic and the size goes up.

Open to settle in the discussion: the exact GPU set (proposed 1-4), and
whether step 0 may start before the discussion concludes.
