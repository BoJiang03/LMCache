# Deployment requirements and the candidate that may meet them

Two things in one place: the standing requirements set by Bo Jiang, and the one
configuration on this hardware that the evidence so far says may satisfy them
jointly.

This is a living reference, not a session log. Companion to
`records/deployment_requirements.md`, which carries the full history and
rationale of each requirement; this document restates them compactly and adds
the candidate.

Last updated 2026-08-30. Status: candidate identified and confirmed to start.
**Not yet demonstrated. No request has been served.**

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

- At most 4 GPUs. All 8 H200 on this box are currently free, but the standing
  budget is 4.
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

**Qwen3.5-397B-A17B-FP8, TP=4, on vllm-main, with the LMCache MP connector.**

Local at `/raid/data/hub/models--Qwen--Qwen3.5-397B-A17B-FP8`.

| property | value |
|---|---|
| total / active | 397B / A17B, fp8, 394 GB resident |
| layers | 60: 15 full attention, 45 gated delta net (`full_attention_interval` 4) |
| experts | 512, top-10, so E/k = 51.2, the highest on the box |
| attention | 2 KV heads, head_dim 256 (vLLM replicates KV heads to TP=4) |
| context | 262,144 |

E/k = 51.2 is what makes it work: it holds the expert read to about 118 GB of
the 192 GB step budget at B = 18, where a denser-activation MoE of the same
size would spend the whole budget on weights and hold 2 to 3 requests.

### Launch configuration

MP server (needs an interpreter with LMCache's deps; the vllm-lazy venv has
them):

```
python -m lmcache.v1.multiprocess.http_server \
  --host 127.0.0.1 --port <MP> --http-host 127.0.0.1 --http-port <HTTP> \
  --l1-size-gb <N> --chunk-size 2096 --eviction-policy LRU \
  --script-allowed-imports hashlib --max-workers 4
```

vLLM (vllm-main, 0.28.1rc1):

```
vllm serve <model> --tensor-parallel-size 4 \
  --max-model-len 262144 --kv-cache-dtype fp8 \
  --gpu-memory-utilization 0.90 \
  --kv-transfer-config '{"kv_connector":"LMCacheMPConnector","kv_role":"kv_both",
    "kv_connector_extra_config":{"lmcache.mp.host":"tcp://127.0.0.1","lmcache.mp.port":<MP>}}'
```

Required environment:

- `CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:...` -- no
  `python3.12-dev` on this box, so every Triton kernel fails to build without
  it. Already in the harness `env.sh:40`.
- `PYTHONPATH` must resolve `lmcache` to this repo, and the vllm-main venv
  needs `sortedcontainers` (a scratchpad dir on `PYTHONPATH` is enough).

Do **not** set `--block-size`; vLLM overrides it. Do **not** set
`--num-gpu-blocks-override` or lower `gpu_memory_utilization` (R4, R5).

### Three non-obvious requirements of this configuration

1. **vllm-main, not the harness's 0.23.0.** This is a hybrid model, and
   `config/model.py:1852` in 0.23.0 refuses prefix caching for hybrids
   ("still experimental"); 0.28.1rc1 at `config/model.py:2128` allows it by
   default. Without the newer vLLM there is no L0 and no L1 for this model.
2. **The connector must resolve to LMCache's own class.** vLLM's shim prefers
   the external `LMCacheMPConnector` and falls back to its builtin
   `LMCacheMPConnectorUpstream` on any ImportError. Only the external one
   declares `SupportsHMA`; vLLM silently disables the hybrid KV cache manager
   when the connector does not, which turns off hybrid prefix caching without
   an error. Check for "Using external LMCacheMPConnector" in the log.
3. **`--chunk-size 2096` on the MP server.** vLLM forces the attention block
   size up so the attention page is at least the mamba page: 2096 on
   0.28.1rc1. LMCache requires `chunk % block == 0`. The default 256 fails,
   and no power-of-two ever works, because 2096 = 2^4 x 131. The value is read
   from the MP server over the message queue, so setting it on the vLLM side
   does nothing.

## Part 4. The predicted operating point

On the measured pool (3,568,733 tokens, 115.7 GB), everything else from the
cost model:

| B | f | tpot ms | tok/s | window s | L1/L0 ceiling | free contexts |
|---|---|---|---|---|---|---|
| 15 | 0.475 | 16.8 | 59.6 | 24.3 | 0.68 | 16.6 |
| 16 | 0.506 | 17.7 | 56.4 | 17.2 | 1.16 | 15.6 |
| **17** | **0.538** | **18.7** | **53.6** | **12.6** | **2.09** | **14.6** |
| 18 | 0.570 | 19.6 | 51.1 | 10.2 | 3.31 | 13.6 |

B = 16 to 18 satisfies R1 and R2 together. R7 should follow: the free pool
still holds 13 to 16 full contexts, and across every archived arm TTFT p50 was
1.0 to 1.2 s whenever the engine was not queueing, against 190 to 300 s when it
was.

L1/L0 here is a **ceiling**, set by the window. What fraction of it L1 actually
returns is the offload policy's job; measured realisation on coder30 ran 20 to
75 percent, and lazy returned 10.0 percent of input tokens against eager's 8.1
at CONC=64.

## Part 5. What is measured, what is not

**Measured on the hardware:**

| | |
|---|---|
| weights load at TP=4 | 98.5 GB/GPU, 394 GB, 150 s |
| pool | 115.7 GB, 3,568,733 tokens, 32,434 B/token |
| contexts held at ISL 107k | 33.4 |
| forced attention block size | 2096 |
| hybrid prefix caching | on by default in 0.28.1rc1 |
| connector | resolves to the `SupportsHMA` class, HMA stays on |

**Computed, not measured.** Every number in Part 4. Both constants of the cost
model were calibrated on a different model:

- `b = 2.4 GB/ms/GPU` was fitted on coder30's 4 KV heads at head_dim 128. This
  model has 2 KV heads at head_dim 256 plus 45 linear-attention layers. Every
  figure in Part 4 scales with it.
- `c(B) = 1 - (1-k/E)^B` was validated at E/k = 16. This model is 51.2.

**Not measured at all.** No request has been served on this model: no observed
tok/s, TTFT, L0 or L1 hit rate, and no Stored or Retrieved.

## Part 6. What would disqualify the candidate

1. **Store and retrieve failing at a 2096-token block.** Unproven, and the next
   thing to check.
2. **The 68 MB storage granule.** At 32,434 B/token a 2096-token chunk is 68 MB,
   two orders of magnitude coarser than the 256 every arm on disk ran. Store
   latency, eviction granularity and the lazy policy's block accounting were all
   designed around the small chunk. The assert passing does not mean L1 behaves.
3. **The bandwidth constant not carrying.** If b is materially below 2.4 on this
   attention shape, the whole Part 4 table shifts and B = 17 may not deliver
   50 tok/s.

## Part 7. The fallback, if hybrid turns out to be unworkable

**MiniMax-M2.7 fp8, TP=4.** 230 GB, 62 layers, GQA 8 x 128, 256 experts top-8,
no mamba, no sparse attention, no chunked local attention: the ordinary
full-attention path every arm on disk has exercised.

The cost of the fallback is R2. At 50 tok/s it reaches an L1/L0 ceiling of 0.24
with fp8 KV, or 0.72 with bf16 KV, against the [1, 3] band. R1 and R7 hold;
R2 misses by roughly 4x.

## Part 8. Next step

Not started. Needs its design discussed first, per the standing constraint.

1. Confirm Stored and Retrieved at block 2096. One server, two requests, with a
   local prefix cache reset between them so the second must come from L1.
2. Only then, a CONC sweep placing Running at roughly 12 / 15 / 17 / 20, in the
   healthy region only, recording from the same samples: `Running`, `GPU KV
   cache usage`, generation throughput over Running, `Waiting`, both hit rates,
   and `tokens_retrieved / isl_sum`. That is R1, R2, R7 and f together, and it
   is what replaces every computed number in Part 4.
