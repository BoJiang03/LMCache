# Probing the Qwen3.5-397B pick: what loaded, what did not

Ran the probe record 8 section 10 specified. The pick survives and its pool is
measured, but the path to it is narrower than record 8 said: it needs a newer
vLLM than the harness runs, and it needs an LMCache chunk size nobody would
guess.

Nothing here is a benchmark. It is a startup and configuration result.

## 1. What the probe measured

Qwen3.5-397B-A17B-FP8, TP=4 on cards 4-7, `gpu_memory_utilization` 0.9,
`max_model_len` 262144, fp8 KV, no block override.

| | vllm-lazy 0.23.0, no connector | vllm-main 0.28.1rc1, LMCache MP |
|---|---|---|
| weights | 98.5 GB/GPU, 394 GB, loaded in 150 s | same |
| attention block size | 2112 (forced) | 2096 (forced) |
| mamba page padding | 1.34% | 0.58% |
| pool | 102.2 GiB = 109.7 GB | 107.8 GiB = 115.7 GB |
| pool tokens | 3,381,857 | 3,568,733 |
| bytes/token | 32,436 | 32,434 |
| 107k contexts held | 31.6 | 33.4 |

Record 8 assumed a 117 GB pool from a 5 GB/GPU overhead fitted on a 30B model
at TP=2. Measured 109.7 to 115.7 GB. The per-token figure agrees to 2 bytes
across two vLLM versions, so the cost model's pool term is sound.

## 2. The operating point moves, and R1 and R2 now overlap

Recomputed on the measured pool (3,568,733 tokens, 115.7 GB), everything else
as in record 8 section 3.

| B | f | tpot ms | tok/s | W s | L1/L0 ceiling | free contexts |
|---|---|---|---|---|---|---|
| 15 | 0.475 | 16.8 | 59.6 | 24.3 | 0.68 | 16.6 |
| 16 | 0.506 | 17.7 | 56.4 | 17.2 | 1.16 | 15.6 |
| 17 | 0.538 | 18.7 | 53.6 | 12.6 | 2.09 | 14.6 |
| 18 | 0.570 | 19.6 | 51.1 | 10.2 | 3.31 | 13.6 |

At the 3.57M-token pool the band shifts to roughly B = 17 to 18. Either way
there are operating points where per request decode is above 50 tok/s and the
L1/L0 ceiling is inside [1, 3]. Record 8 section 8 had them two requests apart
and could not resolve which side; the measured pool closes it.

This is still a ceiling. Realisation is the offload policy's job.

## 3. Hybrid prefix caching needs the newer vLLM

The pick is a hybrid model: 45 gated-delta-net layers, 15 full-attention, and
`full_attention_interval` 4.

- vllm-lazy 0.23.0, `config/model.py:1852`: `attn_type == "hybrid"` returns
  False, "Hybrid models do not support prefix caching since the feature is
  still experimental". `--enable-prefix-caching` overrides it silently.
- vllm-main 0.28.1rc1, `config/model.py:2128`: returns True, "Generative hybrid
  models support prefix caching". Default on.

So the newer vLLM is not a preference. Without it there is no L0 and no L1 for
this model except through an override vLLM itself calls experimental.

## 4. LMCache supports hybrid, and only on one path

The worry was that LMCache would not support a hybrid model. It does, and the
support is deliberate rather than incidental:

- `integration/vllm/kv_cache_group_edits.py` exists only for Mamba-hybrid
  models: per-group store/load masks, and a validator that rejects
  `mamba_cache_mode != "align"` (:129).
- `integration/vllm/kv_cache_groups.py:31` maps an align-mode Mamba spec to a
  one-block sliding window. :184 reads each group's block size from vLLM rather
  than assuming one.
- `v1/multiprocess/group_view.py` is built around vLLM's hybrid groups;
  non-hybrid is the degenerate empty-groups case.
- The lazy offload policy handles it: `lazy_offload_policy/eviction_aware.py:280`
  on hash-less null blocks from sliding-window and mamba layers.

But only one connector path works. vLLM's shim
(`kv_transfer/kv_connector/v1/lmcache_mp_connector.py:1226`) prefers the
external `LMCacheMPConnector` shipped with lmcache, and falls back to its own
`LMCacheMPConnectorUpstream` on any ImportError. The two differ:

| class | HMA |
|---|---|
| `lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPConnector` | `SupportsHMA` |
| vLLM's builtin `LMCacheMPConnectorUpstream` | raises at :79, "only works without hybrid kv cache manager" |
| `lmcache_mp_connector_0201.py` | no `SupportsHMA` |
| `lmcache_connector_v1.py` | no `SupportsHMA` |

vLLM auto-disables the hybrid KV cache manager when the connector does not
declare `SupportsHMA` (`config/vllm.py:1770`). So a silent fallback does not
error, it turns HMA off, and the hybrid model's prefix caching goes with it.
Verified before spending a GPU:

```
resolved to: lmcache.integration.vllm.lmcache_mp_connector.LMCacheMPConnector
SupportsHMA: True
```

and the live log confirms "Using external LMCacheMPConnector".

## 5. The chunk size has to be 2096, and it is set in the wrong-looking place

vLLM forces the attention block size up so the attention page is at least the
mamba page: 2096 on 0.28.1rc1, 2112 on 0.23.0. LMCache requires

```
lmcache_tokens_per_chunk % vllm_block_size == 0
```

at `integration/vllm/vllm_multi_process_adapter.py:616` and :1153, and again as
a `ValueError` at `lmcache_mp_connector.py:450`. The default chunk size is 256
(`v1/config.py:90`), so the engine dies with

```
AssertionError: LMCache chunk size should be a multiple of vLLM block size
```

No power-of-two chunk size can ever satisfy it: 2096 = 2^4 x 131 and
2112 = 2^6 x 33, and 131 and 33 are the odd factors. The chunk size must be a
multiple of the forced block size.

Two traps in setting it:

1. It is read from the LMCache server over the message queue
   (`get_lmcache_chunk_size(self.mq_client)`, :1145), not from the vLLM
   process. `LMCACHE_CHUNK_SIZE` on the vLLM side is inert. It is the MP
   server's `--chunk-size` flag.
2. The failing check is an `assert`, which `docs/coding_standards.md` forbids
   for runtime validation. Under `python -O` it disappears, and the failure
   mode is then misaligned per-group block-id slicing rather than a dead
   engine. Worth a separate fix.

## 6. Corrections to record 8

- Section 8's pool of 117 GB was a guess from a 30B fit. Measured 115.7 GB with
  the connector. The direction was right and the band closes rather than opens.
- Section 9 item 4, "hybrid prefix caching works end to end, not exercised", is
  now half answered: the vLLM side works and the connector resolves correctly,
  but store and retrieve at a 2096-token block is still unproven.
- Section 6 said going to Hugging Face does not change the answer. That holds,
  but for a reason record 8 did not state: the local
  `/raid/data/hub` tree already carries GLM-5.1, GLM-5.3-Flash, Qwen3.5-122B,
  Qwen3.8-Flash-Next, MiniMax-M2.7, Devstral-2-123B, Qwen3-Coder-480B and
  DeepSeek-V4-Flash. The catalogue was not short.

## 7. Environment, for anyone repeating this

Three failures cost most of the session and none were the model:

1. Triton's launcher build shells out to gcc and there is no `python3.12-dev`
   on this box, so `Python.h` is missing and every Triton kernel fails to
   compile. `env.sh:40` already carries the workaround: `CPATH` pointing at
   `/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12`. Launching
   without sourcing `env.sh` loses it.
2. The vllm-main venv lacks `sortedcontainers` (connector import) and
   `opentelemetry.exporter.prometheus` (MP server). Installed the first into a
   scratchpad directory on `PYTHONPATH`; ran the MP server on the vllm-lazy
   interpreter, which has the deps. No shared venv was touched.
3. Self-inflicted: a `p3.sh` still inside its 120 s MP health-wait loop passed
   its check when a later launch brought the server up on the same port, and
   `exec`ed a second `vllm serve` onto the same four cards. Two EngineCores,
   two workers claiming rank 3, CUDA OOM at 139 of 139.8 GiB. Check for a live
   launcher before relaunching, not just for a live server.

## 8. Open

1. Store and retrieve at a 2096-token block. Unproven. This is the only thing
   between here and a benchmark.
2. Whether a 2096-token chunk is workable for L1 at all. At 32,434 B/token that
   is a 68 MB storage granule, two orders of magnitude coarser than the 256
   every arm on disk ran. Store latency, eviction granularity and the lazy
   policy's block accounting all assumed the small chunk.
3. The realisation fraction on this model. Measured only at CONC=64 on coder30.
4. Whether b = 2.4 GB/ms/GPU carries to 2 KV heads at head_dim 256 with 45
   linear layers. Everything in section 2 scales with it.
