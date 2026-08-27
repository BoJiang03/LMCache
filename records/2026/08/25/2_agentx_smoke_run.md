# AgentX smoke run on the lazy offload branch

Date: 2026-08-25 (10:07-11:20 PT)
Branch: `lazy-offload-publish` @ 924e2c1c (worktree `/home/bo/LMCache-worktrees/lazy_offloading`)
Scripts + logs: `records/2026/08/25/artifacts/`

## Goal

First end-to-end run of the AgentX corpus against vLLM + LMCache MP connector with
lazy offload enabled. Toolchain validation, not measurement: does aiperf drive the
server, does the policy engage, do the counters move.

## Environment

| item | value |
|---|---|
| model | `Qwen/Qwen3-Coder-30B-A3B-Instruct`, TP=2 on GPUs 2,3 |
| why this model | plain GQA (no hybrid, no sparse), 262k native ctx, 57 GB weights, boots in ~4 min |
| KV/token | 98,304 B = 96 KiB (2 x 4 kv_heads x 128 head_dim x 48 layers x 2 B); confirmed by the MP server's `cache_size_per_token: 49152` per rank x 2 ranks |
| vLLM | 0.23.0, `/home/bo/venvs/vllm-lazy`, `enable_prefix_caching=True`, block_size 16 |
| lmcache | editable install pointing at the worktree, so the branch code is live |
| aiperf | 0.12.0, `/home/bo/venvs/aiperf` (python3.12) |
| corpus | `--public-dataset semianalysis-cc-traces-weka-062126` |
| scenario | `--scenario inferencex-agentx-mvp --unsafe-override` (locks AGENTIC_REPLAY + ignore_eos + streaming + first-turn-prefix cache-bust; `--unsafe-override` only to run shorter than the 900 s submission floor) |

Two environment blockers fixed on the way in:

- vLLM 0.23 dropped `--disable-log-requests`.
- triton's runtime launcher needs `Python.h`; no `python3.12-dev` on this host.
  `CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:...` (the
  unpacked headers the earlier repro runs already used) is required or every
  worker dies during engine init.

## Corpus facts measured this session

Peak context (max `in`+`out` over all requests including subagents) per trace,
over all 393 traces:

| p0 | p5 | p25 | p50 | p75 | p90 | p99 |
|---|---|---|---|---|---|---|
| 44,048 | 82,073 | 142,160 | 226,315 | 412,037 | 696,834 | 996,579 |

Traces surviving `--max-context-length`:

| cap | traces | requests |
|---|---|---|
| 40,000 | 0 | 0 |
| 64,000 | 6 | 150 |
| 100,000 | 42 | 1,362 |
| 131,072 | 82 | 3,132 |
| 262,144 | 220 | 14,817 |
| 1,000,000 | 393 | 98,827 |

**No trace fits in 40k.** The minimum peak is 44,048 tokens, so Qwen3-8B at its
native 40960 cannot replay a single AgentX trace. That rules out the model the
whole `repro/pr4499` calibration was built on -- any AgentX run needs >=64k
context, and a representative one needs >=131k.

Correction to the previous record: the `-256k` HF repo carries only `stats.txt`,
no `traces.jsonl`. The 256k variant is not a separate corpus; it is the same 393
traces with a client-side `--max-context-length` cap. Its stats.txt is what gave
the 21.6B -> 6.9B input-token figure.

## Runs

`--random-seed 1234` throughout. All aiperf tables in `artifacts/*_aiperf_stdout.log`.

| # | config | pool | conc | max ctx | L1 | reqs | TTFT p50 | latency p50 | local APC hit | external hit | preempt |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | lazy | 24 GiB | 4 | 131,072 | 64 GB | 18 | 711 ms | 4,030 ms | 83% | **0** | 1 |
| 2 | lazy | 24 GiB | 32 | 131,072 | 64 GB | 69 | 39,488 ms | 45,755 ms | 0.25% | **0** | 41 |
| 3 | eager | 48 GiB | 16 | 131,072 | 600 GB | 63 | 630 ms | 7,725 ms | 58% | **0** | 0 |
| 4 | lazy | 24 GiB | 8 | 100,000 | 600 GB | 69 | 531 ms | 2,392 ms | n/m | **78,352 tok** | 9 |
| 5 | lazy | 24 GiB | 8 | 100,000 | 200 GB | 69 | 533 ms | 2,434 ms | n/m | **79,088 tok** | 9 |

Runs 4 and 5 are the same configuration at two L1 budgets; they agree to within
0.4% on TTFT p50 and 1.8% on latency p50, and to 0.9% on external hit tokens. The
seed reproduces.

"n/m" = not meaningful: under preemption vLLM re-queries the prefix cache on every
prefill restart, so `prefix_cache_queries_total` reached 128M for ~4.4M tokens of
actual prompt (a 29x amplification). The ratio stops being a hit rate.

## Findings

### 1. The chain works

aiperf resolves AGENTIC_REPLAY from the scenario, builds one trajectory lane per
`--concurrency`, replays recorded think-time gaps, and the server answers with
0 client errors in runs 1-3. Requests are real: ISL p50 103,898 tokens in run 1.

### 2. Lazy offload engages, and every gate is observable

Run 5's final counters:

```
admitted=445 emitted=329 dropped_evicted=19 rejected_short_prefix=0
rejected_unhashed=0 rejected_prefix_broken=0 dropped_on_request_drop=1
dropped_failed_store=0 dropped_id_reuse=0 deduplicated=0 throttled_drains=1
drain_steps=22140 free_queue_blocks_read=986849 requests_validated=706
blocks_validated=2367099 pending=96 held=0
```

`dropped_evicted=19` is gate 1's quality sensor firing, `throttled_drains=1` is
the drain cap binding once. Coverage is 329/445 = 74%.

### 3. The GPU pool, not the workload, decides whether LMCache can contribute

External hits were zero in runs 1-3 and non-zero in 4-5. The reason is structural,
not a bug: an agent turn's prompt is the previous turn's prompt plus new content,
so the *only* tokens vLLM ever asks LMCache about are the ones its own pool has
evicted. Give the pool room and LMCache is asked exclusively about genuinely-new
tokens and correctly answers 0.

- Run 1 (24 GiB pool, 4 lanes): effective concurrency 0.77, pool 29% used. No
  eviction, so nothing to retrieve. `emitted` froze at 29 while `admitted` climbed
  to 103 -- gate 1 correctly never fired, and 71 ops sat pending. Not a stall.
- Run 3 (48 GiB pool, 16 lanes): 0 preemptions, 58% local hit, still 0 external.
  The pool absorbed the whole reuse set.
- Run 2 (24 GiB pool, 32 lanes): the other failure mode. 32 x ~100k against a
  262k-token pool is ~12x oversubscribed; 41 preemptions, TTFT p50 39 s, local hit
  0.25%. Collapse, and still 0 external hits because L1 was 64 GB against ~307 GiB
  of distinct prefix.
- Runs 4-5 (24 GiB pool, 8 lanes, 100k cap, L1 >= 200 GB): pool small enough to
  evict, L1 large enough to retain. External hits appear.

The operating rule for the real sweep: **pool below the reuse working set, L1
above it.** Both conditions are needed; run 2 had the first and not the second.

### 4. Concurrency has to be much higher than the target in-flight count

AGENTIC_REPLAY honors recorded think time (p99 23 min). Measured duty cycle is
~21%: 4 lanes gave effective concurrency 0.77. So a lane is not a concurrent
request, and `--concurrency N` buys roughly `N/5` in-flight requests. Reaching a
useful in-flight count of ~12 needs ~50 lanes, which puts the reuse working set at
~50 x 70k x 96 KiB = ~320 GiB. That is well above any GPU pool and is the regime
LMCache exists for -- but it also means the earlier sizing note ("12 sessions") was
an order of magnitude too small on the lane count.

Warmup is a fixed tax: it primes one snapshot per lane serially, ~10-14 s each
(339 s for 34 lanes).

### 5. Blocking: the MP server's transfer path fails with cudaErrorInvalidValue

This is what stops the sweep from being run today.

```
File "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py", line 1173, in store
File "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py", line 599, in transfer_kv_per_object_group
File "lmcache/v1/platform/base/device_ops.py", line 175, in multi_layer_block_kv_transfer
File "lmcache/v1/platform/torch_ops.py", line 1233, in multi_layer_block_kv_transfer
File "lmcache/v1/platform/torch_ops.py", line 1797, in _transfer_per_layer_nhd
    selected = layer.index_select(0, eff_idx)
torch.AcceleratorError: CUDA error: invalid argument   (cudaErrorInvalidValue)
```

A second variant of the same failure surfaces one frame lower as
`RuntimeError: cudaMemcpy failed with error code 1` from
`gpu_ops.lmcache_memcpy_async_d2h`. Per-run counts:

| run | config | store exceptions | fatal |
|---|---|---|---|
| 3 | eager | 640 | 0 |
| 4 | lazy | 98 | 1 |
| 5 | lazy | 84 | 1 (server died at 11:15:57) |

Not a lazy offload regression:

- `git diff $(git merge-base HEAD origin/dev)..HEAD -- lmcache/v1/` is empty. The
  branch touches only `lmcache/integration/vllm/`; the failing files are unmodified
  dev code.
- eager fails 6.5x more often than lazy, and eager uses none of the lazy code path.
  Lazy fails *less* simply because it stores less.

Consequence: in runs 4 and 5 the MP server process died mid-run (fatal Python
error after the AcceleratorError), taking the in-flight requests with it -- the
"1 error" in both. Any A/B on this configuration is measuring a server that
does not survive the run.

Excerpt in `artifacts/mp_server_transfer_failure.txt`.

#### Root cause

The `index_select` `AcceleratorError` is a symptom, not the cause. In every log the
first failure is the raw `cudaMemcpy`, with the `AcceleratorError` following ~0.5 s
later on a context that already has a sticky error -- its own message says
"CUDA kernel errors might be asynchronously reported at some other API call".

The real site is `torch_ops.lmcache_memcpy_async`, pointer mode:

```python
ret = libcudart.cudaMemcpy(
    ctypes.c_void_p(dest), ctypes.c_void_p(src),
    ctypes.c_size_t(nbytes), ctypes.c_int(4),  # cudaMemcpyDefault
)
```

It issues one `cudaMemcpy` for the whole object over a host range that can span two
separately `cudaHostRegister`ed pin chunks. `LazyMemoryAllocator.PIN_CHUNK_SIZE` is
`1 << 26` = 64 MiB (the "Expanded 10240 MB" log line is the *reservation* step, not
the registration granularity), and an L1 object here is one 256-token chunk at
96 KiB/token = 24 MiB. Instrumenting the call (`nbytes=25165824`, `align=67108864`):

| | crosses a 64 MiB pin boundary | does not |
|---|---|---|
| `cudaMemcpy` returned 1 | 17 | **0** |
| `cudaMemcpy` returned 0 | 213 | 662 |

Crossing a pin-chunk boundary is a **necessary** condition for the failure -- no
failure occurred without it -- though not a sufficient one (17 of 230 crossings
fail; what selects those 17 is not yet identified, and the pin frontier is ruled
out: L1 was fully expanded to its 200 GB target before any traffic).

Per-object failure rate is 17/892 = 1.9%, and a store batch here carries ~31
objects (8,002 tokens / 256), so P(batch fails) = 1 - 0.981^31 = 45%, which is
the ~37-53% batch failure rate observed.

Why the fallback runs at all: the compiled extension is missing on this host, logged
at every startup -- `lmcache.cuda_ops compiled extension not found; CudaDeviceOps
stays on the torch baseline for all ops`. The C++ implementation splits the copy at
`host_buffer_alignments` boundaries; the Python fallback deliberately does not, and
its own docstring states the premise that turns out to be wrong:

> Unlike the C++ version (which uses cudaMemcpyAsync and must split copies at
> cudaHostRegister boundaries), this Python fallback does NOT need alignment-based
> chunking because cudaMemcpy (synchronous) handles cross-cudaHostRegister
> boundaries internally via staging buffers

Both call sites already pass the two arguments a split would need
(`memory_obj.meta.address`, `LazyMemoryAllocator.PIN_CHUNK_SIZE`); the fallback just
ignores them apart from a power-of-two check. The raw-pointer path is only taken
`if isinstance(memory_obj.parent(), LazyMemoryAllocator)` -- the other branch uses
`tensor.copy_()` and is safe, which is likely why earlier repro runs at smaller L1
budgets never showed this.

#### Confirmed pre-existing on dev

Ruled out as ours three independent ways:

1. `git diff HEAD origin/dev -- lmcache/v1/platform/torch_ops.py` is **empty**. The
   file is byte-identical in today's `origin/dev` (23cca679), same unsplit
   `cudaMemcpy` at the same line numbers.
2. Same failure at TP=1 as TP=2 (`world_size: 1` in `/status`), so it is not a
   multi-rank / IPC-context problem -- that hypothesis was tested and falsified.
3. **Ran `origin/dev` itself.** Worktree `/home/bo/LMCache-worktrees/dev_baseline`
   at 23cca679 (no `lazy_offload_manager.py` in that tree at all), same model, same
   eager config, same 10 x 48k-token probe:

   | | successful stores | store exceptions | cudaMemcpy err | AcceleratorError |
   |---|---|---|---|---|
   | branch `lazy-offload-publish` | 27 | 33 | 17 | 16 |
   | `origin/dev` @ 23cca679 | 27 | 33 | 17 | 16 |

   Identical, and the dev traceback is from
   `dev_baseline/lmcache/v1/platform/torch_ops.py:2210`.

Note for future baselines: the cwd shadows the editable-install finder, because
`$REPO/lmcache` is a real package directory. cwd, not the finder, decides which tree
is served -- `up.sh` now `cd`s to `$REPO` and refuses to start unless
`lmcache.__file__` is under it, so a baseline run cannot silently import the branch.

The reproducer is `artifacts/probe.py` (N distinct long prompts, no aiperf) --
30 seconds per data point instead of a 6-minute warmup.

### 6. Harness hazards worth keeping fixed

- `up.sh` was not idempotent: a second launch starts a second MP server that
  loses the port race, keeps running without a listener, and `/healthcheck`
  answers from whichever one owns the port -- so "mp-server up" was not evidence
  the *new* server was serving. `down.sh` then killed the pid in `server.pid`,
  leaving the other alive. Both scripts are now hardened: `up.sh` refuses to start
  on a bound port or with an MP server already running, and asserts the L1 target
  by reading it out of the log of the server it just started; `down.sh` kills by
  GPU-owner sweep and exits non-zero unless the GPUs and ports are verifiably
  clear.
- `--l1-size-gb` is honored, not clamped. `/status`'s `memory_total_bytes` reports
  the `LazyMemoryAllocator`'s *currently expanded* pinned size, which grows ~10 GB
  per 3 s in the background. Reading it early looks like a clamp (it reported
  22 GiB for a 600 GB target) and it cannot be used to identify a server.
- L1 was set to 200 GB rather than 600: the node is shared (other users held
  GPUs 0, 4, 6, 7 during this session) and 600 GiB of pinned host memory against
  674 GB free is not a neighbourly thing to do. 200 GB covers the largest observed
  use (182 GB).

## Open items

1. File the `_transfer_per_layer_nhd` / `index_select` cudaErrorInvalidValue against
   dev, with the eager-vs-lazy counts as evidence it is not the PR. Needs a minimal
   repro first: which `eff_idx` value is out of range for `layer`, and whether it
   correlates with `--num-gpu-blocks-override`.
2. Re-run the eager/lazy pair at pool 24 GiB / conc 8 / max ctx 100k once the
   transfer bug is fixed. That is the first honest A/B point.
3. Re-derive the sweep plan on the corrected lane arithmetic (~5x lanes per
   in-flight request), which changes the pool and L1 axes from the previous record.
4. The previous record's DeepSeek-V4-Flash KV figure (86 KiB/token) is wrong.
   `config.json` has `compress_ratios` over 44 layers with values {0: 3 layers,
   4: 21, 128: 20} plus `sliding_window=128` and `index_topk=512` -- this is sparse
   /compressed attention, so effective KV is far below the 48 KiB/token the
   uncompressed 576-wide latent would give. It also means DeepSeek-V4-Flash and
   Qwen3.5 (linear-attention hybrid) both add an unvalidated attention backend on
   top of the measurement. Qwen3-Coder-30B-A3B stays the harness-validation model.
5. Still not done: the interactive code walkthrough. Runs 1 and 5 now give it a
   concrete spine -- run 1 is gate 1 correctly refusing to fire (emitted frozen at
   29, pending 71), run 5 is the full cycle with `dropped_evicted` and
   `throttled_drains` both non-zero.

## References

- `records/2026/08/25/1_agentx_workload_and_model_selection.md` -- workload and model choice
- `docs/design/integration/vllm/lazy_offload.md`, `..._decision_model.md`,
  `.../lazy_offload_policy/eviction_aware.md`
- aiperf scenario spec: `aiperf/common/scenario/inferencex_agentx_mvp.py`
