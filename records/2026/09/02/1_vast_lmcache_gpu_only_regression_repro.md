# VAST <> LMCache: reproducing the GPU-only regression

Date: 2026-09-01 / 2026-09-02
Line: `vast_repro` (`vast_repro_dev`)
Task: mentor handed over `vast__LMCache collab.pdf` — reproduce VAST's findings, and if
they hold, find out why LMCache underperforms.

## What VAST reported

1. GPU-only KV cache configs are **slower** with LMCache than with vLLM alone
   (both NVIDIA and AMD). Chart: P99 TTFT, GPT-OSS-120B on MI355X, ISL=60K, OSL=1,
   TP=8, vllm-v0.22.1, warm cache, concurrency 1 -> 1500.
   At c=1500: `1a` GPU-only 761,774 ms vs `1b` GPU-only+LMCache ~885,142 ms.
   `3.` GDS-hipifile to local NVMe is far worse (1,556,769 ms).
   `4-7.` LMCache-fs to VAST is best (204,440 ms at best).
2. LMCache-MP slower than LMCache-IP on AMD, ISL=120k: IP w/ VAST ~505k tok/s vs
   MP w/ VAST ~270k tok/s (1.87x).
3. (configs, reproduced verbatim in `harness/configs/`)
4. Customer wants more Prometheus logging / observability.

## Environment built for the repro

- `/home/bo/vast_profiling_problem/.venv` — uv venv, Python 3.12.13,
  **vllm 0.22.1** (exact version match with VAST), torch 2.11.0+cu130.
- LMCache editable install from worktree `~/LMCache-worktrees/vast_repro`,
  branch `vast_repro_dev`, based on `origin/dev` @ `a12e430c`.
  CUDA extensions rebuilt against torch 2.11 (`cuda_ops`, `lmcache_native`,
  `lmcache_fs`, `lmcache_redis`). `uv pip check` clean.
- Hardware: 8x H200 (not MI355X), 2015 GB RAM, no swap. Shared box with other
  tenants (see "Incident" below). Model: `/raid/rui/gpt-oss-120b` (official
  MXFP4 checkpoint).
- Ports moved off defaults (8765 / MP 5765) — 5555 was already taken by a
  colleague's 15-day-old lmcache server.

## Key numbers

gpt-oss-120b: 36 layers = 18 `sliding_attention` (window 128) + 18 `full_attention`,
`num_key_value_heads: 8`, `head_dim: 64` -> **36 KB/token** of KV with the hybrid
allocator on, 73 KB/token with it off. (Llama-3.1-70B is ~320 KB/token for
comparison — gpt-oss is an unusually KV-light model.)

Measured GPU KV pool (`kv_cache_utils.py:1733`):

| config | pool | max concurrency @131k |
|---|---|---|
| plain vLLM | **25,798,626** tokens | 196.83x |
| `--disable-hybrid-kv-cache-manager` | **13,724,416** tokens | 104.71x |
| plain vLLM **+ LMCacheConnectorV1** | **13,724,160** tokens | 104.71x |

Ratio 1.880x.

## Root cause of finding (1)

`vllm/config/vllm.py:1440-1478` (this is vllm 0.22.1, the version VAST used):
when the user expresses no preference and a `kv_transfer_config` is set, vLLM
checks whether the connector subclasses `SupportsHMA`; if not it **silently
disables the hybrid KV cache manager**:

```
WARNING [vllm.py:1471] Turning off hybrid kv cache manager because connector
  LMCacheConnectorV1 does not subclass `SupportsHMA`. This will reduce
  performance on models with sliding window or Mamba attention.
```

- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:72`
  `class LMCacheConnectorV1(KVConnectorBase_V1)` — **no** `SupportsHMA`.
- `lmcache/integration/vllm/lmcache_mp_connector.py:273`
  `class LMCacheMPConnector(KVConnectorBase_V1, SupportsHMA)` — **has** it
  (added by PR #3419 "[Core] add support for hybrid memory allocator").

So **attaching LMCache IP halves the GPU KV cache pool** on any
sliding-window / Mamba hybrid model (gpt-oss, Gemma 3/4, Qwen3-Next, ...).
This is more general than VAST's framing: it has nothing to do with "GPU-only";
they just happened to test in a config where the pool was the only variable.

`SupportsHMA` requires one method:
`request_finished_all_groups(request, block_ids: tuple[list[int], ...])`.

**A hypothesis that turned out to be wrong** (recorded so it isn't re-derived):
I initially thought MP was the one forced to disable the allocator (its config
passes `--disable-hybrid-kv-cache-manager`, and
`lmcache_mp_connector_0201.py:81` raises without it), and that this explained
finding (2), since 1.880x pool ratio ~ 1.87x throughput gap. That is a
coincidence. Because vLLM 0.22.1 already auto-disables HMA for IP, **both** arms
of VAST's 4-way matrix ran with the allocator off. Finding (2) is unexplained
again; current best hypothesis is a different asymmetry — their MP has
`--l1-size-gb 1600` doing real work while their IP has `local_cpu: false`, i.e.
no tier at all.

## Benchmark results (warm pass, all from the JSON files)

`vllm bench serve`, random dataset, ISL=60000, OSL=1, `--ignore-eos --seed 42`,
`num_prompts == max_concurrency`, cold pass then warm pass.

| conc | 1a plain (25.8M) | 1c no-HMA (13.7M) | 1b +LMCache (13.7M) |
|---|---|---|---|
| 100 | p99 23.0s / 256,659 tok/s | p99 21.1s / 280,531 | p99 23.1s / 257,513 |
| 300 | p99 81.4s / 211,141 | p99 155.6s / 113,470 | (running) |
| 600 | p99 295.9s / 119,206 | p99 368.9s / 95,303 | (running) |
| 1000 | p99 612.7s / 96,383 | p99 616.1s / 95,755 | (running) |
| 1500 | p99 928.2s / 95,509 | (engine crashed, missing) | (running) |

Cost of the halved pool alone (1c/1a): 0.92x, **1.91x**, 1.25x, 1.01x, — .
A clean hump: zero when the working set fits both pools, zero again when it
fits neither, **peaking where it fits only the big one** (c~300). That window is
the common production operating point.

Connector overhead alone (1b/1c) at c=100: **~1.09x**. 1b ~= 1a there.

## Methodology notes

- **The warm pass at low concurrency is client-bound, not engine-bound.** Server
  reports `Avg prompt throughput: 0.0-169 tokens/s` during the warm pass while
  the client sees 23s — the time is API-server tokenization of 60k-token
  prompts (~256k tok/s). Cold pass IS engine-bound (96,000 tok/s prefill,
  matches duration). This is why VAST's chart has all 7 curves collapsed
  below c~400: a shared client-side bottleneck, not equal engine performance.
  vllm 0.22.1 has `--api-server-count`; adding `-asc 8` would remove it.
- `--max-num-seqs 256` is a no-op in this workload: one 60k prefill saturates
  `max_num_batched_tokens`, so the log shows `Running: 1 reqs, Waiting: 70 reqs`.
- H200 prefill 96k tok/s vs VAST's implied 118k tok/s on MI355X — same
  ballpark, so the platforms are comparable.

## LMCache bugs found

1. **`LMCacheConnectorV1` lacks `SupportsHMA`** -> silently halves the GPU KV
   pool on hybrid-attention models. Root cause of finding (1). Fix needs both
   sides: vLLM's `LMCacheConnectorV1` subclassing `SupportsHMA` and forwarding
   `request_finished_all_groups`, and LMCache's `vllm_v1_adapter` handling
   multiple KV cache groups (`kv_cache_groups.py` already exists for MP).
2. **`max_local_cpu_size <= 0` crashes init.** `storage_backend/__init__.py:181-190`
   deliberately skips creating `LocalCPUBackend` (logs an INFO — an expected
   branch), then `storage_manager.py:325` unconditionally does
   `self.storage_backends["LocalCPUBackend"]` -> `KeyError`. `manager.py:235`
   swallows it and the engine runs in "degraded mode (recompute)" — **vLLM keeps
   serving normally and the user is never told**. `LocalCPUBackend` doubles as
   LMCache's chunk allocator, so a positive value is structurally required even
   with `local_cpu: false`.
3. **`local_cpu: false` still allocates `max_local_cpu_size`, per TP rank**, with
   no warning. Empirically: `free` shared went 8 GB -> 171 GB with
   `max_local_cpu_size: 20.0` at TP=8. VAST's own IP config (180.0, TP=8) pins
   **1.44 TB** of host memory; they likely don't know.
4. **Degraded mode is catastrophically slow rather than free.** With the engine
   marked init-failed and every store/retrieve short-circuited, c=100 measured
   19,059 tok/s cold / 22,653 warm vs 88,290 / 256,659 for plain vLLM — 4.6x
   slower cold, and vLLM's own prefix cache stopped helping (cold->warm speedup
   1.2x vs 2.9x). Note `vllm_v1_adapter.py` contains **no** `is_healthy` check —
   the scheduler-side `get_num_new_matched_tokens` / `build_connector_meta` /
   `request_finished` still run in full against a dead storage manager.
   (Confirmed specific to the degraded path: healthy 1b at c=100 costs only ~9%.)

## Incident: I OOM-killed two of someone else's processes

At 17:38:24 I launched 1b with VAST's `max_local_cpu_size: 180.0` copied
verbatim. That is **per TP rank**: 180 x 8 = 1440 GB, against ~1404 GB
available and no swap. The kernel OOM killer took the largest RSS processes:

- my `VllmWorker-7` died at **17:38:24**
- root's two 200 GB k8s `lmcache server` pods restarted at **17:38:24** and
  **17:38:43** (new pids 3203853 / 3204611)

Same second. They self-healed — kubelet restarted them immediately and they have
been up since; rui's lmcache and the DeepSeek-V2-Lite docker service were
unaffected. Net effect was a ~19 second outage of a deployment I don't own.
My error: I had done the "180 x 8 = 1440" arithmetic hours earlier to explain
something else and failed to apply it before acting.

A second, unexplained worker death at 19:14:55 killed the 1c c=1500 warm pass
(`RuntimeError: Executor failed.`, no worker traceback). 1c is plain vLLM with
no host allocation and root's pods were not touched then, so it has a different
cause. c=1500 for 1c still needs a re-run.

## Harness

`harness/scripts/`:
- `env.sh`, `lib.sh` — shared launch / health-check / cold+warm bench / teardown.
  `teardown` only ever kills pids this script spawned (an earlier
  `pkill -f "vllm serve"` plus a "wait for nvidia-smi to go empty" loop would
  have hit other tenants and hung forever on their long-running processes).
  `spawn` sets `$SPAWNED_PID` rather than echoing it — `$(spawn ...)` ran in a
  subshell and silently dropped the pid from `MY_PIDS`.
  `trap on_signal INT TERM` exits; a bare handler returned and the script
  carried on running the next config.
- `phase0_pool.sh` — pool size with/without the allocator, no benchmark.
- `phase1_gpu_only.sh` / `phase1_rest.sh` / `phase1b_only.sh` — 1a / 1b / 1c.
  These abort before benchmarking if `marked as init failed` appears in the
  server log, so a degraded LMCache is never measured again.
- `phase2_mp_vs_ip.sh` — written, **needs redesign** now that IP is known to run
  with the allocator off too.
- `collect.py` — aggregates the result JSONs into comparison tables.

## Open items

- 1b at c=300/600/1000/1500 (running as of this record).
- Re-run 1c c=1500.
- Redesign phase 2 around the L1-asymmetry hypothesis for finding (2).
- Draft the reply to VAST + a LMCache issue for bug 1 (evidence is complete and
  does not depend on the remaining runs).
- Asked for but not yet available: `sudo mkdir -p /raid/bo/kvcache` (phase 2 L2
  needs the 8-disk RAID0; /dev/shm can't be used because tmpfs has no O_DIRECT
  and both VAST configs set `use_odirect: true`), and
  `sudo sysctl -w kernel.yama.ptrace_scope=0` for py-spy attach.
- Question list for VAST: their startup log's `GPU KV cache size: N tokens` line
  (settles the pool, the quantization path, and whether HMA was on in one shot);
  whether their MP and IP runs really used different allocator settings; and
  that their `max_local_cpu_size: 180.0` is per-rank.
