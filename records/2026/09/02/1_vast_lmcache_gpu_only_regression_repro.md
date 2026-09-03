# VAST <> LMCache: reproducing the GPU-only regression

> ## CORRECTION (2026-09-02, end of day) — the halved KV pool is REAL but COSTS NOTHING
>
> The pool halving itself stands: attaching either LMCache connector turns off
> vLLM's hybrid KV cache manager on gpt-oss-120b and the pool goes
> 25,798,626 -> 13,724,416 tokens, 1.880x, byte-for-byte reproducible.
>
> **What is false is that this explains the slowdown.** Measured per forward step
> (every arm runs `max_num_batched_tokens=8192`, so a step is a fixed 8192 tokens):
>
> | | pool 25,798,626 | pool 13,724,416 |
> |---|---|---|
> | no connector | **85.3 ms/step** (1a) | **85.3 ms/step** (1c) |
> | LMCache MP | **91.0 ms/step** (1d) | **91.0 ms/step** (1e) |
>
> The pool changes by 1.88x and the per-step cost does not move at all, in either
> row. At ISL=60,000 and c<=1500 the pool never binds: vLLM reports max concurrency
> 104.71x even with the allocator off. **Any sentence below that treats the halved
> pool as the cause of the regression, or that quantifies a "pool-halving cost" in
> throughput, is void.**
>
> The settled decomposition is in `7_per_step_decomposition_and_the_1f_negative.md`
> and `8_state_of_the_investigation.md`: vLLM's connector plumbing +0.0 ms/step,
> LMCache common to IP and MP +5.7, LMCache IP-only +6.5.
>
> This does not kill the `SupportsHMA` issue draft in record 2 — silently halving a
> user's KV cache is still worth reporting as a resource/correctness problem — but
> **that draft must not claim a throughput regression**, because there isn't one.



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
| plain vLLM **+ LMCacheConnectorV1** | **13,724,416** tokens | 104.71x |

Ratio 1.880x.  The last two rows are byte-identical, which makes 1b/1c a clean
control pair: same pool, same prompts, same seed, differing only in whether the
connector is attached.

(Corrected 2026-09-02: this row previously read 13,724,160, a number that came
from the *discarded* 17:16 1b run -- the one with `max_local_cpu_size: 180.0`
that the OOM killer took down.  The healthy re-run measures 13,724,416.
`scripts/phase1_rest.sh:5-6` still carries the stale figure in a comment.)

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

All three pools verified from the startup logs; 1b and 1c are byte-identical
at 13,724,416 tokens, so they differ only in whether the connector is attached.

| conc | 1a plain (25.8M) | 1c no-HMA (13.7M) | 1b +LMCache (13.7M) | pool cost | conn cost |
|---|---|---|---|---|---|
| 100 | p99 23.0s / 256,659 tok/s | p99 21.1s / 280,531 | p99 23.1s / 257,513 | 0.91x | 1.09x |
| 300 | p99 81.4s / 211,141 | p99 155.6s / 113,470 | p99 118.9s / 148,395 | 1.86x | **0.76x** |
| 600 | p99 295.9s / 119,206 | p99 368.9s / 95,303 | p99 434.6s / 81,477 | 1.25x | 1.17x |
| 1000 | p99 612.7s / 96,383 | p99 616.1s / 95,755 | p99 713.0s / 82,864 | 1.01x | 1.16x |
| 1500 | p99 928.2s / 95,509 | (engine crashed, lost) | p99 1072.1s / 82,696 | — | 1.15x* |

\* against 1a, since 1c's c=1500 warm pass was lost.  1a and 1c agree to 0.1%
on the c=1500 *cold* pass (924.3s vs 925.3s) and to 0.7% at c=1000, so the
substitution is safe here; it would not be at c=300.

**Cost of the halved pool alone (1c/1a): 0.91x, 1.86x, 1.25x, 1.01x.**
A clean hump: zero when the working set fits both pools, zero again when it
fits neither, **peaking where it fits only the big one** (c~300). That window is
the common production operating point.

**Cost of the connector alone (1b/1c): 1.09x, 0.76x, 1.17x, 1.16x, 1.15x.**
Four of the five points sit in 1.09-1.17x.  c=300 is an outlier in the wrong
direction (1b *faster* than 1c on both cold and warm despite the identical
pool), and is not trusted -- see "The c=300 anomaly" below.

### The connector's saturation tax

At c>=600 all three configs are engine-bound and the pool size stops mattering
entirely -- but the connector's cost does not go away:

| | c=600 | c=1000 | c=1500 | mean |
|---|---|---|---|---|
| 1a plain | — | 96,383 | 95,509 | **95,946** tok/s |
| 1c no-HMA | — | 95,755 | (lost) | **95,755** tok/s |
| 1b +LMCache | 81,477 | 82,864 | 82,696 | **82,346** tok/s |

1a and 1c converge to within 0.2% -- the 1.88x pool difference buys nothing once
every config is queueing.  1b sits **14.2% lower**, and flat: the three points
spread only 1.7%.  So with `local_cpu: false` and no remote backend -- LMCache
storing nothing and retrieving nothing -- the connector still costs a constant
~14% of engine throughput.  That is scheduler-side per-request work
(`get_num_new_matched_tokens`, `build_connector_meta`, `request_finished` in
`vllm_v1_adapter.py`) running against an empty store.  Independent of the
`SupportsHMA` bug, and arguably easier to act on: pure overhead, zero benefit.

### The c=300 anomaly

**Resolved by the control re-run, and the culprit was the opposite of what the
symptom suggested.**  At c=300, 1b beat 1c by 24% warm and 8% cold with a
byte-identical pool and bench arguments verified equal field by field.  The
natural reading was that the connector somehow helps.  It does not -- the
*original 1c* was simply a bad measurement:

| c=300 warm | when | p99 TTFT | tok/s |
|---|---|---|---|
| 1c original | Sep 1 17:53, 15 min after the OOM incident below | 155.6s | 113,470 |
| **1c control re-run** | **Sep 2 13:12** | **122.0s** | **143,628** |
| 1b | Sep 2 11:04 | 118.9s | 148,395 |

Measured on the same machine state, 1c and 1b agree within 3%: **the connector
costs approximately nothing at c=300.**  The 47.0% vs 24.2% prefix-cache-hit
gap was a symptom of the perturbed run, not a mechanism.

**This voids the 1.86x pool-halving figure**, and the replacement is *not*
simply 1a/1c_rerun = 1.47x.  1a was measured Sep 1 15:30.  Having just watched
this box move one point by 28% between sessions, pairing a Sep 1 number with a
Sep 2 number is precisely the error that produced the bad figure in the first
place.  `scripts/phase1_control_1a.sh` re-measures 1a at c=300/600 in the same
session; until it lands, the pool-halving magnitude is unquantified.  The
*shape* (a hump, ~1.0x at both ends) and the mechanism are unaffected.

The saturation tax is less exposed to this: 1a's c=1000 (96,383, Sep 1 ~16:30)
and 1c's c=1000 (95,755, Sep 1 ~18:40) agree to 0.7% *across* the OOM incident,
which suggests the engine-bound plateau is insensitive to host-memory pressure
-- as one would expect. The 1c control's c=600 point pairs with 1b's c=600 on
the same day for a direct same-session check.

**Harness lesson: interleave configurations within a session** (A/B/A/B) instead
of running one config to completion and then the next.  The ordering used here
made a machine-state shift indistinguishable from a treatment effect, and it
took a day and a re-run to separate them.

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
