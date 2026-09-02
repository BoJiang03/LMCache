# Drafts: LMCache issue + reply to VAST on the GPU-only regression

Status: **drafts only.** Nothing filed, nothing sent, nothing pushed.
Evidence chain is complete and does not depend on the still-running 1b sweep
or on Phase 2.  Companion to `1_vast_lmcache_gpu_only_regression_repro.md`.

---

## Part A — LMCache issue draft

**Title:** `LMCacheConnectorV1` silently halves the GPU KV cache on hybrid-attention models (no `SupportsHMA`)

**Body:**

### Summary

On any model with sliding-window or Mamba/GDN attention, attaching
`LMCacheConnectorV1` causes vLLM to turn off its hybrid KV cache manager, which
roughly **halves the GPU KV cache pool** — before LMCache has stored or
retrieved a single byte.  The user is told only via a `WARNING` at startup, and
cannot override it.

The `LMCacheMPConnector` (multiprocess path) is unaffected: it declares
`SupportsHMA` and ships the group-handling machinery to back that up.  The
in-process path never got the same port.

### Reproduce

vLLM 0.22.1, LMCache @ `dev`, `openai/gpt-oss-120b` (36 layers: 18 sliding
window @128 + 18 full attention), TP=8, `--max-model-len 131072`,
`--block-size 64`, `--enable-prefix-caching`.  Read the pool off the startup log
(`kv_cache_utils.py:1733`):

| config | GPU KV cache size | max concurrency @131k |
|---|---|---|
| plain vLLM | **25,798,626** tokens | 196.83x |
| plain vLLM `--disable-hybrid-kv-cache-manager` | **13,724,416** tokens | 104.71x |
| plain vLLM + `LMCacheConnectorV1` (no storage tier at all) | **13,724,416** tokens | 104.71x |

The last two are byte-identical: attaching the connector is exactly equivalent
to passing `--disable-hybrid-kv-cache-manager`.  **1.880x** less KV cache.

The LMCache config used for row 3 stores nothing (`local_cpu: false`, no L2), so
this is purely the cost of the connector being *present*.

### Mechanism

`vllm/config/vllm.py:1440-1478` — when the user expresses no preference and a
`kv_transfer_config` is set, vLLM checks the connector class and disables HMA if
it does not subclass `SupportsHMA`:

```
WARNING [vllm.py:1471] Turning off hybrid kv cache manager because connector
LMCacheConnectorV1 does not subclass `SupportsHMA`. This will reduce performance
on models with sliding window or Mamba attention.
```

- `vllm/distributed/kv_transfer/kv_connector/v1/lmcache_connector.py:72`
  `class LMCacheConnectorV1(KVConnectorBase_V1):` — no `SupportsHMA`.
- `lmcache/integration/vllm/lmcache_mp_connector.py:273`
  `class LMCacheMPConnector(KVConnectorBase_V1, SupportsHMA):` — has it.
- `vllm/.../v1/base.py:85` — `SupportsHMA` is a one-method ABC
  (`request_finished_all_groups`); `supports_hma()` at `base.py:117`.

There is **no user-side workaround**: `vllm/config/vllm.py:1483-1490` raises
`ValueError` if you try to force HMA back on while a non-HMA connector is
configured, so `--no-disable-hybrid-kv-cache-manager` fails outright.

### Why this is not a one-line fix

Adding the mixin alone would be worse than the bug.  `SupportsHMA` hands the
connector `block_ids` as a **tuple, one list per KV cache group**.  The
in-process adapter assumes a single group in three places and collapses to group
0, discarding the rest:

- `lmcache/integration/vllm/vllm_v1_adapter.py:175` `if not isinstance(new_request.block_ids[0], list):`
- `lmcache/integration/vllm/vllm_v1_adapter.py:186` `unfolded_block_ids = new_request.block_ids[0].copy()`
- `lmcache/integration/vllm/vllm_v1_adapter.py:235` `new_block_ids = new_block_ids[0]`

The MP path solved this with dedicated machinery the IP path does not import:
`lmcache/integration/vllm/kv_cache_groups.py`
(`create_engine_group_infos_from_vllm`, sliding-window and cachable-Mamba spec
detection, per-layer sliding-window size resolution, group merging) and
`kv_cache_group_edits.py` (Mamba page views, subpaged attention, MLA).  Its
`request_finished_all_groups` (`lmcache_mp_connector.py:1099-1105`) is a
one-line delegation *only because* that machinery already normalised the groups.

So vLLM turning HMA off is defensively correct — silent KV corruption would be
worse than a halved pool.  The gap is that the IP adapter never got the port
that landed for MP.

### Impact

Every hybrid-attention model, not just gpt-oss: Gemma 3, Qwen3-Next, and the
Mamba/GDN families all pay it.  The larger the sliding-window fraction, the
larger the loss.

Measured cost on gpt-oss-120b, ISL=60k, OSL=1, TP=8, warm pass (P99 TTFT / total
tok/s), isolating the pool halving from connector overhead:

| concurrency | plain vLLM (25.8M) | HMA off (13.7M) | pool-halving cost |
|---|---|---|---|
| 100 | 23.0s / 256,659 | 21.1s / 280,531 | 0.91x |
| 300 | 81.4s / 211,141 | 155.6s / 113,470 | **1.86x** |
| 600 | 295.9s / 119,206 | 368.9s / 95,303 | 1.25x |
| 1000 | 612.7s / 96,383 | 616.1s / 95,755 | 1.01x |

The cost is a hump, not a constant: zero at low concurrency (working set fits
either pool) and zero at high concurrency (fits neither, both queue), peaking
where the working set straddles the two pool sizes.  Deployments sized near that
knee lose about half their throughput.

### A separate finding in the same measurements

Attaching the connector costs throughput even where the pool size has stopped
mattering.  At concurrency >= 600 every config is engine-bound and queueing, and
the 1.88x pool difference buys nothing -- plain vLLM and HMA-off converge to
within 0.2% (95,946 vs 95,755 tok/s).  But with `LMCacheConnectorV1` attached
and configured to store nothing at all (`local_cpu: false`, no remote backend),
throughput sits at 82,346 tok/s -- a flat **14.2%** lower, with only 1.7%
spread across c=600/1000/1500.

That is scheduler-side per-request work running against an empty store:
`get_num_new_matched_tokens`, `build_connector_meta` and `request_finished` in
`vllm_v1_adapter.py` all execute in full whether or not there is anything to
match.  Worth a fast path when no backend can serve a hit.

### Suggested fix

Port the group handling from the MP path into `vllm_v1_adapter.py`, then have
`LMCacheConnectorV1` declare `SupportsHMA` and forward
`request_finished_all_groups`.  Note `LMCacheConnectorV1` lives in **vLLM's**
tree, so the declaration lands there while the adapter work lands in LMCache —
worth coordinating so the mixin is never added ahead of the adapter.

Until then, consider making the situation legible from the LMCache side: the
warning currently comes from vLLM and names only the class.

---

## Part B — reply to VAST draft

> Thanks for the detailed writeup and for including the exact configs — being
> able to run your YAML verbatim is what made this quick to pin down.
>
> **Finding (1) reproduces, and the cause is upstream of LMCache's data path.**
> On vLLM 0.22.1 with gpt-oss-120b, simply attaching `LMCacheConnectorV1` makes
> vLLM turn off its hybrid KV cache manager, because that connector class does
> not subclass vLLM's `SupportsHMA` marker. The GPU KV pool goes from 25,798,626
> tokens to 13,724,416 — a 1.88x reduction — before LMCache stores anything. We
> confirmed it is exactly equivalent to `--disable-hybrid-kv-cache-manager`: the
> two pool sizes are byte-identical. Your startup log should carry the same
> `Turning off hybrid kv cache manager because connector LMCacheConnectorV1 does
> not subclass SupportsHMA` warning; if you still have those logs, the
> `GPU KV cache size: N tokens` line would confirm you saw the same halving.
>
> This is broader than the GPU-only case you framed it as — it applies to every
> configuration in your chart, and to any sliding-window or Mamba model, not
> just gpt-oss.
>
> Two things worth knowing about the shape of the cost. It is a hump, not a
> constant: we measure 1.86x at concurrency 300 and ~1.0x at 100 and at 1000,
> because it only bites when the working set fits the larger pool but not the
> smaller.
>
> The second is separate from the allocator and may matter more to you at the
> concurrencies in your matrix. Above concurrency 600 the pool size stops
> mattering entirely — plain vLLM and allocator-off converge to within 0.2%
> (95,946 vs 95,755 tok/s), because everything is engine-bound and queueing. But
> with `LMCacheConnectorV1` attached and configured to store *nothing*
> (`local_cpu: false`, no remote backend, exactly your IP config), throughput
> sits at 82,346 tok/s — a flat 14.2% lower, within 1.7% across concurrency 600,
> 1000 and 1500. That is per-request scheduler-side connector work running
> against an empty store. Your IP-without-L2 curve is paying it.
>
> **Finding (2) we cannot yet explain, and one hypothesis is now ruled out.** We
> first suspected the MP/IP gap came from your MP config passing
> `--disable-hybrid-kv-cache-manager` while IP did not; the 1.88x pool ratio
> matches your ~1.87x throughput gap almost exactly. That turns out to be a
> coincidence: because 0.22.1 auto-disables HMA for the IP connector too, **both
> arms of your matrix ran with the allocator off**. Our current hypothesis is a
> different asymmetry — your MP config sets `--l1-size-gb 1600`, a real CPU
> tier, while your IP config sets `local_cpu: false`, i.e. no tier at all. We
> are setting up that comparison now.
>
> Three notes on the configs themselves:
>
> - `max_local_cpu_size` is **per TP rank**. Your IP config's `180.0` at TP=8
>   pins 1.44 TB of host memory. We hit the OOM killer reproducing it.
> - `local_cpu: false` still allocates `max_local_cpu_size`, because
>   `LocalCPUBackend` doubles as LMCache's chunk allocator. It is not a way to
>   opt out of the allocation.
> - Setting `max_local_cpu_size: 0` does not work either — it raises
>   `KeyError('LocalCPUBackend')` during init, which LMCache swallows and then
>   runs in "degraded mode (recompute)" while vLLM keeps serving normally. We
>   lost an hour of benchmarking to this before noticing. If any of your runs
>   used a zero or unset value, they are worth re-checking; degraded mode was
>   4.6x slower than plain vLLM in our measurements, which would badly distort a
>   comparison.
>
> On the Prometheus request: agreed, and the third point above is the argument
> for it — the failure that cost us an hour was invisible in every metric vLLM
> exposes.

### Questions for VAST

1. The `GPU KV cache size: N tokens` line from your startup logs, for at least
   one IP run and one MP run. This settles whether you saw the same halving.
2. Did MP and IP really run with different `--disable-hybrid-kv-cache-manager`
   settings, or was that flag present in both?
3. What was `max_local_cpu_size` in every run, and did any run log
   `marked as init failed` or `degraded mode`?
4. Were the MP and IP runs L2-matched? The configs you sent have MP with
   `--l1-size-gb 1600` and IP with `local_cpu: false`.

---

## Open questions in our own data

At c=300 the healthy 1b run (connector attached, no storage tier) came out
**24% faster** than 1c (plain vLLM, allocator forced off) despite a byte-identical
13,724,416-token pool, identical prompts, identical seed, and identical bench
arguments (verified field by field).  Windowed to the same benchmark interval,
1b's vLLM prefix-cache hit rate was 47.0% against 1c's 24.2%, with mean in-engine
concurrency 3.2 vs 1.2.

This is unexplained and possibly not real: 1c was measured at 17:53 on Sep 1,
fifteen minutes after the OOM incident, while root's two k8s lmcache pods were
restarting and re-claiming 200 GB each.  `scripts/phase1_control.sh` re-runs 1c
at c=300 and c=600 back-to-back with 1b on the same machine state to settle it.
**Do not cite the 1.91x figure as final until that control lands** — it is the
one number in Part A that this could move.
