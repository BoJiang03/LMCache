# Handoff: making LMCache work with vLLM 0.28 on a Mamba-hybrid model

Written 2026-08-30 for the agent taking this over. Everything below is either
measured on this box or read out of the source; nothing is projected.

Working directory: `/home/bo/LMCache-worktrees/lazy_offloading` (a git worktree,
branch `lazy-offload-dev`). Do not `cd` to the main checkout.

---

## 1. Why this work exists

The standing goal is a deployment on this hardware that satisfies Bo's
requirements R1-R7 jointly. The full statement and history live in
`records/deployment_requirements.md`; the picked configuration and its
predicted operating point live in `records/deployment_candidate.md`; the
reasoning that produced the pick is `records/2026/08/30/8_*.md`, the startup
probe is `records/2026/08/30/9_*.md`, and the session log is
`records/2026/08/30/10_*.md`. **Read those first if you need the deployment
context.** The short version:

- The candidate is **Qwen3.5-397B-A17B-FP8, TP=4** at
  `/raid/data/hub/models--Qwen--Qwen3.5-397B-A17B-FP8/snapshots/ea5b4f81.../`.
  It is the only model on the box where R1 (50 tok/s per request) and R2
  (L1/L0 in [1,3]) have a common operating point, because it is large enough
  next to HBM to push the KV residency fraction f above 0.5 without shrinking
  the pool (R4) or lowering `gpu_memory_utilization` (R5).
- It is a **Mamba/GDN hybrid**: 60 layers, 15 full attention + 45 gated delta
  net, `full_attention_interval` 4. That is what drags in everything below.
- Hybrid prefix caching needs **vllm-main 0.28.1rc1**
  (`/home/bo/venvs/vllm-main`). The harness's vllm-lazy 0.23.0 refuses prefix
  caching for hybrids at `vllm/config/model.py:1852`.

Nothing has been benchmarked. **No request has ever been served on this
model.** The blockers below are why.

## 2. The two blockers

LMCache's vLLM integration has not caught up with vLLM 0.28. Two separate
breakages, in the same code path, both in the Mamba-hybrid registration.

### Blocker 1 -- kv_layout hint (FIXED, see section 3)

```
RuntimeError: Worker failed with error
  'Unsupported kv_layout: none. Only NHD and HND are supported.'
```

Chain: `kv_cache_group_edits.py:423` needs `layout_hints["kv_layout"]` ->
`utils.py:42` fills it from `try_get_vllm_kv_cache_layout()` -> that imports
`get_kv_cache_layout` from `vllm.v1.attention.backends.utils` -> **that
function was removed in 0.28** (it is at `:53` in vllm-lazy 0.23.0, absent from
vllm-main) -> a bare `except Exception` swallowed the ImportError and returned
`None` -> the hint was never set -> default `"none"` -> raise.

In 0.28 the engine core resolves one layout for the whole model
(`resolve_kv_cache_layout`, `v1/attention/backends/utils.py:238`), records it on
`CacheConfig.kv_cache_layout` and ships it to the workers **before** the KV
connector is initialized (`v1/worker/gpu_worker.py:682` precedes
`ensure_kv_transfer_initialized` at `:690`). The layout is now a stride
permutation name over the logical `[L,B,H,N,C]` axes, not `NHD`/`HND`. vLLM
keeps `NHD -> LBNHC` and `HND -> LBHNC` as aliases in
`vllm/config/cache.py:21` (`_LAYOUT_COMPAT_ALIASES`); the other four members of
`KVCacheLayout` have no legacy name.

On this model vLLM resolves **LBNHC**, i.e. NHD.

### Blocker 2 -- Mamba unified view reshape (ROOT CAUSE FOUND, NOT FIXED)

With blocker 1 fixed, startup gets one step further and dies at:

```
RuntimeError: shape '[1797, 2096, 1, -1]' is invalid for input of size 1917413376
```

at `lmcache/integration/vllm/kv_cache_group_edits.py:418`, inside
`_MambaUnifiedViewEdit.apply()`:

```python
kv_layout = layout_hints.get("kv_layout", "none")
if kv_layout == "NHD":
    return kv_cache.view(kv_cache.shape[0], spec.block_size, 1, -1)
elif kv_layout == "HND":
    return kv_cache.view(kv_cache.shape[0], 1, spec.block_size, -1)
```

The edit's stated premise (class docstring) is that the registered Mamba tensor
is `[num_blocks, 1, 1, context_size]` with `context_size == vllm_block_size *
head_size`. **That premise no longer holds in 0.28.**

Measured, by instrumenting `apply_kv_cache_group_edits` and launching (the
instrumentation has since been reverted; section 6 says how to re-add it):

| | Qwen3.5-397B, TP=4 | Qwen3.5-2B, TP=1 |
|---|---|---|
| forced attention block size | 2096 | 1072 |
| `MambaSpec.block_size` | 2096 | 1072 |
| `page_size_bytes` = `page_size_padded` | 1,073,152 | 1,097,728 |
| `state_content_size_bytes` | 1,067,008 | 1,085,440 |
| `num_heads` / `num_states` / `tokens_per_state` | 1 / 1 / -1 | 1 / 1 / -1 |
| registered tensor | `[1797, 1, 1, 1067008]` int8 | `[5542, 1, 1, 1085440]` int8 |
| `stride` | (not captured) | `(1097728, 1085440, 1085440, 1)` |
| `storage_offset` / `is_contiguous` | - | `0` / `False` |
| layers in the group | 15 | 6 |

The decisive facts:

1. The tensor's **last dim is `state_content_size_bytes`, the UNPADDED state**,
   not the padded page. vLLM 0.28 shapes every layer as
   `(B, H, N, C_bytes)` via `compute_layer_kv_cache_shape_bytes`
   (`v1/kv_cache_interface.py:251`), and for a `MambaSpec` that is
   `(num_blocks, 1, 1, state_content_size_bytes)`. The dtype is `int8`: it is a
   byte view.
2. `state_content_size_bytes` **is not divisible by `block_size`**:
   `1,067,008 = 2^11 x 521` against `2096 = 2^4 x 131`;
   `1,085,440` against `1072 = 2^4 x 67`. Hence the reshape error.
3. `page_size_bytes` **is** divisible: `1,073,152 / 2096 = 512` and
   `1,097,728 / 1072 = 1024`, both exact. vLLM pads the Mamba page precisely so
   that it equals the attention page (engine log:
   `Padding mamba page size by 0.58% to ensure that mamba page size and
   attention page size are exactly equal`).
4. **`stride(0) == page_size_bytes`** (1,097,728) with `storage_offset == 0`.
   The tensor is already strided across the full padded page; the pad bytes sit
   between blocks. So re-striding to cover the whole page stays in bounds
   (`untyped_storage().nbytes()` is the whole 36.5 GB pool).

This is not an fp8 corner case. With bf16 KV the forced block size would drop
to ~1048 and `1,067,008 / 1048` still does not divide. **On 0.28 this edit
fails for this model unconditionally.**

## 3. What is already fixed and verified

Uncommitted in the working tree. Nothing has been committed or pushed.

```
 M lmcache/integration/vllm/lmcache_mp_connector.py    (1 line)
 M lmcache/integration/vllm/utils.py                   (the fix)
 M lmcache/integration/vllm/vllm_service_factory.py    (1 line)
?? tests/v1/test_vllm_layout_hints.py                  (9 new tests)
```

`try_get_vllm_kv_cache_layout()` and `vllm_layout_hints()` now take an optional
`vllm_config`. Behaviour:

- Import `get_kv_cache_layout`. If it exists (vLLM < 0.28), call it as before,
  with its own `try/except` so vLLM's failures still degrade to `None`.
- On `ImportError` (vLLM >= 0.28), read `vllm_config.cache_config.kv_cache_layout`
  and map it through `_VLLM_LAYOUT_TO_LEGACY = {"LBNHC": "NHD", "LBHNC": "HND"}`.
  Any other permutation logs an error and returns `None` rather than guessing.
- The resolved value is memoized in a module global, because two of the four
  call sites hold no config. Only the 0.28 path memoizes; the old path keeps
  hitting vLLM's own `lru_cache`, which `set_kv_cache_layout` can invalidate.

Call sites, and which ones now pass a config:

| site | passes config? |
|---|---|
| `lmcache_mp_connector.py:506` `register_kv_caches` | yes, `self._vllm_config` |
| `vllm_service_factory.py:207` `CreateGPUConnector` | yes, `self.vllm_config` |
| `vllm_multi_process_adapter.py:1295` | no -- reads the memo |
| `sdk/qringbuffer.py:638` | no -- reads the memo |

Ordering is safe: the connector resolves at `:506` before calling
`worker_adapter.register_kv_caches` at `:515`, which is what reaches `:1295`.

**Note on the narrowed exception.** The original bare `except Exception` also
swallowed vLLM's own `AssertionError` from `get_kv_cache_layout()`. Narrowing it
turned 8 tests in `tests/v1/test_vllm_mp_adapter.py` red, which exposed a
pre-existing test bug: those tests monkeypatch
`lmcache.integration.vllm.utils.vllm_layout_hints`, but the adapter does
`from ...utils import vllm_layout_hints` at module level (`:20`), so the stub
never applied and the real function ran. The call-site guard was kept (with an
honest message), so they pass again. **The monkeypatch target in those tests is
still wrong and should be fixed separately** -- patch
`lmcache.integration.vllm.vllm_multi_process_adapter.vllm_layout_hints`.

Verified, four layers:

1. **Unit** -- `tests/v1/test_vllm_layout_hints.py`, 9 cases, both vLLM
   generations simulated by monkeypatching `sys.modules`, no vLLM import.
   Passing, together with the neighbouring suites: 62 passed.
   ```bash
   cd /home/bo/LMCache-worktrees/lazy_offloading && PYTHONPATH=$PWD /home/bo/venvs/vllm-lazy/bin/python -m pytest tests/v1/test_vllm_layout_hints.py tests/v1/test_vllm_mp_adapter.py tests/v1/test_vllm_kv_cache_groups.py tests/v1/test_kv_cache_groups.py -q -p no:logging
   ```
2. **Mapping against vLLM's truth table** -- asserted to be the exact inverse of
   `_LAYOUT_COMPAT_ALIASES`. Passing.
3. **End to end startup** -- engine logs `Using LBNHC KV cache layout` and
   `Unsupported kv_layout` is gone, on both the 397B and the 2B.
4. **L1 round trip** -- NOT DONE. Blocked by blocker 2. See section 5; this is
   the only layer that actually proves the mapping is semantically right.

## 4. The proposed fix for blocker 2 (not written)

Follow what `_MambaPageViewEdit` (the pre-0.26 list-form rule,
`kv_cache_group_edits.py:206-263`) already does: address the **whole padded
page** and give it a synthetic attention shape at `block_size` granularity. It
validates `storage_offset() == 0` and
`stride(0) * element_size() == spec.page_size_bytes`, then
`as_strided((num_blocks, elems_per_page), (elems_per_page, 1))` and reshapes.
The measurements in section 2 say those invariants hold for the 0.28 unified
tensor too.

So `_MambaUnifiedViewEdit.apply()` should derive the extent from
`spec.page_size_bytes` instead of from the tensor's own element count:

```python
num_blocks = kv_cache.shape[0]
elems_per_page = spec.page_size_bytes // kv_cache.element_size()
# validate storage_offset == 0 and stride(0)*elem_size == page_size_bytes,
# raising ValueError otherwise (do not silently re-stride)
flat = kv_cache.as_strided((num_blocks, elems_per_page), (elems_per_page, 1))
head_size = elems_per_page // spec.block_size      # 512 / 1024, both exact
# NHD -> flat.reshape(num_blocks, spec.block_size, 1, head_size)
# HND -> flat.reshape(num_blocks, 1, spec.block_size, head_size)
```

Why this is backward compatible: when the registered tensor already spans the
whole page (0.26 / 0.27, where the edit worked), `elems_per_page` equals the
tensor's own element count and the result is identical to today's `view`.

Do **not** reuse `_synthetic_attention_shape` unchanged -- it factors for the
5-D form and carries a `2 *` K/V axis this 4-D view does not have.

Two things to decide, and I did not get to either:

- Whether the divisibility should be asserted or whether a non-dividing page
  should fall through to a pass-through view. Note that the 0.28 tensor is
  *already* in vLLM's canonical `(B, H, N, C)` form with `num_heads == 1` and
  `num_states == 1`, so "leave it alone" is a defensible alternative fix. I
  lean against it: the whole point of the edit is to make the page addressable
  at the same block granularity as the attention group, and the design of
  `_MambaPageViewEdit` says that is deliberate. But this was not settled.
- Whether the same premise break affects `_SubpagedMLAAttentionViewEdit`
  (`:426`) and `_SubpagedAttentionViewEdit` (`:266`), which also assume the
  registered tensor spans the padded page. Not checked. Neither fires on this
  model.

## 5. How to test correctness (the part that matters)

`kv_layout` decides whether KV tensors are re-interpreted as `[..., N, H, C]` or
`[..., H, N, C]`. **A wrong mapping does not raise.** It silently transposes
heads against block tokens, and the model keeps emitting fluent, wrong text. So
"the engine starts" proves nothing. The decisive test is a real L1 round trip:

`<scratchpad>/probe/correctness.py` (already written, never run):
a ~7000-token prompt spanning 3 chunks, `temperature=0`, run once, then
`POST /reset_prefix_cache` to force an L0 miss, then run again. Run 2 must
report non-zero `cached_tokens` **and** reproduce run 1's text byte for byte.
Misaligned KV cannot survive that.

Run it against the 2B first (fast), then the 397B.

```bash
/home/bo/venvs/vllm-main/bin/python <scratchpad>/probe/correctness.py
```

It targets port 8963 (the 397B probe); change `BASE` to 8973 for the 2B.

## 6. Reproduction environment

Scratchpad: `/tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/ec1e8d69-21b5-467d-a788-88cf087ff44f/scratchpad/probe/`

- **`p3.sh`** -- the 397B probe. TP=4 on GPUs 4-7, ports MP 8961 / HTTP 8962 /
  vLLM 8963, L1 64 GB, `--chunk-size 2096`.
- **`q2.sh <chunk>`** -- the 2B probe. **Use this for iteration.** TP=1 on GPU 7,
  ports 8971/8972/8973. `q2.sh 0` starts it without the connector (to read the
  forced block size); `q2.sh 1072` starts it with the MP connector.
- `correctness.py` -- section 5.
- `deps/` -- a `--target` install of `sortedcontainers`, on `PYTHONPATH`.

**Iterate on Qwen3.5-2B, not the 397B.** Same architecture family
(`Qwen3_5ForConditionalGeneration`), same hybrid structure (24 layers, 18 GDN +
6 full, `full_attention_interval` 4), same attention geometry (2 KV heads,
head_dim 256), and it exercises the identical code path. 4.3 GB, starts in about
90 s. The 397B takes **9-11 minutes** because its 394 GB of weights no longer
fit in page cache and are re-read from disk at ~18 s per shard.

To re-add the instrumentation that produced section 2's table, insert a
`logger.info` at the top of the group loop in `apply_kv_cache_group_edits`
(`kv_cache_group_edits.py:537`) printing, for `group.layer_names[0]`'s tensor:
`type(spec).__name__`, `block_size`, `page_size_bytes`, `page_size_padded`,
`num_heads`, `num_states`, `tokens_per_state`, `state_content_size_bytes`,
`shape`, `dtype`, `stride()`, `storage_offset()`, `is_contiguous()`,
`untyped_storage().nbytes()`, `element_size()`.

### Traps, all of which cost time

1. **Triton needs `CPATH`.** No `python3.12-dev` on this box, so every Triton
   kernel fails to build with `Python.h: No such file or directory`. The
   workaround is in `env.sh:40` and is already baked into `p3.sh` / `q2.sh`.
2. **The vllm-main venv lacks `sortedcontainers`** (connector import) and
   `opentelemetry.exporter.prometheus` (MP server). Handled without touching
   shared state: the first via `deps/` on `PYTHONPATH`, and the MP server runs
   on the **vllm-lazy** interpreter, which has them. The two processes talk over
   ZMQ and both import LMCache from this worktree.
3. **The LMCache chunk size is read from the MP server**, not from the vLLM
   process (`vllm_multi_process_adapter.py:1145`,
   `get_lmcache_chunk_size(self.mq_client)`). `LMCACHE_CHUNK_SIZE` on the vLLM
   side is inert. Use the server's `--chunk-size`. It must be a multiple of the
   forced block size, and no power of two ever works
   (2096 = 2^4 x 131, 2112 = 2^6 x 33, 1072 = 2^4 x 67).
4. **Never `pkill -f` a pattern that can match your own shell.** It matches the
   shell's command line and kills the session (exit 144 / lost logs). Enumerate
   with `nvidia-smi --query-compute-apps=pid` or
   `ps -u 1016 -o pid=,args= | grep ...` and kill by PID.
5. **Check for a live launcher, not just a live server, before relaunching.** A
   `p3.sh` still inside its MP health-wait loop will pass its check when a later
   launch brings a server up on the same port and then `exec` a second
   `vllm serve` onto the same cards -> CUDA OOM with two workers both claiming
   rank 3.
6. **Only touch uid 1016 processes.** PIDs 1816225 / 2647600 / 2650232 hold
   small allocations on GPU 0 and belong to other users.

## 7. Standing constraints (these are Bo's, they remain in force)

- At most 4 GPUs; go easy on host memory.
- No shared-environment mutation. Writes stay in the scratchpad. Never rebuild
  `lmcache/*.so`, the venvs, `/raid`, or `/usr/local`.
- **No experiment is launched before its design has been discussed with Bo.**
- Push only to `BoJiang03/LMCache`, only when asked. **Never open a PR** -- hand
  over a title and body draft.
- Branches: `<line>_pr` carries only PR content, `<line>_dev` carries `records/`
  and experiments. `records/` lives on dev and nowhere else.
- Git author and sign-off is always `Bo Jiang <bo.jiang@temple.edu>` via the
  repo-local config. Never pass `-c user.email`. **No `Co-Authored-By: Claude`
  trailer.**
- Records are in English, force-added (`git add -f`). Method and config docs go
  in `records/`, never in `docs/design/`.
- Record and PR text in a terse plain register. No em dashes, no bold, no
  rhetoric.
- Keep chat replies short; detail goes in the records file.
- MP mode is primary.

## 8. Open items beyond the two blockers

1. **Blocker 2 fix**, section 4. Then the layer-4 correctness test, section 5.
2. **Whether a 2096-token chunk is workable for L1 at all.** At 32,434 bytes per
   token that is a 68 MB storage granule, two orders of magnitude coarser than
   the 256 every archived arm ran. Store latency, eviction granularity and the
   lazy policy's block accounting were all designed around the small chunk. The
   chunk-size assert passing does not mean L1 behaves.
3. **A CONC sweep** to replace every computed number in
   `records/deployment_candidate.md` Part 4. Design not discussed with Bo yet.
   Both constants of that cost model (`b = 2.4 GB/ms/GPU`, `c(B)` at E/k = 16)
   were calibrated on Qwen3-Coder-30B, not on this model.
4. **`assert` used for runtime validation** at
   `vllm_multi_process_adapter.py:616` and `:1153`, which
   `docs/coding_standards.md` forbids. Under `python -O` it vanishes and the
   failure mode becomes misaligned per-group block-id slicing instead of a dead
   engine. Separate PR.
5. **The wrong monkeypatch target** in `tests/v1/test_vllm_mp_adapter.py`,
   section 3. Separate fix.
6. **The `max_deferral_seconds` default-to-zero PR**, carried from earlier
   sessions, on a `_pr` branch. Not started.
7. `records/deployment_candidate.md` Part 6 lists three disqualifiers; the vLLM
   0.28 integration breakage is a fourth and more immediate one, and is not
   written down there yet.
