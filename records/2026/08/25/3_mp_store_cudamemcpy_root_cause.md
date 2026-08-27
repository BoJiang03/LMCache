# Root cause: MP store failures are an unsplit cudaMemcpy across pin chunks

Date: 2026-08-25 (11:20-12:25 PT)
Branch under test: `lazy-offload-publish` @ 924e2c1c
Baseline: `origin/dev` @ 23cca679 in worktree `/home/bo/LMCache-worktrees/dev_baseline`
Context: follow-up to `2_agentx_smoke_run.md` §5, which found the failures but not why.

Two questions drove this: what causes it, and is it ours. Answers: an unsplit
`cudaMemcpy` across `cudaHostRegister` boundaries in the Python device-ops fallback,
and no -- it reproduces identically on dev with none of the PR's code.

## Not ours

Three independent checks.

1. `git diff HEAD origin/dev -- lmcache/v1/platform/torch_ops.py` is **empty**. The
   file is byte-identical in today's dev, same unsplit `cudaMemcpy` at the same line
   numbers (2203-2210). The PR touches only `lmcache/integration/vllm/`.
2. The failure is identical at TP=1 and TP=2, with `world_size: 1` in `/status`, so
   it is not a multi-rank or IPC-context problem.
3. Ran dev itself. Same model, same eager config, same probe:

   | tree | successful store batches | store exceptions | `cudaMemcpy` err | `AcceleratorError` |
   |---|---|---|---|---|
   | `lazy-offload-publish` @ 924e2c1c | 27 | 33 | 17 | 16 |
   | `origin/dev` @ 23cca679 | 27 | 33 | 17 | 16 |

   Identical. `dev_baseline/` has no `lazy_offload_manager.py` at all, and its
   traceback is from `dev_baseline/lmcache/v1/platform/torch_ops.py:2210`. Log kept
   at `artifacts/dev_baseline_eager_server.log`.

## The failing call

`index_select` is a symptom. In every log the **first** failure is the raw
`cudaMemcpy`; the `AcceleratorError` at `torch_ops.py:1797` follows ~0.5 s later on
a context that already carries a sticky error -- its own message says so ("CUDA
kernel errors might be asynchronously reported at some other API call"). Chasing
`_transfer_per_layer_nhd` was chasing the second error.

The primary site is `torch_ops.lmcache_memcpy_async`, pointer mode:

```python
ret = libcudart.cudaMemcpy(
    ctypes.c_void_p(dest), ctypes.c_void_p(src),
    ctypes.c_size_t(nbytes), ctypes.c_int(4),  # cudaMemcpyDefault
)
if ret != 0:
    raise RuntimeError(f"cudaMemcpy failed with error code {ret}")
```

Error code 1 is `cudaErrorInvalidValue`. One copy is issued for the whole object,
over a host range that can span two separately `cudaHostRegister`ed pin chunks.

Sizes that make it happen here:

| quantity | value | where from |
|---|---|---|
| `LazyMemoryAllocator.PIN_CHUNK_SIZE` | `1 << 26` = 64 MiB | `lazy_memory_allocator.py:69` |
| L1 object (one LMCache chunk) | 256 tokens x 96 KiB = 24 MiB | `chunk_size` x model KV width |
| observed `nbytes` | 25,165,824 | instrumented |
| observed `align` | 67,108,864 | instrumented |

The "Expanded 10240 MB pinned memory" log line is the *reservation* step, not the
registration granularity -- that misread cost an hour. Registration is per 64 MiB.

## Evidence

Instrumented the call to record, per invocation, whether the host range crosses a
`host_buffer_alignments` boundary and what `cudaMemcpy` returned. One probe run
(10 x 48k-token prompts, TP=1, eager), 892 pointer-mode copies:

| | crosses a 64 MiB pin boundary | does not |
|---|---|---|
| returned 1 (fail) | 17 | **0** |
| returned 0 (ok) | 213 | 662 |

Crossing is a **necessary** condition -- not one failure without it -- and not a
sufficient one: 17 of 230 crossings fail.

Rates, in the right units:

- per copy: 17/892 = 1.9%
- crossings: 230/892 = 25.8% observed (24/64 = 37.5% if offsets were uniform;
  the allocator's placement is not uniform)
- per store batch: 33 of 60 batches failed = 55%. A complete batch carries ~31
  objects (8,002 tokens / 256), and 1 - 0.981^31 = 45%, so the per-copy rate
  explains the batch rate.

Correction to something stated mid-investigation: I matched an observed 37.2% against
the 37.5% straddle probability and called it a decimal-place confirmation. That
compared a per-*batch* number against a per-*object* prediction and the agreement was
coincidence. The evidence for the mechanism is the necessary-condition table above,
which does not depend on that arithmetic; the conclusion is unchanged.

## Why the fallback runs

Logged at every startup on this host:

```
lmcache.cuda_ops compiled extension not found; CudaDeviceOps stays on the torch
baseline for all ops.
```

The message is misleading, and I initially took it at face value. The extension is
**present** -- `lmcache/cuda_ops.cpython-312-x86_64-linux-gnu.so` exists in the tree.
It fails to *load*:

```
ImportError: generic_type: type "PageBufferShapeDesc" is already registered!
```

`device_ops.ensure_native()` catches bare `ImportError` and reports it as "not found",
so a stale build is indistinguishable from a missing one in the log.

The stale `cuda_ops.so` is dated 2026-08-14; `lmcache_native.so` is dated 2026-08-20, and
ab09ffeb "Move common transfer descriptors into lmcache_native (#4515)" landed
2026-08-20. Both `.so` files therefore register `PageBufferShapeDesc`, and pybind11
refuses the duplicate. So `device_ops.lmcache_memcpy_async` dispatches to `torch_ops`. The C++ path splits
the copy at `host_buffer_alignments`; the Python fallback deliberately does not, on a
premise its own docstring states:

> Unlike the C++ version (which uses cudaMemcpyAsync and must split copies at
> cudaHostRegister boundaries), this Python fallback does NOT need alignment-based
> chunking because cudaMemcpy (synchronous) handles cross-cudaHostRegister
> boundaries internally via staging buffers

Both call sites (`gpu_ops.py:45` H2D, `gpu_ops.py:86` D2H) already pass the two
arguments a split needs -- `memory_obj.meta.address` and
`LazyMemoryAllocator.PIN_CHUNK_SIZE` -- and the fallback uses them only for a
power-of-two check.

The raw-pointer path is taken only `if isinstance(memory_obj.parent(),
LazyMemoryAllocator)`; the other branch is `tensor.copy_()` and is safe. That is the
likely reason the earlier `repro/pr4499` runs at 8-40 GB L1 never showed this.

## Hypotheses tested and rejected

| hypothesis | test | result |
|---|---|---|
| multi-rank IPC / wrong current device (predicted ~1/world_size failures) | rerun at TP=1 | rejected -- identical failures at `world_size: 1` |
| copy runs into memory reserved but not yet `cudaHostRegister`ed | correlate failing addresses against the expansion frontier | rejected -- L1 fully expanded to its 200 GB target before any traffic; every address far below the frontier |
| a single larger power-of-two boundary explains it | swept boundaries 1 MiB..16 GiB for one that separates fail from ok | none separates; only <=64 MiB is necessary-but-not-sufficient |
| offset within the pin chunk selects the failures | compare `off_in_chunk` of failing vs succeeding crossings | rejected -- both are exactly {48, 56} MiB |

## Open

1. What selects the 17 of 230 crossings that fail. Next step is
   `cudaPointerGetAttributes` on both ends of the range at failure time (the
   instrumented patch for this is written but was not run:
   `artifacts/torch_ops_straddle_diagnostic.py.txt`).
2. Whether building `lmcache.cuda_ops` on this host sidesteps it entirely -- worth
   knowing before proposing a fallback fix, since it decides whether this is a
   "fallback is wrong" bug or a "fallback is untested" bug.
3. The fix itself: split at `host_buffer_alignments` in the fallback, mirroring the
   native path. Separate PR against dev -- explicitly **not** to be folded into the
   lazy offload PR.
4. `2_agentx_smoke_run.md` open items 2-5 still stand; item 1 (file the bug) now has
   its evidence.

## Reproducer

`artifacts/probe.py` -- N distinct long prompts straight at `/v1/completions`, no
aiperf, no trajectory replay, no warmup. 30 s per data point instead of a 6-minute
aiperf warmup. Used for every measurement above.

```bash
bash up.sh eager && python3 probe.py 28190 10 8000
```

At 8,000 filler words the prompt tokenizes to 48,003 tokens (5.46 tokens/word for
this tokenizer); 24,000 words overshoots the 131,072 context.

## Harness changes made while doing this

- `up.sh` now `cd`s to `$REPO` and refuses to start unless `lmcache.__file__`
  resolves under it. The cwd shadows the editable-install finder, because
  `$REPO/lmcache` is a real package directory -- so cwd, not the finder's MAPPING,
  decides which tree is served. Without the assertion a "dev baseline" run silently
  imports the branch, which would have invalidated the comparison above.
- `dev_baseline` worktree is left in place (detached at 23cca679) as a reusable
  baseline. Remove with `cd /home/bo/LMCache && git worktree remove
  /home/bo/LMCache-worktrees/dev_baseline` when done.
- Verified after teardown: both worktrees report 0 modified files, and the editable
  finder is byte-identical to its backup.


## Fix verified on this host (2026-08-25)

Rebuilding `cuda_ops` against current HEAD removes the fallback entirely, so the
defective copy path is never reached.

Built out-of-tree so no shared artifact was overwritten:

```
PATH=/usr/local/cuda/bin:$PATH CUDA_HOME=/usr/local/cuda \
CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:/raid/data/hub/pr4499_agentic/pydev/usr/include \
TORCH_CUDA_ARCH_LIST=9.0 MAX_JOBS=16 \
python setup.py build_ext --build-lib $SCRATCH/blib --build-temp $SCRATCH/btmp
```

nvcc 13.0 at `/usr/local/cuda` matches torch 2.11.0+cu130. Only
`lmcache/cuda_ops.*.so` was then replaced (it is gitignored, `.gitignore:66`, so the
tree stays clean); the 2026-08-20 `lmcache_native` / `lmcache_fs` / `lmcache_redis`
artifacts were left alone. Originals backed up to `scratchpad/so_bak/`.

After the swap:

| check | before | after |
|---|---|---|
| `compiled extension not found` in MP server log | 1 | 0 |
| `compiled extension not found` in vLLM log | 1 | 0 |
| `lmcache_memcpy_async` bound natively | no | yes |

Two probe runs, lazy config, pool 24 GiB, L1 200 GB, TP=1:

| run | requests | store batches | L1 written | cudaMemcpy failed | AcceleratorError | Traceback |
|---|---|---|---|---|---|---|
| probe 10 | 10/10 | 5 | 23.5 GB | 0 | 0 | 0 |
| probe 40 | 40/40 | 31 | 147.4 GB (6289 objects) | 0 | 0 | 0 |

36 store batches with zero failures. At the previously measured per-batch failure rate
of 55%, P(0 failures in 36) = 0.45^36 ~ 3e-13, so this is not a lucky run.

This closes the open item "does building `lmcache.cuda_ops` sidestep the bug": it does.
It also settles the classification -- the fallback is **untested**, not merely wrong:
on any host where the extension builds, that code never executes.

The dev-side issues remain worth filing separately, still not folded into the lazy
offload PR:

1. `torch_ops` fallback does not split at `host_buffer_alignments` (the actual defect).
2. `except ImportError` in `device_ops.ensure_native()` reports a stale/ABI-mismatched
   build as "not found", silently degrading to the defective path. A duplicate-
   registration `ImportError` should surface, not be swallowed.

## Harness hazard found while verifying

`down.sh` lines 7-8 pattern-kill every `lmcache.v1.multiprocess.http_server` and every
`vllm serve` / `VLLM::EngineCore` on the box. This host is shared: at the time of this
run, pids 2518048 and 3483837 were other users' vLLM servers in containers, and GPU 4
held 91 GB of someone else's work. Do not run that script here.

Replaced by `scratchpad/smoke2/down_safe.sh`, which kills only (a) the pid trees
recorded in `$LOGDIR/{server,vllm}.pid` and (b) compute apps on `$GPUS` owned by our
own uid, skipping anything with a different owner. Verified: it killed our 4 pids,
returned GPU 2 to 4 MiB, and left both other users' servers running.
