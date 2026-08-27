# Session index, harness changes, and what was retracted

Ties together records 3-5 from 2026-08-25 and records the things that live in
none of them: the harness safety fix, the corrections made mid-session, and the
open next steps. No source changed this session -- HEAD is still `924e2c1c` and
`git status` is clean.

## The three records

| record | subject | verdict |
|---|---|---|
| [3](3_mp_store_cudamemcpy_root_cause.md) | `cudaErrorInvalidValue` killing the MP server | fixed: stale `cuda_ops.so` collided with rebuilt `lmcache_native.so` on the pybind type `PageBufferShapeDesc`; `except ImportError` mislabelled it "not found" and fell back to a torch path that does not split at `host_buffer_alignments` |
| [4](4_lazy_smoke_run_after_native_fix.md) | first clean end-to-end smoke run | pass: AgentX replay 62 requests / 187.9 s, both processes alive; concurrent probe 40/40; zero transfer errors |
| [5](5_eager_lazy_ab_trace_driven.md) | eager vs lazy A/B at the workload's own timing | **negative for the branch**: TTFT avg +12.1%, p99 +19.2%, tokens written +138.7%, yield per token written 0.860 -> 0.506; root cause identified |

## The finding, in one paragraph

An APC hit is one-directional in the store-range decision: it raises
`computed_tokens` (how much we are willing to store) via
`max(num_vllm_hit_tokens, num_lmcache_hit_tokens)`, and contributes nothing to
`num_stored_tokens` (how much is already safe), whose only cross-request source
is an LMCache lookup hit. Eager masks this because the predecessor turn really
did reach LMCache, so the follower's lookup hits and the watermark jumps past
the shared prefix. Lazy still has the predecessor pending, the lookup misses,
and the follower stages `[0, full prefix)`. Re-staging is deliberate -- it is
how a prefix survives its predecessor's ops being dropped -- but the dedup that
should collapse the redundant case is an exact hash on `(cache_salt,
prefix_end_tokens, tuple(block_hashes))`, so a longer follower range never
matches a shorter pending one. Both are emitted and the shared range is
transferred twice. Full derivation with line numbers in record 5.

## Harness change: `up.sh` guard scoped to our own port

Not recorded elsewhere. `up.sh` refused to start on
`pgrep -f 'lmcache.v1.multiprocess.http_server'`, a box-wide match. It fired on
MP servers belonging to an unrelated worktree of mine (the multi_modal bisect,
`/home/bo/venvs/vllm-bisect-0.25.1`, ports 29435/29988/27555) while my own
three ports were free. A box-wide guard is both a false positive and an
invitation to a box-wide kill, which is how `down.sh` became a hazard (record
4). Replaced with a uid- and port-scoped check: walk
`pgrep -u "$(id -u)"` matches and keep only those whose
`/proc/<pid>/cmdline` contains `--port $MP_PORT`.

One bug in the first version of that patch, worth noting because it would have
silently disabled the guard: the replacement was written through a Python
string and landed as `tr '\\0' ' '`, which translates backslashes and zeros
rather than NULs, so the cmdline stayed NUL-separated and never matched.

## What was retracted this session

Kept here so nobody re-derives a discarded result from an older message.

1. **"The compiled extension was not built on this host."** It existed; it was
   stale. Changed the fix from "build" to "rebuild" (record 3).
2. **The first A/B, and its headline "lazy phase B 29.5 s vs eager 96.7 s".**
   Discarded. It used `--unsafe-override` to suppress the scenario's
   `min_benchmark_duration_seconds=900` ("requires duration >= 900s to reach
   steady state and trigger KV offloading") and ran 180 s; its timing was
   invented; its reuse shape was whole duplicate prompts rather than
   within-session prefix growth; and it landed on the LRU sequential-scan worst
   case, so its eager 0% hit rate is an artifact.
3. **"`--use-think-time-only` caused the 10,651.9 s spread."** It did not. The
   4.5 s spread came from `--trace-idle-gap-cap-seconds`. And the spread is not
   an arrival rate at all -- it is the alignment of a phase's first requests on
   the trace timeline `t*`.
4. **"Lazy re-writes the whole accumulated prefix every turn."** Read off the
   head of the batch-size distribution, where large writes are correct
   cold-cache first-turn stores; the tail is incremental. Emission coalescing
   contiguous pending ops is by design.
5. **"The pending queue being invisible to lookup is the defect."** Misplaces
   it. A deferred offload means the blocks are still GPU-resident, so a
   follower hits APC and `needs_retrieve()` is false -- no load is needed and
   lookup should not matter. The defect is the write-side asymmetry above.

## Next steps

Recommended order -- evidence, then a failing test, then the fix:

1. **Instrument emission** to log `[prefix_start_tokens, prefix_end_tokens)`
   per store and check it against what is already resident. Turns the root
   cause from code reading into measurement, and shows the shared range going
   out twice.
2. **Regression test for multi-turn growing prefixes.** The present suite
   cannot see this: `min_prefix_tokens=0` plus single-turn synthetic prompts is
   the configuration that hides it.
3. **Fix**, either by advancing the already-stored watermark from the pending
   store so a follower stages only the delta, or by making the policy's dedup
   range-aware (interval containment, trim the follower to the uncovered part).
   Both must preserve the drop-recovery property that re-staging currently
   provides.

Also still open, unchanged:

- Formal testing at the workload's own default of 1800 s, 2-3 repetitions per
  arm. This session's A/B is a smoke run at the 900 s floor, single run each.
- `min_prefix_tokens` sweep above zero. It cannot fix the write amplification,
  but gate 3 is the newest commit on the branch (`924e2c1c`) and is currently
  untested end to end: `held=0` and `rejected_short_prefix=0` throughout.
- Two dev-side issues to file separately, not folded into this PR: the torch
  fallback not splitting at `host_buffer_alignments`, and `except ImportError`
  in `device_ops.ensure_native()` masking a stale build as "not found".
