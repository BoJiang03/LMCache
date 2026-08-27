# Escalation 1 — MP connector aborts vLLM's block pool on a tight pool (mamba-align hybrid)

**Severity:** engine abort (not a wrong answer). **Component:** LMCache MP
connector external-hit accounting. **Status:** reproducible, control
experiment isolates LMCache as the single variable.

This is written to be filed as-is. Everything below is measured, and the
one claim that would normally be assumed — "it's an upstream vLLM bug" —
is explicitly disproved by the control.

---

## Summary

With `LMCacheMPConnector` attached, a Mamba/GDN "align mode" hybrid served
on a small `num_gpu_blocks_override` pool aborts inside vLLM's block-pool
bookkeeping:

```
vllm/v1/core/block_pool.py:273 in cache_full_blocks
    assert blk.block_hash is None
AssertionError
```

**Plain vLLM at the same pool sizes completes cleanly.** The connector is
the only variable changed between the crashing and the clean runs.

## Reproduction

- Model: `Qwen/Qwen3.5-2B` (`mamba_cache_mode="align"`, unified block 544)
- vLLM 0.23.0, `enable_prefix_caching=True`, TP=1, single GPU
- 6 concurrent `ignore_eos` requests of 3518 prompt tokens
- `max_model_len=4352`, `max_num_batched_tokens=550`

| GPU blocks | step budget | connector | result |
|---|---|---|---|
| 128 | 544 | MP | clean, 0 preemptions |
| 48 | 550 | MP | clean, 0 preemptions |
| 32 | 550 | MP | clean, 0 preemptions |
| **28** | 550 | MP | **AssertionError** |
| **24** | 550 | MP | **AssertionError** (`scheduler.py:462`, RUNNING branch) |
| **20** | 550 | MP | **AssertionError** |
| **16** | 544 | MP | **AssertionError** (`scheduler.py:761`, WAITING branch) |
| 24 / 20 / 16 | 550 / 544 | **none** | **clean**, 0 preemptions |
| 32 | 550 | none | clean, 1 preemption |

Call path: `allocate_slots -> coordinator.cache_blocks ->
MambaManager.cache_blocks -> block_pool.cache_full_blocks`.

Reproducer scripts are kept alongside this file (`../plain_preempt_probe.py`
for the no-connector control; the suite's `preemption` scenario with
`preemption_gpu_blocks=24` and `max_num_batched_tokens=550` for the
crashing side).

## Why this points at LMCache, not vLLM

The assertion fires in the branch that caches blocks a request obtained via
`num_external_computed_tokens` — i.e. blocks the *connector* reported as an
external hit. The assertion means a block that already carries a hash is
being cached a second time, so the block-pool bookkeeping is inconsistent
after an external hit landed in a recycled block. Pool pressure is what
makes recycling frequent enough to hit it, which is why only tight pools
abort.

The control run is the argument: same model, same pools, same step budget,
same prompts, connector removed, no crash. If this were vLLM's own
accounting the control would crash too.

## Why it matters beyond the crash

It walls off the `preemption` isolated scenario for the entire
RECURRENT_STATE family, and **not** for want of a measurement — the two
regions do not overlap:

- **Above** the crash region the scenario is vacuous: 128, 48 and 32 blocks
  all yield 0 preemptions.
- **Below** it the engine aborts: 28, 24, 20, 16.

The vacuity has a separate, also-measured cause worth reporting in the same
breath, because it affects what the existing certificates actually prove:
align mode's minimum step budget is one unified block, and vLLM schedules
RUNNING requests before WAITING ones, then truncates a waiting request's
prefill chunk to a block multiple —

```python
# vllm/v1/core/sched/scheduler.py::_mamba_block_aligned_split
num_new_tokens = num_new_tokens // block_size * block_size
```

so one request in decode takes 1 token, leaving 543, and `543 // 544 * 544
== 0` skips the waiting request entirely. Single-variable evidence: budget
544 → 0 preemptions, budget 550 → 1 preemption, same 32-block pool, same
plain vLLM.

**Consequence:** the three certified Qwen hybrids have never had two
requests occupying the GPU simultaneously. Their certificates now say
`concurrent batch submission (vLLM executes it serially -- see
known_not_covered)` rather than "concurrent batches", and the exclusion
list carries the mechanism and the numbers. That correction is already
landed; this escalation is about the crash.

## What is being asked

1. Confirm whether the external-hit path is expected to cache blocks that
   may already carry a hash under pool recycling, and if not, where the
   bookkeeping diverges.
2. Decide whether this is a connector-side fix or needs an upstream vLLM
   guard — the control says connector-side, but the maintainers own that
   call.

Nothing in the test suite is blocked on the answer; the scenario is
correctly excluded with a stated reason either way.
