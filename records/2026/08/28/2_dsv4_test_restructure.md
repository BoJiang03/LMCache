# dsv4 coverage: move the correctness proof to 1 GPU

Date: 2026-08-28 (session 2, continues `1_dsv4_flash_tp_ci_cost.md`)
Worktree: `/home/bo/LMCache-worktrees/k3_mp_dsv4`
Branches: `dsv4_ci_cost_pr` @ `852cbc6c`, `dsv4_ci_cost_dev` (this record)
Base: `852cbc6c` (session 1's commit) on `origin/dev` @ `0c2a6801`

> **Outcome (2026-08-28, after the fact): the L1 test was dropped.** The user
> ruled that correctness stays with the 4-GPU vLLM end-to-end test and L1 is
> not wanted. `dsv4_ci_cost_pr` was moved back to `852cbc6c` (the cost work
> only) and `tests/v1/multiprocess/test_cache_server.py` plus
> `docs/design/integration/vllm/hybrid-kv-cache-groups.md` are byte-identical
> to `origin/dev` again. The dropped commit is `a41726b8`, still reachable via
> reflog if it is ever wanted. Everything below is kept as the record of the
> investigation, not as a description of the current branch.

## Question asked

"Is there a way to verify correctness without launching 4 GPUs?" Then, after
the three-level answer: is the 1-GPU real-vLLM level (L2) better than the
1-GPU pure-transfer level (L1)? Then: drop the nightly entirely? Then: no,
put the nightly and label triggers back.

## Where the DeepGEMM JIT cache path came from (asked separately)

Three steps, recorded because the reasoning is reusable:

1. `grep -rn DG_JIT_CACHE_DIR` over the installed vLLM: one hit,
   `vllm/utils/deep_gemm.py:250`, which sets it to `VLLM_CACHE_ROOT/deep_gemm`
   *only when unset*. `VLLM_CACHE_ROOT` defaults to `~/.cache/vllm`
   (`vllm/envs.py:34`). "Only when unset" is what makes an explicit
   `DG_JIT_CACHE_DIR` win, which is what the pipeline now relies on.
2. DeepGEMM's own fallback is `$HOME/.deep_gemm`; the JIT logic is C++
   (`csrc/jit/compiler.hpp`, `cache.hpp`) read at the pinned commit
   `8b1392b9`. Confirmed against the shipped binary:
   `strings vllm/third_party/deep_gemm/_C.cpython-310-*.so` contains
   `DG_JIT_CACHE_DIR`, `.deep_gemm` and the "Corrupted JIT cache directory"
   template.
3. The cache key (`compiler.hpp:101`) is what justified sharing the mount, not
   what located it.

## Three-level split

| Level | Where | GPUs | Covers |
|---|---|---|---|
| L0 grouping metadata | `tests/v1/test_kv_layer_groups_manager.py::TestKernelAndObjectGroups::test_dsv4_flash_style_mixed_compression` (already existed) | 0 | infos -> kernel groups, ratios, slot arithmetic |
| L1 byte round trip | `tests/v1/multiprocess/test_cache_server.py` (added) | 1 | real server + real paged GPU tensors at the model's geometry; STORE/RETRIEVE restore pages byte for byte |
| L2 served end to end | `.buildkite/.../run-dsv4-flash-tp.sh` (unchanged) | 4 | real weights at TP=4, greedy text equality, live `KVCacheConfig` |

L1 landed in `tests/v1/multiprocess/test_cache_server.py` because that file
already had everything: a module-scoped real MP cache server, real GPU tensors
behind `CudaIPCWrapper`, STORE/RETRIEVE over the message queue, and
`pytestmark = pytest.mark.cuda`. The unit pipeline runs the whole tree on 1 GPU
with no `-m` filter (`.buildkite/k3_tests/unit/run.sh:34`), so it runs on every
PR for free. The gap was narrow and specific: every existing test registers
with `engine_group_infos=[]`, so the server detects `slots_per_block` from the
tensors and treats every group as uncompressed (the file says so at line 357).

## Why L2 cannot replace L1

Asked directly, and the answer is a real constraint rather than a preference.
To make L2 cheap you have to shrink the model (`--load-format dummy`, or
`--hf-overrides num_hidden_layers`), and both destroy logit separation. The
4-GPU script's text comparison is only non-flaky because the full model pins
all 128 tokens to a memorized continuation (measured min top-2 gap 6.75, per
the script's own comment). With random or truncated weights that gap goes to
~0, and the prompt tail is always partly recomputed with a different prefill
kernel shape, so the last-bit logit jitter flips the greedy argmax. Comparing
logprobs has the same problem: the jitter is legitimate, not a bug.

Generalized: the flakiness comes from inferring "was the restore correct" from
the output of a chaotic function. The fix is to compare the restored thing
itself, which is only possible at L1.

So the division of labour is L1 = numerical truth, L2 = interface drift and
smoke (spec equals fixture, registration does not throw, a retrieve happens),
no numerical assertion at L2. L2 is not built yet.

## Trigger decision, and the reversal

Session 1 left the 4-GPU group on `schedule || label "dsv4" || RUN_DSV4_TEST`.
Mid-session the instruction was to drop the nightly and make it manual-only;
that was implemented (env-gate only, nightly section deleted from
`BK_WEB_SETUP.md`) and then reverted on the next instruction back to the
session-1 condition, restored with `git checkout HEAD --` so the `if:` is
byte-identical to `852cbc6c`.

Argument that was made for env-only and is worth keeping on file: a `dsv4`
label persists across pushes, and the pipeline has "rebuild on PR label
change", so a label left on a long-lived PR re-runs a 4-GPU job on every push,
while `RUN_DSV4_TEST` is per-build and cannot be forgotten. This did not win;
the nightly's unattended coverage was judged worth more.

Net effect on cost is unchanged from session 1: off on ordinary
`mp`/`full`-labelled PR builds and dev pushes.

## L1 test design notes

- `DSV4_GROUPS`: four tensors shaped `(2, pages, slots_per_block, 1, head)`;
  `tokens_per_block` 256/256/64/4 over 64/2/64/4 slots, so ratios 4/128/1/1.
  Head sizes are scaled down; only the block/slot geometry has to match.
- The two compressed groups share `engine_group_id=0` with different page
  shapes and dtypes, which is also the "engine group split by physical
  transfer identity" case from the registration design.
- `_fill_position_encoded` writes `(flat_index * 7 + layer * 13) % 251`. Every
  element differs by position, so a block/slot offset bug cannot round-trip;
  values under 251 are exact in `uint8`, `bfloat16` and `float32`, so the
  comparison is `torch.equal`, not `allclose`. (The existing uncompressed
  round-trip test uses `allclose(atol=1e-4)`; for a byte-copy path that is
  weaker than it needs to be.)
- Retrieve region base is `NUM_KEYS * max(blocks_per_chunk)` = 192: one number
  that is above every group's own store extent. Highest page touched is 384,
  under the 512 allocated.
- The geometry is load-bearing. A server that ignored `tokens_per_block` would
  want 4 blocks per chunk from the ratio-4 group and 128 from the ratio-128
  group, be given one, and fail the store closed, so the test goes red instead
  of passing quietly.
- Second test truncates only the 4-token group's block-id list. The
  mixed-block-size caller bug is sizing every list by one group's
  `tokens_per_block`, which under-covers exactly one group.

## Local run and mutation testing

Ran after the fact, on the same day, once the user approved rebuilding this
checkout's extensions. `PATH=/usr/local/cuda/bin:$PATH uv pip install --python
.venv/bin/python -e . --no-build-isolation`, 1m29s, exit 0. It produced
`cuda_ops` and `lmcache_native` (`device_ops` is a proxy object in
`lmcache/v1/platform/base/device_ops.py`, not a compiled module -- the
ImportError was about the proxy's backing extension, not a missing `.so` of
that name). Two side effects worth knowing:

- the venv's editable `lmcache` had been pointing at
  `.claude/worktrees/venv-selection-01459d`, not at this worktree; it now
  points here. That is the same class of hijack as the recorded
  `vllm-lazy venv editable hijack` and would have made any earlier test run
  meaningless even if the extensions had matched.
- uv downgraded opentelemetry (1.44 -> 1.40), prometheus-client and setuptools
  to the versions the project's constraints pin.

Results on GPU 1 (H200, SM90):

```
tests/v1/multiprocess/test_cache_server.py    11 passed   9.11s
tests/v1/test_kv_layer_groups_manager.py      51 passed   8.59s
dsv4 tests only, 10 consecutive runs          10/10 pass  6.84-7.43s
```

No flake in 10 runs, so the inherited store -> lookup race did not fire here.

Three mutations, to check the test is load-bearing rather than decorative:

| mutation | what it models | result |
|---|---|---|
| registration reports `tokens_per_block=0` for every group | engine does not declare its spec; server treats all groups as uncompressed | `STORE` returns False on chunk 0, exactly as the docstring claims |
| `calculate_num_blocks` returns group 0's count for every group | server ignores per-group geometry, but block-id lists are still long enough to pass the fail-closed guard | both dsv4 tests fail; the fail-closed test notices the store no longer fails closed |
| retrieve-side block ids flipped within each chunk (`is_h2d` only, sizes unchanged) | a wrong-order restore that both `STORE` and `RETRIEVE` report as success | only the byte comparison catches it: `Restored pages differ for group 2 (tokens_per_block=64, slots_per_block=64, compress_ratio=1)` |

The third is the one that matters. It confirms the assertion is the restored
bytes and not the `result is True` checks above it, and that the failure names
the offending group -- the two properties the 4-GPU text comparison does not
have. Group 2 is the first group with more than one block per chunk (256/64 =
4), which is why it is the one that reports.

All three were reverted (`git checkout` on `lmcache/v1/kv_layer_groups.py` and
`lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py`); the tree is
clean at `a41726b8` and `ruff check` / `ruff format --check` still pass.

## Not verified

- `assert lookup_all(...) == DSV4_NUM_KEYS` inherits the harness's known
  store -> lookup race (documented in
  `test_store_fails_closed_on_incomplete_block_ids`'s docstring: it can only
  turn a true hit into a miss). Same pattern as the existing
  `test_store_and_lookup`, so not a new risk, but it is the first place to
  look if this test ever flakes.
- Whether DeepSeek-V4-Flash actually reports a sliding window on any group. If
  it does, `separate_object_groups` is not a no-op for it and both fixtures
  cover the merged layout only. `_detect_object_groups` buckets by
  `(extra_tag, recurrent, sw_chunks)` and both fixtures leave
  `sw_size_tokens = -1`. Written into the design doc as a caveat rather than a
  claim.
- L2's feasibility (unchanged). Two unknowns: whether `--load-format dummy`
  works alongside `fp8_ds_mla` and `--tokenizer-mode deepseek_v4`, and how far
  `num_hidden_layers` can be cut while still producing all four group
  geometries (if the model's early layers are dense, cutting too far makes the
  8/4 groups disappear and the fixture stops being exercised). Neither is
  derivable from the config; both need one real run.

## Review pass findings (self-review, fixed before commit)

- The referenced fixture class was wrong: `test_dsv4_flash_style_mixed_compression`
  lives in `TestKernelAndObjectGroups`, not `TestKVLayerGroupsManager`. Wrong
  in both the test file and the design doc.
- `expected` / `block_ids` locals were untyped.
- A `#:` page-count comment was attached to `DSV4_NUM_KEYS`.
- The load-bearing-geometry property was not stated anywhere.

## Files touched (`a41726b8`)

```
 .buildkite/k3_tests/multiprocess/BK_WEB_SETUP.md   |   6 +
 .buildkite/k3_tests/multiprocess/pipeline.yml      |  11 +
 .../multiprocess/scripts/run-dsv4-flash-tp.sh      |  13 +
 .../integration/vllm/hybrid-kv-cache-groups.md     |  38 +++
 tests/v1/multiprocess/test_cache_server.py         | 362 ++++++++++++++++++++-
```

The three `.buildkite` / doc edits are all cross-references so the two levels
explain each other; the design doc gains a `## Test coverage` section with the
table above and an explicit list of what is unguarded between nightlies (the
compressed path at TP>1; fixture geometry drift, which matters because this
pipeline reinstalls vLLM nightly on every job).

## Next

1. Done: rebuilt, ran, mutation-tested. See above.
2. Push `dsv4_ci_cost_pr` / `dsv4_ci_cost_dev` and draft the PR text.
3. Then L2 as a drift canary, after one real run settles the two unknowns.
4. Session 1's open items still stand: a human with Buildkite access has to
   create the nightly Scheduled Build (branch `dev`, `0 4 * * *`, no env) and
   confirm `/data/deep_gemm_jit` is writable.
