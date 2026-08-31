# PR branch gates: build, unit tests, gsm8k on the extracted code

Session continuation of record 1. All gates on `lazy_offloading_policy_pr`
passed; both branches pushed to the fork. Bo creates the PR from
`pr_info.md`.

## 1. The .so build (waiver: PR worktree only)

Bo waived the no-rebuild red line for the PR worktree only ("可以编，只在pr
worktree里"). `setup.py build_ext --inplace` with the venv python,
TORCH_CUDA_ARCH_LIST=9.0a, MAX_JOBS=32; exit 0. Four extensions, not the six
the dev worktree has: upstream HEAD's CUDA profile builds cuda_ops,
lmcache_native, lmcache_fs, lmcache_redis; c_ops survives only in the ascend
profile and native_storage_ops is gone. No non-test import of either missing
module in the PR tree, so 4/4 is complete.

## 2. The import hijack has two more layers than pyguard knew

The known problem: the vllm-lazy venv's editable install points `lmcache` at
the dev worktree. The recorded fix (strip `__editable___lmcache` finders from
`sys.meta_path` in a sitecustomize) turned out to cover one of three
mechanisms:

1. `sys.meta_path` finder -- stripped by the old pyguard. Fine.
2. The .pth also registers a path hook plus a `...__path_hook__` placeholder
   entry on `sys.path` (namespace-package fallback, still mapping to the dev
   worktree). New sitecustomize strips both and clears
   `sys.path_importer_cache`.
3. Not the editable at all: running `python -c` from the dev worktree root
   puts `''` (cwd) first on `sys.path`, and cwd contains `lmcache/`. The
   probe that "proved" the hijack was just resolving cwd. Run from inside
   the PR worktree.

For engine runs there is a fourth wrinkle: `driver.py` launches the MP server
and vLLM with `PYTHONPATH=SMOKE_REPO` (overwrite, not prepend), so a
scratchpad pyguard on the caller's PYTHONPATH never reaches the engine. The
repo root itself is on the engine's sys.path, so the sitecustomize went into
the PR worktree root for the gate and was deleted after. Untracked, never
committed.

## 3. Unit tests on the PR tree

354 passed in 21s: the six lazy suites plus
`multiprocess/test_cache_server.py`, `multiprocess/test_l1_pressure.py`,
`distributed/test_l1_manager.py`. (The l1_manager/cache_server/l1_pressure
files live under subdirectories upstream, not `tests/v1/` flat.)

## 4. GSM8K gate 2 (PR tree, Qwen3-8B TP=4, 120 q, l1 68 GB)

Same pr4499 harness, SMOKE_REPO pointed at the PR worktree.

| config | cold | cached | pass-2 ext |
|--------|------|--------|-----------|
| off    | 0.925 | 0.900 | - |
| eager  | 0.908 | 0.925 | 0.961 |
| lazy   | 0.917 | 0.908 | 0.934 |

All six scores sit inside the off config's own cold/cached spread
(0.900-0.925), apc 0.000 everywhere, l1 peak 0.75 < 0.8 watermark, 0
evictions. Lazy ledger closes exactly: admitted 191 = emitted 178 + pending 2
+ dropped_evicted 11. Scores differ from gate 1's by one question here and
there across engine boots; the harness's own doc says greedy decode
legitimately perturbs with prefill split, and the off arm shows the same
spread with no cache involved.

Proof the engine ran the PR tree (not the hijacked dev tree): the vllm log's
LMCache lines carry file:line tags (`lmcache_mp_connector.py:550`, `:803`,
`vllm_multi_process_adapter.py:157`, `:1339`, `:2025`, ...); each of those
line numbers is a `logger.info` call in the PR tree and unrelated code in the
dev tree. EVICTION_AWARE confirmed in the connector config line.

## 5. Decisions taken with Bo this session

- Pressure stats (commit 3) stays in the PR. Honest framing required: it has
  no in-tree consumer beyond tests (the dev-time consumer was the
  experiment harness estimating L1 residence); pr_info.md's "used by the
  policy's sizing sensors" was wrong and is fixed to "observability probe".
- FIFO diff is interface adaptation only (state machine owns finished
  tracking, shared drain signature); threshold semantics untouched.
- Mechanism correction (Bo): the ext-hit gap at saturation is L1
  residence, not transfer bandwidth. Verified from mp_final.prom lookup
  ratios: hit/requested tokens e48 55.5% vs l48 54.9% (same), e72 30.4%
  vs l72 40.9% (diverged) -- eager's 2.5x write volume into the same
  250 GB churns L1 until entries die before reuse; writing less is an
  effective capacity multiplier. Record 1's bandwidth-contention framing
  overstated; pr_info bullets rewritten to the residence mechanism.
- Docs compressed to "concise but complete": eviction_aware.md 462 -> 306,
  decision_model.md 197 -> 159. Contracts, config keys, the ledger equation
  and its in/out rationale kept; measurement anecdotes (i60F, 448-block
  cap experiment, gemma null-block replay numbers, 1965-token chunking
  demo) and one duplicated pin-cascade section cut. lazy_offload.md's
  upstream 684 lines untouched (colleague's document); our +136 there was
  already tight. l1_pressure_stats.md (52) unchanged. Docs commit amended:
  cb1efd15 -> b8263808.

## 6. Branch state at push

- `lazy_offloading_policy_pr`: 5 commits on origin/dev @117a0b88, head
  d45edab6 after the lint fixes (codespell in l1_pressure_stats.md, a dict
  annotation in test_lazy_offload_pending_store.py). Working tree clean, no untracked files.
- `lazy_offloading_policy_dev`: lazy-offload-dev including this record.

Artifacts: `artifacts/gsm8k_pr/` (gate2.log, env, three ac_*.json, lazy
ledger line).
