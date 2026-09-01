# PR slimming: six mechanisms out, docstrings to the repo norm, tests to dev

Continuation of record 4 (`4_goodput_check_and_the_pr_budget.md`). Bo's
colleague asked for less code: keep only the extremely core tests in the PR
and delete the over-designed policy mechanisms. Bo approved a single PR at
about 2.7k lines. This is what the work actually produced.

## 1. What the PR looks like now

Branch `lazy_offloading_policy_pr`, three commits on upstream `117a0b88`,
head `12c882ed`.

| commit | subject | +/- |
|---|---|---|
| 95cb3abd | [Core] Add eviction-aware lazy offload policy | 1885 / 385 |
| 46e021cf | [Core] Wire lazy offload through the MP connector and worker adapter | 656 / 98 |
| 12c882ed | Docs: lazy offload design and configuration reference | 436 / 24 |
| | **total** | **2977 / 507** |

Before: 5 commits, 9611 / 493 across 29 files. After: 3 commits, 2977 / 507
across 15 files. 69% of the PR is gone.

## 2. Where it lands against the 2.7k projection

Record 4 section 7 projected ~2,750. The first cut of this session landed at
4,782, over on both module python and tests. Bo then said **测试全部搬 dev**
-- move every test out of the PR -- which took it to 2,977.

| part | projected | first cut | after "tests to dev" |
|---|---|---|---|
| module python | ~1,995 | 2,586 | 2,586 |
| tests | ~500 | 2,195 | 160 net (the pre-existing suite, adapted) |
| docs | ~250 | 426 | 446 |
| | **2,750** | **4,782** | **2,977** |

The module-python line is still 590 over the projection, and the reason is
docstrings. The projection assumed the repo aggregate 0.47 doc/code. That
aggregate includes files with almost no docstrings at all, and the repo's own
coding standard requires a complete docstring (summary, args, returns,
raises) on every public function. Measured against modules that actually
follow that standard, the PR sits **below** the norm:

| module | doc/code |
|---|---|
| `lmcache/v1/distributed/eviction.py` | 1.51 |
| `lmcache/v1/platform/cuda/timeline_semaphore_event_ipc.py` | 0.68 |
| `lmcache/v1/mp_coordinator/views/usage_manager.py` | 0.57 |
| `lmcache/v1/mp_coordinator/controllers/eviction_controller.py` | 0.55 |
| `lmcache/v1/multiprocess/session.py` | 0.39 |
| six of those, aggregate | **0.64** |
| this PR's six lazy-offload modules, aggregate | **0.59** |

`eviction_aware.py` went from 756 code / 840 doc (1.11) to 475 / 347 (0.73).
Going lower means deleting Args/Returns blocks the standard asks for.

**Tests.** Bo's instruction was all of them to dev. What that leaves in the
PR is `tests/v1/test_lazy_offload_pending_store.py`, and only because it
already exists upstream and tests interfaces this PR changes
(`mark_req_finished`, `pop_items_for_offload(count)`,
`update_request_gpu_block_ids`). Left untouched it would be a red test on the
merge, so it is adapted in place -- 217 lines against upstream's 275, same
two classes, no new coverage. Everything genuinely new (1,465 lines: the
policy's decision contract, the manager's pin/unpin and session teardown, the
request registry, and the facade's EVICTION_AWARE routing) is on dev.

The consequence to state plainly when the PR goes up: it adds ~1,300 lines of
new production code with no new tests, and the "this PR contains unit tests"
box is false. `AGENTS.md` and `docs/coding_standards.md` both ask for tests
on new features, so a reviewer will raise it. The dev branch has them ready
to move back in one commit if that happens.

## 3. The six mechanisms, deleted

Record 4 section 6 has the ledger evidence. What came out:

| mechanism | what went | 
|---|---|
| `min_prefix_tokens` + held two-stage admission | `_held_short`, `_promote_held`, `_fails_economy_gate`, `rejected_short_prefix`, `DrainResult.dropped_short_prefix`, `num_held_ops`, the finished-id gate-3 drop in `collect_due` |
| `announce_allocation` / `retract_allocation` | both methods, `_announced_blocks`, `announced_bursts`, the manager's `announce_hit_load`, the connector's `lazy_offload_announce_hits` key and its announce-then-admit branch, the tracker's `hit_load_announced` |
| content deduplication | `_content_key`, the content index, `covering_op`, `_chain_intact`, `AdmitResult.DEDUPLICATED`, `AddOutcome.DEDUPLICATED`, `deduplicated`, `PendingStoreOp.cache_salt` |
| covered-prefix advance | `covered_prefix_tokens` at all three layers, `covers_block`, `_covered_blocks`, `covered_prefix_advances`, `covered_prefix_tokens_skipped`, `covered_blocks_probed`, the connector's `_skip_pending_covered_prefix` and its two call sites |
| adaptive danger floor | `_update_danger_floor`, `_danger_depth`'s floor term, five `_DANGER_FLOOR_*` / `_RECENT_ALLOC_STEPS` constants, `_recent_step_allocs`, `danger_floor_raises` |
| block volume cap | `max_drain_blocks_per_step`, the whole `_DrainBudget` class (the op cap is now one integer in the drain loop) |

Two consequences worth naming:

- **`_contiguous_front_run` went too.** A hole in a request's pending list
  could only ever come from a deduplicated chunk. With deduplication gone,
  ops of one request are contiguous by construction, so the cut is dead
  code. The manager's `_coalesce_store_metadata` keeps its non-contiguous
  `ValueError` as a defensive guard.
- **`collect_due` no longer takes `finished_request_ids`.** Its only use was
  dropping gate-3 held chains. FIFO still receives them through
  `LazyOffloadPendingStore.drain()`.

Counters went from 24 fields to 17; the ledger equation lost two terms and is
restated in the design doc.

## 4. What moved to dev

`records/2026/08/31/artifacts/pr_slim/`:

- `lazy_offload_decision_model.md` (159 lines) -- selection rationale, not an
  interface contract, and it was the doc most tied to gates 2 and 3.
- `l1_pressure_stats.md` and `l1_pressure_stats.patch` -- the GET_L1_PRESSURE
  protocol commit (`badf1902`, 112 code lines) plus its 150 lines of tests,
  reverted out of the PR whole. Nothing in the PR consumed it; the dev-time
  consumer was the experiment harness estimating L1 residence. It is a
  standalone follow-up PR if it is worth one.
- `tests_moved_from_pr/` -- the five test files as they stood before the cut,
  so nothing written is lost:
  `test_mp_connector_lazy_offload.py` (846) and
  `test_mp_worker_adapter_lazy_offload.py` (319) left the PR entirely;
  `test_lazy_offload_eviction_aware.py`, `test_lazy_offload_manager.py` and
  `test_lazy_offload_pending_store.py` were trimmed in place.

Test classes deleted with their mechanisms: `TestEconomyGate`,
`TestContentDeduplication`, `TestCoveredPrefix`, `TestBlockVolumeCap`,
`TestAnnouncedBursts`, `TestDangerFloor`, `TestEmissionContiguity`,
`TestAnnouncedHitLoads`, `TestCoveredPrefixRouting`. Classes moved to dev for
size, not because they were wrong: `TestPinCascadeShift` (87),
`TestFreeQueueSnapshotBound` (194), the six block-pool-view parity tests
against the real vLLM `BlockPool` (124), and the FIFO receipt/reuse suite.

## 5. Gates

- **Build**: not rerun. Every change this session is pure Python; the four
  extensions built for gate 2 (`cuda_ops`, `lmcache_native`, `lmcache_fs`,
  `lmcache_redis`) are unchanged in the PR worktree.
- **Lint**: `ruff check` clean, `ruff format` applied (4 files reformatted).
  `mypy` not run -- it is not installed in any venv here and installing it
  would mutate a shared environment.
- **Unit**: 201 passed in 23 s -- the four kept lazy suites plus
  `multiprocess/test_cache_server.py`, `distributed/test_l1_manager.py`
  (both back to their upstream content after the pressure-stats revert) and
  `test_vllm_kv_cache_groups.py`.
- **GSM8K**: rerun on the slimmed tree, see section 6.

## 6. GSM8K gate 3 (slimmed PR tree)

Same pr4499 harness and env as gate 2, `SMOKE_REPO` on the PR worktree,
Qwen3-8B TP=4 on GPUs 1-4, 120 questions, L1 68 GB, one repetition per mode.
The import guard (`sitecustomize.py` stripping the venv's editable finder,
path hook and importer cache) went into the PR worktree root for the run and
was deleted after; untracked, never committed.

| config | cold | cached | pass-2 ext |
|--------|------|--------|-----------|
| off    | 0.908 | 0.908 | - |
| eager  | 0.908 | 0.917 | 0.961 |
| lazy   | 0.917 | 0.925 | 0.961 |

All six scores sit at or inside the historical off spread (0.900-0.925), apc
0.000 everywhere, l1 peak 0.75 under the 0.8 watermark, 0 evictions, every
non-vacuity guard clean. Lazy's pass-2 external hit share is 0.961, up from
0.934 at gate 2 and now equal to eager's -- with content deduplication and
the covered-prefix advance gone, a follower request re-buffers the prefix
itself instead of deduplicating against a cover that could be dropped later,
so nothing is left unreachable.

The ledger closes exactly:

    admitted 189 = emitted 178 + dropped_evicted 8 + pending 3

`emitted_overdue` is 0 because the harness leaves `max_deferral_seconds` at
its default of 0. `emitted_deferral_drains` 25,146 over 178 emissions is a
mean deferral of 141 drains, on 5,504 drain steps. `throttled_drains` 0.

Artifacts: `artifacts/gsm8k_slim/` (gate.log, run.sh, three ac_*.json).

## 7. Correction

Record 4 section 7's arithmetic promised ~2,750 with "core tests (~350 code
lines) -> ~500". That was wrong: the fake block pool and connector harnesses
alone are ~500 lines before a single assertion, and the docstring figure used
a repo aggregate that includes files which do not follow the repo's own
docstring standard. The first cut came out at 4,782. The 2,977 the PR now
carries is reached by having no new tests in it at all, which is a different
trade than the one that arithmetic described.

## 8. Open items

- The CONC sweep in `pr_info.md` was measured with
  `lazy_offload_danger_floor_max_blocks=8192`, a knob this PR no longer has.
  The floor raised 1-6 times in 35-58k drain steps per arm, so the numbers
  should be unchanged, but the body currently reports a config the PR cannot
  express. Either re-run one lazy arm (CONC=40, ~40 min; the e40 eager
  reference already exists) with the shipped defaults, or say so in the body.
- Nothing is pushed yet. Both branches are local.
