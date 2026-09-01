# Code-level refactor: 2,977 -> 2,741 insertions, 1,255 -> 1,089 lines of code

Bo, after the slimming pass: "代码层面难道没有可以重构而减少行数的吗?"

Yes. The slimming pass deleted mechanisms; it did not look at the shape of
what was left. The shape had one structural defect worth 200 lines.

## 1. The defect: the PR had deleted its own polymorphism

Upstream `lazy_offload_policy/base.py` defines an `OffloadPolicy` ABC, and
upstream `LazyOffloadPendingStore` holds exactly one policy behind it and
forwards to it with no branching at all -- five one-line methods.

The PR deleted `base.py` (-68) and replaced that with a 596-line
`LazyOffloadPendingStore` in which every method is

```python
if self._eviction_queue is not None:
    return self._eviction_queue.X(...)
return self._require_fifo_policy().X(...)
```

Eight of those, plus two `_require_*` guards, plus a 30-line config-cast
block, plus the ledger logging (which only one of the two policies has).
That is the layer a reviewer flags first, and it was our own regression
against the module's existing design.

## 2. What was done

| change | effect |
|---|---|
| Restored `base.py` as the `OffloadPolicy` Protocol; both policies implement it | mypy now checks conformance at the factory's return |
| Bundled the five drain arguments into one `DrainSignals` frozen dataclass | the argument list was written out three times (manager -> store -> queue) with three docstrings; now once |
| Deleted `LazyOffloadPendingStore`; policy selection is `create_offload_policy()` in `lazy_offload_policy/__init__.py` | -596 lines of file, +72 of factory |
| Folded `observe_step` into `drain`; `admit` became `add`; `collect_due` became private | one public entry per step instead of two, and the caller no longer assembles `PendingStoreOp` |
| Deleted the `AdmitResult` enum | it was translated one-to-one into `AddOutcome` by the facade |
| Deleted `AddOutcome` as well | the connector discards `add_store_candidate`'s return value; the three-valued outcome was never read by anything |
| `LazyOffloadCounters.decisions()` computed from `_COST_SENSOR_FIELDS` | was a hand-written 13-line tuple that had to be edited whenever a counter was added |
| `_count_new_blocks` + `_allocated_block_ids` -> one `_new_blocks` | two functions walked the same nested scheduler-output structure |
| `_due_front_segment` returns a list, not `tuple[int, list] | None` | the caller discarded the rank half |
| `_snapshot_intact` counts per block instead of via a `checked` accumulator | same counter value, five fewer lines |
| Dropped `moved to dev` dead API: `GPUBlockPoolView.num_free_blocks`, `LazyOffloadRequestRegistry.is_active`, `LazyOffloadPendingStore.mode`/`stats`, `_token_span`, the `allocated_block_ids=None` full-validation branch | each had zero production callers |
| Removed the once-per-process throttle-vs-loss WARNING (27 lines) | `throttled_drains` stays in the ledger; measured 0 in every real arm |
| Trimmed docstrings that restate the protocol, and three over-long log messages | implementations now say "see `OffloadPolicy.add`" |
| `tests/v1/test_lazy_offload_pending_store.py` -> `test_lazy_offload_policy.py` | the module it was named after no longer exists; 12 tests, covers factory selection + FIFO through the protocol |

## 3. Numbers

Diff against `117a0b88`:

| | before | after |
|---|---|---|
| insertions | 2,977 | 2,741 |
| deletions | 507 | 585 |
| files | 15 | 16 |
| **lines of code** (ast/tokenize, excludes docstrings, comments, blanks) | **1,255** | **1,089** |
| docstring lines | 746 | 678 |
| comment lines | 129 | 106 |
| doc+comment / code | 0.70 | 0.72 |

Deletions rose because the refactor removes an upstream file (101 lines) and
renames the test file. Net diff lines: 3,484 -> 3,326.

Per commit:

| commit | subject | +/- |
|---|---|---|
| 661793ce | [Core] Add eviction-aware lazy offload policy | 1631 / 463 |
| 48123f24 | [Core] Wire lazy offload through the MP connector and worker adapter | 676 / 98 |
| a3f4b8b0 | Docs: lazy offload design and configuration reference | 434 / 24 |

## 4. Two things deliberately not cut

- **`max_deferral_seconds` and the whole `_OVERDUE_RANK` path** (~80 lines).
  It looks like a knob that is off by default. It is not dead: the headline
  sweep's lazy arm sets it to 30, and the deadline released 57-77% of all
  emissions there. Cutting it would delete the mechanism the benchmark
  measured.
- **`StoreReleasePlacement` + `_free_blocks_accepts_prepend`** (~60 lines).
  Both placements were measured in-round paired on 2026-08-26; `eviction_head`
  won at 90G, lost at 250G, and the default was flipped to `lru_tail` on that
  evidence. Keeping a measured knob.

The cost sensors (`drain_steps`, `free_queue_blocks_read`,
`requests_validated`, `blocks_validated`) also stayed. They are four int
increments and they are what showed the per-step scheduler cost
(204M block reads over 57k drains in the sweep). Removing them would save
about 40 lines and the only production view of that cost.

## 5. Behaviour

The refactor is intended to be behaviour-preserving. The differences that
exist at all:

- An invalid `lazy_offload_policy` name now raises at `bind_block_pool`
  instead of at connector construction. Both are engine startup.
- The throttle warning no longer prints; the counter it reported still does.

Everything else is a rename, a move, or a discarded return value.

## 6. Gates

- `ruff check` clean, `ruff format` no changes, `codespell` clean.
- `mypy 1.17.1` clean on the policy package, manager, state, metadata. The
  protocol conformance of both policies is checked at
  `create_offload_policy`'s return.
- `isort` flags only `lmcache_mp_connector_0180.py` / `_0201.py`, which fail
  identically at upstream HEAD and are untouched here.
- Unit: 253 passed (the new policy suite plus every `tests/v1` file that
  imports `lmcache.integration.vllm`, plus `multiprocess/test_cache_server.py`
  and `distributed/test_l1_manager.py`).
- GSM8K gate 4: see section 7.

## 7. GSM8K gate 4 (PR tree after the refactor, Qwen3-8B TP=4, 120 q, l1 68 GB)

Artifacts: `artifacts/gsm8k_refactor/` (gate.log, run.sh, three ac_*.json, and
the untracked import guard the engine needed).

| config | cold | cached | pass-2 ext | l1 peak | evictions |
|--------|------|--------|-----------|---------|-----------|
| off    | 0.900 | 0.908 | -     | 0.000 | 0 |
| eager  | 0.908 | 0.908 | 0.961 | 0.752 | 0 |
| lazy   | 0.925 | 0.925 | 0.935 | 0.732 | 0 |

apc 0.000 everywhere, l1 peak under the 0.8 watermark, all non-vacuity guards
clean. Lazy's ledger closes exactly:

```
admitted 190 = emitted 177 + dropped_evicted 10 + pending 3
emitted_overdue 0, throttled_drains 0, rejected_* 0
drain_steps 5646, free_queue_blocks_read 172859, blocks_validated 46229
```

`emitted_overdue` is 0 because the harness leaves `max_deferral_seconds` at
its default of 0.0; the deadline path is exercised by the sweep, not by this
gate.

Lazy accuracy is the highest of the three arms (0.925/0.925). Its pass-2
external share moved 0.961 (gate 3) -> 0.935, with `dropped_evicted` 8 -> 10.
Gate 2 measured 0.934 on the same code path, so this is the run-to-run spread
of a 120-question harness, not a refactor effect: the refactor changes no
decision logic, only where it lives.

Which arms ran the final tree: the `off` arm started at 18:48 and the last
source edit landed at 18:53:16, so `off` ran a tree one commit behind. `off`
never constructs the manager or a policy. The `eager` engine launched at
18:53:50 and the `lazy` engine at 18:54:29, both after the last edit.

## 8. Open items

Unchanged from record 5, plus one:

1. The CONC sweep in `pr_info.md` was measured with
   `lazy_offload_danger_floor_max_blocks=8192`, a knob this PR no longer has.
2. ~1.3k lines of new production code with no new tests; the moved suites sit
   at `records/2026/08/31/artifacts/pr_slim/tests_moved_from_pr/slimmed/`.
   They now need porting to the refactored interface before they could go
   back: `admit(op)` -> `add(meta, block_hashes, epoch)`,
   `observe_step` + `collect_due` -> `drain(DrainSignals)`, and no
   `LazyOffloadPendingStore`.
3. The series is stacked, not independently bisectable: commit 1 removes
   `lazy_offload_pending_store.py` while the connector still imports it until
   commit 2. That was already true of the previous arrangement (commit 1
   changed the pending store's API out from under the connector); the
   deletion makes it an import error instead of a call error. Only the tip is
   green either way.
