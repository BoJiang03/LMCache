# Line state at 2026-09-01, and the two decisions waiting on Bo

No new work today. The whole of the 08-31 session is recorded under
`records/2026/08/31/`; this file exists so a session starting on 09-01 does
not have to reconstruct where the line stands.

## 1. Where the branches are

| branch | head | contents |
|---|---|---|
| `lazy_offloading_policy_pr` | `677764f1` | the PR, three commits on `117a0b88` |
| `lazy-offload-dev` (pushed as `lazy_offloading_policy_dev`) | `c2a3fa53` | records, artifacts, the moved test suites |

Both pushed to `BoJiang03/LMCache`. The PR is not open; Bo opens it.

PR shape, three commits, 2,594 insertions / 585 deletions over 16 files:

| commit | subject | +/- |
|---|---|---|
| `e6764ce1` | [Core] Add eviction-aware lazy offload policy | 1592 / 463 |
| `410e4fb5` | [Core] Wire lazy offload through the MP connector and worker adapter | 587 / 98 |
| `677764f1` | Docs: lazy offload design and configuration reference | 415 / 24 |

## 2. How it got to 2.6k

Three passes, one record each:

| record | pass | insertions | lines of code |
|---|---|---|---|
| `2026/08/31/5_pr_slimming.md` | six dead mechanisms out, tests to dev | 9,611 -> 2,977 | 1,255 |
| `2026/08/31/6_code_refactor.md` | `OffloadPolicy` interface restored, facade deleted | 2,977 -> 2,741 | 1,089 |
| `2026/08/31/7_second_squeeze.md` | `store_release` split out, two indexes removed | 2,741 -> 2,594 | 1,027 |

"Lines of code" is ast/tokenize counting, excluding docstrings, comments and
blanks. Record 7 section 5 argues the floor is here: what remains is the drain
decision, the epoch registry, the receipt/pin/coalesce plumbing, the deferral
deadline, and four counters. Further reduction deletes capability.

## 3. The two open decisions

Both are also checkboxes in `2026/08/31/pr_info.md`.

1. **The CONC sweep config.** The table in the PR body was measured with
   `lazy_offload_danger_floor_max_blocks=8192`, a knob the PR no longer has.
   The floor raised 1-6 times in 35-58k drain steps per arm, so the numbers
   should be unchanged, but the body reports a config the PR cannot express.
   Either re-run one lazy arm (CONC=40, ~40 min; the e40 eager reference
   exists) with the shipped defaults, or say so in the body.
2. **The last test file.** `tests/v1/test_lazy_offload_policy.py` is 174 of
   the 2,594 insertions and the PR's only test. Bo's instruction was all tests
   to dev; the carve-out that kept this one (leaving upstream's file untouched
   would be red on merge) stopped applying once
   `lazy_offload_pending_store.py` was deleted outright. Moving it takes the
   PR to ~2,420 insertions and zero tests, against ~1.3k lines of new
   production code. AGENTS.md and docs/coding_standards.md both ask for tests
   on new features, so a reviewer will raise it either way.

## 4. Two things parked, not lost

- `lazy_offload_store_release` is out of this PR and belongs in a follow-up.
  Re-add patch: `2026/08/31/artifacts/pr_squeeze/store_release_readd.patch`,
  applies with `git apply`. Its justification is the 08-26 paired measurement
  (`eviction_head` wins at 90G, loses at 250G) in
  `2026/08/26/7_placement_verdict_eviction_head_is_the_bill.md`.
- The test suites moved off the PR are at
  `2026/08/31/artifacts/pr_slim/tests_moved_from_pr/slimmed/` (1,465 lines).
  They need porting before they could go back: `admit(op)` is now
  `add(meta, block_hashes, epoch)`, `observe_step` + `collect_due` are now
  `drain(DrainSignals)`, and `LazyOffloadPendingStore` is gone.

## 5. Gate history on the PR tree

| gate | tree | off | eager | lazy | lazy ext |
|---|---|---|---|---|---|
| 2 | pre-slim | 0.925/0.900 | 0.908/0.925 | 0.917/0.908 | 0.934 |
| 3 | slimmed (record 5) | 0.908/0.908 | 0.908/0.917 | 0.917/0.925 | 0.961 |
| 4 | refactored (record 6) | 0.900/0.908 | 0.908/0.908 | 0.925/0.925 | 0.935 |
| 5 | squeezed (record 7) | 0.925/0.917 | 0.917/0.917 | 0.917/0.917 | 0.942 |

Every score across all four gates sits inside the off arm's own cold/cached
spread, and the ledger closed exactly in each. The lazy pass-2 external share
(0.934 / 0.961 / 0.935 / 0.942) is the spread of a 120-question harness, not a
signal about any of the three changes.

## 6. Reproducing a gate

`2026/08/31/artifacts/gsm8k_squeeze/run.sh` is the current runner. It needs
`sitecustomize.py` in the PR worktree root for the duration of the run --
`driver.py` overwrites `PYTHONPATH` with `SMOKE_REPO`, so that file is the
only way to keep the venv's editable install from resolving `lmcache` to the
dev worktree. A copy is archived at
`2026/08/31/artifacts/gsm8k_refactor/sitecustomize.py`. It is untracked and
must be deleted before committing.
