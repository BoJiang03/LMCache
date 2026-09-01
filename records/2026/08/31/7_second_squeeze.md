# Second squeeze: 2,741 -> 2,594 insertions, 1,089 -> 1,027 lines of code

Bo: "还有能榨代码行数的吗?"

Yes, three more things, and then the floor. Numbers against `117a0b88`:

| | after record 5 | after record 6 | after this |
|---|---|---|---|
| insertions | 2,977 | 2,741 | 2,594 |
| deletions | 507 | 585 | 585 |
| **lines of code** | **1,255** | **1,089** | **1,027** |
| docstring lines | 746 | 678 | 623 |
| doc+comment / code | 0.70 | 0.72 | 0.71 |

Commits: `e6764ce1` policy core (1592/463), `410e4fb5` connector wiring
(587/98), `677764f1` docs (415/24).

## 1. `lazy_offload_store_release` split out of this PR (-110)

`StoreReleasePlacement`, `_accepts_prepend`, the config parse, the branch in
`on_store_results`, the `configuration.rst` entry and the `lazy_offload.md`
paragraph: about 110 diff lines for a knob that answers a different question
from the rest of the PR ("where do completed-store blocks re-enter the free
queue", not "when do we offload").

The deciding fact: **the shipped benchmark runs its default.** `pr_info.md`
records the sweep's lazy arm as `store_release=lru_tail`, and `lru_tail` is
exactly plain `pool.free_blocks(blocks)` -- vLLM's own placement. Removing the
knob changes no measured number. `docs/coding_standards.md` asks for small
focused PRs, so this belongs in its own follow-up with the 08-26 paired
evidence (`eviction_head` won at 90G, lost at 250G) as its justification.

Re-add patch: `artifacts/pr_squeeze/store_release_readd.patch`, applies to the
PR tree with `git apply`.

## 2. `_PendingOperations`: two indexes deleted (-20 code)

- `_request_block_refs: dict[tuple[str, int], int]` counted, per request, how
  many of its ops covered each block, so the reverse index only dropped an
  entry when the last one departed. `_reindex` gets the same answer by
  comparing the departed ops' blocks against what the surviving list still
  covers -- one set difference, no second dict to keep in sync.
- `_request_order` + `_next_request_order` + `admission_order(request_id)`
  maintained an integer per request purely as the drain's sort tie-break.
  `_by_request` is already in admission order (dict insertion order, and a
  request that empties and returns re-enters at the back), so the tie-break is
  `enumerate(self._by_request)`, built once per drain.

Two invariants fewer to hold.

## 3. Smaller items

- `DrainResult.ops_held_back` -> a local counter in the drain loop; only
  `throttled_drains` ever read it.
- `_FreeQueueWindow.depth()` inlined to `len(window.ranks)`.
- Docstring and comment trims in `eviction_aware.py` and `base.py`: narrative
  that `docs/design/integration/vllm/lazy_offload_policy/eviction_aware.md`
  already carries. Every Args/Returns/Raises section is intact; density stayed
  at 0.71, above the 0.64 aggregate of comparable repo modules.

## 4. Verifying the index rewrite

Removing the refcount changes an invariant that no test in the PR covers (the
suites are on this branch). `artifacts/pr_squeeze/check_pending_index.py` is a
scratch driver -- not part of the PR -- that runs the queue through five
scenarios and asserts, after every mutation, that `_requests_by_block` equals
what the pending lists imply and that the counter ledger closes:

1. two requests sharing a block across a chunk boundary; dropping one must
   leave the block indexed for the other (the case the refcount guarded);
2. an evicted block dropping its op and every later op of the request;
3. `drop_request` / `mark_store_failed` clearing the index completely;
4. admission order after a request empties and is re-admitted;
5. the deferral deadline releasing with no window pressure at all;
6. window widening: emissions pin blocks, a second `discover` round brings
   fresh requests into view, and they must still be found in the
   admission-order snapshot taken at the top of the drain.

All six pass. The first run failed on scenario 1 -- the assertion was wrong,
not the code: the second request shared the block that had come into the
danger window, so it was correctly due in the same drain.

## 5. Where the floor is

What is left is the algorithm and its plumbing:

| block | code lines | why it stays |
|---|---|---|
| `_collect_due` + `discover` | ~145 | the drain decision itself |
| `LazyOffloadRequestRegistry` | 94 | store epochs; preemption reset and finished-id reuse both need them |
| `on_store_results` / `_drain` / `_coalesce_store_metadata` | ~180 | receipts, pinning, one store op per request |
| `max_deferral_seconds` + `_OVERDUE_RANK` | ~80 | released 57-77% of emissions in the sweep |
| cost sensors (4 counters) | ~10 | the only production view of the per-step scheduler cost |

Further reduction from here means deleting one of those, not restructuring.

One lever is not mine to pull: `tests/v1/test_lazy_offload_policy.py` is 174
of the 2,594 insertions and is the PR's only test. Bo's instruction was all
tests to dev, and the reason for the carve-out (leaving the upstream file
untouched would be red on merge) no longer applies now that
`lazy_offload_pending_store.py` is deleted outright. Removing it takes the PR
to ~2,420 insertions and zero tests. Flagged, not done.

## 6. Gates

- ruff check + format clean, codespell clean, mypy 1.17.1 clean.
- Unit: 253 passed.
- GSM8K gate 5: see section 7.

## 7. GSM8K gate 5 (PR tree after the second squeeze)

Artifacts: `artifacts/gsm8k_squeeze/`. Same harness as gates 3 and 4
(Qwen3-8B TP=4, 120 questions, l1 68 GB, cold/cached two-pass).

| config | cold | cached | pass-2 ext | l1 peak | evictions |
|--------|------|--------|-----------|---------|-----------|
| off    | 0.925 | 0.917 | -     | 0.000 | 0 |
| eager  | 0.917 | 0.917 | 0.961 | 0.751 | 0 |
| lazy   | 0.917 | 0.917 | 0.942 | 0.737 | 0 |

apc 0.000 everywhere, l1 peak under the 0.8 watermark, all non-vacuity guards
passed. All six scores sit inside the off arm's own cold/cached spread. The
ledger closes exactly:

```
admitted 189 = emitted 178 + dropped_evicted 10 + pending 1
emitted_overdue 0, throttled_drains 0, rejected_* 0
drain_steps 5672, free_queue_blocks_read 169043, blocks_validated 46293
```

The reverse index rewritten in section 2 is what feeds `requests_validated`
(270) and the eviction drops (10); both land where gate 4 put them (270 and
10), which is the end-to-end confirmation the scratch driver could not give.

Lazy pass-2 external share across the four gates on this code path: 0.934,
0.961, 0.935, 0.942. The spread is the harness, not the change.
