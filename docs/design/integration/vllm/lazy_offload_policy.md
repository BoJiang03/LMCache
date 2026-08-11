# `lazy_offload_policy.py`: Eviction-Aware Store Queue

Implements gates 1 and 3 of the store decision defined in
[lazy_offload_decision_model.md](lazy_offload_decision_model.md); the buffering
/ protection mechanism it plugs into is described in
[lazy_offload.md](lazy_offload.md). This document is the module's contract.

## Scope and non-scope

The module is **pure policy**: no vLLM imports at runtime, no I/O, no lock (it
runs on the scheduler thread). It decides *which* buffered store operations to
release *when*; the connector executes everything (snapshots, pinning,
submission). Gate 2 (reuse prediction) is out of scope — phase 1 stores every
admitted op whose blocks come under eviction pressure.

## Objects

- **`BlockPoolReader`** (protocol) — read-only pool view: `free_queue_ranks()`
  (block id → LRU eviction rank, rank 0 = next victim; absent = not free) and
  `block_hash(block_id)`. Production impl `GPUBlockPoolView` wraps the
  `BlockPool` bound via the vLLM `bind_gpu_block_pool` hook; both must never
  mutate pool state.
- **`PendingStoreOp`** — one deferred store: opaque `store_metadata` (the
  ready `LMCacheMPRequestMetadata`), the covered blocks' hash snapshot taken
  at admission, and `prefix_end_tokens` (prefix length once this op lands).
- **`EvictionAwareStoreQueue`** — the policy object, one per connector.

## Per-step protocol (connector obligations)

1. Route each `GetStoreMetadata` result to `admit(op)` instead of the step
   metadata. Handle the outcome:
   - `ADMITTED` → nothing now.
   - `REJECTED_UNHASHED_BLOCK` → **store eagerly** (a hash-less block's later
     eviction is undetectable: evicted-and-reallocated also reads `None`).
   - `REJECTED_PREFIX_BROKEN` → **skip** (an earlier chunk was dropped; this
     chunk would be unreachable on retrieval).
2. Once per step: `observe_step(gross_blocks_allocated, est_next_step_blocks)`
   then `collect_due()`.
3. For every op in `DrainResult.to_store` (already ordered): re-verify + pin
   (`touch`) its blocks, put its metadata into this step's connector metadata,
   and unpin (`free_blocks(prepend=True)`) when the worker reports the store
   complete. `dropped_*` lists need no action beyond accounting.
4. On `request_finished`: call `mark_request_finished(id)`; if it returns
   True, defer `end_session` until the id appears in a later
   `DrainResult.released_requests`. On abort/error paths use
   `drop_request(id)`.

## Decision rule

- **Danger depth** = `ceil(max(EMA(gross allocation/step), next-step
  feedforward) × horizon_steps)`; below half a block over the horizon it is 0
  (a decayed EMA must not hold the depth at 1 forever). An idle engine never
  drains — free-queue *position* alone is never a trigger (that would be the
  inverted-gate-1 anti-pattern, decision model §6).
- An op is **due** when any covered block's rank < danger depth. Blocks not
  in the free queue (in use / resurrected) are not at risk.
- **Prefix closure** (amendment A1): a due op releases the request's ops from
  the front through the last due one; a data-loss drop (hash mismatch) drops
  from the first lost op through the tail and blacklists the request's later
  chunks; the intact stored prefix stays pending and stays valuable.
- **Gate 3**: when a request comes due with known prefix <
  `min_prefix_tokens`, all its ops are dropped (the due front is dying, which
  breaks the chain for the rest). The threshold is the offline break-even
  prefix length; 0 disables.
- Cross-request drain order = min due rank ascending; `max_drain_per_step`
  bounds the per-step D2H burst, cutting whole segments from the tail.

## Observability

`stats()` returns cumulative counters. `dropped_evicted` is the gate-1 sensor
(drop rate: data lost before we drained — lower the horizon is too tight);
`emitted / admitted` is store precision's denominator; `rejected_short_prefix`
audits gate 3. Tests: `tests/v1/test_lazy_offload_policy.py` (pure, no vLLM).
