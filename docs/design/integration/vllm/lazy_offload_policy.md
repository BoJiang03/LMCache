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
  at admission, `prefix_start_tokens` / `prefix_end_tokens` (the op's token
  range; the start detects deduplication holes in the pending list), and
  `cache_salt` (part of the op's content identity).
- **`EvictionAwareStoreQueue`** — the policy object, one per connector.

## Per-step protocol (connector obligations)

1. Route each `GetStoreMetadata` result to `admit(op)` instead of the step
   metadata. Handle the outcome:
   - `ADMITTED` → nothing now.
   - `REJECTED_UNHASHED_BLOCK` → **skip and warn** (a hash-less block's later
     eviction is undetectable: evicted-and-reallocated also reads `None`).
     The tracker has already advanced past the skipped range by the time the
     op reaches `admit`, so the request's later chunks are unreachable; the
     queue blacklists the request and rejects them as prefix-broken. With
     plain prefix caching (enforced at connector init) chunk-aligned ranges
     never cover unhashed blocks, but hybrid-attention models (sliding
     window, mamba) can place hash-less null blocks in block tables.
   - `REJECTED_PREFIX_BROKEN` → **skip** (an earlier chunk was dropped; this
     chunk would be unreachable on retrieval).
   - `DEDUPLICATED` → nothing now (identical content — same salt, range, and
     block-hash chain — is already buffered under another request and will be
     stored or dropped with that op; this op must not defer its own request's
     teardown). Deduplication is what bounds the queue: without it every
     request over a hot shared prefix (blocks never in the free queue, so
     never due) would buffer its own copy indefinitely; with it the queue is
     bounded by the unique cached content on the GPU. A hit is validated
     against the pool: if the covering op's block snapshot is no longer
     intact (its blocks were recycled while it waited for its eviction
     drop — e.g. behind an in-flight batch, or by this very step's
     allocation), or an earlier pending op of the covering op's request has
     lost a block (the next drain then prefix-closes over the cover too),
     the new op is admitted instead and takes over the content key; a
     doomed op never absorbs a live copy. Past that check it is
     optimistic: if the covering op is dropped later, chunks the
     deduplicated request stores past that point are unreachable until a
     future request re-buffers the prefix — wasted storage, never
     corruption. A deduplicated chunk also
     leaves a *hole* in its request's pending list; emission never spans a
     hole (each batch is coalesced into one contiguous store op), so the ops
     on each side go out in separate batches.
2. Once per step: `observe_step(gross_blocks_allocated, est_next_step_blocks)`
   then `collect_due()`.
3. For every op in `DrainResult.to_store` (already ordered): pin (`touch`)
   its blocks, **coalesce each request's released ops into one store op**
   (the worker adapter tracks a single in-flight store future per request),
   and put it into this step's connector metadata. `dropped_*` lists need no
   action beyond accounting.
4. On the store-completion receipt: unpin with `free_blocks(prepend=True)`
   (a stored block has a copy below the GPU, so among free blocks it should
   die first) and call `notify_stored(id)` — the queue holds back a
   request's remaining ops while a batch is in flight; a True return means
   the request is finished and fully drained, so its session may end.
5. On `request_finished`: call `mark_request_finished(id)`; True means
   stores are pending or in flight — defer `end_session` until the id
   appears in `DrainResult.released_requests` (remaining ops all dropped) or
   `notify_stored` returns True (stored).
6. When the request's buffered state goes stale — today only the preemption
   tracker reset (the recreated tracker re-produces metadata from token
   zero, overlapping anything buffered) — call `drop_request(id)`. It
   discards pending ops only: an in-flight batch stays tracked until its
   receipt, so a re-admitted op cannot be emitted while the worker still
   holds an outstanding store for the request. The surviving batch is
   marked *stale* (see step 7). An abort is **not** a drop: it routes
   through `request_finished` → `mark_request_finished`, and the aborted
   request's buffered ops stay storable until drained or evicted.
7. When a receipt reports the store **failed** (worker-side failure signal):
   call `mark_store_failed(id)` before `notify_stored(id)`. It drops the
   request's held-back ops and rejects its later chunks (without the failed
   prefix they would be stored unreachable), while leaving the finished and
   in-flight markers alone so the accompanying receipt still tears the
   request down through `notify_stored` as usual. A failure of a batch
   marked stale by `drop_request` is ignored: ops admitted after the reset
   were re-produced from token zero and do not depend on the failed prefix.
   `notify_stored` clears the stale mark along with the in-flight marker.
8. When a **new** request's id is first seen (tracker creation): call
   `reclaim_finished_request(id)`. In lazy mode a finished request leaves
   vLLM's request table immediately (`request_finished` returns False), so a
   client-supplied id can return while its previous owner's teardown is
   still deferred; without the reclaim the two requests' pending lists
   conflate (the predecessor's eviction drop prefix-closes over the
   successor's intact ops, and the deferred release fires while the
   successor is live). The reclaim discards the predecessor's buffered ops
   and its finished marker; a True return means the caller must
   `end_session(id)` now, before the successor's first operation. With an
   in-flight predecessor batch it returns False instead: the batch is
   marked stale and the id-keyed session, which now covers both requests,
   ends once through the successor's own lifecycle — the predecessor's
   receipt only clears the in-flight hold. The marker must not ride the
   receipt: the successor is live when the reclaim fires, so any teardown
   the marker later authorizes (the predecessor's receipt, the successor's
   own receipt, or an eviction drop landing the id in
   `released_requests`) would end a running request's session.

## Decision rule

- **Danger depth** = `ceil(max(EMA(gross allocation/step), next-step
  feedforward) × horizon_steps)`; below half a block over the horizon it is 0
  (a decayed EMA must not hold the depth at 1 forever). An idle engine never
  drains — free-queue *position* alone is never a trigger (that would be the
  inverted-gate-1 anti-pattern, decision model §6).
- An op is **due** when any covered block's rank < danger depth. Blocks not
  in the free queue (in use / resurrected) are not at risk.
- **Pin-cascade shift**: emitting a segment pins its blocks out of the free
  queue, moving every block behind them toward the head by the segment's
  size before the next step's allocation runs. Within one `collect_due`
  call, each later candidate is therefore checked against
  `danger depth + blocks emitted so far in this call`; without this, a
  candidate teleported into the danger window by an earlier emission loses
  its tail to the next allocation before the next drain can see it
  (observed as `dropped_evicted` under back-to-back drains). The first
  emission still requires a plain danger-depth hit, so this never opens
  the gate on an idle system; dropped (unpinned) segments do not extend
  the shift.
- **Prefix closure** (amendment A1): a due op releases the request's ops from
  the front through the last due one; a data-loss drop (hash mismatch) drops
  from the first lost op through the tail and blacklists the request's later
  chunks; the intact stored prefix stays pending and stays valuable.
- **Gate 3**: when a request comes due with known prefix <
  `min_prefix_tokens`, all its ops are dropped (the due front is dying, which
  breaks the chain for the rest). The threshold is the offline break-even
  prefix length; 0 disables.
- A due segment is cut at the first deduplication hole before emission
  (the batch must coalesce into one contiguous token range); the request
  keeps its due-rank urgency, and the post-hole ops follow in a later
  batch once the front run's receipt arrives.
- Cross-request drain order = min due rank ascending; `max_drain_per_step`
  bounds the per-step D2H burst. The cap may split a request's due segment,
  but only ever emits a front slice of it, so within-request prefix order is
  preserved and the remainder stays pending.
- **Idle consequences**: receipts travel in worker metadata, which only
  flows on steps that schedule tokens. If the engine goes idle with a
  batch in flight, its pins and its request's session stay held until the
  next non-empty step delivers the receipt; finished requests whose ops
  never come due likewise hold their sessions open. Both resolve on the
  next activity — nothing leaks permanently, by design ("idle never
  drains" also means "idle never settles").

## Observability

`stats()` returns cumulative counters. `dropped_evicted` is the gate-1 sensor
(drop rate: data lost before we drained — lower the horizon is too tight);
`emitted / admitted` is store precision's denominator; `rejected_short_prefix`
audits gate 3. Tests: `tests/v1/test_lazy_offload_policy.py` (pure, no vLLM).

The counters surface in the scheduler process log, not in vLLM's
`get_kv_connector_stats` plumbing (that hook is polled worker-side, where the
policy does not live). Three hooks, all on the pending-store facade:

- each `dropped_evicted` op logs one INFO line at drain time
  (`dropped store for request ... blocks evicted before drain`), so the drop
  ledger is visible without running at DEBUG;
- every drain re-logs the whole ledger as one greppable `key=value` line
  (`Lazy offload counters: admitted=... emitted=...`) when the counters
  changed, throttled to one line per 5s;
- connector `shutdown()` (invoked by vLLM's scheduler shutdown) calls
  `log_final_stats()`, which emits the exact final ledger
  (`Lazy offload final counters: ...`). Best-effort: `vllm serve` under
  SIGINT force-kills the engine core (abort mode) and can beat scheduler
  shutdown to it -- that is why the periodic line exists. A log reader
  should take the last line matching `Lazy offload (final )?counters:`.
