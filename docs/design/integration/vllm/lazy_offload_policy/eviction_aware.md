# `lazy_offload_policy/eviction_aware.py`: Eviction-Aware Store Queue

Implements gates 1 and 3 of the store decision defined in
[lazy_offload_decision_model.md](../lazy_offload_decision_model.md); the
buffering / protection mechanism it plugs into is described in
[lazy_offload.md](../lazy_offload.md). This document is the module's contract.

## Scope and non-scope

The module is **pure policy**: no vLLM imports at runtime, no I/O, no lock (it
runs on the scheduler thread). It decides *which* buffered store operations to
release *when*; `LazyOffloadManager` executes the integration side effects
(snapshots, pinning, submission) and returns explicit actions to the connector.
Gate 2 (reuse prediction) is out of scope — phase 1 stores every admitted op
whose blocks come under eviction pressure.

## Objects

- **`BlockPoolReader`** (protocol) — read-only pool view:
  `free_queue_block_ids()` (a *lazy* iterator over the free queue from the
  eviction head; a block's position in it is its LRU rank, rank 0 = next
  victim), `is_free(block_id)` (O(1) queue membership — vLLM keeps exactly
  the unreferenced blocks in the queue, so the reference count answers it),
  and `block_hash(block_id)`. Production impl `GPUBlockPoolView` wraps the
  `BlockPool` bound via the vLLM `bind_gpu_block_pool` hook; both must never
  mutate pool state.

  Laziness is not an optimisation detail the policy may ignore: this walk is
  on the scheduler's critical path once per step, and reading the whole
  queue is O(free blocks) — tens of thousands on a pool sized to fill the
  GPU. `collect_due` consumes the iterator through `_FreeQueueWindow`, which
  opens at `danger_depth` and widens only by the blocks an emission has
  *already* pinned out of the queue (see "Pin cascade" below), so a step
  reads the ranks its decisions compare. It reads nothing at all at danger
  depth 0.

  The depth deliberately does **not** scale with `max_drain_per_step`. A
  bound of `danger_depth + max_drain_per_step × largest pending op` is
  sound, but it charges every step for a full-budget drain: measured on an
  agentic replay at the default cap of 64, the mean drain emitted 0.2 ops
  while the bound sized the read for 64, and the read — with the request
  validation it pulls in behind it — became the policy's dominant cost late
  in a run. The budget bounds the D2H burst; it is not a statement about
  ranks.

  `is_free` exists for the same accounting. Whether pinning a block shifts
  the queue is a property of the pool, not of how far this step happened to
  read: asking the window instead would miss a pin deeper than the window
  and stall the widening that would have revealed it.
- **`PendingStoreOp`** — one deferred store: opaque `store_metadata` (the
  ready `LMCacheMPRequestMetadata`), the covered blocks' hash snapshot taken
  at admission, `prefix_start_tokens` / `prefix_end_tokens` (the op's token
  range; the start detects deduplication holes in the pending list), and
  `cache_salt` (part of the op's content identity).
- **`EvictionAwareStoreQueue`** — the policy object, one per connector.

## Per-step protocol (policy-caller obligations)

`LazyOffloadManager` is the production caller that fulfills this contract; the
connector only forwards lifecycle events to the manager.

1. Route each `GetStoreMetadata` result to `admit(op)` instead of the step
   metadata. Handle the outcome:
   - `ADMITTED` → nothing now. The op is in the policy's custody; while the
     request's known prefix is below `min_prefix_tokens` that custody is a
     gate-3 *holding pen* outside the pending machine (see "Gate 3" below),
     indistinguishable to the caller.
   - `REJECTED_UNHASHED_BLOCK` → **skip and warn** (a hash-less block's later
     eviction is undetectable: evicted-and-reallocated also reads `None`).
     The tracker has already advanced past the skipped range by the time the
     op reaches `admit`, so the request's later chunks are unreachable; the
     queue blacklists the request and rejects them as prefix-broken. With
     plain prefix caching (enforced at connector init) chunk-aligned ranges
     never cover unhashed blocks, but hybrid-attention models (sliding
     window, mamba) can place hash-less null blocks in block tables.

     Measured on `google/gemma-3-270m-it` (18 layers, 5 sliding-window
     layers of 512 tokens for every full-attention layer, so vLLM builds
     six kernel groups). The case that reaches it is not the long request
     itself — its blocks are in the window as each chunk is buffered — but a
     request whose prefix comes back from vLLM's *own* prefix cache:
     `SlidingWindowManager.find_longest_cache_hit` prepends
     `block_pool.null_block` for every out-of-window position, and those
     positions hold no KV for the sliding-window layers. The eager path has
     no admission step and stores them: on a 2166-token prompt replayed
     against an empty LMCache, 7 of the prefix's 8 chunks came back with
     different bytes under the same content-addressed key (the one that
     matched is the chunk still inside the attention window). This is what
     the rejection avoids, and why it is a skip rather than a best-effort
     store. Both counters are exercised by layer-1 scenario S18.
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

     **Range equality is required, and a capped step budget breaks it.** An
     op covers the range one step made known, so the same tokens produce
     different ops depending on how the prefill was chunked. Measured with
     the same 1965-token prompt sent twice: with `max_num_batched_tokens`
     capped at 512 the first request admits four per-step ops
     (512/1024/1536/1792) while the repeat prefix-cache-hits the whole
     prompt in one step and admits a single op over the whole range —
     `deduplicated` stays 0 and the content is buffered twice, under two
     different chunkings. Uncapped (one step per prefill), the two match and
     the repeat deduplicates. So the queue is bounded by the distinct
     (range, content) pairs resident, not by the request count: still a
     bound that does not grow with load, but the constant is the number of
     distinct chunkings of a hot prefix, not 1.

     A consequence for the doomed-cover check above: on vLLM it is defensive
     rather than load-bearing. `add()` runs before `collect_due()` in a step,
     so an op that becomes doomed in a step is dropped in that same step
     unless its request holds an in-flight batch — and the follower whose op
     could hit a doomed cover has to share its exact range, which (per the
     paragraph above) means an uncapped step budget, where a request has a
     single op and so never holds a batch in flight with siblings pending.
     The two conditions pull against each other; the branch is covered at
     layer 0 (`test_lazy_offload_eviction_aware.py`) and was not reachable on
     hardware.
2. Once per step, call `observe_step(gross_blocks_allocated,
   est_next_step_blocks, allocated_block_ids)` and then
   `collect_due(in_flight_request_ids, finished_request_ids)`. The controller
   obtains all three sets of ids from scheduler output and its request
   registry. Finished ids drive one gate-3 obligation: a finished request's
   prefix can no longer grow, so a chain still held below the break-even
   length is dropped by that drain (its request appears in
   `emptied_requests`); *pending* ops of finished requests are untouched --
   waiting out their eviction clock is the point of lazy offload. The queue keeps a
   block-to-request reverse index and revalidates only requests touched by
   allocations or represented in the bounded free-queue snapshot. Requests
   blocked by a submitted batch retain pending validation until a receipt.
3. For every op in `DrainResult.to_store` (already ordered): pin (`touch`)
   its blocks, **coalesce each request's released ops into one store op**,
   register the submitted batch and its epoch in the controller, and put it
   into this step's connector metadata. `dropped_*` lists need no action
   beyond accounting. `emptied_requests` is only a buffer transition; the
   controller may end those sessions only when its registry says they are
   finished with no submitted batch.
4. On a store-completion receipt, the controller completes the submitted
   batch and unpins with `free_blocks(prepend=True)` (a stored block has a
   copy below the GPU, so among free blocks it should die first). It ends the
   session only when the request registry says the current request is
   finished and the policy reports no pending operations. The policy has no
   receipt or in-flight lifecycle hook.
5. On `request_finished`, the controller records `FINISHED` in the request
   registry. It ends immediately only when the policy has no pending
   operations and the registry has no submitted batch; otherwise drain or
   receipt processing performs the same predicate later.
6. On preemption tracker reset, the controller advances the epoch and calls
   `drop_request(id)`. This discards buffered operations and prefix-validity
   state only. An already submitted batch remains in the registry and blocks
   new emission until its receipt. An abort is **not** a drop: its buffered
   operations remain storable until drained or evicted.
7. For a failed receipt, the controller first compares the submitted batch
   epoch with the current request epoch. A current-epoch failure calls
   `mark_store_failed(id)`, which drops held-back operations and marks the
   prefix broken. An old-epoch failure does not enter the policy because
   operations admitted after reset or reuse do not depend on it. Both paths
   still complete the receipt and unpin the old batch.
8. On request-id reuse, the controller detects the `FINISHED` predecessor in
   its registry, advances the epoch, and calls `discard_for_reuse(id)`. With
   no submitted batch it releases the predecessor session immediately. With
   one submitted batch, the id-keyed session spans both epochs and ends once
   through the successor's lifecycle; the predecessor receipt only removes
   the block. This prevents predecessor state from conflating with successor
   operations or authorizing teardown while the successor is live. vLLM's
   HTTP input processor normally appends eight random characters to external
   ids, but direct engine callers and deployments with
   `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` can exercise this path.

**Prerequisite for 1 and the dedup path**: the connector must record vLLM's
prefix-cache hit in the tracker even when the LMCache lookup misses. In lazy
mode a follower over a hot APC-shared prefix always misses the lookup (the
predecessor's ops are buffered, not stored); without the vllm-hit share,
`GetStoreMetadata` stages under one chunk and the follower never reaches
`admit` — deduplication is dead code for followers, and a dropped
predecessor op is never re-buffered while APC keeps hitting. The recording
is mode-independent by design: in eager mode, an APC-hit request whose
lookup misses (predecessor's store in flight, or data evicted from LMCache)
now issues a store covering its full prefix at once instead of accumulating it
over decode steps. That backfills the under-store the old behavior left
when LMCache had really evicted the data, at the cost of duplicate stores
in the in-flight window (eager has no client-side dedup; content-addressed
keys make them idempotent server-side). It also makes the
`cached_token_stats` reported through `kv_transfer_params` show the true
vLLM hit instead of 0 on a lookup miss.

## Decision rule

- **Danger depth** = `ceil(max(EMA(gross allocation/step), next-step
  feedforward) × horizon_steps)`; below half a block over the horizon it is 0
  (a decayed EMA must not hold the depth at 1 forever). An idle engine never
  drains under pressure — free-queue *position* alone is never a trigger
  (that would be the inverted-gate-1 anti-pattern, decision model §6). The
  opt-in backlog and idle drains below emit by *age*, which is not position
  either.
- An op is **due** when any covered block's rank < danger depth. Blocks not
  in the free queue (in use / resurrected) are not at risk.
- **Horizon calibration.** The default is 2.5 scheduler steps. A fine sweep
  over 2.0–8.0 on two opposing Qwen3-8B/H200 workloads selected it as the
  measured compromise: three 120-request hot/cold runs at 2.5 completed in
  27.0–27.1 s with 0.952–0.957 cache coverage and three lower-tier eviction
  cycles, while 2.0 took 31.6–32.2 s and 4.0 took 36.2–36.3 s. On the
  no-hot-set GSM8K workload, 2.5 retained 0.945–0.961 coverage versus 0.961
  once the horizon reached 3.0–4.0. The value is a calibrated default, not a
  universal optimum: increase it when eviction loss is more important than
  filtering, and decrease it when lower-tier write or eviction pressure is
  the limiting cost.
- **Pin-cascade shift**: emitting a segment pins its blocks out of the free
  queue, moving every block behind them toward the head before the next
  step's allocation runs. The shift is the number of **unique emitted
  blocks that were in the free-queue snapshot**: an in-use block does not
  leave the queue, and a block shared by multiple emitted ops leaves it only
  on the first touch. Within one `collect_due` call, each later candidate is
  therefore checked against `danger depth + free blocks removed so far in
  this call`; without this, a candidate teleported into the danger window by
  an earlier emission loses its tail to the next allocation before the next
  drain can see it (observed as `dropped_evicted` under back-to-back drains).
  The first emission still requires a plain danger-depth hit, so this never
  opens the gate on an idle system; dropped (unpinned) segments do not extend
  the shift.
- **Prefix closure** (amendment A1): a due op releases the request's ops from
  the front through the last due one; a data-loss drop (hash mismatch) drops
  from the first lost op through the tail and blacklists the request's later
  chunks; the intact stored prefix stays pending and stays valuable.
- **Gate 3, at admission**: while a request's known prefix is below
  `min_prefix_tokens` (the offline break-even length; 0 disables), its ops
  are held in a side pen outside the pending machine -- unindexed, so the
  per-step validation and free-queue walk never pay for them. This placement
  is the decision model's (§5: gate 3 is static and decidable early) and it
  is what the emission-side variant measurably was not: rejecting at drain
  time left every sub-break-even op a queue resident until its eviction due
  date, roughly doubling per-step validation on a mixed-length replay. The
  chunk that lifts the prefix past the threshold promotes the whole held
  chain into the pending machine. Held ops are invisible to the per-step
  loss check, so promotion validates the chain's snapshots once; a block
  lost during the wait kills the whole chain (the intact front is still
  below break-even and the break stops it from ever growing past it) and
  the promoting chunk is rejected prefix-broken. Promotion skips the
  deduplication check on purpose: held ops already entered the admission
  ledger, and deduplicating one away would leave that entry matched by
  neither a pending op nor a drop counter -- the cost is a rare duplicate
  store, never corruption. A request that finishes below the threshold has
  its held chain dropped by the next drain (`rejected_short_prefix`). Held
  and pending are mutually exclusive per request: nothing emits before
  promotion, and every chunk after promotion ends past the threshold.
  An emission-time check remains as a backstop for the one path back below
  the threshold: eviction truncating a promoted chain. When such a request
  comes due, all its ops are dropped (the due front is dying, which breaks
  the chain for the rest).
- A due segment is cut at the first deduplication hole before emission
  (the batch must coalesce into one contiguous token range); the request
  keeps its due-rank urgency, and the post-hole ops follow in a later
  batch once the front run's receipt arrives.
- Cross-request drain order = min due rank ascending; `max_drain_per_step`
  bounds the per-step D2H burst. The cap may split a request's due segment,
  but only ever emits a front slice of it, so within-request prefix order is
  preserved and the remainder stays pending.
- **Sizing the cap.** It bounds emissions per step while a prefilling request
  *admits* one op per step, and a request with a batch in flight is skipped
  entirely until its receipt arrives (one more step). So the cap has to sit
  above the concurrent prefill admission rate, or the queue cannot work off
  a backlog and buffered ops are lost to eviction instead of stored.
  Measured on a 448-block pool with a 512-token step budget, one 4-op
  request buffered ahead of five prefilling fillers: at the default 64 the
  workload emitted 21 of 24 admitted ops, dropped 1 and left none pending;
  at 1 it emitted 11 of 26, dropped 6 and left 9 pending at shutdown, and
  the buffered request stored its first two ops while losing the other two
  (prefix closure held — the replay retrieved exactly the surviving 1024 of
  1792 tokens). A cap near 1 is a steady-state loss setting, not a
  burst-shaping one.

  There is no static validation for this: the break-even depends on how
  many requests prefill concurrently, which the policy learns only at
  runtime. The sensor is therefore a runtime one. `DrainResult.ops_held_back`
  reports what a drain found due and did not emit (a lower bound: candidates
  the loop never reached are not counted, their due-ness being unevaluated),
  `throttled_drains` counts the drains that held anything back, and the
  pending store logs one WARNING per process once `throttled_drains` and
  `dropped_evicted` are *both* nonzero — the pair that separates a cap
  merely delaying a burst from one below the workload's admission rate.
  Neither symptom alone warns: ops lost without the cap binding is ordinary
  pressure, and a cap that binds without loss is the knob doing its job.

  The pair is read cumulatively, not within one drain. The two are causally
  linked but not simultaneous: the cap builds a backlog on the steps where
  it binds, and the backlog dies on whatever later step reallocates its
  blocks. Requiring both in the same drain turns the warning into a
  coincidence — a measured run at a cap of 1 held ops back on 8 drains and
  lost 5 ops to eviction without the two ever landing on the same step, so
  nothing was reported.
- **Pin cascade.** Emitting a segment pins its blocks out of the free queue,
  which moves every block behind them toward the head by that many positions
  before the next step's allocation runs. Each candidate is therefore tested
  against `danger_depth` extended by the blocks this drain has already
  pinned, so a request an emission teleports into the danger window drains
  now instead of losing the race. The first emission still needs a plain
  `danger_depth` hit — an idle system never starts draining — which is what
  keeps the extension from being a way to drain early on its own.

  The threshold and the read grow together, alternating: emit, count the
  pins that left the queue, widen the window to the new threshold, and let
  the newly revealed blocks name more candidates. This terminates because
  each round either emits (and `max_drain_per_step` is finite) or finds
  nothing due. Only pins that *were* in the queue count, and a shared block
  counts once.
- **Backlog cap (`max_pending_ops`)**: the danger depth is a forecast, and
  no forecast built from the preceding steps can see a single admission
  that consumes thousands of blocks at once. A request whose prefix comes
  back from a lower tier is exactly that: vLLM allocates blocks for the
  whole external hit in one step, so the step that would have to be
  predicted *is* the step that pays for the prediction. Measured on the
  agentic replay below, 60–80% of all operations lost to eviction were
  destroyed within 1.5 s of a retrieve, against an 11% baseline for a
  random instant — a 6–7× enrichment, and the dominant loss mode.

  What cannot be forecast can still be bounded. Above the cap the oldest
  pending operations are emitted regardless of their free-queue rank, at
  `max_drain_per_step` per step and only down to the cap (the cap bounds
  the backlog; it is not an instruction to empty it). Requests are taken
  in admission order, each contributing the contiguous front run of its
  surviving operations, so prefix closure, the deduplication-hole cut, the
  loss check and the one-batch-per-request constraint hold exactly as they
  do for a pressure-driven emission; a request with a batch in flight, or
  one this drain already emitted for, is skipped. `0` (the default) leaves
  the backlog unbounded and the policy behaves exactly as it did before
  the cap existed.

  Age is the ordering because prefix closure already forces it *within* a
  request — a batch must start at the front of its pending list — and a
  request's front operations are both its oldest and the ones whose blocks
  reached the free queue first, so age is the exposure proxy that costs
  nothing to compute. Targeting exposure directly (asking `is_free` of
  every pending block, or reading the queue past the danger depth to rank
  them) would buy a sharper choice of *which* request goes first while
  paying per-step for a decision the closure rule has already mostly made.

  The cap trades the wait's filtering for bounded loss. The wait only
  filters content that is *never* evicted from the GPU while the engine
  runs — anything evicted later is stored either way, just later — so on
  the replay the filtering it bought was the 9–22 operations still pending
  at shutdown, against 114–140 lost to eviction. A backlog deep enough to
  lose operations is deeper than the workload can defend, which is what
  makes `dropped_evicted` the sizing sensor: raise the cap while it stays
  near zero, lower it while it does not. `backlog_emitted` reports how
  many stores the cap timed rather than the forecast; its share of
  `emitted` rising towards 1 means the cap has taken over the timing
  entirely, which is eager offload with extra steps.

- **Idle drain (`idle_drain_max_ops`)**: the pressure trigger has a phase
  problem it cannot fix alone. It times an emission to the moment the
  operation's blocks are about to be reallocated, and the allocator is
  busiest exactly when a prefill burst runs — so deferred copies are
  submitted in phase with the burst they waited out, competing with it for
  the step. A step whose allocation rate is at or below
  `idle_threshold_blocks` is the opposite moment: the drain emits up to
  `idle_drain_max_ops` of the oldest operations, taken exactly as the
  backlog drain takes them (admission order, contiguous front run, loss
  check, economy backstop, one batch per request), so the backlog is
  worked off in the gaps between bursts instead of inside them. `0` (the
  default) disables it.

  The rate is `max(EMA, next-step feedforward)`: the feedforward vetoes
  the first step of a burst before the EMA has seen it, and the EMA
  vetoes the trailing steps after the feedforward has fallen. Decode-only
  traffic allocates about (running requests / tokens per block) blocks
  per step, so the default threshold of 1.0 separates decode-only steps
  from prefill at typical concurrency.

  The trade is the backlog cap's: an idle emission gives up the wait's
  remaining filtering, since content evicted after the emission is stored
  either way. The sensors mirror it — `idle_emitted` is the share of
  `emitted` the idle path timed rather than the forecast, and
  `idle_drain_steps` counts the drains in which it emitted at all.

- **Block volume cap (`max_drain_blocks_per_step`)**: `max_drain_per_step`
  bounds a drain in operations, but a deferred backlog coalesces into one
  contiguous copy per batch, so an op-count cap alone lets one step submit
  an arbitrarily long prefix as a single D2H burst — a burst shape eager
  offload never produces, storing chunk by chunk as the prefill runs. The
  block cap bounds the same drain in blocks. One budget serves all three
  emission paths (pressure, backlog, idle), and the bound is soft at the
  boundary: the operation that crosses it still emits — progress must not
  depend on an operation fitting under the cap — the overshoot is charged,
  and everything after it waits for the next step. `0` (the default)
  leaves the volume unbounded. What it holds back reports through the
  same `ops_held_back` / `throttled_drains` sensors as the op-count cap.

- **Idle consequences**: receipts travel in worker metadata, which only
  flows on steps that schedule tokens. If the engine goes idle with a
  batch in flight, its pins and its request's session stay held until the
  next non-empty step delivers the receipt; finished requests whose ops
  never come due likewise hold their sessions open. Both resolve on the
  next activity — nothing leaks permanently, by design. An engine that is
  not stepping runs no drains at all; the idle drain above runs *within* a
  low-allocation step, so it never changes this.

## Adaptive degradation

Deferral has two separable effects. It can *filter* stores -- operations
die or shrink while pending because a later request covers them or their
content never needs L1 at all -- and it *re-times* the stores that do
happen, moving them from "while the request runs" to "when the blocks
approach eviction". Under L1 churn the re-timing is a pure cost: the
free-queue pressure that triggers a deferred emission is the same
pressure the allocator is under, so the emission's pins collide with
allocation bursts by construction. Whether deferral is worth that cost
depends entirely on whether it is filtering: measured on an
agent-shaped workload deferral filters nothing (lazy and eager store
the same bytes) and degrading to immediate emission recovers the whole
gap, while on a hot/cold document workload deferral withholds the hot
set's useless stores and that filtering is precisely what keeps the
cold set alive in L1 -- degrading there destroys the win and, worse,
the extra volume creates the very churn that a churn-only signal reads
as justification. No fixed residence threshold separates the two cases.

The policy therefore enforces one invariant -- **degrading may change
the timing of stores, never their volume** -- and, because the volume a
regime *would* produce is a counterfactual no passive signal can see,
it measures both sides by briefly running them:

- **Signal**: the caller feeds `observe_l1_pressure()` snapshots of the
  server's cumulative evicted-bytes counter and L1 capacity (see
  `docs/design/v1/multiprocess/l1_pressure_stats.md`). Rates are read
  off a sliding window of these snapshots -- never a per-sample EMA:
  eviction arrives in watermark bursts tens of seconds apart, and
  per-sample smoothing decays between bursts and flaps across any
  threshold. Residence is `capacity / windowed rate`, infinite while
  nothing evicts. Emission volume is the policy's own emitted-block
  ledger, snapshotted on the same heartbeat, so trials and probes need
  no server-side counterfactual.
- **Churn gate**: residence below `degrade_l1_residence_secs` says the
  re-timing cost is being paid and opens a *trial*; it never degrades
  by itself. `0` (the default) disables the controller entirely.
- **Loss gate**: the windowed residence estimate needs a couple of burst
  cycles to cross, and at churn onset that latency is paid by the
  pending backlog -- the first eviction wave harvests exactly the
  oldest deferred stores. The policy's own loss ledger is the faster
  and sharper signal: deferral losing a material share of its windowed
  intake to eviction (`dropped_evicted` reaching the neutrality margin
  of windowed admissions, over one trial-length window) also opens a
  trial. Incidental drops stay silent: on workloads where the losses
  are the tail-release economy's expected cost (a few percent of
  intake), the share never reaches the material line. The trigger may
  be eager because a trial bounds the cost of a false alarm.
- **Trial**: a bounded window of immediate emission. At its end the
  trial's emitted-block rate is compared against the deferred baseline
  (the trailing window before the trial): within the neutrality factor
  it commits to DEGRADED -- deferral was only re-timing -- and beyond
  it the trial reverts and enters a cooldown, because the volume jump
  means deferral was filtering stores out.
- **Recovery**: only through a *probe* -- a bounded deferred window,
  after which the controller returns to NORMAL when the probe's
  emission rate drops below the degraded baseline by the neutrality
  factor. Probes run periodically, and residence recovering past the
  hysteresis factor arms one early (subject to a minimum retry
  spacing, so a failed probe is respected as evidence). The residence
  estimate alone never lifts a committed degradation: bursts spacing
  out past the rate window read as infinite residence -- a lull is
  indistinguishable from genuine recovery -- and lifting on that
  estimate hands the re-deferred backlog to the next burst. Immediate
  emission also forfeits the filtering window, so filtering value
  returning is invisible from inside the regime; the probe measures
  both questions at once.
- **DEGRADED semantics** (also during a trial): every drain call emits
  all pending operations in admission order -- the backlog-drain walk
  with an unbounded allowance -- subject to the same validation, prefix
  closure, dedup-hole cut, one-in-flight batch per request, and the
  shared per-step budget. Admission gates are unchanged. The effect is
  emission on the first step after admission: the store happens while
  the request is still running, so its blocks are not yet in the free
  queue and the pins that protect the copy are inert bookkeeping.
- **Constants**: all controller constants are properties of the
  measurement, not workload tunables -- the rate window spans at least
  two eviction burst cycles, the trial/probe length spans several store
  cycles, the neutrality factor covers the sampling noise of comparing
  two short windows, and the probe interval and revert cooldown bound
  the counterfactual-measurement duty cycle to under ~10%.
- **Counters**: `degraded_emitted`/`degraded_drain_steps` (immediate
  emissions and the drains that made them), `degrade_transitions`
  (behavior flips between deferred and immediate), and the decision
  ledger `degrade_trials`, `degrade_commits`, `degrade_reverts`,
  `degrade_probes`, `degrade_probe_recoveries`. Reverts ticking once
  per cooldown is the signature of a workload whose deferral filters;
  a commit with no probe recoveries is one whose deferral only
  re-times.

## Scheduler-path complexity

A dedicated pending-operation owner maintains the primary per-request lists
together with their content covers, admission order, block-to-request reverse
index, per-request block reference counts, and bounded operation-size multiset.
Admission, replacement, and departure update primary storage and every derived
index through one atomic API. A production drain therefore walks only the
bounded free-queue window and the requests represented in that window or
touched by this step's allocations. Its cost is proportional to the pressure
window and drain cap, not total pending queue depth. The pure-policy tests
retain an `allocated_block_ids=None` compatibility path that performs a full
validation pass.

Request lifecycle is not policy state. The controller registry owns request
phase, epoch, and submitted batches; the policy receives blocked request ids as
an input to each drain. The queue retains only prefix validity because that is
a consequence of its store decisions. `release_request`, `drop_request`, and
`discard_for_reuse` clear that non-pending state at controller-defined epoch
boundaries, so completed request ids do not accumulate.

## Observability

`stats()` returns cumulative counters. `dropped_evicted` is the gate-1 sensor
(drop rate: data lost before we drained — lower the horizon is too tight, or
the backlog cap is too loose); `emitted / admitted` is store precision's
denominator; `backlog_emitted` is the share of `emitted` that the backlog cap
timed rather than the eviction forecast, `idle_emitted` the share the idle
drain timed (`idle_drain_steps` counts the drains in which it fired);
`rejected_short_prefix` audits gate 3. Tests: `tests/v1/test_lazy_offload_eviction_aware.py` (pure, no vLLM).

Four counters measure the decision loop's own cost rather than any op's
fate: `drain_steps`, `free_queue_blocks_read`, `requests_validated` and
`blocks_validated`. Divided by `drain_steps` they give the mean free-queue
depth a step walks and the mean number of block-hash comparisons it makes —
the quantities that turn a policy which saves prefill time into one that
spends more decode time than it saves. They are excluded from the ledger's
change test (`LazyOffloadCounters.decisions()`) because they advance on
every drain: gating the log line on them would never let it go quiet.

The counters surface in the scheduler process log, not in vLLM's
`get_kv_connector_stats` plumbing (that hook is polled worker-side, where the
policy does not live). Three hooks, all on the pending-store facade:

- each drain that dropped ops logs one aggregate INFO line **per cause**
  (`dropped N store op(s): blocks evicted before drain (req (prefix P), ...)`
  and `dropped N store op(s): request prefix below the break-even length
  (...)`, each naming at most 8 ops and counting the rest), so both kinds of
  cache-quality loss are attributable to a request without running at DEBUG,
  while a burst that evicts a large queue cannot flood the scheduler hot
  path; per-op detail logs at DEBUG. The later chunks a broken request keeps
  producing are rejected at admission and log at DEBUG only: their cause was
  already reported, and one broken request produces many of them;
- every drain re-logs the whole ledger as one greppable `key=value` line
  (`Lazy offload counters: admitted=... emitted=... pending=N held=M`) when
  the counters changed, throttled to one line per 5s. `pending` and `held`
  are the two custody depths at the same instant (the pending machine and
  the gate-3 holding pen), which close the line as an equation over
  exactly six outcome counters:

  ```
  admitted == pending + held + emitted + dropped_evicted
              + rejected_short_prefix + dropped_on_request_drop
              + dropped_failed_store + dropped_id_reuse
  ```

  so a reader can separate an operation still waiting for pressure from one
  that left the queue without incrementing any outcome counter. The set is
  neither "every `dropped_*` counter" nor "every drop and reject":
  `rejected_short_prefix` belongs in it although it is not named `dropped_*`
  (gate 3 discards ops that were admitted into the pen), while `rejected_unhashed`,
  `rejected_prefix_broken` and `deduplicated` must stay out of it — those ops
  are turned away at admission and never counted in `admitted`. Summing by
  name instead of by this list makes the equation fail the moment gate 3
  fires. `throttled_drains` stays out for a different reason: it counts
  drains, not operations, so it belongs alongside the step count rather
  than in an equation over ops;
- connector `shutdown()` (invoked by vLLM's scheduler shutdown) calls
  `log_final_stats()`, which emits the exact final ledger
  (`Lazy offload final counters: ...`). Best-effort: `vllm serve` under
  SIGINT force-kills the engine core (abort mode) and can beat scheduler
  shutdown to it -- that is why the periodic line exists. A log reader
  should take the last line matching `Lazy offload (final )?counters:`.
