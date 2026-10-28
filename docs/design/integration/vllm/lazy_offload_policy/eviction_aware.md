# `lazy_offload_policy/eviction_aware.py`: Eviction-Aware Store Queue

The buffering / protection mechanism this policy plugs into is described in
[lazy_offload.md](../lazy_offload.md). This document is the module's contract.

## Scope and non-scope

The module is **pure policy**: no vLLM imports at runtime, no I/O, no lock (it
runs on the scheduler thread). It decides *which* buffered store operations to
release *when*; `LazyOffloadManager` executes the integration side effects
(snapshots, pinning, submission) and returns explicit actions to the
connector. Reuse prediction is out of scope: the policy stores every admitted
operation whose blocks come under eviction pressure.

## Objects

- The **`BlockPool`** bound via the vLLM `bind_gpu_block_pool` hook is only
  ever read: the free queue's links from the eviction head (a block's
  position is its LRU rank, rank 0 = next victim) and block hashes. The walk
  runs on the scheduler's critical path once per step, so it follows the
  queue's links and stops at the danger depth instead of materialising the
  whole queue -- nothing at danger depth 0.
- **`PendingStoreOp`** -- one deferred store: opaque `store_metadata` (the
  ready `LMCacheMPRequestMetadata`), the covered blocks' hash snapshot taken
  at admission, whether it holds a real block of an anchor group
  (`has_anchor`, see the anchor rule), whether it is held, and whether a
  finish or lookup expedited it.
  Its request id and token range are not copied out: the buffer is keyed by
  request id and holds each request's ops in token order.
- **`EvictionAwareStoreQueue`** -- the policy object, one per connector.

## Lazy offload covers the blocks vLLM retains

Lazy offload stores a block at the latest moment before vLLM reuses it.
vLLM frees hash-less blocks to the head of the free queue, reused first,
and hashed blocks to the tail in LRU order (`BlockPool.free_blocks`,
vLLM `v1/core/block_pool.py:777-806`); an allocation takes blocks from the
head (`get_new_blocks`, `:684`), and a touch removes a block from the queue
(`touch`, `:766-767`).

- A **hashed block has slack**: it is reused only after the blocks ahead of
  it, so the policy tracks it as it moves toward the head and stores it
  before reuse.
- A **hash-less block has none**: the next allocation takes it. Pinning it
  to store it later would make vLLM take the next free block instead,
  evicting a hashed, still reusable block early: a reusable block traded
  for one vLLM does not intend to reuse. So lazy offload covers only hashed
  blocks, and the manager replaces a hash-less block by the null block when
  it buffers an op (`add_store_candidate`). It never enters a hash snapshot
  and never breaks a request's chain, which is defined over hashed blocks.
- The **hashed blocks are exactly those vLLM retains** past the step: it
  decides and hashes them in `allocate_slots` (`kv_cache_manager.py:607-616`)
  before the connector builds its metadata. Under sparse retention
  (`--prefix-cache-retention-interval 0`) the hash-less non-null blocks in
  a store op are the align-mode recurrent states it does not keep
  (`single_type_kv_cache_manager.py:1572-1630`); `remove_skipped_blocks`
  frees each two schedules after the step that wrote it (`:1653-1681`,
  called from `allocate_slots`, `kv_cache_manager.py:555`), in the same
  call that then allocates. Full attention blocks in an op are always
  hashed: an op covers whole computed blocks, and lazy offload requires
  prefix caching. Sliding-window blocks out of the window are already null.
- **Hash-less blocks ahead of a victim still count** for the window. They
  occupy free-queue positions and absorb allocations first, so the window
  walk counts every free block outside buffered ops toward the danger depth
  `D`, hash-less ones included.

The invariant (no missed store): L0 (vLLM's prefix cache) plus L1 covers
every block vLLM retains, that is, every hashed block. A hashed block
leaves L0 only through an allocation, and the rule below stores it before
an allocation can reach it.

### Known limits

This comes from the separate Mamba null-page commit bug (a recurrent chunk
whose last block is null commits null page 0 as its state):

- When a prompt's kept state lands mid-chunk (prompt length modulo the
  chunk in `1..block_size-1` with chunk = 2 blocks), the chunk ending at
  the hit point carries that state in its first block and null in its
  last; storing it hits the bug.

## The invariant, and the one rule built on it

> A buffered block is recycled only if an allocation pops it from the
> free-queue head.

Every way this vLLM version changes the free queue, and what it does to a
buffered block:

| change                      | who                                  | buffered block                         |
|-----------------------------|--------------------------------------|----------------------------------------|
| `popleft_n` (allocation)    | running requests, admissions, loads  | the only way one is lost               |
| `append_n` (free, cached)   | vLLM `free_blocks`, `unpin_blocks`   | joins the tail                         |
| `prepend_n` (free, no hash) | vLLM `free_blocks`                   | never buffered (nulled at admission); counted by the walk |
| `remove` (touch)            | prefix hit, our pin                  | leaves the queue                       |
| head return on unpin        | `LazyOffloadManager._unpin`          | stored blocks only, never a buffered one (step 4); not for `hold_past` or lookup-flush pins |
| tail move                   | `LazyOffloadManager._move_to_tail`   | joins the tail                         |

So nothing is lost if, before every allocation of `n` blocks, the first
`n` blocks of the queue hold no buffered block. The policy keeps that with
one rule -- **store an op when an allocation can reach it, and no
earlier**:

1. Each drain walks the queue from the head until it has counted `D` blocks
   outside buffered ops, `D` = running requests' bound + unscheduled
   resumed loads + one admission (`admission_blocks`). Every op met is due:
   emitted, or held (pinned) when its request has a batch in flight or the
   drain budget is spent. The budget stores at most `B` ops per drain
   (`MAX_DRAIN_PER_STEP`, 64), so the window also widens when it would fall
   behind: if the first `m` buffered ops in queue order lie within `k`
   drains' allocations (`k * D` blocks outside buffered ops),
   `m - (k - 1) * B` of them are due now. Those past `D` only take budget
   the window left over and are
   never held, since no allocation reaches them before the next drain.
2. A second admission in the same step reaches past `D`; `protect()` moves
   the ops it would reach to the tail, still buffered.
3. A walk that reaches the tail first (a squeezed drain) means blocks freed
   before the next step join a tail already in reach, so every op the drain
   does not emit is held. If the next drain stops short of the tail, it
   releases those holds, unpinned, onto the tail; if it reaches the tail
   too, the pins are what keeps the queue short, so the held ops come due
   like any other hold. A lasting squeeze otherwise pins the pool without
   storing any of it.
4. A failed store is buffered again and its blocks go to the tail.
5. A stored block returns to the head on unpin unless an op still buffers it.
6. The one exception to "no earlier": `hold_past` and `flush_for_lookup`
   (below) hold and expedite ops wherever their blocks sit, to store chunks
   eager offload would have stored that the GPU no longer serves. Expedited
   ops are served after held ones, closed into their segment, and once a
   squeeze releases their hold they only take budget left over, like the
   ops past `D`; the mark persists until a drain emits them.

The proof obligations are then: `D` is an upper bound on what can allocate
before the next drain (step 2 and the decision rule below), and no path
moves a buffered block toward the head (the table).
`admission_blocks` is derived from `max_model_len`, not tuned: an
admission's local hit plus the slots it allocates hold at most
`max_model_len + 1 + L` tokens, and splitting them between the hit and the
allocation costs a group at most one block (two for an align-mode Mamba
group's hit state blocks). On Qwen3.5-122B-A10B (`max_model_len` 262144,
`L` 3) it is 86 of 2551 blocks.

An earlier version stored far earlier than this rule asks: ops protected
from an admission stayed *due* at the tail and were paced out, and a
squeezed drain made every op due. On a replay of the c100 run (simulated),
58% of emitted ops were stored in such a squeeze flush and 37-41% from the
tail, against 0.2-1% from the window; with the rule above every emitted op
is in the window or prefix-closed behind one.

Storing a request's last full block at once was tried and dropped. With
MTP, vLLM drops the last matched block of every local prefix hit, so that
block never serves a later request from GPU and deferring it looked like
pure recompute. But its presence is what lets the block before it hit:
stored early, it lost its tail position, was recycled first, and every
continuation's local hit ended one block sooner. On the c100 run (measured,
two runs each) local hits fell from 33.4-34.6% to 25.0-25.2% of prompt
tokens and TTFT p50 rose from 31.0-38.0 s to 42.9-60.6 s.

The chunk holding that last block is still one the GPU cannot serve: with
a chunk of two blocks, the next turn's local hit stops mid-chunk, and only
an L1 copy saves recomputing the other half. So when a request finishes,
`hold_past` holds every buffered op ending past vLLM's hit over the
request's own chain, before vLLM frees the blocks. The pin keeps them out
of the free queue and the receipt frees them to the tail as vLLM would
have. The ops the hit covers back to the previous anchor op (see the
anchor rule) are held with them and stored in the same segment: a state
stored ahead of the prefix its hit serves would age first in L1, and its
eviction would void that prefix.

A request's own hit cannot foresee the next turn's: on Qwen3.8-27B (MTP 1,
chunk of two blocks) a turn of 204,431 tokens kept its only state at 128
blocks, so `hold_past` held nothing, and the next turn matched those 128
attention blocks, lost one to the MTP drop and found no state at or below
127: its local hit was 0, its lookup missed the buffered chunks, and it
recomputed 202,752 tokens that the eager arm hit in L1. So the store is
also triggered by the request that needs it. Before submitting a lookup,
the connector calls `LazyOffloadManager.flush_for_lookup` with vLLM's local
hit. The manager finds, by the chunk-ending block hashes of the request's
chain (an index filled as ops are offered, so the cost is linear in the
request's chunks), which other requests buffer chunks of its prefix past
that hit, and `flush_for_lookup(start, ends)` holds and expedites those
ops, with the ops before them back to the previous anchor op, as
`hold_past` does. With anchors only the ops up to the last anchor op are
flushed, and none without one: the server's hit, like vLLM's, ends at a
chunk carrying a state.
While a flushed op is
buffered held or expedited (`is_flushing`) or in an open batch, the lookup
reports "not yet known" (`None`), which vLLM answers by skipping the
request for the step, without allocating for it or blocking the requests
behind it. Once the receipts settle, the lookup goes out and hits L1. The
check runs once per tracker (again after a preemption); a lookup already submitted (eager
prefetch) is not held back.

Neither pin may change vLLM's eviction order, since lazy offload decides
when a chunk is stored, never whether vLLM keeps it. Their receipts free
the blocks to the tail, unlike the drain's, which return a block pinned
while free to the head (step 4 of the per-step protocol below): a drain
only pins blocks its window met near the head, but these pins reach
anywhere. An align-mode Mamba group frees a request's prompt-end state
block before
`request_finished`, so `hold_past` pins it while it is free; returned to
the head, the next allocation evicted it, and the next turn's local hit
fell back to an earlier state (measured offline on the real scheduler:
from 19 blocks to 15, and to 10 when a lookup flush did the same to the
waiting request's own hit-point state).

## Stores before an allocation reaches the op

Lazy offload stores as late as possible, and four paths store an op before
an allocation can reach it. Each is required by something the window alone
cannot see:

| path | why it cannot wait for the window |
|------|-----------------------------------|
| `_budget_reach` | A drain stores at most `MAX_DRAIN_PER_STEP` ops. If more ops lie within `k` drains' allocations than `k` drains can store, waiting for the window would force holds (pins), which take blocks from vLLM. It picks the latest drain that still meets every deadline under that cap, so it is "as late as possible" for a bounded drain, not an early store. |
| squeeze hold-all | A window that reaches the tail also reaches blocks freed before the next step (a running request's older align-mode states), which no walk sees. Every op is held, not stored; the holds come due only if the next drain is squeezed too, because then the pins are what keeps the queue short and only storing frees them. |
| `hold_past` (finish) | Ops past what vLLM's hit over the finished request serves are not served from the GPU, which under MTP includes the chunk holding the last cached block, since the hit drops it. They are stored as eager offload would. |
| `flush_for_lookup` | A waiting request whose local hit stops short of chunks other requests still buffer would recompute them: they are neither served by vLLM nor in L1. |

## Per-step protocol (policy-caller obligations)

`LazyOffloadManager` is the production caller; the connector only forwards
lifecycle events to it.

1. Route each `GetStoreMetadata` result to `add(meta, block_hashes)`
   instead of the step metadata. The queue takes custody or drops the
   operation itself. `block_hashes` holds only hashed blocks: the manager
   has replaced each hash-less block by the null block (see "Lazy offload
   covers the blocks vLLM retains"), and the null block, which hybrid
   models put at positions that hold no data, is never recycled. One case
   is dropped here and logged: a **broken prefix**, where an earlier chunk
   of the request was dropped, so this one would be unreachable on
   retrieval.
2. Once per step call `drain(signals)` with that step's `DrainSignals`.
   Pending ops of finished requests wait out their eviction clock -- that is
   the point of lazy offload. Every drain revalidates every pending request's
   hash snapshots, including those of requests blocked by a submitted batch.
   A blocked request is never emitted -- the worker accepts one store future
   per request -- but the ops it would release now (its due segment)
   are returned in
   `LazyOffloadDrain.holds`: the manager pins those blocks so they cannot be
   recycled while the batch is in flight, and once the receipt arrives the
   held ops are due and their batch takes the pins over. Its other ops stay
   buffered and unpinned: the window is a bound, so they cannot be recycled
   before a later drain sees them. Without holds, a slow store let the
   blocks behind it go: on Qwen3.5-122B-A10B at N=150 with
   `prefix-cache-retention-interval` 4240, 49% of admitted ops were dropped,
   91% of them while their request was blocked. Holding only the front
   segment then still lost 27%, all to allocations the estimated window
   missed, so holds covered every op of a blocked request until the window
   became a bound and admissions and resumed loads got their own
   protection. Holding everything held 4843 of 5790 ops (84%), and each
   was stored on unblock, at risk or not, which made the hold, not
   eviction, decide when most ops were stored.
   The danger window does not cover allocations by requests that are not
   yet running: an admission with a long prefix hit allocates its whole
   prefix in one step, and vLLM evicts while allocating, before the step's
   drain. vLLM asks the connector for the external hit right before that
   allocation, so `get_num_new_matched_tokens` reports the admission to
   `LazyOffloadManager.on_admission`, which bounds its tokens: hit + one
   chunk + draft lookahead, or the hit alone for an asynchronous load,
   whose rest `on_loads_finished` bounds when the load completes. Per KV
   group the blocks are those the tokens span, plus a partial block at the
   start, one for a partial hit's copy-on-write block or a Mamba
   checkpoint, and `L` Mamba speculative state blocks (vLLM reserves one
   per draft token on a request's first allocation in align mode). An
   align-mode Mamba group is the exception: it nulls every block the
   tokens span but the state block, and `MambaManager` caps an allocation
   at `1 + extra + L` blocks, so that is its count whatever the hit's
   length. The `L` only applies to a first allocation: a resumed load's
   admission already allocated for its hit, so `on_loads_finished` counts
   `1 + extra` per Mamba group. The local prefix hit adds its own block
   count: vLLM touches those blocks before allocating, which takes the free
   ones out of the queue and brings every block behind them closer to the
   head. An align-mode Mamba hit is null blocks but its last, so it adds at
   most one, and an external hit one more:
   `allocate_external_computed_blocks` gives it a state block before the
   capped allocation. Leaving that block out lost 6 of 5285 ops in a c150
   run. Counting Mamba groups by the tokens spanned made a 100k-token hit
   on Qwen3.5-122B-A10B (3 Mamba groups of 4) count about 116 blocks for
   at most about 47, and every admission past the free blocks outside
   buffered ops is squeezed.
   The drain keeps `admission_blocks` of the head clear, so a step's first
   admission meets no buffered block. `protect(num_blocks)` covers the
   rest: it walks the free queue until it has counted that many blocks
   outside buffered ops. The ops met move to the free-queue **tail**
   (`touch` then `free_blocks`, hash intact) and stay plainly buffered:
   the window meets them again when the head reaches the tail. Without
   the reserve almost every admission walked into buffered ops: on a
   closed-loop replay of the c100 workload (simulated, 3 seeds), 18-20
   squeezed admissions per run against 3-6 with it. The moved ops are not
   pinned: a pin shrinks the free queue the admission allocates from, and
   when the rest no longer fits vLLM refuses the admission and every
   waiting request behind it for that step. Only when the walk reaches the
   tail before counting `num_blocks` -- the allocation would take the tail
   too -- is every buffered op held (pinned) instead: deferring the
   admission is cheaper than losing the op and, with it, the request's
   prefix chain. Every op, not only
   the ones met: a block freed between the walk and the allocation joins
   a tail the allocation already reaches. A resumed load allocates after
   the running requests, and those free blocks first -- align-mode Mamba
   frees a running request's older state blocks, hash intact, on each
   step. Pinning a block its request still owns costs no free block.
   A preempted request's ops are victims in eviction order like any
   other: vLLM freed its hashed blocks to the free-queue tail with their
   hashes (step 7).
3. For every item in `LazyOffloadDrain.items` (already ordered): pin
   (`touch`) its blocks, coalesce each request's released ops into one store
   op, register the submitted batch, and put it into this
   step's connector metadata. `emptied_request_ids` is only a buffer
   transition -- teardown additionally requires the registry to say the
   request is finished with no submitted batch.
4. On a store-completion receipt, complete the batch and unpin. A block
   the drain pinned while it sat in the free queue goes back to the queue
   **head**, in order: `free_blocks` would append it as most recently used,
   while eviction order had it next, so every lazy store would otherwise
   refresh the GPU prefix cache's LRU. A block a request still owns is left
   to vLLM, and so is one another request's op still buffers
   (`buffered_block_ids()`): requests sharing a prefix share its blocks, and
   at the head the block would land inside the danger window after the
   drain walked the queue, so the next allocation would recycle it and drop
   that request's op and its whole suffix. At the tail it waits for the
   window like any buffered block. End the session only when the registry
   says the request is finished and the policy has nothing pending.
5. On `request_finished`, call `hold_past(id, hit)` with vLLM's hit over
   the request's chain and pin what it returns, while the request still
   owns those blocks; on its receipt they go to the tail, also a block
   vLLM already freed. Record `FINISHED` in the registry; end immediately
   only under the same predicate, otherwise a later drain or receipt applies
   it.
6. Before a waiting request's lookup, `flush_for_lookup(start, ends)` with
   its local hit and the requests buffering chunks of its prefix past it;
   pin the returned holds (released to the tail, like step 5's), and
   defer the lookup while `is_flushing` or an open batch covers a flushed
   range.
7. On preemption tracker reset, call `restart_request(id)`: buffered and
   held ops and an already submitted batch all stay the request's own;
   only its broken-chain marker is cleared, since the restart computes the
   lost range again. The restarted request offers its tokens again from
   where its hit ends, and `add()` drops the buffered ops each offer
   overlaps (`dropped_superseded`); the manager unpins those that were
   held. `requeue_failed` drops an op that overlaps a newer offer the same
   way. A buffered op lost before the restart
   offers its range again is dropped without breaking the chain. An abort
   is **not** a drop: its buffered ops remain storable until drained.
8. On a failed receipt, mark the in-flight batch failed and wait for the
   rest of its receipts: another worker may still be reading its blocks.
   Once the batch settles, unpin it; a failed batch that is not orphaned
   goes back to the policy with `requeue_failed(item)`, in token order
   among the request's buffered ops, and its blocks move to the free-queue
   tail.
   The data is still on the GPU, so the request's prefix chain stays whole.
   The batch failed because a worker failed, the server was unhealthy, or
   L1 had no room for one of its chunks; the retry waits for the window
   like any buffered op rather than being resubmitted into the L1 that just
   refused it. Keys already resident count as stored, so retrying a partly
   stored batch is safe. An orphaned batch is never retried: a different
   request now uses the id (step 9). After three failed batches in a row
   the manager gives up instead: `mark_store_failed` drops the request's
   ops (`dropped_failed_store`) and breaks its prefix chain.
9. On request-id reuse, detect the `FINISHED` predecessor in the registry and
   call `discard_for_reuse(id)`. With a submitted batch outstanding, that
   batch is marked orphaned and the id-keyed session spans both generations,
   ending once through the successor's lifecycle. A failed orphaned batch
   cannot be retried: the worker keeps one store per request id and the
   successor's ops are a different token sequence under that id, so the
   retry could only go out after the successor's session, which ends the
   id's L1 session for both. (vLLM's HTTP layer
   randomizes external ids, but direct engine callers and
   `VLLM_DISABLE_REQUEST_ID_RANDOMIZATION=1` reach this path.)

**Prerequisite for 1**: the connector must record vLLM's prefix-cache hit in
the tracker even when the LMCache lookup misses. A lazy follower over a hot
APC-shared prefix always misses the lookup (the predecessor's ops are
buffered, not stored); without the vllm-hit share it never reaches `add` and
stores nothing at all. The recording is mode-independent by design: in eager
mode an APC-hit request whose lookup misses now stores its full prefix at once
instead of accumulating it over decode steps, and `cached_token_stats` reports
the true vLLM hit instead of 0.

## Losses, and which can be repaired

The invariant (L0 plus L1 covers every block vLLM retains) has three
places where an op can go unstored. Each is either repaired or provably
loses nothing servable:

- **A failed batch** (a worker failed, the server was unhealthy, or L1 had
  no room for one of its chunks) is requeued and stored again when it comes
  due; its blocks were pinned, so the data is intact. A preempted request's
  batch is its own and is retried too (protocol step 7). Only an
  *orphaned* batch, whose finished request's id a new request took over, is
  not retried (step 9): the worker keeps one store per id, and the
  successor's ops are another token sequence under that id.
- **A hash mismatch at emission** (the manager re-reads hashes before it
  pins). For this policy it is unreachable: `drain()` checked the same
  hashes synchronously and nothing allocates in between. For FIFO, which
  does not walk the queue, it is the only check, and a mismatch means the
  block was already recycled: the data is gone from the GPU, so there is
  nothing to store.
- **An evicted suffix.** When op `k`'s block is recycled before it comes
  due, its data is in neither tier. Both vLLM and LMCache match prefixes,
  so ops after `k` cannot be hit until `k` is computed again, and the
  request that computes it computes and offers them too. Dropping them
  loses nothing a hit could read. A nonzero `dropped_evicted` means an
  allocation escaped the bound (see Observability).

## Decision rule

- **Running-request bound** = `DrainSignals.next_step_blocks`, a bound,
  not an estimate, on the blocks the running requests can allocate before
  the next drain. With `c` a request's tokens computed after this step and
  `L` the draft lookahead, the next step schedules it at most
  `min(remaining, B)` tokens while prefilling or `1 + L` while decoding,
  and all requests together at most `B` (`max_num_scheduled_tokens`,
  defaulting to `max_num_batched_tokens`); its slots run `L` past what it
  schedules. Per KV group, the first new block of a request costs the
  tokens that fill the blocks its table holds, less `L` (at least 1); each
  further block costs `bs` tokens. The table length, tracked per group from
  the scheduler's block ids, already covers the block the last step's
  lookahead reached, which `ceil(c / bs)` would count again. Two vLLM
  rules for align-mode Mamba hybrids tighten that. A prefill chunk short
  of the prompt's end ends on the hash grid (`_mamba_block_aligned_split`,
  when a block fits in a chunk), so a prefilling crossing costs up to the
  next grid point, or the rest of the prompt if that is less. A Mamba
  group takes one state block per step, plus a checkpoint block if the
  spec has them and a copy-on-write block after a partial hit
  (`1 + partial hit + checkpoint`), whatever the chunk covers, and ignores
  `L`. The connector derives both from the KV-cache config
  (`get_prefill_alignment`, `get_mamba_step_extra_blocks`). Without them,
  every prefilling request at a block boundary counted 2 blocks for 1
  token per group, and on Qwen3.5-122B-A10B at N=104 the bound came out
  3.83x the measured allocation. The bound is the fractional knapsack over
  every crossing with budget `B`: crossings sorted by tokens per block,
  taken whole while they fit, and the one the budget ends in rounded up to
  its full blocks. It ignores that a further crossing needs its request's
  first one, which can only overcount, so it is at least what any schedule
  allocates; the rounding costs at most one crossing per group, on a step
  `B` binds. On a replay of that run (simulated), the table length and the
  token budget bring the bound from 1.47x to 1.01x the allocation, with no
  step over it (measured with crossings paid whole, before the rounded-up
  one). It is the token budget that keeps it tight: summing each request's
  own reach instead counts a full chunk for every prefilling request,
  several times what one step can schedule. Spec-decode rejections only
  lower `c`. Free-queue *position* alone is never a trigger -- an idle
  engine never drains.
- The danger depth `D` adds `admission_blocks` and `resuming_blocks` to
  that bound, so the first admission of a step finds the head clear. The
  running-request bound
  itself excludes the two allocations the scheduler makes for requests
  that are not yet running, which the manager protects right before they
  happen (step 2): admissions, via `get_num_new_matched_tokens`, and
  asynchronous loads resuming, via `update_connector_output`'s
  `finished_recving` -- vLLM schedules a resumed load without asking for a
  hit again. vLLM allocates for the running requests before either, so the
  resume notice protects `next_step_blocks` plus the resumed requests'
  chunks. vLLM schedules the resumed request only when its chunk fits,
  which with a full cache can be steps later; meanwhile the head moves
  past the blocks the notice moved to the tail. Until a step schedules
  it, each drain adds its chunk to the danger depth (`resuming_blocks`,
  kept out of the running-request bound). Protecting only at the notice
  lost 18 ops in one event on Qwen3.5-122B-A10B, N=150: one op's three
  Mamba state blocks were recycled by a new allocation while no admission
  exceeded its estimate and no step its bound. An admission with an
  asynchronous load allocates for its hit alone. The EMA-and-burst-floor
  estimate this replaced lost 31 of 5790 ops on Qwen3.5-122B-A10B, N=150,
  most of them to steps that allocated more than it predicted.
- An op is **due** when the window walk below meets any covered block. Blocks
  not in the free queue (in use / resurrected) are not at risk.
- Emitting a segment pins its blocks out of the free queue, moving every
  block behind them toward the head. The drain accounts for this
  shift by walking until it has counted `D` blocks outside
  buffered ops: every buffered block met is due, so it leaves the queue
  and the blocks behind it are what the next step can reach. Due ops past
  `MAX_DRAIN_PER_STEP` are held (pinned), and later drains emit them
  first. A walk that reaches the tail before counting the danger depth
  (a squeezed drain) means the next step may take the whole queue, and
  every block freed before it allocates. Every op the drain does not emit
  is then held, due or not; the holds of ops that were not due are
  released by the first drain whose walk stops short of the tail, and
  their blocks join the tail behind everything that walk counted. In a
  c150 benchmark run the ops lost before squeezes held were: only Mamba
  groups, requests still running, and every loss during a burst of
  squeezed walks. Making every op due there instead stored the whole
  buffer in one burst.
- **No deadline**: eviction pressure alone decides when an op is stored;
  how long it has waited does not. A time bound would store ops that no
  allocation is about to reach.
- **Prefix closure** (attention-only models; see the anchor rule for
  hybrids): a due op releases the request's ops from the front
  through the last due one; a data-loss drop (hash mismatch) drops from the
  first lost op through the tail and blacklists the request's later chunks;
  the held ops among those dropped go in `LazyOffloadDrain.dropped_holds`,
  for the manager to unpin; the intact stored prefix stays pending and
  stays valuable.
- **Anchor segments** (align-mode Mamba hybrids): vLLM resumes a hit at `p`
  only with the attention KV of `[0, p)` and the Mamba state block at `p`
  (the *anchor*), so chunks up to a cached anchor `p'` are still a GPU hit
  and front closure only copied them from L0 into L1 (0.10-0.16 of L1 bytes
  on a hybrid run). An op with a real block in a Mamba group
  (`anchor_groups`) is an anchor op; with `f` the first due op, the segment
  starts past the last anchor op before `f`, whose blocks are all outside
  the window, and ends at the first anchor op at or after the last due op
  (a segment not ending at an anchor cannot be hit), else at the last due
  op. The ops before it stay buffered and come due on their own blocks: the
  rule changes when they are stored, not whether. A suffix store is
  servable because the server looks up from vLLM's hit (`key.start`) and
  chunk keys hash from token zero. Attention-only models have no anchors and
  keep front closure.

  ```
  ops      0    1*   2    3    4*        * = anchor op
  window                       ##        chunk 4's state is due
  emitted            [2 .. 4]            0..1 served from L0 via 1*
  later         ##                       chunk 1's state is due
  emitted  [0 .. 1]
  ```

  Holds are per op (`PendingStoreOp.held`), not a front prefix, and a
  requeued op is inserted where its start sorts.
- Cross-request drain order = min due rank ascending; `MAX_DRAIN_PER_STEP`
  bounds the per-step D2H burst in operations. It is a constant, not a
  setting: rule 1 widens the window by it, so ops come due early enough
  that none is recycled unstored. The cap may split a request's due
  segment but only ever emits a front slice of it, so within-request prefix
  order is preserved. Due ops past the cap are not lost: they are held for
  the next drain, which emits them first.
- **Idle consequences**: receipts travel in worker metadata, which flows on
  every step, including steps that schedule no tokens. Those steps drain as
  well, which a squeezed admission needs: its pins can leave no waiting
  request room to run, and with nothing running only storing the pinned ops
  frees the blocks; a squeezed drain's holds come due for the same reason
  once the next drain is squeezed too (rule 3). A benchmark run
  (Qwen3.5-122B-A10B, N=150) deadlocked there after 19 minutes when
  zero-token steps returned no actions. Pins, in-flight
  sessions, and finished requests whose ops never come due all settle on the
  next activity -- nothing leaks permanently. An engine that is not stepping
  runs no drains at all.

## Scheduler-path complexity

A drain walks the free queue to the danger depth and re-checks every pending
request's hash snapshots: cost proportional to the pressure window plus the
buffered blocks, both small -- pending depth is bounded by concurrent
requests, and each check is a dict lookup and a comparison.

Request lifecycle is not policy state. The manager's registry owns request
phase and submitted batches; the policy receives blocked request ids as a
drain input and retains only prefix validity, a consequence of its own store
decisions. `release_request`, `restart_request`, and `discard_for_reuse` clear
that non-pending state at manager-defined lifecycle boundaries, so
completed request ids do not accumulate.

## Observability

Counters are cumulative and surface in the scheduler process log, not in
vLLM's `get_kv_connector_stats` plumbing (that hook is polled worker-side,
where the policy does not live). `bound_exceeded` counts steps whose
running requests allocated more blocks than the previous drain's
`next_step_blocks`, which the bound says cannot happen: nonzero means it is
wrong for this model or vLLM version. `bound_blocks` over
`bound_allocated_blocks` is the bound's overcount, how many blocks it
reserves per one allocated. `dropped_evicted` is the quality sensor --
data lost before we drained, meaning an allocation escaped both the
running-request bound and the admission protection, so nonzero means a
bound is wrong for this model or vLLM version. `requeued` counts stores
L1 rejected and the policy retried.

Two log hooks:

- A drain that dropped ops logs one aggregate INFO line naming up to 8 of the
  requests, and a drain whose counters changed re-logs the whole ledger as one
  greppable `key=value` line, throttled to one per 5 s. `pending` is the
  custody depth at the same instant, which closes the line as an equation:

  ```
  admitted + requeued == pending + emitted + dropped_evicted
                         + dropped_superseded + dropped_id_reuse
                         + dropped_failed_store
  ```

  `requeued` counts emitted ops whose store failed and came back to be
  retried; each retry counts in `emitted` again.

  `rejected_prefix_broken` stays out (those ops are
  turned away before `admitted`), and the `bound_*` counters are weights
  beside the equation, not terms of it. `emitted` means handed to the manager, not
  submitted: the manager re-reads the hashes before pinning and can still
  drop an op there. That drop is a WARNING line and no counter, and for this
  policy it is unreachable, since `drain()` checked the same hashes
  synchronously a moment earlier.
- Connector `shutdown()` calls `log_final_stats()`, which emits the exact
  final ledger. Best-effort: `vllm serve` under SIGINT can beat scheduler
  shutdown to it -- that is why the periodic line exists. A log reader takes
  the last line matching `Lazy offload (final )?counters:`.
