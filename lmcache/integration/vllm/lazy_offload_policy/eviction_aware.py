# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction-aware lazy offload policy.

Store operations are buffered and released when the GPU blocks holding their
data approach the free queue's eviction head. Pure policy: it decides,
``LazyOffloadManager`` acts. See
``docs/design/integration/vllm/lazy_offload_policy/eviction_aware.md``.
"""

# Standard
from collections.abc import Container, Iterator
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING
import bisect
import time

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    AdmissionProtection,
    BlockHashes,
    BufferedOpView,
    DrainSignals,
    LazyOffloadDrain,
    LookupFlush,
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import KVCacheBlock

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestMetadata,
    )

logger = init_logger(__name__)

# Caps the store work one drain triggers. ``_budget_reach`` reads it to make
# ops past the window due early enough that none is recycled unstored.
MAX_DRAIN_PER_STEP = 64

# Rank of an unblocked request whose front operations are held because they
# came due earlier: they are safe for now, so they go after every block about
# to be recycled.
# Sorts behind every real rank (free-queue positions, bounded by the queue
# length).
_HELD_RANK = 1 << 61

# Rank of a request due only because some of its ops were expedited (see
# ``hold_past`` and ``flush_for_lookup``) and their holds were released: no
# allocation reaches them before the next drain, so it goes after the held
# ones and is never pinned.
_EXPEDITED_RANK = _HELD_RANK + 1

# Rank base of a request due only because the budget would fall behind: no
# allocation reaches it before the next drain, so it goes after the expedited
# ones, in queue order, and is never pinned.
_AHEAD_RANK = _EXPEDITED_RANK + 1

_STATS_LOG_INTERVAL_S = 5.0  # Minimum seconds between ledger log lines.
_DROP_LOG_SAMPLE_REQUESTS = 8  # Dropped requests named in the drop line.


def _pending_item(request_id: str, ops: "list[PendingStoreOp]") -> PendingStoreItem:
    """Hand one request's buffered ops to the manager, in prefix order.

    Args:
        request_id: The request the ops belong to.
        ops: The ops, each paired in the item with its hash snapshot.

    Returns:
        The item carrying every op of ``ops``.
    """
    item = PendingStoreItem(request_id=request_id)
    item.metadatas.extend((op.store_metadata, op.block_hashes) for op in ops)
    return item


def _min_rank(ops: "list[PendingStoreOp]", ranks: dict[int, int]) -> int | None:
    """The most imminent rank any block of ``ops`` has.

    Args:
        ops: One request's buffered operations.
        ranks: Block id to its position from the free-queue head.

    Returns:
        The lowest rank among the blocks of ``ops`` found in ``ranks``, or
        None when none of them is there.
    """
    return min(
        (
            ranks[block_id]
            for op in ops
            for block_id in op.block_hashes
            if block_id in ranks
        ),
        default=None,
    )


def _in_window(op: "PendingStoreOp", window: "Container[int]") -> bool:
    """Whether a block of ``op`` is in ``window``.

    Args:
        op: A buffered operation.
        window: The block ids in reach. Only membership is read.

    Returns:
        True when any block the op covers is in ``window``.
    """
    return any(block_id in window for block_id in op.block_hashes)


def _overlaps(op: "PendingStoreOp", start: int, end: int) -> bool:
    """Whether ``op`` stores a token of ``[start, end)``.

    Args:
        op: A buffered operation.
        start: First token of the range.
        end: Token the range ends at.

    Returns:
        True when the op's token range intersects the range.
    """
    return op.store_metadata.op.start < end and op.store_metadata.op.end > start


def _due_segment(
    ops: "list[PendingStoreOp]", first_due: int, last_due: int
) -> tuple[int, int]:
    """The ops to store so that every due op is stored, as a slice.

    vLLM resumes a hybrid model only at a cached Mamba state (an anchor), so
    the ops up to the last anchor before ``first_due`` are still served from
    the GPU through it: the segment starts past that anchor. A segment that
    does not end at an anchor cannot be hit, so it runs on to the first
    anchor at or after ``last_due``. Without anchors (attention-only models)
    the segment is the front up to ``last_due``, which keeps every stored
    op's prefix stored. An op a finished request's hit covers is stored
    with the segment too: L1 evicts by store age, so an anchor stored ahead
    of its prefix ages first and its loss voids the whole prefix.

    Args:
        ops: One request's buffered operations, in prefix order.
        first_due: Index of the first due op.
        last_due: Index of the last due op, at least ``first_due``.

    Returns:
        ``(start, end)``: ``ops[start:end]`` is the segment.
    """
    start = next(
        (index + 1 for index in range(first_due - 1, -1, -1) if ops[index].has_anchor),
        0,
    )
    end = next(
        (index + 1 for index in range(last_due, len(ops)) if ops[index].has_anchor),
        last_due + 1,
    )
    return start, end


@dataclass
class PendingStoreOp:
    """A deferred store operation with the state needed to validate it.

    Attributes:
        store_metadata: The operation itself, handed back unchanged when it
            is emitted: the token range and the GPU blocks holding its KV.
        block_hashes: Prefix-cache hash of every block the range covers,
            snapshotted at admission. It never holds a None -- an unhashed
            operation is rejected rather than buffered -- so a later
            mismatch means the block was recycled and the data is gone.
        has_anchor: Whether the range holds a real block of an anchor group
            (an align-mode Mamba state), so vLLM can resume a hit there.
        held: Whether the manager pinned its blocks for a hold.
        expedited: Whether a finish or a lookup made it due (see
            :meth:`EvictionAwareStoreQueue.hold_past`); it stays due until
            emitted.
    """

    store_metadata: "LMCacheMPRequestMetadata"
    block_hashes: BlockHashes
    has_anchor: bool
    held: bool = False
    expedited: bool = False


@dataclass
class LazyOffloadCounters:
    """Cumulative policy counters for observability.

    The counts close as a ledger: ``admitted`` plus ``requeued`` equals the
    pending depth plus ``emitted`` plus every ``dropped_*`` counter.
    The ``bound_*`` counters stand beside the equation, not in it.

    Attributes:
        admitted: Operations taken into the buffer.
        emitted: Operations released to the manager for submission.
        dropped_evicted: Operations dropped because a covered block was
            recycled before they were emitted, plus the later operations of
            the same request that the loss made unreachable.
        rejected_prefix_broken: Operations refused at admission because the
            request's prefix chain was already broken.
        dropped_superseded: Buffered or requeued operations
            replaced by a later offer of the same tokens: a preempted
            request restarts and offers its tokens again.
        requeued: Emitted operations whose store failed, buffered again to
            be retried. A retried operation counts as emitted once more.
        bound_exceeded: Steps whose running requests allocated more blocks
            than the previous drain bounded. Nonzero means the danger depth
            is not a bound for this model, and buffered blocks may have
            been recycled unseen.
        bound_blocks: The previous drain's bound, summed over the steps it
            was checked against. Over ``bound_allocated_blocks`` it is the
            bound's slack: how many blocks it reserves per one allocated.
        bound_allocated_blocks: Blocks the running requests allocated,
            summed over the same steps.
        dropped_id_reuse: Buffered operations of a finished request
            discarded when a new request took over its id.
    """

    admitted: int = 0
    emitted: int = 0
    requeued: int = 0
    bound_exceeded: int = 0
    bound_blocks: int = 0
    bound_allocated_blocks: int = 0
    dropped_evicted: int = 0
    rejected_prefix_broken: int = 0
    dropped_superseded: int = 0
    dropped_id_reuse: int = 0


class EvictionAwareStoreQueue(OffloadPolicy):
    """Buffers store operations and releases them by eviction imminence.

    One rule decides when to store: an operation is due once one of its
    blocks sits within the *danger depth* of the free queue -- what the
    running requests, the resumed loads and one admission can allocate
    before the next drain. A due operation is emitted with the segment
    :func:`_due_segment` closes it into: its request's front, or, on a model
    with Mamba anchors, only the ops vLLM cannot serve through an earlier
    anchor. A due segment is emitted, or, when its request has a batch in
    flight or the drain budget is spent, held: the manager pins its blocks
    until a later drain emits it. When the
    window reaches the free-queue tail, every block freed before the next
    step is in reach, so every operation the drain does not emit is held
    too, and released unpinned by the next drain whose window stops short
    of the tail. A second admission in one step reaches past the window;
    the operations it would recycle are moved to the free-queue tail and
    stay buffered. A preempted request's operations stay buffered like any
    other: vLLM keeps the hashes of the blocks it freed, so they are victims
    in eviction order, until the restarted request offers the same tokens
    again. An engine with no requests never drains;
    operations whose blocks are evicted before they come due are dropped,
    never stored stale. Scheduler thread only.
    """

    def __init__(
        self,
        pool: "BlockPool",
        anchor_groups: frozenset[int],
    ) -> None:
        """Create an empty queue over ``pool``.

        Args:
            pool: The scheduler's GPU block pool. It is only ever read here:
                free-queue order and block hashes.
            anchor_groups: KV-cache groups whose real blocks are states vLLM
                resumes a hit at (align-mode Mamba); empty for a model
                without them, whose due ops are stored as front segments.
        """
        self._pool = pool
        self._anchor_groups = anchor_groups
        # Insertion order is admission order (a re-entering request goes to
        # the back), so the dict is also the drain's tie-break order.
        self._pending: dict[str, list[PendingStoreOp]] = {}
        # Prefix validity is policy; phase and batches are the manager's.
        self._broken_prefixes: set[str] = set()
        # Per request, the token its offers since it last (re)started reach:
        # a lost op starting there or later is one the request offers again.
        self._offered_end: dict[str, int] = {}
        # Requests whose held ops a squeezed drain pinned without them being
        # due: the next unsqueezed drain releases them rather than emitting.
        self._squeeze_held: set[str] = set()
        # The last drain's bound, which this drain's allocations are checked
        # against.
        self._next_step_blocks = 0
        self._now = 0.0  # This step's clock, set at the top of drain().
        self._counters = LazyOffloadCounters()
        self._last_logged = LazyOffloadCounters()
        self._last_stats_log_time = 0.0

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
    ) -> PendingStoreItem:
        """Buffer one store operation; see ``OffloadPolicy.add``.

        Args:
            meta: The store operation offered by the manager.
            block_hashes: Hash of every protectable block the operation
                covers, read at admission: the manager has already replaced
                hash-less blocks, which vLLM never evicts, by the null block.

        Returns:
            The buffered operations the offer superseded.
        """
        request_id = meta.request_id
        start, end = meta.op.start, meta.op.end
        self._offered_end[request_id] = max(self._offered_end.get(request_id, 0), end)
        superseded = PendingStoreItem(request_id=request_id)
        ops = self._pending.get(request_id, [])
        if any(_overlaps(op, start, end) for op in ops):
            kept = [op for op in ops if not _overlaps(op, start, end)]
            superseded.metadatas = [
                (op.store_metadata, op.block_hashes)
                for op in ops
                if _overlaps(op, start, end)
            ]
            self._counters.dropped_superseded += len(superseded.metadatas)
            self._pending[request_id] = kept
        if meta.request_id in self._broken_prefixes:
            self._counters.rejected_prefix_broken += 1
            # DEBUG, not INFO: the site that broke the chain already logged
            # the cause, and every later operation lands here.
            logger.debug(
                "Lazy offload: skipping store for request %s tokens [%d, %d): "
                "the request's prefix chain is already broken",
                meta.request_id,
                meta.op.start,
                meta.op.end,
            )
            if not self._pending.get(request_id, True):
                del self._pending[request_id]
            return superseded
        self._insert_sorted(PendingStoreItem(request_id, [(meta, block_hashes)]))
        self._counters.admitted += 1
        return superseded

    def protect(self, num_blocks: int) -> AdmissionProtection:
        """Protect the buffered ops an admission's allocation would recycle.

        The danger depth covers one admission, so the first admission of a
        step meets no buffered block; a second one reaches past the window.
        The walk runs until ``num_blocks`` blocks outside buffered ops are
        counted: the ops it meets move to the tail, so the allocation
        reaches that much further. They stay buffered, and the window meets
        them again once the head reaches the tail. Per request, the ops moved
        run up to the last op met, stopping at an op whose data is already
        gone, which the next drain drops.

        A walk that reaches the queue tail first means the allocation would
        recycle the tail as well, and with it any block freed before the
        allocation runs. So every buffered op not yet held is held instead:
        the pin may defer the admission, but loses nothing.
        Pinning a block a request still owns does not shrink the queue.

        Args:
            num_blocks: Upper bound on the blocks the allocation consumes.

        Returns:
            Per request, the ops to move to the tail or newly held.
        """
        # One owner tuple per op, not per block: an op can cover hundreds of
        # blocks.
        owners: dict[int, tuple[str, int]] = {}
        for request_id, ops in self._pending.items():
            for index, op in enumerate(ops):
                owners.update(dict.fromkeys(op.block_hashes, (request_id, index)))
        ranks, reaches_tail = self._free_queue_ranks(num_blocks, owners.keys())
        last_met: dict[str, int] = {}
        for block_id in ranks:
            owner = owners.get(block_id)
            if owner is not None:
                request_id, index = owner
                last_met[request_id] = max(last_met.get(request_id, -1), index)
        if reaches_tail and self._pending:
            # Blocks freed between this walk and the allocation, as running
            # requests free theirs ahead of a resumed load, join a tail the
            # walk already reached: every buffered op is in reach.
            last_met = {
                request_id: len(ops) - 1 for request_id, ops in self._pending.items()
            }
        protection = AdmissionProtection()
        for request_id, last in last_met.items():
            met: list[PendingStoreOp] = []
            for op in self._pending[request_id][: last + 1]:
                if not self._snapshot_intact(op):
                    break
                # Held ops are pinned, so the walk never meets them.
                if not op.held:
                    met.append(op)
            if not met:
                continue
            item = _pending_item(request_id, met)
            if reaches_tail:
                for op in met:
                    op.held = True
                protection.holds.append(item)
            else:
                protection.to_tail.append(item)
        return protection

    def hold_past(self, request_id: str, start: int) -> PendingStoreItem:
        """Hold and expedite the request's ops ending past ``start``; see
        ``OffloadPolicy.hold_past``.

        Held, they are due like any hold. They are expedited too, so a
        squeeze that releases the holds leaves them due. The ops are taken
        in token order up to the first one whose data is already gone, which
        the next drain drops with its suffix. The ops ending by ``start``
        back to the previous anchor are held too: the segment storing the
        held ones takes them along (see :func:`_due_segment`), and pinned
        now, while the request owns their blocks, they rejoin the free queue
        at the tail instead of the head.

        Args:
            request_id: The finishing request.
            start: Token vLLM's prefix cache serves the request up to.

        Returns:
            The ops newly held, in token order.
        """
        ops = self._pending.get(request_id, [])
        first = next(
            (index for index, op in enumerate(ops) if op.store_metadata.op.end > start),
            len(ops),
        )
        past: list[PendingStoreOp] = []
        for op in ops[self._segment_start(ops, first) :]:
            if not self._snapshot_intact(op):
                break
            op.expedited = True
            if not op.held:
                op.held = True
                past.append(op)
        return _pending_item(request_id, past)

    def flush_for_lookup(self, start: int, ends: dict[str, int]) -> LookupFlush:
        """Hold and expedite the ops a lookup needs; see
        ``OffloadPolicy.flush_for_lookup``.

        Per request, the ops ending past ``start`` and starting before its
        end are taken in token order, up to the first one whose data is
        already gone, which the next drain drops with its suffix. With
        anchors, the server's hit, like vLLM's, ends at a chunk carrying a
        state, so only the ops up to the last anchor op taken over all
        requests are flushed: storing ops past it would serve nothing. The
        flushed ops are held and expedited, as by :meth:`hold_past`, with
        the ops before them back to the previous anchor, which the segment
        storing them takes along.

        Args:
            start: Token vLLM's prefix cache serves the waiting request up
                to.
            ends: Request id -> token up to which its ops cover the waiting
                request's prefix.

        Returns:
            The flushed ops' ranges and the newly held ones.
        """
        taken: dict[str, list[PendingStoreOp]] = {}
        for request_id, end in ends.items():
            ops: list[PendingStoreOp] = []
            for op in self._pending.get(request_id, []):
                if op.store_metadata.op.end <= start:
                    continue
                if op.store_metadata.op.start >= end or not self._snapshot_intact(op):
                    break
                ops.append(op)
            if ops:
                taken[request_id] = ops
        if self._anchor_groups:
            hit_end = max(
                (
                    op.store_metadata.op.end
                    for ops in taken.values()
                    for op in ops
                    if op.has_anchor
                ),
                default=start,
            )
            taken = {
                request_id: kept
                for request_id, ops in taken.items()
                if (kept := [op for op in ops if op.store_metadata.op.start < hit_end])
            }
        flush = LookupFlush()
        for request_id, ops in taken.items():
            pending = self._pending[request_id]
            first = pending.index(ops[0])
            prefix = pending[self._segment_start(pending, first) : first]
            if not all(self._snapshot_intact(op) for op in prefix):
                prefix = []
            newly_held = [op for op in prefix + ops if not op.held]
            for op in prefix + ops:
                op.expedited = True
                op.held = True
            if newly_held:
                flush.holds.append(_pending_item(request_id, newly_held))
            flush.ends[request_id] = ops[-1].store_metadata.op.end
        return flush

    def is_flushing(self, request_id: str, start: int, end: int) -> bool:
        """Whether a held or expedited op overlapping ``[start, end)`` is
        buffered; see ``OffloadPolicy.is_flushing``.

        Args:
            request_id: A request whose ops a lookup flushed.
            start: First token of the range.
            end: Token the range ends at.

        Returns:
            True while such an op waits for a drain.
        """
        return any(
            (op.held or op.expedited)
            and op.store_metadata.op.start < end
            and op.store_metadata.op.end > start
            for op in self._pending.get(request_id, [])
        )

    def has_pending_request(self, request_id: str) -> bool:
        """Whether this request currently owns buffered operations.

        Args:
            request_id: The request id to query.

        Returns:
            True while at least one of its operations is buffered.
        """
        return request_id in self._pending

    def buffered_ops(self) -> list[BufferedOpView]:
        """Copy every buffered operation; see ``OffloadPolicy``.

        Returns:
            One view per buffered operation, requests in admission order
            (the drain's tie-break order), ops in token order.
        """
        return [
            BufferedOpView(
                request_id=request_id,
                start=op.store_metadata.op.start,
                end=op.store_metadata.op.end,
                block_ids=tuple(
                    tuple(group) for group in op.store_metadata.op.block_ids
                ),
                held=op.held,
                expedited=op.expedited,
            )
            for request_id, ops in self._pending.items()
            for op in ops
        ]

    def buffered_block_ids(self) -> set[int]:
        """Every GPU block a buffered operation covers.

        Returns:
            The block ids, possibly empty.
        """
        return set().union(
            *(op.block_hashes for ops in self._pending.values() for op in ops)
        )

    def restart_request(self, request_id: str) -> None:
        """Note that a preempted request restarts; see ``OffloadPolicy``.

        Its buffered operations stay. The restarted request offers its
        tokens again, so an earlier loss no longer breaks its chain, and an
        op it loses before offering that range again is one it offers again.

        Args:
            request_id: The preempted request.
        """
        self._broken_prefixes.discard(request_id)
        self._offered_end.pop(request_id, None)

    def discard_for_reuse(self, request_id: str) -> None:
        """Discard the buffered state of the id's previous holder.

        Args:
            request_id: The reused request id, whose previous holder's
                buffer and broken-chain marker are dropped.
        """
        dropped = self._pending.pop(request_id, [])
        self._broken_prefixes.discard(request_id)
        self._offered_end.pop(request_id, None)
        self._squeeze_held.discard(request_id)
        self._counters.dropped_id_reuse += len(dropped)

    def release_request(self, request_id: str) -> None:
        """Forget non-pending policy state after session teardown.

        Only the broken-chain marker is held outside the buffer, so this is
        what keeps finished request ids from accumulating.

        Args:
            request_id: The request whose session was torn down.
        """
        self._broken_prefixes.discard(request_id)
        self._offered_end.pop(request_id, None)

    def requeue_failed(self, item: PendingStoreItem) -> None:
        """Buffer a failed batch in token order among its request's ops.

        The manager moves the ops' blocks to the free-queue tail, so they
        come due again when the head reaches them, rather than being
        resubmitted at once into an L1 that just rejected them. A failed
        later segment goes behind the earlier ops that stayed buffered.

        Args:
            item: The failed batch's operations, in token order. The
                manager filters out orphaned batches first.
        """
        self._counters.requeued += len(item.metadatas)
        self._insert_sorted(item)

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Record the step's block pressure and release what it made due.

        Per request: the suffix from the first op whose blocks were recycled
        is dropped, then the ops with a block in the danger window are due,
        closed into the segment :func:`_due_segment` gives: the front up to
        the last due op, or, with Mamba anchors, the ops past the last
        anchor before the first due op through the first anchor at or after
        the last one. Requests in ``blocked_request_ids`` stay pending, since
        the worker holds one in-flight batch per request; their due segment
        is held instead, so a slow store cannot let it be recycled. Once
        unblocked, a request with held ops is due, held ops counting as due
        ones. Requests are served most imminent first until
        ``MAX_DRAIN_PER_STEP``; due ops past the budget are held. Expedited
        ops (see :meth:`hold_past`) whose holds a squeeze released are due
        too, served after the held ones, and wait unheld once the budget is
        spent. Past the window, ops come due as soon as the budget could not
        otherwise store them before allocations reach them (see
        :meth:`_budget_reach`); they take budget the window left over, after
        the held ones, and are never held.

        A window that reaches the free-queue tail also covers blocks not yet
        in the queue: vLLM frees blocks of running requests (align-mode
        Mamba drops older states) before the next step allocates, and they
        join a tail already in reach. So every op this drain leaves buffered
        is held, not just the due ones. If the next drain stops short of the
        tail, it releases those holds unpinned: their blocks then join the
        tail behind every block the window counts. If it reaches the tail
        too, the pins are what keeps the queue short, so the held ops are
        due like any other hold; otherwise a lasting squeeze would pin the
        whole pool without storing any of it.

        Args:
            signals: The manager's bounds on the blocks the running
                requests, the resumed loads and one admission allocate
                before the next drain, which set the danger depth, plus the
                finished and blocked request ids. Finished ids
                are not read by this policy: imminence alone decides,
                whether or not the request is still running.

        Returns:
            The stores to submit, the holds to pin and release, and the
            requests left with nothing buffered.
        """
        self._now = time.monotonic()
        self._counters.bound_blocks += self._next_step_blocks
        self._counters.bound_allocated_blocks += signals.running_blocks_allocated
        if signals.running_blocks_allocated > self._next_step_blocks:
            self._counters.bound_exceeded += 1
            logger.info(
                "Lazy offload: running requests allocated %d blocks, over the "
                "previous drain's bound of %d",
                signals.running_blocks_allocated,
                self._next_step_blocks,
            )
        self._next_step_blocks = signals.next_step_blocks
        if not self._pending:
            self._maybe_log_stats()
            return LazyOffloadDrain()

        # A second admission in one step allocates further, but the manager
        # protects buffered blocks from it right before it does. A resumed
        # load still unscheduled was protected for a step that has passed, so
        # its allocation counts here.
        depth = (
            signals.next_step_blocks
            + signals.resuming_blocks
            + signals.admission_blocks
        )
        # Depth zero makes nothing due and the queue is not walked; the loss
        # check below reads hashes, not ranks, and still runs. Built once:
        # both walks below read it, and nothing changes the buffer between.
        op_of_block = self._op_of_block()
        ranks, reaches_tail = self._free_queue_ranks(depth, op_of_block)
        drain = LazyOffloadDrain()
        ranks_ahead: dict[int, int] = {}
        if reaches_tail:
            # A squeeze hold bets the squeeze passes by the next drain. Lost,
            # the pins are what keeps the queue short, so they come due.
            self._squeeze_held.clear()
        else:
            # Pinned, these blocks were not walked; unpinned, they join the
            # tail, past the depth just counted.
            for request_id in self._squeeze_held:
                squeezed = self._pending.get(request_id, [])
                if any(op.held for op in squeezed):
                    for op in squeezed:
                        op.held = False
                    drain.releases.append(request_id)
            self._squeeze_held.clear()
            ranks_ahead = self._budget_reach(depth, op_of_block)
        dropped_ops = 0
        dropped_ids: list[str] = []
        # Due now, as (imminence rank, admission order, request id).
        candidates: list[tuple[int, int, str]] = []
        # Of those, the ones only the budget or an expedite made due: they go
        # if budget is left, unpinned, since no allocation reaches them before
        # the next drain.
        ahead_ids: set[str] = set()
        expedited_ids: set[str] = set()
        for order, request_id in enumerate(list(self._pending)):
            ops = self._pending[request_id]
            surviving = self._drop_evicted_suffix(request_id, ops)
            if len(surviving) != len(ops):
                dropped_ops += len(ops) - len(surviving)
                dropped_ids.append(request_id)
                if not surviving:
                    del self._pending[request_id]
                    drain.emptied_request_ids.append(request_id)
                    continue
                self._pending[request_id] = surviving
            if request_id in signals.blocked_request_ids:
                # Pin what an unblocked request would release now; the rest
                # stays where the window can see it, unless the window reached
                # the tail and so sees everything.
                if reaches_tail:
                    self._hold(drain, request_id, surviving)
                    self._squeeze_held.add(request_id)
                else:
                    start, end = self._segment(
                        surviving, ranks, held_due=False, expedited_due=False
                    )
                    self._hold(drain, request_id, surviving[start:end])
                continue
            if (imminent := _min_rank(surviving, ranks)) is not None:
                # TODO: free-queue pressure is the only retirement mechanism
                # modelled here. A sliding-window layer retires a block on a
                # token schedule instead, which is predictable but invisible
                # in the free queue, so such an op is not released when its
                # data is about to go.
                # Size the release per attention type: one due-predictor per
                # mechanism, emit up to the last index any of them marks due.
                candidates.append((imminent, order, request_id))
            elif (
                any(op.held for op in surviving)
                and request_id not in self._squeeze_held
            ):
                candidates.append((_HELD_RANK, order, request_id))
            elif any(op.expedited for op in surviving):
                candidates.append((_EXPEDITED_RANK, order, request_id))
                expedited_ids.add(request_id)
            elif (ahead := _min_rank(surviving, ranks_ahead)) is not None:
                candidates.append((_AHEAD_RANK + ahead, order, request_id))
                ahead_ids.add(request_id)
        if dropped_ops:
            # INFO, not DEBUG: each drop is a unit of cache-quality loss and
            # production rarely runs at DEBUG. One line per drain, so a burst
            # evicting a large queue cannot flood the scheduler hot path.
            logger.info(
                "Lazy offload: dropped %d store op(s), blocks evicted before "
                "drain (%s)",
                dropped_ops,
                ", ".join(dropped_ids[:_DROP_LOG_SAMPLE_REQUESTS]),
            )
        candidates.sort()
        ops_left = MAX_DRAIN_PER_STEP
        for _, _, request_id in candidates:
            ops = self._pending[request_id]
            is_ahead = request_id in ahead_ids or request_id in expedited_ids
            if is_ahead and ops_left <= 0:
                # Nothing about to be recycled made it due: it waits in place.
                continue
            # A squeeze hold is not due; only the window makes its ops due.
            start, end = self._segment(
                ops,
                ranks_ahead if is_ahead else ranks,
                held_due=request_id not in self._squeeze_held,
            )
            cut = start + min(end - start, max(ops_left, 0))
            emitted = ops[start:cut]
            remaining = ops[:start] + ops[cut:]
            if not is_ahead:
                # Over budget, yet the running requests may recycle these
                # before the next drain: pin them.
                self._hold(drain, request_id, ops[cut:end])
            ops_left -= len(emitted)
            if not emitted:
                continue
            self._counters.emitted += len(emitted)
            drain.items.append(_pending_item(request_id, emitted))
            if remaining:
                self._pending[request_id] = remaining
            else:
                del self._pending[request_id]
                drain.emptied_request_ids.append(request_id)
        if reaches_tail:
            for request_id, ops in self._pending.items():
                if request_id in signals.blocked_request_ids:
                    continue
                if not all(op.held for op in ops):
                    self._squeeze_held.add(request_id)
                    self._hold(drain, request_id, ops)
        self._maybe_log_stats()
        return drain

    def _free_queue_ranks(
        self, depth: int, pending_blocks: "Container[int]"
    ) -> tuple[dict[int, int], bool]:
        """The free-queue head an allocation can recycle, id -> eviction rank.

        A buffered block met in the walk is due, or protected, so it leaves
        the queue or moves to the tail, and every block behind it moves up:
        the walk runs until ``depth`` blocks outside buffered ops are
        counted.

        Args:
            depth: How many blocks the allocation can take from the queue
                head. Zero or less returns an empty mapping without
                touching the queue.
            pending_blocks: Every block a buffered op covers.

        Returns:
            Block id to its position from the head, lowest evicted first,
            and whether the walk reached the queue tail before counting
            ``depth`` blocks, so the allocation may take the whole queue.

        Raises:
            RuntimeError: If the fake head has no successor.
        """
        ranks: dict[int, int] = {}
        if depth <= 0:
            return ranks, False
        counted = 0
        for block in self._free_blocks():
            if counted >= depth:
                break
            ranks[block.block_id] = len(ranks)
            if block.block_id not in pending_blocks:
                counted += 1
        return ranks, counted < depth

    def _budget_reach(self, depth: int, op_of_block: dict[int, int]) -> dict[int, int]:
        """Buffered blocks past the danger window that the budget needs now.

        Allocations take at most ``depth`` blocks per drain and a drain
        stores at most ``MAX_DRAIN_PER_STEP`` ops, so the ops an allocation
        reaches within ``k`` drains must be stored within those ``k``
        drains. If the first ``m`` ops in queue order lie within ``k``
        drains, ``m - (k - 1) * MAX_DRAIN_PER_STEP`` of them must go in
        this one; past the window, the first ops up to the largest such
        count are due. A window holding no more than one drain stores, with
        nothing behind it outpacing the budget, needs none.

        Args:
            depth: The danger depth, blocks an allocation takes per drain.
            op_of_block: Every buffered block to the ``id`` of its op, as
                :meth:`_op_of_block` builds it.

        Returns:
            Block id past the window to its position from the head, one
            block per op the budget needs now; empty when it needs none.
        """
        budget = MAX_DRAIN_PER_STEP
        pending_ops = sum(len(ops) for ops in self._pending.values())
        # One drain stores them all: nothing past the window is needed.
        if depth <= 0 or pending_ops <= budget:
            return {}
        # Past this many drains the budget stores every op there is in time.
        horizon = depth * (-(-pending_ops // budget) + 1)
        # Each op once, first block met, as (position, block id, in window).
        met: list[tuple[int, int, bool]] = []
        seen: set[int] = set()
        need = 0
        counted = 0
        for position, block in enumerate(self._free_blocks()):
            if counted >= horizon:
                break
            op = op_of_block.get(block.block_id)
            if op is None:
                counted += 1
            elif op not in seen:
                seen.add(op)
                drains_ahead = counted // depth
                met.append((position, block.block_id, drains_ahead == 0))
                need = max(need, len(met) - drains_ahead * budget)
        return {
            block_id: rank for rank, block_id, in_window in met[:need] if not in_window
        }

    def _op_of_block(self) -> dict[int, int]:
        """Map every buffered block to the ``id`` of the op covering it.

        A block two ops cover maps to the later one, in buffer order.

        Returns:
            Block id to ``id(op)``, possibly empty.
        """
        op_of_block: dict[int, int] = {}
        for ops in self._pending.values():
            for op in ops:
                op_of_block.update(dict.fromkeys(op.block_hashes, id(op)))
        return op_of_block

    def _drop_evicted_suffix(
        self, request_id: str, ops: list[PendingStoreOp]
    ) -> list[PendingStoreOp]:
        """Drop ops from the first one whose data was lost; return survivors.

        A hash mismatch means the block was recycled; that op and every later
        op of the request go. The data is in neither vLLM's prefix cache nor
        L1, and both are read as prefixes, so the later ops are unreachable
        until the lost range is computed again; whoever computes it computes
        and offers them too. The request's further offers are rejected
        unless it offers the lost range again, which only a restarted
        (preempted) request does. The caller installs the survivors and
        reports the count.

        Args:
            request_id: The request the operations belong to, marked
                prefix-broken when anything is dropped.
            ops: Its buffered operations, in prefix order.

        Returns:
            The leading operations whose snapshots are still intact, which
            is ``ops`` itself when nothing was lost.
        """
        first_lost = next(
            (i for i, op in enumerate(ops) if not self._snapshot_intact(op)),
            len(ops),
        )
        if first_lost == len(ops):
            return ops
        self._counters.dropped_evicted += len(ops) - first_lost
        lost_start = ops[first_lost].store_metadata.op.start
        if lost_start < self._offered_end.get(request_id, 0):
            self._broken_prefixes.add(request_id)
        return ops[:first_lost]

    def _free_blocks(self) -> "Iterator[KVCacheBlock]":
        """Walk the free queue from its eviction head, one block at a time.

        Follows the queue's links rather than ``get_all_free_blocks()``,
        which materialises the whole queue every step.

        Yields:
            Each free block, the next to be recycled first.

        Raises:
            RuntimeError: If the fake head has no successor.
        """
        block = self._pool.free_block_queue.fake_free_list_head.next_free_block
        if block is None:
            raise RuntimeError("free_block_queue.fake_free_list_head has no successor")
        # The fake tail has no successor, which is what ends the walk.
        while block.next_free_block is not None:
            yield block
            block = block.next_free_block

    def _hold(
        self,
        drain: LazyOffloadDrain,
        request_id: str,
        ops: list[PendingStoreOp],
    ) -> None:
        """Hold every op of ``ops`` that is not held yet.

        Args:
            drain: The drain result; the newly held ops join its holds, for
                the manager to pin.
            request_id: The request the ops belong to.
            ops: Some of its buffered operations, in prefix order.
        """
        newly_held = [op for op in ops if not op.held]
        if not newly_held:
            return
        for op in newly_held:
            op.held = True
        drain.holds.append(_pending_item(request_id, newly_held))

    def _segment(
        self,
        ops: list[PendingStoreOp],
        window: "Container[int]",
        held_due: bool,
        expedited_due: bool = True,
    ) -> tuple[int, int]:
        """The due segment of one request's ops, as a slice.

        Args:
            ops: The request's buffered operations, in prefix order.
            window: The block ids in reach. Only membership is read.
            held_due: Whether held ops count as due.
            expedited_due: Whether expedited ops count as due; not when the
                segment is held, since an expedite never pins.

        Returns:
            ``(start, end)`` per :func:`_due_segment`, or ``(0, 0)`` when no
            op is due.
        """
        due = [
            index
            for index, op in enumerate(ops)
            if (held_due and op.held)
            or (expedited_due and op.expedited)
            or _in_window(op, window)
        ]
        if not due:
            return 0, 0
        return _due_segment(ops, due[0], due[-1])

    def _has_anchor(self, meta: "LMCacheMPRequestMetadata") -> bool:
        """Whether an op holds a real block of an anchor group.

        Args:
            meta: A store operation.

        Returns:
            True when one of its anchor-group blocks is not the null block.
        """
        null_block_id = self._pool.null_block.block_id
        return any(
            block_id != null_block_id
            for group_idx in self._anchor_groups
            for block_id in meta.op.block_ids[group_idx]
        )

    def _segment_start(self, ops: list[PendingStoreOp], first: int) -> int:
        """Index of the first op a segment containing ``ops[first]`` stores.

        Args:
            ops: One request's buffered operations, in prefix order.
            first: Index of the first op the caller needs stored; may be
                ``len(ops)`` when it needs none.

        Returns:
            The segment start per :func:`_due_segment`, or ``first`` when
            ``first`` is past the last op.
        """
        if first >= len(ops):
            return first
        return _due_segment(ops, first, first)[0]

    def _insert_sorted(self, item: PendingStoreItem) -> None:
        """Buffer ``item``'s ops, unheld, where their starts sort.

        An op overlapping a buffered one is dropped: the buffered one is
        the later offer of the same tokens (a preempted request restarted).

        Args:
            item: Ops of one request, in token order, with their snapshots.
        """
        ops = self._pending.setdefault(item.request_id, [])
        for meta, block_hashes in item.metadatas:
            if any(_overlaps(op, meta.op.start, meta.op.end) for op in ops):
                self._counters.dropped_superseded += 1
                continue
            index = bisect.bisect_right(
                [op.store_metadata.op.start for op in ops], meta.op.start
            )
            ops.insert(
                index,
                PendingStoreOp(
                    store_metadata=meta,
                    block_hashes=block_hashes,
                    has_anchor=self._has_anchor(meta),
                ),
            )

    def _snapshot_intact(self, op: PendingStoreOp) -> bool:
        """Whether every covered block still holds its admission-time hash.

        Args:
            op: The buffered operation to validate.

        Returns:
            False once any covered block was recycled, which means the
            operation's data is gone.
        """
        # TODO: an operation's blocks span every KV cache group, so on a
        # hybrid model the shortest-lived group decides for all layers: one
        # recycled sliding-window block drops the operation and the request's
        # whole tail, including full-attention layers whose data is still
        # live. Splitting the check needs per-group token ranges in
        # LoadStoreOp.
        # A plain loop over a local, not all() over a generator: every drain
        # checks every buffered block.
        blocks = self._pool.blocks
        for block_id, snapshot in op.block_hashes.items():
            if blocks[block_id].block_hash != snapshot:
                return False
        return True

    def log_final_stats(self) -> None:
        """Log the cumulative counters at shutdown as one INFO line.

        Best-effort: a force-killed engine may not reach it, which is why
        :meth:`drain` also logs the ledger periodically.
        """
        logger.info("Lazy offload final counters: %s", self._ledger_line())

    def _maybe_log_stats(self) -> None:
        """Log the counter ledger if it changed and the throttle allows."""
        if self._counters == self._last_logged:
            return
        if self._now - self._last_stats_log_time < _STATS_LOG_INTERVAL_S:
            return
        logger.info("Lazy offload counters: %s", self._ledger_line())
        self._last_logged = replace(self._counters)
        self._last_stats_log_time = self._now

    def _ledger_line(self) -> str:
        """Render the ledger as one greppable ``key=value`` line body.

        Ends with the pending depth, so the counts close as
        ``admitted + requeued == pending + emitted + every dropped_* counter``.

        Returns:
            The counter names and values, space separated.
        """
        fields = " ".join(
            f"{name}={value}" for name, value in asdict(self._counters).items()
        )
        pending = sum(len(ops) for ops in self._pending.values())
        return f"{fields} pending={pending}"
