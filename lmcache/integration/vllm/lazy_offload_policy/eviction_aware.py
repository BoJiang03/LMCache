# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction-aware lazy offload policy.

Store operations are buffered and released when the GPU blocks holding their
data approach the free queue's eviction head, or when they hit the deferral
deadline. Pure policy: it decides, ``LazyOffloadManager`` acts. See
``docs/design/integration/vllm/lazy_offload_policy/eviction_aware.md``.
"""

# Standard
from collections.abc import Container
from dataclasses import asdict, dataclass, replace
from typing import TYPE_CHECKING, cast
import time

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    AdmissionProtection,
    BlockHashes,
    ConfigValue,
    DrainSignals,
    LazyOffloadDrain,
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestMetadata,
    )

logger = init_logger(__name__)

# Rank of a request released only by the deferral deadline. Sorts behind
# every real rank (free-queue positions, bounded by the queue length), so a
# deadline that has passed still makes the request due on a step whose danger
# depth is zero, but never at the expense of a block about to be recycled:
# missing the deadline costs latency, missing an eviction costs the data.
_OVERDUE_RANK = 1 << 62

# Rank of an unblocked request whose front operations are held because they
# came due earlier: their pinned blocks are safe for now, so they go after
# every block about to be recycled and before a mere deadline.
_HELD_RANK = 1 << 61

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
        admitted_at_time: Monotonic clock read when the operation was
            buffered, against which ``max_deferral_seconds`` is measured.
    """

    store_metadata: "LMCacheMPRequestMetadata"
    block_hashes: BlockHashes
    admitted_at_time: float = 0.0


@dataclass(frozen=True)
class EvictionAwarePolicyConfig:
    """Tunables of the eviction-aware drain policy.

    Attributes:
        max_drain_per_step: Upper bound on the operations one drain emits,
            which caps the store work a single step can trigger.
        max_deferral_seconds: How long an operation may stay buffered before
            it is emitted regardless of eviction pressure. Zero leaves
            emission entirely to the danger window.

    The effect of raising each is documented for users in
    ``docs/source/mp/configuration.rst``.
    """

    max_drain_per_step: int = 64
    max_deferral_seconds: float = 0.0

    @classmethod
    def from_configs(
        cls, configs: dict[str, ConfigValue]
    ) -> "EvictionAwarePolicyConfig":
        """Read the tunables from ``lmcache.mp.lazy_offload_<field name>``.

        Args:
            configs: The connector's ``kv_connector_extra_config``. Only the
                ``lmcache.mp.lazy_offload_*`` keys are read; a missing key
                keeps the field default.

        Returns:
            The validated config.

        Raises:
            ValueError: If a value is outside its documented range.
        """

        def number(name: str, default: float) -> float:
            """Read one tunable as a float, falling back to ``default``."""
            key = f"lmcache.mp.lazy_offload_{name}"
            return float(cast("str | int | float", configs.get(key, default)))

        return cls(
            max_drain_per_step=int(number("max_drain_per_step", 64)),
            max_deferral_seconds=number("max_deferral_seconds", 0.0),
        )

    def __post_init__(self) -> None:
        """Validate field ranges, raising ValueError on any out of range."""
        if self.max_drain_per_step < 1:
            raise ValueError(
                f"max_drain_per_step must be >= 1, got {self.max_drain_per_step}"
            )
        if self.max_deferral_seconds < 0.0:
            raise ValueError(
                f"max_deferral_seconds must be >= 0.0, got {self.max_deferral_seconds}"
            )


@dataclass
class LazyOffloadCounters:
    """Cumulative policy counters for observability.

    The counts close as a ledger: ``admitted`` plus ``requeued`` equals the
    pending depth plus ``emitted`` plus every ``dropped_*`` counter.
    ``emitted_overdue`` is a weight beside the equation, not a term in it.

    Attributes:
        admitted: Operations taken into the buffer.
        emitted: Operations released to the manager for submission.
        emitted_overdue: Of those, the ones released by the deferral
            deadline rather than by eviction pressure.
        held: Operations whose blocks were pinned while their request had a
            batch in flight. Like ``emitted_overdue``, a weight beside the
            equation: a held operation is still pending until emitted.
        protected: Operations an admission would have recycled, moved to
            the free-queue tail and left buffered. A weight beside the
            equation too.
        squeezed_admissions: Admissions whose allocation would also have
            reached the free-queue tail, so every buffered operation was
            held instead; each may have been deferred by vLLM for the pins.
        squeezed_drains: Drains whose window reached the free-queue tail,
            so every buffered operation the drain did not emit was held,
            until the next drain whose window stops short of the tail.
        bound_exceeded: Steps whose running requests allocated more blocks
            than the previous drain bounded. Nonzero means the danger depth
            is not a bound for this model, and buffered blocks may have
            been recycled unseen.
        bound_blocks: The previous drain's bound, summed over the steps it
            was checked against. Over ``bound_allocated_blocks`` it is the
            bound's slack: how many blocks it reserves per one allocated.
        bound_allocated_blocks: Blocks the running requests allocated,
            summed over the same steps.
        dropped_evicted: Operations dropped because a covered block was
            recycled before they were emitted, plus the later operations of
            the same request that the loss made unreachable.
        rejected_unhashed: Operations refused at admission because a covered
            block carried no prefix-cache hash.
        rejected_prefix_broken: Operations refused at admission because the
            request's prefix chain was already broken.
        dropped_on_request_drop: Buffered operations discarded when vLLM
            preempted the request and freed its blocks.
        requeued: Emitted operations whose store failed, buffered again to
            be retried. A retried operation counts as emitted once more.
        dropped_id_reuse: Buffered operations of a finished request
            discarded when a new request took over its id.
    """

    admitted: int = 0
    emitted: int = 0
    emitted_overdue: int = 0
    requeued: int = 0
    held: int = 0
    protected: int = 0
    squeezed_admissions: int = 0
    squeezed_drains: int = 0
    bound_exceeded: int = 0
    bound_blocks: int = 0
    bound_allocated_blocks: int = 0
    dropped_evicted: int = 0
    rejected_unhashed: int = 0
    rejected_prefix_broken: int = 0
    dropped_on_request_drop: int = 0
    dropped_id_reuse: int = 0


class EvictionAwareStoreQueue(OffloadPolicy):
    """Buffers store operations and releases them by eviction imminence.

    One rule decides when to store: an operation is due once one of its
    blocks sits within the *danger depth* of the free queue -- what the
    running requests, the resumed loads and one admission can allocate
    before the next drain -- and nothing else makes it due but its request's
    deferral deadline or preemption. A due operation is emitted, or, when
    its request has a batch in flight or the drain budget is spent, held:
    the manager pins its blocks until a later drain emits it. When the
    window reaches the free-queue tail, every block freed before the next
    step is in reach, so every operation the drain does not emit is held
    too, and released unpinned by the next drain whose window stops short
    of the tail. A second admission in one step reaches past the window;
    the operations it would recycle are moved to the free-queue tail and
    stay buffered. A preempted request releases its whole buffer, which its
    restart would discard. An engine with no requests never drains;
    operations whose blocks are evicted before they come due are dropped,
    never stored stale. Scheduler thread only.
    """

    def __init__(self, config: EvictionAwarePolicyConfig, pool: "BlockPool") -> None:
        """Create an empty queue over ``pool`` with the tunables in ``config``.

        Args:
            config: The validated tunables of this policy.
            pool: The scheduler's GPU block pool. It is only ever read here:
                free-queue order and block hashes.
        """
        self._config = config
        self._pool = pool
        # Insertion order is admission order (a re-entering request goes to
        # the back), so the dict is also the drain's tie-break order.
        self._pending: dict[str, list[PendingStoreOp]] = {}
        # Prefix validity is policy; phase and batches are the manager's.
        self._broken_prefixes: set[str] = set()
        # Request id -> length of the held prefix of its buffer.
        self._held: dict[str, int] = {}
        # Requests whose held prefix a squeezed drain pinned without it being
        # due: the next unsqueezed drain releases it rather than emitting it.
        self._squeeze_held: set[str] = set()
        self._next_step_blocks = 0
        self._resuming_blocks = 0
        self._admission_blocks = 0
        self._now = 0.0  # This step's clock, set at the top of drain().
        self._counters = LazyOffloadCounters()
        self._last_logged = LazyOffloadCounters()
        self._last_stats_log_time = 0.0

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
    ) -> None:
        """Buffer one store operation; see ``OffloadPolicy.add``.

        Args:
            meta: The store operation offered by the manager.
            block_hashes: Hash of every block the operation covers, read at
                admission. A None among them rejects the operation and
                breaks the request's prefix chain, since a block whose hash
                is gone cannot be checked for eviction later.
        """
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
            return
        if any(block_hash is None for block_hash in block_hashes.values()):
            # The tracker has advanced past this range, so later operations
            # would be stored without their prefix: the chain is broken.
            self._broken_prefixes.add(meta.request_id)
            self._counters.rejected_unhashed += 1
            logger.warning(
                "Lazy offload: skipping store for request %s tokens [%d, %d) "
                "and every later operation of it: covered blocks carry no "
                "prefix-cache hash, so their eviction could not be detected. "
                "Prefix caching is off, or a sliding-window or hybrid model "
                "left a hash-less null block in the block table. "
                "rejected_unhashed and rejected_prefix_broken count these.",
                meta.request_id,
                meta.op.start,
                meta.op.end,
            )
            return
        self._pending.setdefault(meta.request_id, []).append(
            PendingStoreOp(
                store_metadata=meta,
                block_hashes=block_hashes,
                # Read here rather than reusing the step clock: before
                # the first drain, and after an idle gap, that clock is
                # stale and would make a fresh op instantly overdue.
                admitted_at_time=time.monotonic(),
            )
        )
        self._counters.admitted += 1

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
        allocation runs. So every buffered op is held instead, back to the
        held prefix: the pin may defer the admission, but loses nothing.
        Pinning a block a request still owns does not shrink the queue.

        Args:
            num_blocks: Upper bound on the blocks the allocation consumes.

        Returns:
            Per request, the ops to move to the tail or newly held.
        """
        owners = {
            block_id: (request_id, index)
            for request_id, ops in self._pending.items()
            for index, op in enumerate(ops)
            for block_id in op.block_hashes
        }
        ranks, reaches_tail = self._free_queue_ranks(num_blocks, owners.keys())
        last_met: dict[str, int] = {}
        for block_id in ranks:
            owner = owners.get(block_id)
            if owner is not None:
                request_id, index = owner
                last_met[request_id] = max(last_met.get(request_id, -1), index)
        if reaches_tail and self._pending:
            self._counters.squeezed_admissions += 1
            # Blocks freed between this walk and the allocation, as running
            # requests free theirs ahead of a resumed load, join a tail the
            # walk already reached: every buffered op is in reach.
            last_met = {
                request_id: len(ops) - 1 for request_id, ops in self._pending.items()
            }
        protection = AdmissionProtection()
        for request_id, last in last_met.items():
            ops = self._pending[request_id]
            # Held ops are pinned, so the walk never meets them.
            start = self._held.get(request_id, 0)
            end = start
            while end <= last and self._snapshot_intact(ops[end]):
                end += 1
            if end <= start:
                continue
            item = _pending_item(request_id, ops[start:end])
            if reaches_tail:
                self._held[request_id] = end
                self._counters.held += end - start
                protection.holds.append(item)
            else:
                self._counters.protected += end - start
                protection.to_tail.append(item)
        return protection

    def has_pending_request(self, request_id: str) -> bool:
        """Whether this request currently owns buffered operations.

        Args:
            request_id: The request id to query.

        Returns:
            True while at least one of its operations is buffered.
        """
        return request_id in self._pending

    def buffered_block_ids(self) -> set[int]:
        """Every GPU block a buffered operation covers.

        Returns:
            The block ids, possibly empty.
        """
        return {
            block_id
            for ops in self._pending.values()
            for op in ops
            for block_id in op.block_hashes
        }

    def drop_request(self, request_id: str) -> int:
        """Discard buffered operations invalidated by a tracker reset.

        The request keeps its id and restarts at token zero, so its
        broken-chain marker is cleared too.

        Args:
            request_id: The preempted request.

        Returns:
            The number of buffered operations discarded.
        """
        dropped = self._pending.pop(request_id, [])
        self._broken_prefixes.discard(request_id)
        self._held.pop(request_id, None)
        self._squeeze_held.discard(request_id)
        self._counters.dropped_on_request_drop += len(dropped)
        return len(dropped)

    def discard_for_reuse(self, request_id: str) -> None:
        """Discard the buffered state of the id's previous holder.

        Args:
            request_id: The reused request id, whose previous holder's
                buffer and broken-chain marker are dropped.
        """
        dropped = self._pending.pop(request_id, [])
        self._broken_prefixes.discard(request_id)
        self._held.pop(request_id, None)
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

    def requeue_failed(self, item: PendingStoreItem) -> None:
        """Buffer a failed batch ahead of its request's later ops.

        The manager moves the ops' blocks to the free-queue tail, so they
        come due again when the head reaches them, rather than being
        resubmitted at once into an L1 that just rejected them. A held
        prefix of the later ops grows by the requeued ops, which precede it.

        Args:
            item: The failed batch's operations, in token order. The
                manager filters out orphaned batches first.
        """
        request_id = item.request_id
        now = time.monotonic()
        ops = [
            PendingStoreOp(
                store_metadata=meta,
                block_hashes=block_hashes,
                # A fresh deadline: an op made overdue would be resubmitted
                # on the next drain.
                admitted_at_time=now,
            )
            for meta, block_hashes in item.metadatas
        ]
        self._pending[request_id] = ops + self._pending.get(request_id, [])
        if request_id in self._held:
            self._held[request_id] += len(ops)
        self._counters.requeued += len(ops)

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Record the step's block pressure and release what it made due.

        Per request: the suffix from the first op whose blocks were recycled
        is dropped, then the front up to the last op with a block in the
        danger window is due (prefix closure). Requests in
        ``blocked_request_ids`` stay pending, since the worker holds one
        in-flight batch per request; their due ops are held instead, so a
        slow store cannot let them be recycled. Once unblocked, a request
        with held ops is due, up to the last of them. A request in
        ``preempted_request_ids`` releases its whole buffer: its tracker
        restarts when it is scheduled again, which discards what is still
        buffered, so it is not held to the budget. Requests are served most
        imminent first until ``max_drain_per_step``; due ops past the budget
        are held. One whose deadline passed while no block of it sits in the
        window is due anyway, but it is served after every request holding
        one, so an expired backlog cannot spend the budget that a block
        about to be recycled needs.

        A window that reaches the free-queue tail also covers blocks not yet
        in the queue: vLLM frees blocks of running requests (align-mode
        Mamba drops older states) before the next step allocates, and they
        join a tail already in reach. So every op this drain leaves buffered
        is held, not just the due ones. Those holds are released, unpinned,
        by the first drain whose window stops short of the tail: their
        blocks then join the tail behind every block the window counts.

        Args:
            signals: The manager's bounds on the blocks the running
                requests, the resumed loads and one admission allocate
                before the next drain, which set the danger depth, plus the
                finished, blocked and preempted request ids. Finished ids
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
        self._resuming_blocks = signals.resuming_blocks
        self._admission_blocks = signals.admission_blocks
        if not self._pending:
            self._maybe_log_stats()
            return LazyOffloadDrain()

        # Depth zero makes nothing due and the queue is not walked; the loss
        # check below reads hashes, not ranks, and still runs.
        ranks, reaches_tail = self._free_queue_ranks(
            self._danger_depth(), self.buffered_block_ids()
        )
        drain = LazyOffloadDrain()
        if reaches_tail:
            self._counters.squeezed_drains += 1
        else:
            # Pinned, these blocks were not walked; unpinned, they join the
            # tail, past the depth just counted.
            for request_id in self._squeeze_held:
                if self._held.pop(request_id, 0):
                    drain.releases.append(request_id)
            self._squeeze_held.clear()
        overdue_cutoff = self._now - self._config.max_deferral_seconds
        dropped_ops = 0
        dropped_ids: list[str] = []
        # Due now, as (imminence rank, admission order, request id).
        candidates: list[tuple[int, int, str]] = []
        # Of those, the ones the deadline alone made due: they release their
        # whole buffer, not just the segment sitting in the danger window.
        overdue_ids: set[str] = set()
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
            held = self._held.get(request_id, 0)
            if request_id in signals.blocked_request_ids:
                # Pin what an unblocked request would release now; the rest
                # stays where the window can see it, unless the window reached
                # the tail and so sees everything.
                target = (
                    len(surviving)
                    if reaches_tail
                    else max(held, len(self._due_front_segment(surviving, ranks)))
                )
                if target > held:
                    self._held[request_id] = target
                    self._counters.held += target - held
                    drain.holds.append(
                        _pending_item(request_id, surviving[held:target])
                    )
                if reaches_tail:
                    self._squeeze_held.add(request_id)
                continue
            in_window = [
                rank
                for op in surviving
                for block_id in op.block_hashes
                if (rank := ranks.get(block_id)) is not None
            ]
            if in_window:
                # A block the engine is about to recycle outranks a passed
                # deadline, and sizes the release too: a request past both
                # emits its due front segment, which always holds the op that
                # made it overdue. The ops behind it keep their own admission
                # clocks and come due on their own deadlines.
                # TODO: free-queue pressure is the only retirement mechanism
                # modelled here. A sliding-window layer retires a block on a
                # token schedule instead, which is predictable but invisible
                # in the free queue, so such an op waits for its deadline
                # rather than being released when its data is about to go.
                # Size the release per attention type: one due-predictor per
                # mechanism, emit up to the last index any of them marks due.
                candidates.append((min(in_window), order, request_id))
            elif (
                held and request_id not in self._squeeze_held
            ) or request_id in signals.preempted_request_ids:
                candidates.append((_HELD_RANK, order, request_id))
            elif (
                self._config.max_deferral_seconds > 0.0
                and surviving[0].admitted_at_time <= overdue_cutoff
            ):
                candidates.append((_OVERDUE_RANK, order, request_id))
                overdue_ids.add(request_id)
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
        ops_left = self._config.max_drain_per_step
        for _, _, request_id in candidates:
            ops = self._pending[request_id]
            is_overdue = request_id in overdue_ids
            if is_overdue and ops_left <= 0:
                # Nothing about to be recycled made it due: it waits in place.
                continue
            held = self._held.pop(request_id, 0)
            # A squeeze hold is not due; only the window makes its ops due.
            due_held = 0 if request_id in self._squeeze_held else held
            if request_id in signals.preempted_request_ids or is_overdue:
                due_ops = ops
            else:
                due_ops = ops[: max(due_held, len(self._due_front_segment(ops, ranks)))]
            if request_id in signals.preempted_request_ids:
                emitted = due_ops
            else:
                emitted = due_ops[: max(ops_left, 0)]
            ops_left -= len(emitted)
            kept_held = held
            if not is_overdue and len(due_ops) > max(held, len(emitted)):
                # Over budget, yet the running requests may recycle these
                # before the next drain: pin them.
                moved = due_ops[max(held, len(emitted)) :]
                kept_held = len(due_ops)
                self._counters.held += len(moved)
                drain.holds.append(_pending_item(request_id, moved))
            if kept_held > len(emitted):
                self._held[request_id] = kept_held - len(emitted)
            if not emitted:
                continue
            self._counters.emitted += len(emitted)
            if is_overdue:
                self._counters.emitted_overdue += len(emitted)
            drain.items.append(_pending_item(request_id, emitted))
            remaining = ops[len(emitted) :]
            if remaining:
                self._pending[request_id] = remaining
            else:
                del self._pending[request_id]
                drain.emptied_request_ids.append(request_id)
        if reaches_tail:
            for request_id, ops in self._pending.items():
                if request_id in signals.blocked_request_ids:
                    continue
                held = self._held.get(request_id, 0)
                if len(ops) > held:
                    self._held[request_id] = len(ops)
                    self._counters.held += len(ops) - held
                    self._squeeze_held.add(request_id)
                    drain.holds.append(_pending_item(request_id, ops[held:]))
        self._maybe_log_stats()
        return drain

    def _danger_depth(self) -> int:
        """Free-queue depth at risk before the next drain.

        A second admission in one step allocates further, but the manager
        protects buffered blocks from it right before it does. A resumed
        load that is still unscheduled was protected for a step that has
        passed, so its allocation counts here.

        Returns:
            The manager's bound on the blocks the running requests, the
            unscheduled resumed loads and one admission can allocate before
            the next drain.
        """
        return self._next_step_blocks + self._resuming_blocks + self._admission_blocks

    def _free_queue_ranks(
        self, depth: int, pending_blocks: "Container[int]"
    ) -> tuple[dict[int, int], bool]:
        """The free-queue head an allocation can recycle, id -> eviction rank.

        A buffered block met in the walk is due, or protected, so it leaves
        the queue or moves to the tail, and every block behind it moves up:
        the walk runs until ``depth`` blocks outside buffered ops are
        counted. Walks the queue's links lazily rather than
        ``get_all_free_blocks()``, which materialises the whole queue every
        step.

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
        block = self._pool.free_block_queue.fake_free_list_head.next_free_block
        if block is None:
            raise RuntimeError("free_block_queue.fake_free_list_head has no successor")
        counted = 0
        # The fake tail has no successor, which is what ends the walk.
        while block.next_free_block is not None and counted < depth:
            ranks[block.block_id] = len(ranks)
            if block.block_id not in pending_blocks:
                counted += 1
            block = block.next_free_block
        return ranks, counted < depth

    def _drop_evicted_suffix(
        self, request_id: str, ops: list[PendingStoreOp]
    ) -> list[PendingStoreOp]:
        """Drop ops from the first one whose data was lost; return survivors.

        A hash mismatch means the block was recycled; that op and every later
        op of the request go, and further admissions are rejected. The caller
        installs the survivors and reports the count.

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
        self._broken_prefixes.add(request_id)
        return ops[:first_lost]

    def _due_front_segment(
        self, ops: list[PendingStoreOp], ranks: dict[int, int]
    ) -> list[PendingStoreOp]:
        """The front segment up to the last op with a block in the window.

        Taking from the front keeps a stored operation's prefix stored.

        Args:
            ops: One request's buffered operations, in prefix order.
            ranks: The danger window, as returned by
                :meth:`_free_queue_ranks`. Only membership is read here.

        Returns:
            The due prefix of ``ops``, empty when none of them is due.
        """
        last_due = -1
        for index, op in enumerate(ops):
            if any(block_id in ranks for block_id in op.block_hashes):
                last_due = index
        return ops[: last_due + 1]

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
        return all(
            self._pool.blocks[block_id].block_hash == snapshot
            for block_id, snapshot in op.block_hashes.items()
        )

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
        ``admitted == pending + emitted + every drop counter``.

        Returns:
            The counter names and values, space separated.
        """
        fields = " ".join(
            f"{name}={value}" for name, value in asdict(self._counters).items()
        )
        pending = sum(len(ops) for ops in self._pending.values())
        return f"{fields} pending={pending}"
