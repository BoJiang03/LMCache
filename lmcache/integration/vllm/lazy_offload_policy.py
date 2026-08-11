# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction-aware lazy offload policy (gates 1 and 3 of the store decision).

Implements the drain policy described in
``docs/design/integration/vllm/lazy_offload_decision_model.md``: store
operations are buffered instead of submitted eagerly, and are released only
when the GPU blocks holding their data are about to be evicted (gate 1,
"replace prediction with timing") and the covered prefix is long enough for
the store to beat recomputation (gate 3, static break-even threshold).

This module is pure policy: it never touches vLLM at runtime (vLLM types
appear only in annotations) and performs no I/O, so it is unit-testable
without a GPU or a vLLM installation. The connector owns execution: taking
block-hash snapshots at admission, calling :meth:`EvictionAwareStoreQueue.
observe_step` / :meth:`EvictionAwareStoreQueue.collect_due` once per
scheduler step, pinning (``touch``) the blocks of emitted operations, and
submitting them to the worker.
"""

# Standard
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol
import enum
import math

# First Party
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestMetadata,
    )

logger = init_logger(__name__)

# Smoothing factor for the per-step block-consumption EMA. Not a config knob:
# the horizon (in steps) is the tunable quantity; the EMA only smooths noise.
_EMA_ALPHA = 0.3


class BlockPoolReader(Protocol):
    """Read-only view over the GPU block pool required by the policy.

    The production implementation is :class:`GPUBlockPoolView`; tests provide
    a fake. Both must be side-effect free: the policy never mutates pool
    state.
    """

    def free_queue_ranks(self) -> dict[int, int]:
        """Return the current eviction order of the free queue.

        Returns:
            Mapping from block id to its rank in the free queue, where rank
            0 is the next eviction victim. Blocks that are not in the free
            queue (in use, or pinned) are absent from the mapping.
        """
        ...

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the current prefix-cache hash of a GPU block.

        Args:
            block_id: The GPU block id to inspect.

        Returns:
            The block's current hash, or None if the block holds no cached
            (full, hashed) content -- e.g. it was evicted and reallocated,
            or never completed.
        """
        ...


class GPUBlockPoolView:
    """Production :class:`BlockPoolReader` over a bound vLLM ``BlockPool``.

    All accesses are read-only. ``free_queue_ranks`` snapshots the free
    queue in O(number of free blocks); callers should invoke it at most once
    per scheduler step.
    """

    def __init__(self, block_pool: "BlockPool") -> None:
        """Wrap a vLLM block pool obtained via ``bind_gpu_block_pool``.

        Args:
            block_pool: The scheduler's GPU block pool.
        """
        self._block_pool = block_pool

    def free_queue_ranks(self) -> dict[int, int]:
        """Snapshot the free queue into a block-id -> eviction-rank map.

        Returns:
            Mapping from block id to rank (0 = next eviction victim).
        """
        free_blocks = self._block_pool.free_block_queue.get_all_free_blocks()
        return {block.block_id: rank for rank, block in enumerate(free_blocks)}

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the current hash of the block, or None if uncached.

        Args:
            block_id: The GPU block id to inspect.
        """
        return self._block_pool.blocks[block_id].block_hash

    def num_free_blocks(self) -> int:
        """Return the number of blocks currently in the free queue."""
        return self._block_pool.get_num_free_blocks()


class AdmitResult(enum.Enum):
    """Outcome of admitting a store operation into the lazy queue.

    The connector maps each outcome to an action:

    - ADMITTED: nothing to do now; the operation will be emitted later.
    - REJECTED_UNHASHED_BLOCK: a covered block has no hash, so eviction of
      that block could not be detected later (a reallocated block would also
      read None, masking the loss). The connector must fall back to an eager
      store for this operation.
    - REJECTED_PREFIX_BROKEN: an earlier chunk of this request was already
      dropped, so this chunk would be unreachable on retrieval. The
      connector must skip the store entirely.
    """

    ADMITTED = enum.auto()
    REJECTED_UNHASHED_BLOCK = enum.auto()
    REJECTED_PREFIX_BROKEN = enum.auto()


@dataclass
class PendingStoreOp:
    """A deferred store operation with the state needed to validate it.

    Attributes:
        request_id: The vLLM request this operation belongs to.
        store_metadata: The ready-to-send store metadata produced by
            ``LMCacheMPRequestMetadata.GetStoreMetadata``; opaque to the
            policy.
        block_hashes: Hash of every GPU block covering the operation's token
            range, snapshotted at admission. All values are non-None
            (enforced by admission); a later mismatch against the pool means
            the block was evicted or reallocated.
        prefix_end_tokens: Token index one past the end of this operation's
            range, i.e. the request-prefix length covered once this
            operation and all earlier ones are stored.
    """

    request_id: str
    store_metadata: "LMCacheMPRequestMetadata"
    block_hashes: dict[int, "BlockHashWithGroupId"]
    prefix_end_tokens: int


@dataclass(frozen=True)
class LazyOffloadPolicyConfig:
    """Tunables of the eviction-aware drain policy.

    Attributes:
        horizon_steps: How many scheduler steps of estimated block
            consumption to treat as "imminent eviction". Larger values drain
            earlier (closer to eager, fewer drops); smaller values drain
            later (better filtering, more drops).
        min_prefix_tokens: Break-even prefix length (gate 3): a request
            whose known prefix is shorter than this when its blocks come due
            is dropped instead of stored. 0 disables the gate.
        max_drain_per_step: Upper bound on operations emitted per step, to
            bound the D2H burst. Must be >= 1.
    """

    horizon_steps: float = 2.0
    min_prefix_tokens: int = 0
    max_drain_per_step: int = 64

    def __post_init__(self) -> None:
        """Validate field ranges.

        Raises:
            ValueError: If any field is outside its documented range.
        """
        if self.horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be > 0, got {self.horizon_steps}")
        if self.min_prefix_tokens < 0:
            raise ValueError(
                f"min_prefix_tokens must be >= 0, got {self.min_prefix_tokens}"
            )
        if self.max_drain_per_step < 1:
            raise ValueError(
                f"max_drain_per_step must be >= 1, got {self.max_drain_per_step}"
            )


@dataclass
class LazyOffloadCounters:
    """Cumulative policy counters for observability.

    ``dropped_evicted`` is the gate-1 quality sensor (drop rate): operations
    lost because their blocks were evicted before the policy drained them.
    ``rejected_short_prefix`` counts gate-3 rejections.
    """

    admitted: int = 0
    emitted: int = 0
    dropped_evicted: int = 0
    rejected_short_prefix: int = 0
    rejected_unhashed: int = 0
    rejected_prefix_broken: int = 0
    dropped_on_request_drop: int = 0


@dataclass
class DrainResult:
    """Operations released by one :meth:`EvictionAwareStoreQueue.collect_due`.

    Attributes:
        to_store: Operations to submit now, ordered by eviction imminence
            across requests and by prefix order within a request. The
            connector must pin (``touch``) their blocks before the store and
            unpin after completion.
        dropped_evicted: Operations whose data was lost (block evicted or
            reallocated before drain), including later same-request
            operations dropped for prefix closure.
        dropped_short_prefix: Operations dropped by gate 3 (request prefix
            below the break-even length at the time its blocks came due).
        released_requests: Finished requests that no longer have any pending
            operations after this drain; the connector may now end their
            sessions.
    """

    to_store: list[PendingStoreOp] = field(default_factory=list)
    dropped_evicted: list[PendingStoreOp] = field(default_factory=list)
    dropped_short_prefix: list[PendingStoreOp] = field(default_factory=list)
    released_requests: list[str] = field(default_factory=list)


class EvictionAwareStoreQueue:
    """Buffers store operations and releases them by eviction imminence.

    Gate 1 realization: an operation is emitted when any of its blocks sits
    within the *danger depth* of the free queue -- the number of blocks the
    engine is expected to consume within ``horizon_steps`` steps, estimated
    from an EMA of observed per-step allocation and a one-step feedforward
    supplied by the connector. An idle engine (no allocation pressure) never
    triggers a drain; operations whose blocks are evicted before they come
    due are dropped and counted, never stored stale.

    Not thread-safe: all methods must be called from the scheduler thread
    (the vLLM connector scheduler-side call pattern).
    """

    def __init__(self, config: LazyOffloadPolicyConfig, pool: BlockPoolReader) -> None:
        """Create an empty queue.

        Args:
            config: Policy tunables.
            pool: Read-only view of the GPU block pool.
        """
        self._config = config
        self._pool = pool
        # Per-request pending operations in prefix (admission) order.
        self._pending: dict[str, list[PendingStoreOp]] = {}
        # Requests whose prefix chain was broken by a drop; further chunks
        # of these requests are unreachable and must not be admitted.
        self._prefix_broken: set[str] = set()
        # Requests reported finished by the engine; used to compute
        # DrainResult.released_requests.
        self._finished: set[str] = set()
        self._blocks_per_step_ema: float = 0.0
        self._ema_initialized = False
        self._next_step_estimate = 0
        self._counters = LazyOffloadCounters()

    def admit(self, op: PendingStoreOp) -> AdmitResult:
        """Admit a store operation into the pending queue.

        Args:
            op: The operation to buffer. ``op.block_hashes`` must cover every
                GPU block of the operation's token range.

        Returns:
            The admission outcome; see :class:`AdmitResult` for the action
            the caller must take on each value.
        """
        if op.request_id in self._prefix_broken:
            self._counters.rejected_prefix_broken += 1
            return AdmitResult.REJECTED_PREFIX_BROKEN
        if any(block_hash is None for block_hash in op.block_hashes.values()):
            self._counters.rejected_unhashed += 1
            return AdmitResult.REJECTED_UNHASHED_BLOCK
        self._pending.setdefault(op.request_id, []).append(op)
        self._counters.admitted += 1
        return AdmitResult.ADMITTED

    def observe_step(
        self, new_blocks_allocated: int, est_next_step_blocks: int
    ) -> None:
        """Record one scheduler step's block-consumption signals.

        Must be called once per step, before :meth:`collect_due`.

        Args:
            new_blocks_allocated: GPU blocks newly allocated in the step
                that just finished scheduling (gross allocation, counted
                from the scheduler output).
            est_next_step_blocks: Estimated blocks the next step will
                allocate (e.g. scheduled tokens divided by block size).
        """
        if self._ema_initialized:
            self._blocks_per_step_ema = (
                _EMA_ALPHA * new_blocks_allocated
                + (1 - _EMA_ALPHA) * self._blocks_per_step_ema
            )
        else:
            self._blocks_per_step_ema = float(new_blocks_allocated)
            self._ema_initialized = True
        self._next_step_estimate = est_next_step_blocks

    def mark_request_finished(self, request_id: str) -> bool:
        """Record that the engine finished a request.

        Args:
            request_id: The finished request.

        Returns:
            True if the request still has pending operations (the caller
            must defer session teardown until the request appears in a
            :class:`DrainResult`'s ``released_requests``); False if nothing
            is pending and the caller may tear down immediately.
        """
        if request_id in self._pending:
            self._finished.add(request_id)
            return True
        self._prefix_broken.discard(request_id)
        return False

    def drop_request(self, request_id: str) -> int:
        """Discard all pending operations of a request (e.g. on abort).

        Args:
            request_id: The request to discard.

        Returns:
            The number of operations discarded.
        """
        dropped = self._pending.pop(request_id, [])
        self._counters.dropped_on_request_drop += len(dropped)
        self._finished.discard(request_id)
        self._prefix_broken.discard(request_id)
        return len(dropped)

    def num_pending_ops(self) -> int:
        """Return the total number of buffered store operations."""
        return sum(len(ops) for ops in self._pending.values())

    def stats(self) -> LazyOffloadCounters:
        """Return a copy of the cumulative policy counters."""
        return replace(self._counters)

    def collect_due(self) -> DrainResult:
        """Release the operations whose blocks face imminent eviction.

        For every pending request, first drops the suffix of its operation
        list starting at the first operation whose data is already lost
        (current block hash differs from the admission snapshot) -- storing
        a later chunk without its prefix would be unreachable. Then, if any
        surviving operation has a block within the danger depth of the free
        queue, the request's operations are released from the front up to
        the last due one (prefix closure), subject to gate 3.

        Returns:
            The operations to store and to drop this step; see
            :class:`DrainResult`.
        """
        result = DrainResult()
        if not self._pending:
            return result

        ranks = self._pool.free_queue_ranks()
        danger_depth = self._danger_depth()

        # Per request: (min due rank, ops released from the front).
        # Iterate over a copy: helpers may drop entries from self._pending.
        due_segments: list[tuple[int, str, list[PendingStoreOp]]] = []
        for request_id, ops in list(self._pending.items()):
            surviving = self._drop_evicted_suffix(request_id, ops, result)
            if not surviving:
                continue
            segment = self._due_front_segment(surviving, ranks, danger_depth)
            if segment is None:
                continue
            min_rank, due_ops = segment
            if self._fails_economy_gate(surviving):
                # Gate 3: the whole known prefix is below break-even. The due
                # front is about to die, which breaks the prefix chain for
                # the rest -- drop everything, not just the due segment.
                result.dropped_short_prefix.extend(surviving)
                self._counters.rejected_short_prefix += len(surviving)
                self._prefix_broken.add(request_id)
                self._replace_pending(request_id, [], result)
                continue
            due_segments.append((min_rank, request_id, due_ops))

        # Most imminent requests first; cap cuts whole-request segments from
        # the tail so within-request prefix order is never violated.
        due_segments.sort(key=lambda seg: seg[0])
        budget = self._config.max_drain_per_step
        for _, request_id, due_ops in due_segments:
            if budget <= 0:
                break
            emitted = due_ops[:budget]
            budget -= len(emitted)
            result.to_store.extend(emitted)
            self._counters.emitted += len(emitted)
            remaining = self._pending[request_id][len(emitted) :]
            self._replace_pending(request_id, remaining, result)
        return result

    def _danger_depth(self) -> int:
        """Free-queue depth considered at risk within the horizon.

        Expected consumption below half a block over the whole horizon is
        treated as idle (depth 0): the EMA decays asymptotically after a
        burst and would otherwise keep a ceil'd depth of 1 forever.
        """
        per_step = max(self._blocks_per_step_ema, float(self._next_step_estimate))
        horizon_blocks = per_step * self._config.horizon_steps
        if horizon_blocks < 0.5:
            return 0
        return math.ceil(horizon_blocks)

    def _drop_evicted_suffix(
        self,
        request_id: str,
        ops: list[PendingStoreOp],
        result: DrainResult,
    ) -> list[PendingStoreOp]:
        """Drop ops from the first one whose data was lost; return survivors.

        A hash mismatch on any covered block means the block was evicted (or
        reallocated); the op and every later op of the request are dropped
        for prefix closure, and further admissions are rejected.
        """
        first_lost = len(ops)
        for index, op in enumerate(ops):
            if any(
                self._pool.block_hash(block_id) != snapshot
                for block_id, snapshot in op.block_hashes.items()
            ):
                first_lost = index
                break
        if first_lost == len(ops):
            return ops
        dropped = ops[first_lost:]
        result.dropped_evicted.extend(dropped)
        self._counters.dropped_evicted += len(dropped)
        self._prefix_broken.add(request_id)
        surviving = ops[:first_lost]
        self._replace_pending(request_id, surviving, result)
        return surviving

    def _due_front_segment(
        self,
        ops: list[PendingStoreOp],
        ranks: dict[int, int],
        danger_depth: int,
    ) -> tuple[int, list[PendingStoreOp]] | None:
        """Find the front segment of ops to release for one request.

        An op is due when any of its blocks is within ``danger_depth`` of
        the free-queue head. Blocks absent from ``ranks`` (in use or pinned)
        are not at risk. The released segment runs from the front to the
        last due op, so a stored chunk never lacks its stored prefix.

        Returns:
            (min rank across the segment's due blocks, the segment), or
            None when no op is due.
        """
        if danger_depth <= 0:
            return None
        last_due = -1
        min_rank = danger_depth
        for index, op in enumerate(ops):
            op_ranks = [
                rank
                for block_id in op.block_hashes
                if (rank := ranks.get(block_id)) is not None
            ]
            due_ranks = [rank for rank in op_ranks if rank < danger_depth]
            if due_ranks:
                last_due = index
                min_rank = min(min_rank, min(due_ranks))
        if last_due < 0:
            return None
        return min_rank, ops[: last_due + 1]

    def _fails_economy_gate(self, ops: list[PendingStoreOp]) -> bool:
        """Gate 3: is the request's known prefix below break-even length?"""
        if self._config.min_prefix_tokens == 0:
            return False
        known_prefix = ops[-1].prefix_end_tokens
        return known_prefix < self._config.min_prefix_tokens

    def _replace_pending(
        self,
        request_id: str,
        remaining: list[PendingStoreOp],
        result: DrainResult,
    ) -> None:
        """Update a request's pending list, releasing it if drained empty."""
        if remaining:
            self._pending[request_id] = remaining
            return
        self._pending.pop(request_id, None)
        if request_id in self._finished:
            self._finished.discard(request_id)
            self._prefix_broken.discard(request_id)
            result.released_requests.append(request_id)
