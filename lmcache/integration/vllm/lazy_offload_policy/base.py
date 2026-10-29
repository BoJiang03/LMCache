# SPDX-License-Identifier: Apache-2.0
"""Base types for lazy-offload policies.

Terms used throughout this package:

- **Store operation**: one contiguous token range of one request whose KV is
  ready to be written to LMCache. vLLM's request tracker produces these; a
  lazy-offload policy buffers them instead of submitting them at once.
- **Buffered**: held by the policy, not yet handed to the worker. A buffered
  operation still points at live GPU blocks and is only valid while those
  blocks hold the data snapshotted at admission.
- **Emitted**: chosen by a drain and handed back to ``LazyOffloadManager``,
  which pins the blocks and submits the store to the worker.
- **Prefix chain**: the sequence of a request's stored token ranges, in token
  order. LMCache retrieval walks it from token zero, so a range whose
  earlier neighbour was never stored is unreachable; once a range is lost,
  the chain is *broken* and every later range of that request is worthless.
  On align-mode Mamba hybrids a range past an anchor is reachable from
  vLLM's hit at that state, so a policy may store it before its earlier
  neighbour.
- **Store batch**: the emitted operations of one request, coalesced into the
  one store the worker accepts per request. A batch is *in flight* from its
  submission until every worker has reported it, and the request's later
  operations stay buffered for as long as that lasts.
- **Held**: a buffered operation whose GPU blocks the manager pinned so
  they cannot be recycled before it is stored: its request's batch is in
  flight, the drain budget is spent, the free-queue walk reached the tail
  (a *squeeze*), or a finish or lookup expedited it. A held operation cannot
  lose its data; a later drain emits it, and the batch takes over the pin.
- **Anchor**: an operation holding a real block of an align-mode Mamba
  group, a state vLLM can resume a hit at.
- **Due segment**: the contiguous operations a drain stores together so
  that every due one is stored.
- **Expedited**: due because a finish or a lookup needs it stored,
  independent of eviction pressure.
- **Protected**: a buffered operation an allocation the danger depth does
  not cover (a further admission, or an asynchronous load resuming) would
  recycle. The manager moves its blocks to the free-queue tail instead of
  pinning them, so the allocation still has every free block to allocate
  from, and it stays buffered until the window meets it again. Only when
  the allocation would reach the tail too are they held instead.
- **Requeued**: an emitted operation whose store failed, handed back to the
  policy once its batch settled. Its blocks were pinned in flight, so its
  hash snapshot still holds, and it is buffered again in token order among
  the request's buffered operations.
- **Session**: the LMCache-side state a request owns while it is storing.
  ``LazyOffloadManager`` ends it once the request has finished and has
  nothing buffered and nothing in flight.
"""

# Standard
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata

#: Value type of vLLM's ``kv_connector_extra_config`` entries.
ConfigValue = str | int | float | bool | list[str] | None

#: Prefix-cache hash of every GPU block covering one store operation, keyed
#: by block id. ``None`` means the block carries no hash.
BlockHashes = dict[int, "BlockHashWithGroupId | None"]


@dataclass
class PendingStoreItem:
    """Store metadata selected for one request by any drain policy.

    Attributes:
        request_id: The vLLM request id these operations belong to. One item
            carries operations of exactly one request, so the manager can
            coalesce them into a single store submission.
        metadatas: The selected operations in token order, each paired with
            the block-hash snapshot taken when it was buffered. The manager
            re-reads the hashes before submitting to prove the blocks still
            hold the same data.
    """

    request_id: str
    metadatas: list[tuple["LMCacheMPRequestMetadata", BlockHashes]] = field(
        default_factory=list
    )


@dataclass(frozen=True)
class DrainSignals:
    """One scheduler step's inputs to a drain decision.

    Attributes:
        next_step_blocks: Upper bound on the blocks the requests now running
            can allocate before the next drain. Admissions and resumed
            loads are not in it: the manager protects buffered blocks from
            those right before they allocate.
        admission_blocks: Upper bound on the blocks one admission can
            allocate, from the model's maximum length. A policy that keeps
            this much of the free-queue head clear of buffered blocks leaves
            the admission protection nothing to move for the first
            admission of a step.
        resuming_blocks: Upper bound on the blocks that requests whose load
            finished, but that no step has scheduled yet, allocate once
            scheduled. The protection when their load finished assumed the
            next step; until a step schedules them, the head moves on and
            the danger depth counts them too.
        finished_request_ids: Requests whose generation has ended. Their
            buffered operations are still storable; the id is reported so a
            policy may treat them differently from running requests.
        blocked_request_ids: Requests that already have a store batch in
            flight. The worker tracks one store future per request id, so
            these must stay buffered until their receipt arrives; a policy
            may ask the manager to hold their blocks meanwhile.
        running_blocks_allocated: Blocks this step allocated to requests
            that were already running, which the previous drain's
            ``next_step_blocks`` bounds. Read only to check that bound.
    """

    next_step_blocks: int
    resuming_blocks: int
    admission_blocks: int
    finished_request_ids: set[str]
    blocked_request_ids: set[str]
    running_blocks_allocated: int


@dataclass
class LazyOffloadDrain:
    """Policy-neutral output consumed by ``LazyOffloadManager``.

    Attributes:
        items: What to submit now, one entry per request.
        emptied_request_ids: Requests whose buffer became empty during this
            drain, whether by emission or by dropping invalidated
            operations. This is a buffer fact only: the manager combines it
            with the request's phase and in-flight batch before tearing a
            session down.
        holds: Operations whose blocks the manager must pin now, one entry
            per request. Each entry adds to the request's held operations,
            which need not be a prefix of its buffer. They stay buffered.
        releases: Requests whose held operations the manager must unpin
            before pinning ``holds``. They stay buffered, unheld.
        dropped_holds: Held operations the drain dropped because an earlier
            operation of their request lost its data, one entry per request.
            The manager unpins their blocks; they are no longer buffered.
    """

    items: list[PendingStoreItem] = field(default_factory=list)
    emptied_request_ids: list[str] = field(default_factory=list)
    holds: list[PendingStoreItem] = field(default_factory=list)
    releases: list[str] = field(default_factory=list)
    dropped_holds: list[PendingStoreItem] = field(default_factory=list)


@dataclass
class AdmissionProtection:
    """What the manager must do before an admission or a resumed load allocates.

    Attributes:
        to_tail: Operations to move to the free-queue tail, one entry per
            request, in prefix order. They stay allocatable, so the
            admission is not refused for them, and stay buffered: the
            danger window meets them again when the head reaches the tail.
        holds: Operations to pin, because the allocation would reach the
            tail too. Like ``LazyOffloadDrain.holds``, each entry adds to
            the request's held operations; pinning may make vLLM defer the
            admission, which is cheaper than losing the store.
    """

    to_tail: list[PendingStoreItem] = field(default_factory=list)
    holds: list[PendingStoreItem] = field(default_factory=list)


@dataclass
class LookupFlush:
    """Buffered operations a request's lookup made due.

    Attributes:
        holds: The operations newly held, one entry per request, for the
            manager to pin; ones already held keep their pins.
        ends: Per request whose operations were flushed, the token the
            flushed range ends at. The lookup waits while one of them
            overlapping the range from its local hit is still buffered due
            or in flight.
    """

    holds: list[PendingStoreItem] = field(default_factory=list)
    ends: dict[str, int] = field(default_factory=dict)


class OffloadPolicy(ABC):
    """Scheduler-side decision logic for deferring cache stores.

    Implementations buffer store metadata by request and decide, once per
    step, what to release. ``LazyOffloadManager`` owns every GPU and
    connector side effect and calls these from the scheduler thread only.
    """

    @abstractmethod
    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
    ) -> PendingStoreItem:
        """Buffer one store operation instead of submitting it.

        One request's operations arrive in token order, each starting
        where the previous one ended, and a policy must keep them in that
        order: they are submitted as one coalesced store. A preempted
        request restarts and offers its tokens again; such an offer
        supersedes the request's buffered operations it overlaps.

        An operation belonging to a request whose prefix chain is already
        broken is dropped here rather than buffered.

        Args:
            meta: The store operation: the request id, the token range, and
                the GPU blocks holding that range's KV.
            block_hashes: Prefix-cache hash of every block covering
                ``meta``'s token range, read now. The policy keeps this
                snapshot and compares against it later; a block whose hash
                changed was recycled, so the buffered data is gone.

        Returns:
            The superseded operations, for the manager to release their
            pins; empty when the offer superseded none.
        """

    @abstractmethod
    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Decide which buffered operations one scheduler step releases.

        Args:
            signals: This step's block-allocation pressure and the request
                ids the manager reports as finished or blocked.

        Returns:
            The stores to submit and the requests left with nothing buffered.
        """

    def protect(self, num_blocks: int) -> AdmissionProtection:
        """Protect what an imminent allocation of ``num_blocks`` would recycle.

        Called right before vLLM allocates blocks for a request it admits or
        whose asynchronous load just finished, which evicts from the
        free-queue head before the step's drain runs. The default protects
        nothing, for a policy that accepts losing buffered operations to
        such an allocation.

        Args:
            num_blocks: Upper bound on the blocks the allocation consumes.

        Returns:
            The operations to move to the tail and the ones to pin.
        """
        return AdmissionProtection()

    def hold_past(self, request_id: str, start: int) -> PendingStoreItem:
        """Hold a request's buffered operations ending past ``start``, due now.

        Called when the request finishes, before vLLM frees its blocks, with
        ``start`` the prefix vLLM's prefix cache can serve from them. The
        operations past it are not served from the GPU, so deferring them
        only risks their data; pinned now, while the request still owns
        their blocks, they keep their place in eviction order until stored.
        A policy that does not defer by what the GPU cache serves holds
        nothing.

        Args:
            request_id: The finishing request. An id with nothing buffered
                is ignored.
            start: Token the GPU cache serves the request's prefix up to.

        Returns:
            The newly held operations, in token order, for the manager to
            pin; empty when there are none.
        """
        return PendingStoreItem(request_id=request_id)

    def flush_for_lookup(self, start: int, ends: dict[str, int]) -> LookupFlush:
        """Hold the buffered operations a waiting request's lookup needs.

        Called before a request's LMCache lookup, with ``start`` the prefix
        vLLM's prefix cache serves it and ``ends`` the other requests whose
        buffered operations cover chunks of its prefix past ``start``.
        Deferred, those chunks are neither on the GPU for this request nor
        in L1, so it would recompute them. A policy that
        does not defer by what the GPU cache serves flushes nothing.

        Args:
            start: Token vLLM's prefix cache serves the waiting request up
                to.
            ends: Request id -> token up to which its operations cover the
                waiting request's prefix.

        Returns:
            The operations flushed and the newly held ones to pin.
        """
        return LookupFlush()

    def is_flushing(self, request_id: str, start: int, end: int) -> bool:
        """Whether a held or expedited op of a request in a range is buffered.

        Args:
            request_id: A request whose operations a lookup flushed.
            start: First token of the range.
            end: Token the range ends at.

        Returns:
            True while a held or expedited operation overlapping
            ``[start, end)`` is buffered; such an operation is stored by a
            later drain.
        """
        return False

    @abstractmethod
    def has_pending_request(self, request_id: str) -> bool:
        """Whether the request currently owns buffered operations.

        Args:
            request_id: The vLLM request id to query.

        Returns:
            True while at least one of its operations is still buffered.
        """

    @abstractmethod
    def buffered_block_ids(self) -> set[int]:
        """Every GPU block a buffered operation covers, of any request.

        Requests that share a prefix share its blocks, so a block one
        request's store just released can still back another request's
        buffered operation.

        Returns:
            The block ids, possibly empty.
        """

    @abstractmethod
    def restart_request(self, request_id: str) -> None:
        """Note that a preempted request restarts under its id.

        vLLM freed the request's blocks when it preempted it. Its hashed
        blocks keep their hashes in the free queue, so they are prefix-cache
        victims like any other and its buffered operations stay, to be
        stored if they come due. The restarted request offers its tokens
        again from where its hit ends, and those offers supersede the
        buffered operations they overlap (see :meth:`add`). The policy
        forgets that the request's prefix chain was broken: the restart
        computes the lost range again.

        Args:
            request_id: The preempted request.
        """

    @abstractmethod
    def discard_for_reuse(self, request_id: str) -> None:
        """Discard what a finished request left behind before its id is reused.

        Called when a new, unrelated request arrives under a request id whose
        previous holder has finished but whose policy state is still around
        (its stores were deferred, or its teardown waits on a store receipt).
        Without this, the arriving request would inherit the finished one's
        buffered operations and broken-chain marker.

        Args:
            request_id: The reused request id, whose previous holder's state
                is discarded.
        """

    @abstractmethod
    def release_request(self, request_id: str) -> None:
        """Forget a request's leftover state once its session is torn down.

        Called by the manager when it ends the session of a request that
        has nothing buffered and nothing in flight. Only state outside the
        pending buffer is affected -- the broken-chain marker -- so that
        finished request ids do not accumulate.

        Args:
            request_id: The request whose session is being torn down.
        """

    @abstractmethod
    def requeue_failed(self, item: PendingStoreItem) -> None:
        """Buffer again the operations of a store batch that failed.

        An operation that overlaps one the request offered since is dropped:
        the newer offer stores the same tokens.

        The store failed (a worker failed, or the server was unhealthy),
        but the data is still on the GPU: the batch pinned its blocks until
        every worker reported. Every later operation of the request depends
        on this range, so the policy buffers ``item`` in token order among
        them; it is emitted again under the policy's normal rule. A store of
        keys that are already resident succeeds, so a retry of a partly
        stored batch is safe.
        After repeated failures the manager follows this call with
        ``mark_store_failed``.

        Args:
            item: The failed batch's operations in token order, with the
                hash snapshots taken when they were first buffered.
        """

    @abstractmethod
    def mark_store_failed(self, request_id: str) -> int:
        """Give up a request's stores after its store failed for good.

        The manager calls this once a request's store batches failed
        ``MAX_STORE_ATTEMPTS`` times in a row, after requeueing the last
        one. Every buffered operation of the request is dropped. A policy
        that keeps prefix-chain state also refuses the request's later
        operations, which would be unreachable on retrieval without the
        failed range.

        Args:
            request_id: The request whose stores are given up.

        Returns:
            The number of buffered operations dropped.
        """

    @abstractmethod
    def log_final_stats(self) -> None:
        """Write the policy's final counters at connector shutdown."""
