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
- **Store batch**: the emitted operations of one request, coalesced into the
  one store the worker accepts per request. A batch is *in flight* from its
  submission until every worker has reported it, and the request's later
  operations stay buffered for as long as that lasts.
- **Held**: a buffered operation whose GPU blocks the manager pinned while
  the request's batch was in flight, so they cannot be recycled behind a
  slow store. A held operation cannot lose its data; it is submitted once
  the batch settles, and the batch takes over the pin.
- **Protected**: a buffered operation an imminent admission would recycle.
  The manager moves its blocks to the free-queue tail instead of pinning
  them, so the admission still has every free block to allocate from, and
  the next drain emits it. Only when the allocation would reach the tail
  too are they held instead.
- **Requeued**: an emitted operation whose store failed, handed back to the
  policy once its batch settled. Its blocks were pinned in flight, so its
  hash snapshot still holds, and it is buffered again in token order among
  the request's buffered operations.
- **Readmitted**: a stored operation the manager buffers again because
  vLLM's GPU prefix cache still held its blocks when it was emitted, so
  exclusive retrieve stored it at L1's evict-first end. The manager keeps it
  aside, unpinned, until the holder releases one of its blocks, and then
  hands it back to the policy, which stores it again like any other
  operation before that block is recycled.
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
        reaches_tail: Whether the next step can allocate the whole free
            queue, so blocks freed before it are in reach too. The manager
            then pins what it keeps aside for readmission until the next
            drain.
    """

    items: list[PendingStoreItem] = field(default_factory=list)
    emptied_request_ids: list[str] = field(default_factory=list)
    holds: list[PendingStoreItem] = field(default_factory=list)
    releases: list[str] = field(default_factory=list)
    reaches_tail: bool = False


@dataclass
class AdmissionProtection:
    """What the manager must do before an admission allocates.

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

        Called right before vLLM allocates blocks for a request it admits,
        which evicts from the free-queue head before the step's drain runs.
        A policy that never lets buffered blocks be recycled returns nothing.

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
        """Whether a due operation of a request in a range is still buffered.

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

        The worker rejected the store (L1 was out of space, or the server
        was unhealthy), but the data is still on the GPU: the batch pinned
        its blocks until every worker reported. Every later operation of the
        request depends on this range, so the policy buffers ``item`` in
        token order among them and submits it again. A store of keys that are
        already resident succeeds, so a retry of a partly stored batch is safe.

        Args:
            item: The failed batch's operations in token order, with the
                hash snapshots taken when they were first buffered.
        """

    @abstractmethod
    def readmit(self, item: PendingStoreItem) -> None:
        """Buffer again stored operations whose blocks vLLM still held.

        As in :meth:`requeue_failed`, an operation that overlaps a newer
        offer of the request is dropped.

        Exclusive retrieve stored them at L1's evict-first end because
        vLLM's GPU prefix cache held their blocks; the holder has since
        released one of them, so they are stored again, like any buffered
        operation, before that block is recycled. They precede every
        operation of the request emitted after them, so the policy buffers
        them in token order among the request's buffered operations; ranges
        between them may be missing.

        Args:
            item: The operations, in token order, each with the hash
                snapshot of the blocks it still covers.
        """

    @abstractmethod
    def expedite(self, request_id: str, end: int) -> None:
        """Make a request's buffered operations below ``end`` due now.

        Exclusive retrieve keeps a chunk that two running requests hold in
        L1's most recently used end, written there if missing, and lets it
        fall back to the evict-first end once one holder is left; the
        manager stores the chunk again for each change. The operations stay
        buffered until a drain emits them, which is the next one that has
        budget and finds the request without a batch in flight; they are
        never held for it, since nothing is about to recycle their blocks.

        Args:
            request_id: The request whose operations are expedited. An id
                with nothing buffered is ignored.
            end: Token the expedited range ends at: every buffered
                operation of the request starting before it is due.
        """

    @abstractmethod
    def log_final_stats(self) -> None:
        """Write the policy's final counters at connector shutdown."""
