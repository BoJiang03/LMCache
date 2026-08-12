# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
# Standard
from abc import ABC, abstractmethod
from collections import defaultdict
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, cast
import enum

# First Party
from lmcache.integration.vllm.lazy_offload_policy import (
    AdmitResult,
    DrainResult,
    EvictionAwareStoreQueue,
    GPUBlockPoolView,
    LazyOffloadPolicyConfig,
    PendingStoreOp,
)
from lmcache.utils import init_logger as lmcache_init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    # First Party
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPRequestMetadata


logger = lmcache_init_logger(__name__)


class LazyOffloadMode(enum.Enum):
    """Which drain policy drives the pending store.

    - FIFO: count-triggered whole-request drain (legacy placeholder).
    - EVICTION_AWARE: pressure-triggered drain in free-queue LRU order
      (see ``lazy_offload_policy.py`` and the decision-model design doc).
    """

    FIFO = "FIFO"
    EVICTION_AWARE = "EVICTION_AWARE"


class AddOutcome(enum.Enum):
    """Result of buffering a store operation.

    - BUFFERED: the operation is pending; it will be drained later.
    - SKIPPED_UNHASHED: a covered block has no prefix-cache hash, so its
      later eviction would be undetectable; the operation is not buffered
      and must not be stored (lazy offload requires prefix caching).
    - SKIPPED_PREFIX_BROKEN: an earlier chunk of the request was already
      dropped; storing this one would be unreachable on retrieval.
    """

    BUFFERED = enum.auto()
    SKIPPED_UNHASHED = enum.auto()
    SKIPPED_PREFIX_BROKEN = enum.auto()


@dataclass
class PendingStoreItem:
    """
    Represents a pending store operation in the lazy offload queue.

    Attributes:
        request_id: The request id of the pending store request.
        metadatas: The store metadata to be submitted.
        is_finished: Whether the request is finished.
    """

    request_id: str
    metadatas: list[tuple["LMCacheMPRequestMetadata", dict[int, bytes]]] = field(
        default_factory=list
    )
    is_finished: bool = False


# TODO(chunxiaozheng): support more offload policies
class OffloadPolicy(ABC):
    """
    Abstract base class for lazy offload policies.

    Subclasses define when to trigger offload (should_offload) and
    which items to return (select_items).
    """

    @abstractmethod
    def add(self, meta: "LMCacheMPRequestMetadata", block_hashes: dict[int, bytes]):
        """Add a pending store item to the pending store."""
        ...

    @abstractmethod
    def mark_req_finished(self, req_id: str) -> bool:
        """Mark the pending store item finished.

        Returns:
            True if the request has buffered operations awaiting drain;
            False if nothing is pending for it.
        """
        ...

    @abstractmethod
    def should_offload(self) -> bool:
        """Determine whether the queue should be drained.

        Returns:
            True if offload should be triggered.
        """
        ...

    @abstractmethod
    def select_items(self, count: int) -> list[PendingStoreItem]:
        """Select which items to offload from the queue.

        Args:
            count: The number of items to select.

        Returns:
            A list of PendingStoreItem.
        """
        ...


class FIFOOffloadPolicy(OffloadPolicy):
    """
    FIFO offload policy: triggers when pending count reaches threshold,
    and returns a fixed batch_size number of items from the front.
    """

    def __init__(self, configs: dict | None = None):
        """
        Args:
            configs: The configuration for the FIFO offload policy.
        """
        self._pending_items: dict[str, PendingStoreItem] = {}
        self._threshold = (
            configs.get("lmcache.mp.lazy_offload_threshold", 100) if configs else 100
        )
        self._finished_requests_count = 0
        logger.info(
            "lazy offload enabled with FIFO policy, offload threshold: %d",
            self._threshold,
        )

    def add(self, meta: "LMCacheMPRequestMetadata", block_hashes: dict[int, bytes]):
        if meta.request_id not in self._pending_items:
            self._pending_items[meta.request_id] = PendingStoreItem(
                request_id=meta.request_id
            )
        self._pending_items[meta.request_id].metadatas.append((meta, block_hashes))

    def mark_req_finished(self, req_id: str) -> bool:
        # A request may finish without ever producing store metadata (e.g.
        # shorter than one chunk); that is not an error.
        if req_id in self._pending_items:
            self._pending_items[req_id].is_finished = True
            self._finished_requests_count += 1
            return True
        return False

    def should_offload(self) -> bool:
        return self._finished_requests_count >= self._threshold

    def select_items(self, count: int) -> list[PendingStoreItem]:
        to_offload = []
        for req_id in list(self._pending_items.keys()):
            if self._pending_items[req_id].is_finished:
                to_offload.append(self._pending_items[req_id])
                del self._pending_items[req_id]
                self._finished_requests_count -= 1
            if len(to_offload) >= count:
                break
        return to_offload


class LazyOffloadPendingStore:
    """
    Buffering store operations in lazy offload mode.

    Store metadata is accumulated here instead of being immediately submitted.
    When the offload policy decides it's time, a batch of items is drained
    and returned for submission.
    """

    def __init__(
        self,
        configs: dict | None = None,
    ):
        """
        Initialize the pending store queue.

        Args:
            configs: The kv_connector_extra_config dict. Recognized keys:
                ``lmcache.mp.lazy_offload_policy`` ("EVICTION_AWARE" default,
                or "FIFO"), ``lmcache.mp.lazy_offload_horizon_steps`` (float),
                ``lmcache.mp.lazy_offload_min_prefix_tokens`` (int),
                ``lmcache.mp.lazy_offload_max_drain_per_step`` (int), and the
                FIFO-only ``lmcache.mp.lazy_offload_threshold`` /
                ``lmcache.mp.lazy_offload_select_count``.

        Raises:
            ValueError: If the configured policy name is unknown.
        """
        configs = configs or {}
        policy = configs.get("lmcache.mp.lazy_offload_policy", "EVICTION_AWARE")
        try:
            self._mode = LazyOffloadMode(policy)
        except ValueError as e:
            raise ValueError(f"Unknown offload policy: {policy}") from e

        self._fifo_policy: FIFOOffloadPolicy | None = None
        # Built when the GPU block pool is bound (it needs the pool view).
        self._eviction_queue: EvictionAwareStoreQueue | None = None
        self._eviction_config = LazyOffloadPolicyConfig(
            horizon_steps=float(
                configs.get("lmcache.mp.lazy_offload_horizon_steps", 2.0)
            ),
            min_prefix_tokens=int(
                configs.get("lmcache.mp.lazy_offload_min_prefix_tokens", 0)
            ),
            max_drain_per_step=int(
                configs.get("lmcache.mp.lazy_offload_max_drain_per_step", 64)
            ),
        )
        if self._mode is LazyOffloadMode.FIFO:
            self._fifo_policy = FIFOOffloadPolicy(configs)
        else:
            logger.info(
                "lazy offload enabled with EVICTION_AWARE policy: %s",
                self._eviction_config,
            )

        self._select_count = configs.get("lmcache.mp.lazy_offload_select_count", 10)

        # GPU block pool reference
        self._gpu_block_pool: "BlockPool | None" = None

        # save all request block ids for free
        self._request_block_ids: dict[str, list[int]] = defaultdict(list)

    @property
    def mode(self) -> LazyOffloadMode:
        """The configured drain mode."""
        return self._mode

    def bind_gpu_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """Bind the GPU block pool to the pending store."""
        self._gpu_block_pool = gpu_block_pool
        if self._mode is LazyOffloadMode.EVICTION_AWARE:
            self._eviction_queue = EvictionAwareStoreQueue(
                self._eviction_config, GPUBlockPoolView(gpu_block_pool)
            )

    def add(self, meta: "LMCacheMPRequestMetadata") -> AddOutcome:
        """Buffer a store operation produced by ``GetStoreMetadata``.

        Args:
            meta: The store metadata to buffer.

        Returns:
            The buffering outcome; see :class:`AddOutcome` for the action
            the caller must take on each value.

        Raises:
            ValueError: If the GPU block pool has not been bound.
        """
        if not self._gpu_block_pool:
            raise ValueError("gpu block pool not bound")
        block_hashes = {
            bid: self._gpu_block_pool.blocks[bid].block_hash
            for bid in meta.op.flat_block_ids
        }
        if self._fifo_policy is not None:
            self._fifo_policy.add(meta, block_hashes)
            return AddOutcome.BUFFERED
        queue = self._require_eviction_queue()
        op = PendingStoreOp(
            request_id=meta.request_id,
            store_metadata=meta,
            # admit() rejects any None value in the snapshot.
            block_hashes=cast("dict[int, BlockHashWithGroupId]", block_hashes),
            prefix_end_tokens=meta.op.end,
        )
        admit = queue.admit(op)
        if admit is AdmitResult.ADMITTED:
            return AddOutcome.BUFFERED
        if admit is AdmitResult.REJECTED_UNHASHED_BLOCK:
            logger.warning(
                "Lazy offload: skipping store for request %s tokens [%d, %d): "
                "covered blocks lack prefix-cache hashes (is prefix caching "
                "enabled?)",
                meta.request_id,
                meta.op.start,
                meta.op.end,
            )
            return AddOutcome.SKIPPED_UNHASHED
        return AddOutcome.SKIPPED_PREFIX_BROKEN

    def observe_step(
        self, new_blocks_allocated: int, est_next_step_blocks: int
    ) -> None:
        """Forward one step's block-consumption signals to the policy.

        No-op in FIFO mode.

        Args:
            new_blocks_allocated: GPU blocks newly allocated this step.
            est_next_step_blocks: Estimated allocation of the next step.
        """
        if self._eviction_queue is not None:
            self._eviction_queue.observe_step(
                new_blocks_allocated, est_next_step_blocks
            )

    def collect_due(self) -> DrainResult:
        """Release the operations facing imminent eviction (EVICTION_AWARE).

        Returns:
            The policy's drain decision for this step.

        Raises:
            ValueError: If called in FIFO mode or before the pool is bound.
        """
        queue = self._require_eviction_queue()
        result = queue.collect_due()
        for dropped_op in result.dropped_evicted:
            logger.debug(
                "Lazy offload: dropped store for request %s (prefix %d): "
                "blocks evicted before drain",
                dropped_op.request_id,
                dropped_op.prefix_end_tokens,
            )
        return result

    def notify_store_complete(self, req_id: str) -> bool:
        """Record a completed store batch for a request.

        Args:
            req_id: The request whose store completion was reported.

        Returns:
            True if the request's session may now be torn down.
        """
        if self._eviction_queue is not None:
            return self._eviction_queue.notify_stored(req_id)
        # FIFO drains a request's buffered ops all at once, so the receipt
        # always ends the session.
        return True

    def should_offload(self) -> bool:
        """Check if the queue should be drained (FIFO mode only)."""
        return self._require_fifo_policy().should_offload()

    def select_items(self) -> list[PendingStoreItem]:
        """
        Drain items from the queue according to the policy (FIFO mode only).

        Returns:
            The pending store items to be submitted.
        """
        return self._require_fifo_policy().select_items(self._select_count)

    def mark_req_finished(self, req_id: str) -> bool:
        """Record that the engine finished a request.

        Args:
            req_id: The finished request.

        Returns:
            True if stores are still pending or in flight for the request
            (session teardown must wait); False otherwise.
        """
        if self._eviction_queue is not None:
            return self._eviction_queue.mark_request_finished(req_id)
        return self._require_fifo_policy().mark_req_finished(req_id)

    def _require_eviction_queue(self) -> EvictionAwareStoreQueue:
        if self._eviction_queue is None:
            raise ValueError(
                "EVICTION_AWARE queue unavailable: wrong mode or GPU block "
                "pool not bound"
            )
        return self._eviction_queue

    def _require_fifo_policy(self) -> FIFOOffloadPolicy:
        if self._fifo_policy is None:
            raise ValueError("FIFO policy unavailable in EVICTION_AWARE mode")
        return self._fifo_policy

    def has_in_flight_store(self, req_id: str) -> bool:
        """Whether a drained store batch of the request awaits its receipt.

        True from the drain that pinned and submitted the request's batch
        until the full set of worker completion receipts has been processed.
        A receipt for a request outside this window is a duplicate or stale
        resend and must be ignored (processing it would unpin blocks that
        are no longer pinned and tear the session down twice).

        Args:
            req_id: The request to check.

        Returns:
            True if a submitted store batch is awaiting completion.
        """
        return req_id in self._request_block_ids

    def update_request_gpu_block_ids(self, req_id: str, block_ids: list[int]):
        self._request_block_ids[req_id].extend(block_ids)

    def get_request_gpu_block_ids(self, req_id: str) -> list[int]:
        return self._request_block_ids[req_id]

    def remove_request_gpu_block_ids(self, req_id: str):
        if req_id in self._request_block_ids:
            del self._request_block_ids[req_id]
