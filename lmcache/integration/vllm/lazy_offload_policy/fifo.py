# SPDX-License-Identifier: Apache-2.0
"""FIFO lazy-offload drain policy."""

# Standard
from typing import TYPE_CHECKING, cast

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    BufferedOpView,
    ConfigValue,
    DrainSignals,
    LazyOffloadDrain,
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.utils import init_logger as lmcache_init_logger

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


logger = lmcache_init_logger(__name__)


class FIFOOffloadPolicy(OffloadPolicy):
    """Buffer by request and drain eligible requests in FIFO order.

    A drain happens once enough finished requests have accumulated, and
    releases whole requests in admission order. It reads no GPU state, so it
    never drops an operation whose blocks were recycled.
    """

    def __init__(self, configs: dict[str, ConfigValue]) -> None:
        """Read ``lazy_offload_threshold`` and ``_select_count`` from configs.

        Args:
            configs: The connector's ``kv_connector_extra_config``. Only the
                two keys above are read; missing keys keep their defaults of
                100 eligible requests and 10 requests per drain.
        """
        self._pending_items: dict[str, PendingStoreItem] = {}
        # int() does the conversion; the cast only narrows ConfigValue to
        # what int() accepts, since a JSON config may carry the number as a
        # string.
        self._threshold = int(
            cast(
                "str | int | float",
                configs.get("lmcache.mp.lazy_offload_threshold", 100),
            )
        )
        self._select_count = int(
            cast(
                "str | int | float",
                configs.get("lmcache.mp.lazy_offload_select_count", 10),
            )
        )
        logger.info(
            "lazy offload enabled with FIFO policy, offload threshold: %d",
            self._threshold,
        )

    def add(
        self,
        meta: "LMCacheMPRequestMetadata",
        block_hashes: BlockHashes,
    ) -> PendingStoreItem:
        """Queue one store operation under its request id.

        Args:
            meta: The store operation offered by the manager.
            block_hashes: Its admission-time block hashes, kept untouched
                for the manager to re-validate. This policy does not read
                them, so it never drops an operation whose blocks were
                recycled.

        Returns:
            The queued operations of the request the offer overlaps, which
            it supersedes.
        """
        item = self._pending_items.setdefault(
            meta.request_id, PendingStoreItem(request_id=meta.request_id)
        )
        superseded = PendingStoreItem(request_id=meta.request_id)
        kept: list[tuple["LMCacheMPRequestMetadata", BlockHashes]] = []
        for entry in item.metadatas:
            overlaps = (
                entry[0].op.start < meta.op.end and entry[0].op.end > meta.op.start
            )
            (superseded.metadatas if overlaps else kept).append(entry)
        kept.append((meta, block_hashes))
        kept.sort(key=lambda entry: entry[0].op.start)
        item.metadatas = kept
        return superseded

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        """Release eligible finished requests once the threshold is met.

        Args:
            signals: This step's signals from the manager. Only the
                request-id sets are read: a request is eligible when it
                is finished and not blocked, and the drain happens once
                enough eligible requests have buffered operations. Block
                pressure is ignored.

        Returns:
            Up to ``select_count`` whole requests in admission order, each
            also reported as emptied.
        """
        eligible_ids = signals.finished_request_ids - signals.blocked_request_ids
        eligible_count = sum(
            request_id in self._pending_items for request_id in eligible_ids
        )
        if eligible_count < self._threshold:
            return LazyOffloadDrain()
        items: list[PendingStoreItem] = []
        for request_id in list(self._pending_items):
            if request_id not in eligible_ids:
                continue
            items.append(self._pending_items.pop(request_id))
            if len(items) >= self._select_count:
                break
        return LazyOffloadDrain(
            items=items,
            emptied_request_ids=[item.request_id for item in items],
        )

    def has_pending_request(self, request_id: str) -> bool:
        """Whether the request currently owns buffered operations.

        Args:
            request_id: The request id to query.

        Returns:
            True while at least one of its operations is buffered.
        """
        return request_id in self._pending_items

    def buffered_ops(self) -> list[BufferedOpView]:
        """Copy every queued operation; see ``OffloadPolicy``.

        FIFO neither holds nor expedites, so both flags are False.

        Returns:
            One view per queued operation, requests in admission order, ops
            in token order.
        """
        return [
            BufferedOpView(
                request_id=request_id,
                start=meta.op.start,
                end=meta.op.end,
                block_ids=tuple(tuple(group) for group in meta.op.block_ids),
                held=False,
                expedited=False,
            )
            for request_id, item in self._pending_items.items()
            for meta, _ in item.metadatas
        ]

    def buffered_block_ids(self) -> set[int]:
        """Every GPU block a buffered operation covers.

        Returns:
            The block ids, possibly empty.
        """
        return {
            block_id
            for item in self._pending_items.values()
            for _, block_hashes in item.metadatas
            for block_id in block_hashes
        }

    def restart_request(self, request_id: str) -> None:
        """FIFO keeps a restarted request's operations as they are.

        Args:
            request_id: The preempted request. Unused.
        """

    def discard_for_reuse(self, request_id: str) -> None:
        """Discard what the finished holder of this id left buffered.

        Args:
            request_id: The id a new request is taking over.
        """
        self._pending_items.pop(request_id, None)

    def release_request(self, request_id: str) -> None:
        """FIFO has no non-pending per-request state to release.

        Args:
            request_id: The request whose session was torn down. Unused.
        """

    def requeue_failed(self, item: PendingStoreItem) -> None:
        """Buffer a failed batch again, ahead of the request's later ops.

        Args:
            item: The failed batch's operations, in token order.
        """
        pending = self._pending_items.setdefault(
            item.request_id, PendingStoreItem(request_id=item.request_id)
        )
        # An op overlapping a queued one predates it: a preempted request
        # restarted and offered the same tokens again.
        pending.metadatas.extend(
            [
                entry
                for entry in item.metadatas
                if not any(
                    entry[0].op.start < queued.op.end
                    and entry[0].op.end > queued.op.start
                    for queued, _ in pending.metadatas
                )
            ]
        )
        pending.metadatas.sort(key=lambda entry: entry[0].op.start)

    def log_final_stats(self) -> None:
        """FIFO keeps no counters."""
