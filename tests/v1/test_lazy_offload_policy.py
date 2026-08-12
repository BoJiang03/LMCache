# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the eviction-aware lazy offload policy.

Pure policy tests: no vLLM, no torch, no GPU. The block pool is faked
through the ``BlockPoolReader`` protocol.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_policy import (
    AdmitResult,
    EvictionAwareStoreQueue,
    LazyOffloadPolicyConfig,
    PendingStoreOp,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


class FakePoolView:
    """In-memory BlockPoolReader: a free queue (head first) and a hash map."""

    def __init__(self) -> None:
        self.free_queue: list[int] = []
        self.hashes: dict[int, bytes] = {}

    def free_queue_ranks(self) -> dict[int, int]:
        return {block_id: rank for rank, block_id in enumerate(self.free_queue)}

    def block_hash(self, block_id: int) -> bytes | None:
        return self.hashes.get(block_id)

    def evict(self, block_id: int) -> None:
        """Simulate eviction + reallocation: hash reset, out of the queue."""
        self.free_queue.remove(block_id)
        del self.hashes[block_id]


@dataclass
class FakeStoreMetadata:
    """Opaque payload standing in for LMCacheMPRequestMetadata."""

    label: str


def make_op(
    request_id: str,
    block_ids: list[int],
    pool: FakePoolView,
    prefix_end_tokens: int,
    cache_salt: str = "",
    prefix_start_tokens: int = -1,
) -> PendingStoreOp:
    """Build a pending op whose hash snapshot matches the pool's state.

    ``prefix_start_tokens`` defaults to one 256-token chunk before the end,
    so consecutive ops built with 256-spaced ends form a contiguous chain;
    pass it explicitly to model a deduplication hole.
    """
    if prefix_start_tokens < 0:
        prefix_start_tokens = max(0, prefix_end_tokens - 256)
    return PendingStoreOp(
        request_id=request_id,
        store_metadata=cast(
            "LMCacheMPRequestMetadata",
            FakeStoreMetadata(label=f"{request_id}:{prefix_end_tokens}"),
        ),
        block_hashes={block_id: pool.hashes[block_id] for block_id in block_ids},
        prefix_start_tokens=prefix_start_tokens,
        prefix_end_tokens=prefix_end_tokens,
        cache_salt=cache_salt,
    )


def seed_blocks(pool: FakePoolView, block_ids: list[int], free: bool) -> None:
    """Give each block a distinct hash; optionally append to the free queue."""
    for block_id in block_ids:
        pool.hashes[block_id] = f"hash-{block_id}".encode()
        if free:
            pool.free_queue.append(block_id)


def make_queue(
    pool: FakePoolView,
    horizon_steps: float = 1.0,
    min_prefix_tokens: int = 0,
    max_drain_per_step: int = 64,
) -> EvictionAwareStoreQueue:
    config = LazyOffloadPolicyConfig(
        horizon_steps=horizon_steps,
        min_prefix_tokens=min_prefix_tokens,
        max_drain_per_step=max_drain_per_step,
    )
    return EvictionAwareStoreQueue(config, pool)


class TestConfigValidation:
    def test_rejects_non_positive_horizon(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(horizon_steps=0)

    def test_rejects_negative_min_prefix(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(min_prefix_tokens=-1)

    def test_rejects_zero_drain_cap(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(max_drain_per_step=0)


class TestAdmission:
    def test_admits_fully_hashed_op(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        result = queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED
        assert queue.num_pending_ops() == 1

    def test_rejects_op_with_unhashed_block(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        queue = make_queue(pool)
        op = PendingStoreOp(
            request_id="req",
            store_metadata=cast(
                "LMCacheMPRequestMetadata", FakeStoreMetadata(label="req")
            ),
            block_hashes={1: pool.hashes[1], 2: None},  # type: ignore[dict-item]
            prefix_start_tokens=0,
            prefix_end_tokens=256,
        )
        assert queue.admit(op) is AdmitResult.REJECTED_UNHASHED_BLOCK
        assert queue.num_pending_ops() == 0
        assert queue.stats().rejected_unhashed == 1


class TestPressureTrigger:
    def test_idle_engine_never_drains(self) -> None:
        """Free-queue position alone is never a trigger; pressure is required."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        result = queue.collect_due()
        assert result.to_store == []
        assert queue.num_pending_ops() == 1

    def test_depth_returns_to_zero_after_burst(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        for _ in range(10):
            queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert queue.collect_due().to_store == []

    def test_pressure_drains_blocks_within_danger_depth(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [3, 4], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        result = queue.collect_due()
        # danger depth 2: only blocks at ranks 0-1 (the first op) are at risk.
        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 1

    def test_feedforward_alone_triggers_drain(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=3)
        assert len(queue.collect_due().to_store) == 1

    def test_in_use_blocks_are_not_at_risk(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)  # hashed but not in free queue
        queue = make_queue(pool)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=8)
        assert queue.collect_due().to_store == []
        assert queue.num_pending_ops() == 1


class TestPrefixClosure:
    def test_due_later_op_flushes_earlier_ops_first(self) -> None:
        """A due chunk pulls its whole stored prefix out with it, in order."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)  # first chunk's blocks in use
        seed_blocks(pool, [3, 4], free=True)  # second chunk's blocks at risk
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [3, 4], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.to_store] == [256, 512]

    def test_eviction_drops_suffix_and_keeps_prefix(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.admit(make_op("req", [3], pool, prefix_end_tokens=768))
        pool.evict(2)  # middle chunk's data is lost
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.dropped_evicted] == [512, 768]
        assert queue.num_pending_ops() == 1  # the intact prefix stays pending
        assert queue.stats().dropped_evicted == 2

    def test_admission_rejected_after_prefix_break(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        pool.evict(1)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        queue.collect_due()
        result = queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        assert result is AdmitResult.REJECTED_PREFIX_BROKEN


class TestEconomyGate:
    def test_short_prefix_dropped_not_stored(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        result = queue.collect_due()
        assert result.to_store == []
        assert len(result.dropped_short_prefix) == 2
        assert queue.num_pending_ops() == 0
        assert queue.stats().rejected_short_prefix == 2

    def test_long_prefix_passes_gate(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=2048))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1

    def test_gate_disabled_by_default(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=16))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1


class TestStoreFailure:
    def test_store_failure_breaks_prefix_and_drops_held_back_ops(self) -> None:
        """A failed in-flight store leaves the request without its stored
        prefix: held-back operations must be dropped and later chunks
        rejected, or they would be stored unreachable."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=2.0, max_drain_per_step=1)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1  # first op in flight

        assert queue.mark_store_failed("req") == 1  # held-back op dropped
        assert queue.stats().dropped_failed_store == 1
        assert queue.notify_stored("req") is False  # request still running

        result = queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        assert result is AdmitResult.REJECTED_PREFIX_BROKEN

    def test_store_failure_of_finished_request_still_allows_teardown(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        assert queue.mark_request_finished("req") is True
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1

        queue.mark_store_failed("req")
        assert queue.notify_stored("req") is True

    def test_prefix_broken_cleared_when_finished_request_releases(self) -> None:
        """Teardown clears the broken-prefix mark, so a reused request id
        does not inherit it."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        assert queue.mark_request_finished("req") is True
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1  # in flight
        queue.mark_store_failed("req")  # prefix broken while in flight
        assert queue.notify_stored("req") is True  # teardown clears the mark

        seed_blocks(pool, [2], free=False)
        result = queue.admit(make_op("req", [2], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED


class TestContentDeduplication:
    """One pending op per unique content: requests sharing a hot prefix
    must not each buffer their own copy (the unbounded-growth case), and
    a deduplicated request must not defer its session teardown."""

    def test_identical_content_from_other_request_is_deduplicated(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        assert (
            queue.admit(make_op("req-a", [1, 2], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )
        assert (
            queue.admit(make_op("req-b", [1, 2], pool, prefix_end_tokens=256))
            is AdmitResult.DEDUPLICATED
        )
        assert queue.num_pending_ops() == 1
        assert queue.stats().deduplicated == 1
        # req-b has nothing pending: its session may be torn down now.
        assert queue.mark_request_finished("req-b") is False

    def test_hot_prefix_requests_keep_one_pending_op(self) -> None:
        """The round-2 repro: a hot shared prefix must not grow the queue
        by one op per request."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)  # hot: never in the free queue
        queue = make_queue(pool)
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=8)
        for i in range(100):
            request_id = f"req-{i}"
            queue.admit(make_op(request_id, [1, 2], pool, prefix_end_tokens=256))
            queue.collect_due()
            deferred = queue.mark_request_finished(request_id)
            assert deferred is (i == 0), "only the buffering request defers"
        assert queue.num_pending_ops() == 1

    def test_different_salt_is_not_deduplicated(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1, 2], pool, prefix_end_tokens=256))
        result = queue.admit(
            make_op("req-b", [1, 2], pool, prefix_end_tokens=256, cache_salt="s")
        )
        assert result is AdmitResult.ADMITTED
        assert queue.num_pending_ops() == 2

    def test_content_admittable_again_after_emission(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1
        result = queue.admit(make_op("req-b", [1], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED

    def test_content_admittable_again_after_eviction_drop(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        op_b = make_op("req-b", [1], pool, prefix_end_tokens=256)
        pool.evict(1)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert len(queue.collect_due().dropped_evicted) == 1
        # req-b snapshotted before the eviction; only req-a's chain broke.
        assert queue.admit(op_b) is AdmitResult.ADMITTED

    def test_content_admittable_again_after_short_prefix_drop(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().dropped_short_prefix) == 1
        result = queue.admit(make_op("req-b", [1], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED

    def test_content_admittable_again_after_drop_request(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        assert queue.drop_request("req-a") == 1
        result = queue.admit(make_op("req-b", [1], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED


class TestEmissionContiguity:
    """An emitted batch is coalesced into one contiguous store operation, so
    emission must never span a deduplication hole in the pending list."""

    def _queue_with_hole(self) -> "tuple[FakePoolView, EvictionAwareStoreQueue]":
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        seed_blocks(pool, [3], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        # [256, 512) was deduplicated under another request: the pending
        # list has a hole between the two admitted ops.
        queue.admit(
            make_op("req", [3], pool, prefix_end_tokens=768, prefix_start_tokens=512)
        )
        queue.observe_step(new_blocks_allocated=4, est_next_step_blocks=4)
        return pool, queue

    def test_emission_stops_at_dedup_hole(self) -> None:
        _, queue = self._queue_with_hole()
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 1

    def test_post_hole_op_emitted_in_next_batch_after_receipt(self) -> None:
        _, queue = self._queue_with_hole()
        queue.collect_due()
        queue.notify_stored("req")
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.to_store] == [768]
        assert queue.num_pending_ops() == 0


class TestDrainOrderingAndCap:
    def test_most_imminent_request_drains_first(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)  # ranks 0..3
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-late", [3, 4], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-soon", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-soon", "req-late"]

    def test_drain_cap_cuts_from_the_tail(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        first = queue.collect_due()
        assert [op.prefix_end_tokens for op in first.to_store] == [256]
        assert queue.num_pending_ops() == 1
        queue.notify_stored("req")  # first batch completes
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        second = queue.collect_due()
        assert [op.prefix_end_tokens for op in second.to_store] == [512]


class TestRequestLifecycle:
    def test_finished_request_released_only_after_store_completes(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        assert queue.mark_request_finished("req") is True
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        result = queue.collect_due()
        assert len(result.to_store) == 1
        # The emitted batch is in flight: not released until completion.
        assert result.released_requests == []
        assert queue.notify_stored("req") is True

    def test_finish_while_in_flight_defers_release(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1
        # Request finishes while its only batch is still in flight.
        assert queue.mark_request_finished("req") is True
        assert queue.notify_stored("req") is True

    def test_in_flight_request_held_back_until_notify(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        first = queue.collect_due()
        assert [op.prefix_end_tokens for op in first.to_store] == [256]
        # Second op is due but the request has a batch in flight.
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        assert queue.collect_due().to_store == []
        assert queue.notify_stored("req") is False  # still one op pending
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        second = queue.collect_due()
        assert [op.prefix_end_tokens for op in second.to_store] == [512]

    def test_finish_with_nothing_pending_releases_immediately(self) -> None:
        queue = make_queue(FakePoolView())
        assert queue.mark_request_finished("req") is False

    def test_drop_request_discards_pending_ops(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        assert queue.drop_request("req") == 2
        assert queue.num_pending_ops() == 0
        assert queue.stats().dropped_on_request_drop == 2

    def test_drop_request_keeps_in_flight_batch_tracked(self) -> None:
        """Dropping a request's pending ops must not forget its in-flight
        batch: an op re-admitted afterwards (e.g. after a preemption
        resume) stays blocked until the completion receipt arrives via
        ``notify_stored`` -- the worker allows one outstanding store per
        request."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        # horizon covers both queue ranks: block 2 is due at every step.
        queue = make_queue(pool, horizon_steps=2.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1  # now in flight
        assert queue.drop_request("req") == 0  # nothing left pending

        queue.admit(make_op("req", [2], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert queue.collect_due().to_store == []  # held back

        assert queue.notify_stored("req") is False
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1
