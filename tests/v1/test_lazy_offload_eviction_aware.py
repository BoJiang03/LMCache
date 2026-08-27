# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Unit tests for the eviction-aware lazy offload policy.

Pure policy tests: no vLLM, no torch, no GPU. The block pool is faked
through the ``BlockPoolReader`` protocol.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Iterator, cast

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_policy.eviction_aware import (
    AdmitResult,
    EvictionAwareStoreQueue,
    LazyOffloadPolicyConfig,
    PendingStoreOp,
)

if TYPE_CHECKING:
    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestMetadata


class FakePoolView:
    """In-memory BlockPoolReader: a free queue (head first) and a hash map.

    ``free_queue_block_ids`` is a generator that counts the blocks the
    policy actually consumes into ``blocks_walked``. The production view
    walks a linked list on the scheduler's critical path, so a fake that
    handed over the whole queue at once would hide how deep a step reads --
    the quantity every token's decode latency pays for.
    """

    def __init__(self) -> None:
        self.free_queue: list[int] = []
        self.hashes: dict[int, bytes] = {}
        self.blocks_walked = 0
        self.hash_requests: list[int] = []

    def free_queue_block_ids(self) -> Iterator[int]:
        for block_id in self.free_queue:
            self.blocks_walked += 1
            yield block_id

    def is_free(self, block_id: int) -> bool:
        return block_id in self.free_queue

    def block_hash(self, block_id: int) -> bytes | None:
        self.hash_requests.append(block_id)
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
    epoch: int = 0,
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
        epoch=epoch,
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
    max_pending_ops: int = 0,
    max_drain_blocks_per_step: int = 0,
    idle_drain_max_ops: int = 0,
    idle_threshold_blocks: float = 1.0,
    degrade_l1_residence_secs: float = 0.0,
) -> EvictionAwareStoreQueue:
    config = LazyOffloadPolicyConfig(
        horizon_steps=horizon_steps,
        min_prefix_tokens=min_prefix_tokens,
        max_drain_per_step=max_drain_per_step,
        max_pending_ops=max_pending_ops,
        max_drain_blocks_per_step=max_drain_blocks_per_step,
        idle_drain_max_ops=idle_drain_max_ops,
        idle_threshold_blocks=idle_threshold_blocks,
        degrade_l1_residence_secs=degrade_l1_residence_secs,
    )
    return EvictionAwareStoreQueue(config, pool)


class TestConfigValidation:
    def test_default_horizon_uses_calibrated_value(self) -> None:
        assert LazyOffloadPolicyConfig().horizon_steps == 2.5

    def test_rejects_non_positive_horizon(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(horizon_steps=0)

    def test_rejects_negative_min_prefix(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(min_prefix_tokens=-1)

    def test_rejects_zero_drain_cap(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(max_drain_per_step=0)

    def test_backlog_is_unbounded_by_default(self) -> None:
        assert LazyOffloadPolicyConfig().max_pending_ops == 0

    def test_rejects_negative_backlog_cap(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(max_pending_ops=-1)

    def test_block_volume_is_unbounded_by_default(self) -> None:
        assert LazyOffloadPolicyConfig().max_drain_blocks_per_step == 0

    def test_rejects_negative_block_volume_cap(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(max_drain_blocks_per_step=-1)

    def test_idle_drain_is_disabled_by_default(self) -> None:
        assert LazyOffloadPolicyConfig().idle_drain_max_ops == 0

    def test_rejects_negative_idle_allowance(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(idle_drain_max_ops=-1)

    def test_rejects_non_positive_idle_threshold(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(idle_threshold_blocks=0)

    def test_degradation_is_disabled_by_default(self) -> None:
        assert LazyOffloadPolicyConfig().degrade_l1_residence_secs == 0.0

    def test_rejects_negative_degrade_residence(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(degrade_l1_residence_secs=-1.0)


class TestAdmission:
    def test_admits_fully_hashed_op(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        result = queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        assert result is AdmitResult.ADMITTED
        assert queue.num_pending_ops() == 1

    def test_rejects_mixed_epochs_for_one_request(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        assert (
            queue.admit(make_op("req", [1], pool, 256, epoch=3)) is AdmitResult.ADMITTED
        )

        with pytest.raises(RuntimeError, match="mixed store epochs 3 and 4"):
            queue.admit(make_op("req", [2], pool, 512, epoch=4))

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

    def test_unhashed_rejection_breaks_prefix_chain(self) -> None:
        """The caller's tracker has already advanced past the skipped range,
        so a later chunk would be stored without its prefix (the retrieval
        prefix lookup stops at the hole) -- it must be rejected."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        unhashed = PendingStoreOp(
            request_id="req",
            store_metadata=cast(
                "LMCacheMPRequestMetadata", FakeStoreMetadata(label="req")
            ),
            block_hashes={1: None},  # type: ignore[dict-item]
            prefix_start_tokens=0,
            prefix_end_tokens=256,
        )
        assert queue.admit(unhashed) is AdmitResult.REJECTED_UNHASHED_BLOCK
        later = make_op("req", [2], pool, prefix_end_tokens=512)
        assert queue.admit(later) is AdmitResult.REJECTED_PREFIX_BROKEN
        assert queue.num_pending_ops() == 0

    def test_chunks_admitted_before_unhashed_rejection_stay_storable(self) -> None:
        """Only chunks past the skipped range are unreachable; the prefix
        buffered before the rejection is intact and still emits."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        seed_blocks(pool, [2], free=False)
        queue = make_queue(pool)
        first = make_op("req", [1], pool, prefix_end_tokens=256)
        assert queue.admit(first) is AdmitResult.ADMITTED
        unhashed = PendingStoreOp(
            request_id="req",
            store_metadata=cast(
                "LMCacheMPRequestMetadata", FakeStoreMetadata(label="req")
            ),
            block_hashes={2: None},  # type: ignore[dict-item]
            prefix_start_tokens=256,
            prefix_end_tokens=512,
        )
        assert queue.admit(unhashed) is AdmitResult.REJECTED_UNHASHED_BLOCK
        assert queue.num_pending_ops() == 1
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.to_store] == [256]


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
    def test_short_chain_is_held_out_of_the_pending_machine(self) -> None:
        """Sub-break-even chunks are admitted into a side pen, not the
        pending machine: pressure neither emits nor validates them, so
        short requests cost nothing on the per-step path."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        assert (
            queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )
        assert (
            queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
            is AdmitResult.ADMITTED
        )
        assert queue.num_pending_ops() == 0
        assert queue.num_held_ops() == 2
        assert queue.has_pending_request("req")
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        result = queue.collect_due()
        assert result.to_store == []
        assert result.dropped_short_prefix == []
        assert queue.stats().requests_validated == 0
        assert queue.num_held_ops() == 2

    def test_finish_below_threshold_drops_the_held_chain(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        result = queue.collect_due(finished_request_ids={"req"})
        assert [op.prefix_end_tokens for op in result.dropped_short_prefix] == [
            256,
            512,
        ]
        assert result.emptied_requests == ["req"]
        assert queue.stats().rejected_short_prefix == 2
        assert queue.num_held_ops() == 0
        assert not queue.has_pending_request("req")

    def test_crossing_the_threshold_promotes_the_whole_chain(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool, horizon_steps=1.0, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        assert (
            queue.admit(
                make_op(
                    "req",
                    [3],
                    pool,
                    prefix_end_tokens=1024,
                    prefix_start_tokens=512,
                )
            )
            is AdmitResult.ADMITTED
        )
        assert queue.num_held_ops() == 0
        assert queue.num_pending_ops() == 3
        queue.observe_step(new_blocks_allocated=3, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.prefix_end_tokens for op in result.to_store] == [256, 512, 1024]

    def test_eviction_while_held_drops_chain_and_breaks_prefix(self) -> None:
        """Held ops are invisible to the per-step loss check, so the chain
        is validated at promotion; a block lost during the wait kills the
        whole chain (the intact front can never reach break-even)."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        pool.evict(2)
        result = queue.admit(make_op("req", [3], pool, prefix_end_tokens=1024))
        assert result is AdmitResult.REJECTED_PREFIX_BROKEN
        assert queue.num_pending_ops() == 0
        assert queue.num_held_ops() == 0
        stats = queue.stats()
        assert stats.rejected_short_prefix == 1  # the intact front (gate 3)
        assert stats.dropped_evicted == 1  # the lost op (gate 1)
        later = queue.admit(make_op("req", [3], pool, prefix_end_tokens=1280))
        assert later is AdmitResult.REJECTED_PREFIX_BROKEN

    def test_finished_request_with_pending_ops_is_untouched(self) -> None:
        """Finishing ends prefix growth, not the eviction clock: pending
        ops of a finished request wait for their blocks to come due."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=2048))
        result = queue.collect_due(finished_request_ids={"req"})
        assert result.dropped_short_prefix == []
        assert queue.num_pending_ops() == 1

    def test_drop_request_clears_the_held_chain(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        assert queue.drop_request("req") == 1
        assert queue.num_held_ops() == 0
        assert not queue.has_pending_request("req")
        assert queue.stats().dropped_on_request_drop == 1

    def test_rejects_mixed_epochs_across_the_held_chain(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool, min_prefix_tokens=1024)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256, epoch=0))
        with pytest.raises(RuntimeError):
            queue.admit(make_op("req", [2], pool, prefix_end_tokens=512, epoch=1))

    def test_truncated_chain_below_threshold_dropped_at_emission(self) -> None:
        """Backstop: eviction cuts a promoted chain back under break-even,
        so storing the surviving stub would cost more than recomputing it."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0, min_prefix_tokens=1024)
        queue.admit(
            make_op("req", [1], pool, prefix_end_tokens=512, prefix_start_tokens=0)
        )
        queue.admit(
            make_op("req", [2], pool, prefix_end_tokens=1024, prefix_start_tokens=512)
        )
        pool.evict(2)
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        result = queue.collect_due()
        assert result.to_store == []
        assert [op.prefix_end_tokens for op in result.dropped_evicted] == [1024]
        assert [op.prefix_end_tokens for op in result.dropped_short_prefix] == [512]
        assert queue.num_pending_ops() == 0

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

        result = queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        assert result is AdmitResult.REJECTED_PREFIX_BROKEN

    def test_failure_of_fresh_batch_after_reset_is_honored(self) -> None:
        """A current-epoch failure after reset breaks the chain as usual."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=2.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1
        queue.drop_request("req")

        seed_blocks(pool, [2], free=True)
        seed_blocks(pool, [3], free=False)
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [3], pool, prefix_end_tokens=512))
        assert len(queue.collect_due().to_store) == 1  # fresh batch in flight

        assert queue.mark_store_failed("req") == 1  # held-back op dropped
        result = queue.admit(make_op("req", [3], pool, prefix_end_tokens=768))
        assert result is AdmitResult.REJECTED_PREFIX_BROKEN


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
        assert not queue.has_pending_request("req-b")

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
            assert queue.has_pending_request(request_id) is (i == 0)
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
        finished = queue.collect_due(finished_request_ids={"req-a"})
        assert len(finished.dropped_short_prefix) == 1
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

    def test_cover_with_recycled_blocks_does_not_absorb_live_copy(self) -> None:
        """A dedup hit must verify the covering op's snapshot is still live.
        The covering op can already be a corpse while it waits in the pending
        list (its blocks recycled by this step's allocation, or its cleanup
        skipped while its request holds an in-flight batch); deduplicating
        against it would discard the only live copy of the content."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        corpse = make_op("req-a", [1], pool, prefix_end_tokens=256)
        assert queue.admit(corpse) is AdmitResult.ADMITTED
        # Block 1 evicted; req-b recomputed the same content into block 2
        # (block hashes are content-derived, so the chains are equal).
        content_hash = pool.hashes[1]
        pool.evict(1)
        pool.hashes[2] = content_hash
        live = make_op("req-b", [2], pool, prefix_end_tokens=256)
        assert queue.admit(live) is AdmitResult.ADMITTED
        assert queue.num_pending_ops() == 2

    def test_content_key_follows_live_copy_after_corpse_drop(self) -> None:
        """Dropping the corpse must not release the live copy's content key:
        a third identical admission still deduplicates against the live op."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        content_hash = pool.hashes[1]
        pool.evict(1)
        pool.hashes[2] = content_hash
        assert (
            queue.admit(make_op("req-b", [2], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        result = queue.collect_due()
        assert len(result.dropped_evicted) == 1  # req-a's corpse
        assert (
            queue.admit(make_op("req-c", [2], pool, prefix_end_tokens=256))
            is AdmitResult.DEDUPLICATED
        )
        assert queue.num_pending_ops() == 1

    def test_cover_with_corpse_earlier_sibling_does_not_absorb_live_copy(self) -> None:
        """The dedup liveness check must cover the whole prefix chain, not
        just the covering op: a cover whose earlier sibling is already a
        corpse is deterministically dropped by prefix closure on the next
        drain, so it must not absorb a live copy either. Requires the front
        block to die before the tail block, which hybrid/sliding-window
        block freeing can produce."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-a", [2], pool, prefix_end_tokens=512))
        # Front block recycled (corpse), tail block intact. req-b recomputed
        # the same content into fresh blocks (hashes are content-derived,
        # so the chains are equal).
        front_hash, tail_hash = pool.hashes[1], pool.hashes[2]
        pool.evict(1)
        pool.hashes[11] = front_hash
        pool.hashes[12] = tail_hash
        assert (
            queue.admit(make_op("req-b", [11], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )
        assert (
            queue.admit(make_op("req-b", [12], pool, prefix_end_tokens=512))
            is AdmitResult.ADMITTED
        )
        # req-a's doomed chain is swept; req-b now owns both content keys,
        # so a third identical request deduplicates against the live copy.
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert len(queue.collect_due().dropped_evicted) == 2
        assert (
            queue.admit(make_op("req-c", [12], pool, prefix_end_tokens=512))
            is AdmitResult.DEDUPLICATED
        )

    def test_cover_with_corpse_later_sibling_still_absorbs_duplicates(self) -> None:
        """Only corpses before the cover doom it: a later sibling's loss
        prefix-closes from its own position, leaving the cover storable, so
        the cover remains a valid deduplication target."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-a", [2], pool, prefix_end_tokens=512))
        front_hash = pool.hashes[1]
        pool.evict(2)  # tail corpse; the front op survives prefix closure
        pool.hashes[11] = front_hash
        assert (
            queue.admit(make_op("req-b", [11], pool, prefix_end_tokens=256))
            is AdmitResult.DEDUPLICATED
        )


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

    def test_equal_rank_preserves_request_admission_order(self) -> None:
        """Incremental set discovery must not change the historical tie break.

        This matters under sustained pressure: arbitrary request-id ordering
        changes which shared hot prefixes remain pending long enough to dedup.
        """
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-z-first", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-a-second", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(
            new_blocks_allocated=1,
            est_next_step_blocks=0,
            allocated_block_ids=set(),
        )
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == [
            "req-z-first",
            "req-a-second",
        ]

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
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        second = queue.collect_due()
        assert [op.prefix_end_tokens for op in second.to_store] == [512]

    def test_cap_reports_what_it_held_back(self) -> None:
        """The sizing sensor: a drain the cap cut reports the ops it did not
        emit, and counts itself once regardless of how many it held."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        for index, block in enumerate([1, 2, 3]):
            queue.admit(
                make_op("req", [block], pool, prefix_end_tokens=256 * (index + 1))
            )
        queue.observe_step(new_blocks_allocated=3, est_next_step_blocks=0)
        result = queue.collect_due()
        assert len(result.to_store) == 1
        assert result.ops_held_back == 2
        assert queue.stats().throttled_drains == 1

    def test_uncapped_drain_holds_nothing_back(self) -> None:
        """The sensor must stay silent on the default cap, or it would read
        as a misconfiguration on every healthy deployment."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
        result = queue.collect_due()
        assert len(result.to_store) == 2
        assert result.ops_held_back == 0
        assert queue.stats().throttled_drains == 0


class TestPinCascadeShift:
    """Emitting a segment pins its blocks out of the free queue, shifting
    every block behind them toward the head before the next allocation runs.
    collect_due extends the due threshold by the blocks already emitted in
    the same call so shifted candidates drain now instead of losing the race.
    """

    def test_emission_shift_pulls_next_candidate_into_the_window(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4, 5], free=True)  # ranks 0..4
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1, 2, 3], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [4, 5], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # danger_depth = 1: req-a is due (rank 0). Pinning its 3 blocks
        # will move req-b (min rank 3) to rank 0 before the next step's
        # allocation, so req-b must drain in the same call.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a", "req-b"]

    def test_in_use_blocks_do_not_expand_the_shift(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        seed_blocks(pool, [9], free=False)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1, 9], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [3], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # req-a pins two blocks, but only block 1 leaves the free queue.
        # Block 3 moves from rank 2 to rank 1, exactly outside depth 1.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a"]
        assert queue.num_pending_ops() == 1

    def test_shared_blocks_expand_the_shift_only_once(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4, 5], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1, 2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [2, 3], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-c", [5], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # req-a removes blocks 1 and 2; req-b then removes only block 3,
        # because shared block 2 is already pinned. Block 5 moves from rank
        # 4 to rank 1, which is exactly outside depth 1.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a", "req-b"]
        assert queue.num_pending_ops() == 1

    def test_shift_never_opens_the_gate_by_itself(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [2, 3], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [4], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # danger_depth = 1 but no candidate reaches it (min ranks 1 and 3):
        # with no first emission there is no shift, and nothing drains.
        result = queue.collect_due()
        assert result.to_store == []
        assert queue.num_pending_ops() == 2

    def test_candidate_beyond_the_shifted_window_stays_pending(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [3, 4], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # danger_depth = 1, req-a due at rank 0 and pins one block; req-b's
        # min rank 2 is exactly at the shifted threshold (1 + 1), not below.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a"]
        assert queue.num_pending_ops() == 1

    def test_drain_cap_stops_the_cascade(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        queue.admit(make_op("req-a", [1, 2, 3], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [4], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        # req-b sits inside the shifted window (3 < 1 + 3) but the per-step
        # cap is exhausted by req-a's op, so req-b waits for the next step.
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a"]
        assert queue.num_pending_ops() == 1


class TestControllerEligibilityInputs:
    def test_blocked_request_is_held_until_controller_unblocks_it(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=1)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)

        assert len(queue.collect_due().to_store) == 1
        assert queue.collect_due({"req"}).to_store == []
        assert len(queue.collect_due().to_store) == 1

    def test_discard_for_reuse_clears_buffer_and_prefix_state(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.mark_store_failed("req")

        assert queue.discard_for_reuse("req") == 0
        assert (
            queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )

    def test_release_request_clears_non_pending_prefix_state(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)
        queue = make_queue(pool)
        queue.mark_store_failed("req")
        queue.release_request("req")

        assert (
            queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
            is AdmitResult.ADMITTED
        )


class TestFreeQueueSnapshotBound:
    """The per-step free-queue read is bounded, and the bound decides nothing.

    ``collect_due`` runs once per scheduler step on the critical path, so
    what it reads is paid by every request's TTFT and every token's TPOT.
    Reading the whole free queue is O(free blocks) -- tens of thousands on a
    pool with room to spare -- while the only ranks any decision compares
    are those within the danger depth, extended by the blocks this same call
    *does* pin out of the queue. Everything deeper is indistinguishable from
    a block that is not in the queue at all, which the policy already treats
    as not at risk.

    "Does", not "can": the depth a full-budget drain could reach is not what
    a step should pay for, because a step almost never drains a full budget.
    The read therefore follows the emissions rather than anticipating them,
    which leaves ``max_drain_per_step`` bounding the D2H burst and nothing
    else.
    """

    def test_snapshot_stops_at_danger_depth_plus_pending_blocks(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 101)), free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [90, 91], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=4)
        assert queue.collect_due().to_store == []
        # 4 blocks per step over a 1-step horizon, plus the 2 pending blocks
        # that an emission in this call could shift the queue by.
        assert pool.blocks_walked == 4

    def test_incremental_step_checks_only_bounded_candidate_requests(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 1001)), free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        for block_id in range(1, 1001):
            queue.admit(
                make_op(
                    f"req-{block_id}",
                    [block_id],
                    pool,
                    prefix_end_tokens=256,
                )
            )
        queue.observe_step(
            new_blocks_allocated=1,
            est_next_step_blocks=0,
            allocated_block_ids=set(),
        )
        queue.collect_due()

        # danger depth 1 plus at most 64 one-block emissions: neither the
        # free-list walk nor hash validation reaches the other 935 requests.
        assert pool.blocks_walked == 64
        assert set(pool.hash_requests) <= set(range(1, 66))

    def test_allocated_block_signal_revalidates_a_nonfree_op(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [3], pool, prefix_end_tokens=256))
        pool.evict(3)
        queue.observe_step(
            new_blocks_allocated=1,
            est_next_step_blocks=0,
            allocated_block_ids={3},
        )
        result = queue.collect_due()

        assert [op.request_id for op in result.dropped_evicted] == ["req"]
        assert queue.num_pending_ops() == 0

    def test_idle_step_reads_no_ranks_at_all(self) -> None:
        """No expected consumption means no rank can be below the danger
        depth, so the whole snapshot is dead work."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert queue.collect_due().to_store == []
        assert pool.blocks_walked == 0
        assert queue.num_pending_ops() == 1

    def test_idle_step_still_drops_ops_whose_blocks_were_evicted(self) -> None:
        """Skipping the snapshot must not skip the loss check: whether an
        op's data is still there is read from block hashes, not from ranks,
        and a request that lost its blocks while the engine idled has to be
        dropped on that step like any other."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        pool.evict(1)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        result = queue.collect_due()
        assert len(result.dropped_evicted) == 1
        assert queue.stats().dropped_evicted == 1
        assert pool.blocks_walked == 0

    def test_bound_covers_the_candidate_that_only_an_emission_shifts_into_reach(
        self,
    ) -> None:
        """The bound has to include the shift the call itself causes.

        `b` sits one rank past the danger depth, so it is not due on the
        depth alone; emitting `a` pins one block out of the queue ahead of
        it, which brings it inside. A snapshot cut at the danger depth would
        have shown `b`'s block as absent -- read as "not in the free queue,
        not at risk" -- and lost it to the next allocation.
        """
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("a", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("b", [3], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=2)
        result = queue.collect_due()
        assert {op.request_id for op in result.to_store} == {"a", "b"}
        # Danger depth 2, widened by the one block emitting `a` pinned.
        assert pool.blocks_walked == 3

    def test_read_depth_does_not_scale_with_the_drain_budget(self) -> None:
        """The budget bounds the emissions, not the read.

        Both settings face the same queue, the same backlog and the same
        danger depth, and neither has anything due; the ranks either one
        compares are the same ranks, so the walk has to be the same length.
        """
        walked = []
        for budget in (1, 64):
            pool = FakePoolView()
            seed_blocks(pool, list(range(1, 501)), free=True)
            queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=budget)
            for block_id in range(400, 500):
                queue.admit(
                    make_op(f"req-{block_id}", [block_id], pool, prefix_end_tokens=256)
                )
            queue.observe_step(new_blocks_allocated=3, est_next_step_blocks=0)
            assert queue.collect_due().to_store == []
            walked.append(pool.blocks_walked)
        assert walked == [3, 3]

    def test_read_depth_follows_the_shift_an_emission_causes(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 501)), free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_per_step=64)
        queue.admit(make_op("req", [1, 2, 3], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        assert len(queue.collect_due().to_store) == 1
        # Danger depth 1 plus the three blocks the emission actually pinned
        # out of the queue -- not 64 times the largest pending operation.
        assert pool.blocks_walked == 4

    def test_a_pin_deeper_than_the_window_still_counts_as_a_shift(self) -> None:
        """Queue membership is read from the pool, not from the window.

        `req-a` is due on its head block, but pinning it removes its deep
        block from the queue as well, and both moves `req-b` toward the
        head. Counting only the pins the window happened to cover would
        widen the window by one instead of two, leave `req-b`'s block
        unread, and lose it to the next allocation -- and the deeper the
        pin, the less likely the window is to have covered it.
        """
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 21)), free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req-a", [1, 15], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-b", [3], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["req-a", "req-b"]
        # Danger depth 1, widened to 3 by `req-a`'s two pins and to 4 by
        # `req-b`'s one -- the walk stops where the emissions stop.
        assert pool.blocks_walked == 4

    def test_counters_report_what_each_step_read_and_validated(self) -> None:
        """The decision loop's own cost is observable, not inferred.

        Nothing here is due on either step, which is the case that has to be
        cheap: the backlog sits far from the eviction head and the step
        still pays a walk and a validation pass for it.
        """
        pool = FakePoolView()
        seed_blocks(pool, list(range(1, 101)), free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.admit(make_op("req", [90, 91], pool, prefix_end_tokens=256))
        for _ in range(2):
            queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)
            assert queue.collect_due().to_store == []
        stats = queue.stats()
        assert stats.drain_steps == 2
        assert stats.free_queue_blocks_read == 4
        assert stats.requests_validated == 2
        assert stats.blocks_validated == 4


class TestCoveredPrefix:
    """A deferred store must be visible to the next request over that prefix.

    ``num_stored_tokens`` in the request tracker advances from an LMCache
    lookup, which only ever reports what the *server* holds. A buffered
    operation is by definition not there yet, so without these queries the
    follower request over a shared prefix re-stages all of it: the whole
    range is hashed, one reservation round-trip is paid per chunk, and the
    result is a single oversized store on the transfer thread that
    retrieves share. The server discards the duplicate content, so the
    symptom is latency rather than a wrong cache.
    """

    def test_empty_queue_covers_nothing(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        assert queue.covered_prefix_tokens("", {0: [1, 2]}, [16], 32, 0) == 0

    def test_pending_op_covers_a_shared_prefix(self) -> None:
        """The case the A/B run measured: turn N is buffered, turn N+1 asks."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2], pool, prefix_end_tokens=32))

        # Turn 2 shares blocks 1-2 through vLLM's prefix cache and adds 3-4.
        assert queue.covered_prefix_tokens("", {0: [1, 2, 3, 4]}, [16], 32, 0) == 32

    def test_uncovered_block_stops_the_run(self) -> None:
        """Only a leading run counts: a hole cannot be skipped over."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4, 5, 6], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2], pool, prefix_end_tokens=32))
        # Blocks 5-6 are covered too, but 3-4 are not, so the run ends at 2.
        queue.admit(make_op("other", [5, 6], pool, prefix_end_tokens=32))

        covered = queue.covered_prefix_tokens("", {0: [1, 2, 3, 4, 5, 6]}, [16], 32, 0)
        assert covered == 32

    def test_result_is_floored_to_a_chunk(self) -> None:
        """A store range must be chunk-aligned, so a partial chunk is not covered."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2, 3], pool, prefix_end_tokens=48))

        # Three 16-token blocks are covered; one 32-token chunk fits.
        assert queue.covered_prefix_tokens("", {0: [1, 2, 3]}, [16], 32, 0) == 32

    def test_a_different_salt_covers_nothing(self) -> None:
        """Two salts are two key namespaces; neither covers the other."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool)
        queue.admit(
            make_op("turn-1", [1, 2], pool, prefix_end_tokens=32, cache_salt="tenant-a")
        )

        assert queue.covered_prefix_tokens("tenant-b", {0: [1, 2]}, [16], 32, 0) == 0
        assert queue.covered_prefix_tokens("tenant-a", {0: [1, 2]}, [16], 32, 0) == 32

    def test_stale_snapshot_covers_nothing(self) -> None:
        """A pending op whose block was recycled has no data left to cover with."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2], pool, prefix_end_tokens=32))
        pool.evict(1)

        assert queue.covered_prefix_tokens("", {0: [1, 2]}, [16], 32, 0) == 0

    def test_dropping_an_op_restores_coverage_to_the_next_request(self) -> None:
        """Drop recovery: a lost op must not leave its range permanently skipped.

        The follower that already skipped the range cannot be recalled, but
        the range stops counting as covered the moment the operation dies,
        so the next request over that prefix stages it again.
        """
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2], pool, prefix_end_tokens=32))
        assert queue.covered_prefix_tokens("", {0: [1, 2]}, [16], 32, 0) == 32

        pool.evict(2)
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=1)
        result = queue.collect_due()
        assert len(result.dropped_evicted) == 1

        assert queue.covered_prefix_tokens("", {0: [1, 2]}, [16], 32, 0) == 0

    def test_emitting_an_op_releases_its_cover(self) -> None:
        """An emitted op leaves the pending index, so it stops answering here.

        The range is on its way to the server and a lookup will report it
        once the store lands; the gap between the two is the length of one
        store, whereas the gap this query exists to close is the whole
        eviction horizon.
        """
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2], pool, prefix_end_tokens=32))
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=2)
        result = queue.collect_due()
        assert len(result.to_store) == 1

        assert queue.covered_prefix_tokens("", {0: [1, 2]}, [16], 32, 0) == 0

    def test_least_covered_group_bounds_the_answer(self) -> None:
        """Hybrid models store a prefix only as far as every group has it."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 5], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2], pool, prefix_end_tokens=32))

        # Group 1's 32-token block is not covered, so nothing is storable.
        assert (
            queue.covered_prefix_tokens("", {0: [1, 2], 1: [5]}, [16, 32], 32, 0) == 0
        )

        queue.admit(make_op("turn-1b", [5], pool, prefix_end_tokens=32))
        assert (
            queue.covered_prefix_tokens("", {0: [1, 2], 1: [5]}, [16, 32], 32, 0) == 32
        )

    def test_probing_starts_at_the_callers_watermark(self) -> None:
        """Cost property: a request pays for each block once, not once per step.

        Walking from zero on every scheduler step would make the query
        O(prefix) per step on the scheduler's critical path -- thousands of
        block probes per step for the long prefixes this fix exists for.
        """
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2, 3, 4], pool, prefix_end_tokens=64))
        pool.hash_requests.clear()

        assert queue.covered_prefix_tokens("", {0: [1, 2, 3, 4]}, [16], 32, 32) == 64
        assert pool.hash_requests == [3, 4]

    def test_counters_separate_effect_from_cost(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2], pool, prefix_end_tokens=32))

        assert queue.covered_prefix_tokens("", {0: [1, 2, 3]}, [16], 32, 0) == 32
        stats = queue.stats()
        assert stats.covered_prefix_advances == 1
        assert stats.covered_prefix_tokens_skipped == 32
        # Blocks 1 and 2 matched, block 3 ended the run.
        assert stats.covered_blocks_probed == 3

    def test_a_skipped_prefix_is_outside_the_admission_ledger(self) -> None:
        """The skipped range never becomes an op, so no ledger counter moves."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=False)
        queue = make_queue(pool)
        queue.admit(make_op("turn-1", [1, 2], pool, prefix_end_tokens=32))
        before = queue.stats()

        queue.covered_prefix_tokens("", {0: [1, 2, 3]}, [16], 32, 0)
        after = queue.stats()

        assert after.admitted == before.admitted
        assert after.deduplicated == before.deduplicated
        assert after.emitted == before.emitted

    def test_rejects_invalid_arguments(self) -> None:
        pool = FakePoolView()
        queue = make_queue(pool)
        with pytest.raises(ValueError):
            queue.covered_prefix_tokens("", {0: [1]}, [16], 0, 0)
        with pytest.raises(ValueError):
            queue.covered_prefix_tokens("", {0: [1]}, [16], 32, -1)


class TestBacklogCap:
    """``max_pending_ops``: bound what one allocation burst can destroy.

    The danger depth is a forecast built from an EMA of per-step
    allocation, so a single admission that consumes thousands of blocks --
    the step that pays for the forecast is the step that destroys the
    backlog -- cannot be anticipated. These tests cover the second line of
    defence: capping how much content waits at once.
    """

    def _pin_ops(self, count: int) -> tuple[FakePoolView, EvictionAwareStoreQueue]:
        """Admit ``count`` ops of one request whose blocks are not free.

        No block is in the free queue, so no op is ever due under
        eviction pressure: only the backlog cap can release them.
        """
        pool = FakePoolView()
        blocks = list(range(1, count + 1))
        seed_blocks(pool, blocks, free=False)
        queue = make_queue(pool, max_pending_ops=2)
        for index, block in enumerate(blocks):
            queue.admit(
                make_op("req", [block], pool, prefix_end_tokens=256 * (index + 1))
            )
        return pool, queue

    def test_unbounded_backlog_emits_nothing_without_pressure(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4, 5], free=False)
        queue = make_queue(pool, max_pending_ops=0)
        for index, block in enumerate([1, 2, 3, 4, 5]):
            queue.admit(
                make_op("req", [block], pool, prefix_end_tokens=256 * (index + 1))
            )
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        result = queue.collect_due()

        assert result.to_store == []
        assert queue.num_pending_ops() == 5
        assert queue.stats().backlog_emitted == 0

    def test_cap_releases_the_oldest_until_the_backlog_fits(self) -> None:
        _, queue = self._pin_ops(5)
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        result = queue.collect_due()

        # Oldest first, and only down to the cap: 5 pending - 3 emitted = 2.
        assert [op.prefix_end_tokens for op in result.to_store] == [256, 512, 768]
        assert queue.num_pending_ops() == 2
        stats = queue.stats()
        assert stats.backlog_emitted == 3
        assert stats.emitted == 3

    def test_cap_never_binds_while_the_backlog_fits(self) -> None:
        _, queue = self._pin_ops(2)
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        result = queue.collect_due()

        assert result.to_store == []
        assert queue.stats().backlog_emitted == 0

    def test_cap_respects_the_per_step_drain_budget(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4, 5], free=False)
        queue = make_queue(pool, max_drain_per_step=2, max_pending_ops=1)
        for index, block in enumerate([1, 2, 3, 4, 5]):
            queue.admit(
                make_op("req", [block], pool, prefix_end_tokens=256 * (index + 1))
            )
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        first = queue.collect_due()

        assert [op.prefix_end_tokens for op in first.to_store] == [256, 512]
        assert queue.num_pending_ops() == 3
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        second = queue.collect_due()

        assert [op.prefix_end_tokens for op in second.to_store] == [768, 1024]
        assert queue.num_pending_ops() == 1

    def test_pressure_emission_leaves_the_budget_it_spent(self) -> None:
        """One batch per request: a request the pressure loop drained is not
        drained again by the backlog pass in the same step."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        seed_blocks(pool, [2, 3], free=False)
        queue = make_queue(pool, horizon_steps=1.0, max_pending_ops=1)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.admit(make_op("req", [3], pool, prefix_end_tokens=768))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)

        result = queue.collect_due()

        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 2
        assert queue.stats().backlog_emitted == 0

    def test_cap_skips_a_request_with_a_batch_in_flight(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=False)
        queue = make_queue(pool, max_pending_ops=1)
        for block, end in ((1, 256), (2, 512)):
            queue.admit(make_op("blocked", [block], pool, prefix_end_tokens=end))
        for block, end in ((3, 256), (4, 512)):
            queue.admit(make_op("free-to-go", [block], pool, prefix_end_tokens=end))
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        result = queue.collect_due(blocked_request_ids={"blocked"})

        assert {op.request_id for op in result.to_store} == {"free-to-go"}

    def test_cap_walks_requests_in_admission_order(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=False)
        queue = make_queue(pool, max_pending_ops=2)
        queue.admit(make_op("req-z-first", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-a-second", [2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req-m-third", [3], pool, prefix_end_tokens=256))
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        result = queue.collect_due()

        assert [op.request_id for op in result.to_store] == ["req-z-first"]

    def test_cap_drops_a_lost_op_instead_of_storing_stale_data(self) -> None:
        """The loss check runs on the backlog path too: an op whose block was
        reallocated must not be emitted under its stale snapshot."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4], free=False)
        queue = make_queue(pool, max_pending_ops=1)
        for index, block in enumerate([1, 2, 3, 4]):
            queue.admit(
                make_op("req", [block], pool, prefix_end_tokens=256 * (index + 1))
            )
        # Block 3 is reallocated: its hash no longer matches the snapshot.
        pool.hashes[3] = b"reallocated"
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        result = queue.collect_due()

        # Prefix closure: op 3 and everything after it is dropped, and the
        # cap releases the intact front down to its bound.
        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert [op.prefix_end_tokens for op in result.dropped_evicted] == [768, 1024]
        assert queue.num_pending_ops() == 1

    def test_cap_stops_at_a_deduplication_hole(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=False)
        queue = make_queue(pool, max_pending_ops=1)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        # A hole: this op starts past where the previous one ended.
        queue.admit(
            make_op("req", [2], pool, prefix_end_tokens=768, prefix_start_tokens=512)
        )
        queue.admit(make_op("req", [3], pool, prefix_end_tokens=1024))
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        result = queue.collect_due()

        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 2

    def test_backlog_emitted_is_a_subset_of_emitted(self) -> None:
        """The ledger equation must still close: the new counter reports how
        a store was timed, not a new way for an op to leave the queue."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=True)
        seed_blocks(pool, [2, 3, 4], free=False)
        queue = make_queue(pool, horizon_steps=1.0, max_pending_ops=1)
        queue.admit(make_op("pressure", [1], pool, prefix_end_tokens=256))
        for index, block in enumerate([2, 3, 4]):
            queue.admit(
                make_op("backlog", [block], pool, prefix_end_tokens=256 * (index + 1))
            )
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)

        queue.collect_due()

        # The pressure emission already shrank the backlog, so the cap only
        # has to release the remaining overflow: 4 admitted - 1 pressure - 1
        # cap = 2 backlog emissions.
        stats = queue.stats()
        assert stats.emitted == 3
        assert stats.backlog_emitted == 2
        assert queue.num_pending_ops() == 1
        assert stats.admitted == stats.emitted + queue.num_pending_ops()


class TestIdleDrain:
    """``idle_drain_max_ops``: work the backlog off in the gaps.

    Pressure times an emission to the moment its blocks are about to be
    reallocated, which is exactly when a prefill burst is allocating, so
    the copy lands in phase with the burst. These tests cover the
    complementary trigger: a step whose allocation rate is at or below
    ``idle_threshold_blocks`` emits the oldest waiting operations instead.
    """

    def _pinned_backlog(
        self,
        ops_per_request: dict[str, int],
        idle_drain_max_ops: int,
        **queue_kwargs: float,
    ) -> tuple[FakePoolView, EvictionAwareStoreQueue]:
        """Admit per-request chains whose blocks are all in use.

        No block enters the free queue, so eviction pressure never fires
        and only the idle path can emit.
        """
        pool = FakePoolView()
        queue = make_queue(
            pool,
            idle_drain_max_ops=idle_drain_max_ops,
            **queue_kwargs,  # type: ignore[arg-type]
        )
        next_block = 1
        for request_id, count in ops_per_request.items():
            for index in range(count):
                seed_blocks(pool, [next_block], free=False)
                queue.admit(
                    make_op(
                        request_id,
                        [next_block],
                        pool,
                        prefix_end_tokens=256 * (index + 1),
                    )
                )
                next_block += 1
        return pool, queue

    def test_idle_step_emits_the_oldest_request_first(self) -> None:
        _, queue = self._pinned_backlog({"old": 1, "young": 1}, idle_drain_max_ops=1)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)

        result = queue.collect_due()

        assert [op.request_id for op in result.to_store] == ["old"]
        stats = queue.stats()
        assert stats.idle_emitted == 1
        assert stats.idle_drain_steps == 1
        assert stats.emitted == 1

    def test_disabled_by_default_an_idle_step_emits_nothing(self) -> None:
        _, queue = self._pinned_backlog({"req": 2}, idle_drain_max_ops=0)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)

        result = queue.collect_due()

        assert result.to_store == []
        assert queue.num_pending_ops() == 2
        assert queue.stats().idle_emitted == 0

    def test_busy_step_emits_nothing(self) -> None:
        _, queue = self._pinned_backlog({"req": 2}, idle_drain_max_ops=8)
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)

        result = queue.collect_due()

        assert result.to_store == []
        assert queue.stats().idle_drain_steps == 0

    def test_next_step_estimate_vetoes_the_first_step_of_a_burst(self) -> None:
        """A prefill about to allocate is visible in the estimate before the
        EMA has seen a single busy step; that step must not count as idle."""
        _, queue = self._pinned_backlog({"req": 2}, idle_drain_max_ops=8)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=50)

        result = queue.collect_due()

        assert result.to_store == []
        assert queue.stats().idle_drain_steps == 0

    def test_ema_vetoes_the_trailing_steps_of_a_burst(self) -> None:
        """Right after a burst the smoothed rate is still high, so the step
        is not idle; the rate decays back under the threshold eventually."""
        _, queue = self._pinned_backlog({"req": 1}, idle_drain_max_ops=8)
        queue.observe_step(new_blocks_allocated=100, est_next_step_blocks=0)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)

        assert queue.collect_due().to_store == []

        emitted_after = 0
        for step in range(2, 30):
            queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
            if queue.collect_due().to_store:
                emitted_after = step
                break
        assert emitted_after > 2
        assert queue.num_pending_ops() == 0

    def test_allowance_bounds_each_idle_step(self) -> None:
        _, queue = self._pinned_backlog({"req": 5}, idle_drain_max_ops=2)

        emissions: list[int] = []
        for _ in range(3):
            queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
            emissions.append(len(queue.collect_due().to_store))

        assert emissions == [2, 2, 1]
        stats = queue.stats()
        assert stats.idle_emitted == 5
        assert stats.idle_drain_steps == 3
        assert queue.num_pending_ops() == 0

    def test_blocked_request_stays_pending(self) -> None:
        """One in-flight batch per request: an idle emission must not put a
        second one in flight, and the next-oldest request goes instead."""
        _, queue = self._pinned_backlog(
            {"in-flight": 1, "waiting": 1}, idle_drain_max_ops=1
        )
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)

        result = queue.collect_due(blocked_request_ids={"in-flight"})

        assert [op.request_id for op in result.to_store] == ["waiting"]
        assert queue.num_pending_ops() == 1

    def test_emission_stops_at_a_deduplication_hole(self) -> None:
        """An idle batch is coalesced like any other: it must not span a
        hole left by deduplication, whatever the allowance."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=False)
        queue = make_queue(pool, idle_drain_max_ops=8)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(
            make_op(
                "req",
                [2],
                pool,
                prefix_end_tokens=1024,
                prefix_start_tokens=768,
            )
        )
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)

        result = queue.collect_due()

        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 1

    def test_stale_snapshot_is_dropped_not_emitted(self) -> None:
        """The idle path validates like the others: content whose block was
        evicted and reallocated must never be stored."""
        pool, queue = self._pinned_backlog({"req": 1}, idle_drain_max_ops=8)
        pool.hashes[1] = b"reallocated"
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)

        result = queue.collect_due()

        assert result.to_store == []
        assert len(result.dropped_evicted) == 1
        assert queue.stats().idle_emitted == 0

    def test_pressure_emission_is_not_doubled_by_the_idle_path(self) -> None:
        """A slow trickle can be under the idle threshold and still produce
        a positive danger depth; a request the pressure pass emitted must
        not emit a second batch from the idle pass in the same drain."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0, idle_drain_max_ops=8)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)

        result = queue.collect_due()

        # Rank 0 is due under pressure (danger depth 1). The request now has
        # a batch in flight, so the idle pass must leave its second op
        # pending rather than submit a second batch.
        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 1
        stats = queue.stats()
        assert stats.emitted == 1
        assert stats.idle_emitted == 0

    def test_idle_emitted_is_a_subset_of_emitted(self) -> None:
        """The ledger equation must close: idle emission times a store, it
        is not a new way for an op to leave the queue."""
        _, queue = self._pinned_backlog({"req": 3}, idle_drain_max_ops=8)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)

        queue.collect_due()

        stats = queue.stats()
        assert stats.emitted == 3
        assert stats.idle_emitted == 3
        assert stats.admitted == stats.emitted + queue.num_pending_ops()


class TestBlockVolumeCap:
    """``max_drain_blocks_per_step``: bound the D2H burst in bytes.

    Operations coalesce into one contiguous copy per batch, so the op-count
    cap alone lets a step submit an arbitrarily long prefix. The block cap
    cuts the emitted front at an operation boundary; the bound is soft --
    the operation that crosses it still emits, so progress never depends
    on an operation fitting under the cap.
    """

    def test_cap_cuts_a_due_segment_at_an_op_boundary(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4, 5, 6], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_blocks_per_step=3)
        queue.admit(make_op("req", [1, 2], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [3, 4], pool, prefix_end_tokens=512))
        queue.admit(make_op("req", [5, 6], pool, prefix_end_tokens=768))
        queue.observe_step(new_blocks_allocated=6, est_next_step_blocks=0)

        first = queue.collect_due()

        # 3 blocks of budget: op one spends 2, op two crosses the bound and
        # still emits, op three waits.
        assert [op.prefix_end_tokens for op in first.to_store] == [256, 512]
        assert first.ops_held_back == 1
        assert queue.stats().throttled_drains == 1
        queue.observe_step(new_blocks_allocated=6, est_next_step_blocks=0)

        second = queue.collect_due()

        assert [op.prefix_end_tokens for op in second.to_store] == [768]
        assert queue.num_pending_ops() == 0

    def test_an_op_larger_than_the_cap_still_emits_alone(self) -> None:
        """Progress must not depend on an op fitting under the cap, or a
        long chunk would sit due until eviction destroyed it."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3, 4, 5], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_drain_blocks_per_step=1)
        queue.admit(make_op("req", [1, 2, 3, 4], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [5], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=5, est_next_step_blocks=0)

        result = queue.collect_due()

        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert result.ops_held_back == 1

    def test_cap_is_shared_with_the_backlog_drain(self) -> None:
        """One budget per drain: blocks the pressure pass spent are gone
        for the backlog pass of the same step."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        seed_blocks(pool, [3, 4, 5], free=False)
        queue = make_queue(
            pool,
            horizon_steps=1.0,
            max_pending_ops=1,
            max_drain_blocks_per_step=3,
        )
        queue.admit(make_op("pressure", [1, 2], pool, prefix_end_tokens=256))
        for index, block in enumerate([3, 4, 5]):
            queue.admit(
                make_op("backlog", [block], pool, prefix_end_tokens=256 * (index + 1))
            )
        queue.observe_step(new_blocks_allocated=2, est_next_step_blocks=0)

        result = queue.collect_due()

        # Pressure spends 2 of 3 blocks; the backlog overflow of 2 ops gets
        # one block of budget, so one op crosses the bound and one waits.
        assert [op.request_id for op in result.to_store] == ["pressure", "backlog"]
        stats = queue.stats()
        assert stats.backlog_emitted == 1
        assert queue.num_pending_ops() == 2

    def test_cap_is_shared_with_the_idle_drain(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [1, 2, 3], free=False)
        queue = make_queue(
            pool,
            idle_drain_max_ops=8,
            max_drain_blocks_per_step=2,
        )
        for index, block in enumerate([1, 2, 3]):
            queue.admit(
                make_op("req", [block], pool, prefix_end_tokens=256 * (index + 1))
            )
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)

        result = queue.collect_due()

        assert [op.prefix_end_tokens for op in result.to_store] == [256, 512]
        assert queue.stats().idle_emitted == 2
        assert queue.num_pending_ops() == 1


CAPACITY = 1_000_000
"""L1 capacity used by the degradation tests, in bytes."""


class TestAdaptiveDegradation:
    """``degrade_l1_residence_secs``: the volume-neutrality controller.

    The signal is fed through ``observe_l1_pressure`` as (monotonic time,
    capacity, cumulative evicted bytes) snapshots. Residence -- capacity
    over the windowed eviction rate -- below the threshold opens a bounded
    trial of immediate emission; the trial commits only when its
    emitted-block rate stays neutral against the deferred baseline, and
    reverts with a cooldown when degrading would have increased volume. A
    committed degradation lifts when residence recovers past the
    hysteresis factor, or when a periodic deferred probe shows filtering
    value has returned.

    Timeline used by the helpers, with a threshold of 60 seconds and a
    capacity of ``CAPACITY``: churn samples every 10 seconds deleting
    200_000 bytes give a 20_000 B/s windowed rate once the history spans
    the 60-second minimum, hence residence 50 < 60 -- the trial opens at
    t=60 and, with nothing emitted during it, commits at t=110.
    """

    def _churn(
        self,
        queue: EvictionAwareStoreQueue,
        start_tick: int,
        end_tick: int,
        evicted_start: int,
    ) -> int:
        """Feed churn samples at 10-second ticks, 200_000 bytes each.

        Args:
            queue: The queue under test.
            start_tick: First tick to feed, inclusive.
            end_tick: Last tick to feed, inclusive.
            evicted_start: Cumulative evicted bytes before the first tick.

        Returns:
            The cumulative evicted total after the last tick.
        """
        evicted = evicted_start
        for tick in range(start_tick, end_tick + 1):
            evicted += 200_000
            queue.observe_l1_pressure(tick * 10.0, CAPACITY, evicted)
        return evicted

    def _degrade(self, queue: EvictionAwareStoreQueue) -> None:
        """Walk the controller into a committed degradation.

        Baseline at t=0, churn through t=110: the trial opens at t=60
        (residence 50 < 60) and, with nothing emitted during it, commits
        at t=110 as neutral against an idle deferred baseline.
        """
        queue.observe_l1_pressure(0.0, CAPACITY, 0)
        self._churn(queue, 1, 11, 0)

    def _degraded_backlog(
        self,
        ops_per_request: dict[str, int],
        **queue_kwargs: float,
    ) -> tuple[FakePoolView, EvictionAwareStoreQueue]:
        """A pinned backlog on a queue committed to immediate emission.

        No block enters the free queue, so eviction pressure never fires and
        only the degraded drain can emit.
        """
        pool = FakePoolView()
        queue = make_queue(
            pool,
            degrade_l1_residence_secs=60.0,
            **queue_kwargs,  # type: ignore[arg-type]
        )
        self._degrade(queue)
        next_block = 1
        for request_id, count in ops_per_request.items():
            for index in range(count):
                seed_blocks(pool, [next_block], free=False)
                queue.admit(
                    make_op(
                        request_id,
                        [next_block],
                        pool,
                        prefix_end_tokens=256 * (index + 1),
                    )
                )
                next_block += 1
        return pool, queue

    # ------------------------------------------------------------------
    # Regime state machine
    # ------------------------------------------------------------------

    def test_churn_opens_a_trial_not_a_commitment(self) -> None:
        queue = make_queue(FakePoolView(), degrade_l1_residence_secs=60.0)
        queue.observe_l1_pressure(0.0, CAPACITY, 0)

        self._churn(queue, 1, 6, 0)  # t=60: the gate opens

        assert queue.degraded  # a trial behaves degraded
        stats = queue.stats()
        assert stats.degrade_trials == 1
        assert stats.degrade_commits == 0
        assert stats.degrade_transitions == 1

    def test_volume_neutral_trial_commits(self) -> None:
        queue = make_queue(FakePoolView(), degrade_l1_residence_secs=60.0)

        self._degrade(queue)

        assert queue.degraded
        stats = queue.stats()
        assert stats.degrade_commits == 1
        assert stats.degrade_reverts == 0

    def test_volume_increasing_trial_reverts_and_cools_down(self) -> None:
        """A backlog only the trial's immediate emission would flush: the
        deferred baseline emitted nothing, the trial emits, so the trial
        reads as a volume increase and reverts."""
        pool = FakePoolView()
        queue = make_queue(pool, degrade_l1_residence_secs=60.0)
        queue.observe_l1_pressure(0.0, CAPACITY, 0)
        for index in range(3):
            seed_blocks(pool, [index + 1], free=False)
            queue.admit(
                make_op(
                    "req",
                    [index + 1],
                    pool,
                    prefix_end_tokens=256 * (index + 1),
                )
            )
        evicted = self._churn(queue, 1, 6, 0)  # t=60: the trial opens
        assert queue.degraded
        queue.collect_due()  # the trial flushes the backlog
        evicted = self._churn(queue, 7, 11, evicted)  # t=110: decision

        assert not queue.degraded
        stats = queue.stats()
        assert stats.degrade_reverts == 1
        assert stats.degrade_commits == 0

        # The revert cooldown holds even though the churn continues.
        self._churn(queue, 12, 70, evicted)  # t=120..700 < 110 + cooldown
        assert queue.stats().degrade_trials == 1

    def test_degraded_lifts_when_residence_recovers(self) -> None:
        queue = make_queue(FakePoolView(), degrade_l1_residence_secs=60.0)
        self._degrade(queue)
        assert queue.degraded

        # Eviction stops; the window slides off the churn and residence
        # returns to infinity.
        evicted = 11 * 200_000
        for tick in range(12, 25):
            queue.observe_l1_pressure(tick * 10.0, CAPACITY, evicted)

        assert not queue.degraded

    def test_probe_recovers_when_filtering_returns(self) -> None:
        """While degraded, a periodic probe defers again; emission drying
        up during the probe (relative to the degraded baseline) shows
        deferral is filtering volume once more, and the regime lifts."""
        pool = FakePoolView()
        queue = make_queue(pool, degrade_l1_residence_secs=60.0)
        self._degrade(queue)  # committed at t=110
        evicted = 11 * 200_000
        next_block = 1
        # Sustained churn holds residence at 50; steady admissions flush
        # through the degraded drain, giving the probe a non-zero baseline.
        for tick in range(12, 59):  # t=120..580
            seed_blocks(pool, [next_block], free=False)
            queue.admit(
                make_op(f"r{next_block}", [next_block], pool, prefix_end_tokens=256)
            )
            queue.collect_due()
            next_block += 1
            evicted += 200_000
            queue.observe_l1_pressure(tick * 10.0, CAPACITY, evicted)
        assert queue.degraded

        # t=590: the probe interval since the commit elapses.
        evicted += 200_000
        queue.observe_l1_pressure(590.0, CAPACITY, evicted)
        assert not queue.degraded  # probing: deferred
        assert queue.stats().degrade_probes == 1

        # During the probe nothing forces emission (pinned blocks, no
        # eviction pressure): deferral is filtering again.
        for tick in range(60, 65):  # t=600..640: the probe concludes
            seed_blocks(pool, [next_block], free=False)
            queue.admit(
                make_op(f"r{next_block}", [next_block], pool, prefix_end_tokens=256)
            )
            queue.collect_due()
            next_block += 1
            evicted += 200_000
            queue.observe_l1_pressure(tick * 10.0, CAPACITY, evicted)

        assert not queue.degraded
        assert queue.stats().degrade_probe_recoveries == 1

    def test_probe_without_filtering_returns_to_degraded(self) -> None:
        """A probe during which emission continues at the degraded pace
        (eviction pressure forces the stores out anyway) shows deferral
        buys nothing, and the regime resumes."""
        pool = FakePoolView()
        queue = make_queue(pool, horizon_steps=1.0, degrade_l1_residence_secs=60.0)
        self._degrade(queue)  # committed at t=110
        evicted = 11 * 200_000
        next_block = 1
        for tick in range(12, 59):  # t=120..580: degraded, steady emission
            seed_blocks(pool, [next_block], free=False)
            queue.admit(
                make_op(f"r{next_block}", [next_block], pool, prefix_end_tokens=256)
            )
            queue.collect_due()
            next_block += 1
            evicted += 200_000
            queue.observe_l1_pressure(tick * 10.0, CAPACITY, evicted)
        assert queue.degraded

        evicted += 200_000
        queue.observe_l1_pressure(590.0, CAPACITY, evicted)
        assert not queue.degraded  # probing: deferred
        assert queue.stats().degrade_probes == 1

        # Free blocks under a one-step horizon, with an allocation rate
        # whose due window spans the whole free queue: the pressure path
        # emits during the probe at the same pace the degraded drain did.
        for tick in range(60, 65):  # t=600..640: the probe concludes
            seed_blocks(pool, [next_block], free=True)
            queue.admit(
                make_op(f"r{next_block}", [next_block], pool, prefix_end_tokens=256)
            )
            queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=0)
            queue.collect_due()
            next_block += 1
            evicted += 200_000
            queue.observe_l1_pressure(tick * 10.0, CAPACITY, evicted)

        assert queue.degraded
        assert queue.stats().degrade_probe_recoveries == 0

    def test_disabled_threshold_ignores_pressure(self) -> None:
        queue = make_queue(FakePoolView(), degrade_l1_residence_secs=0.0)

        self._degrade(queue)

        assert not queue.degraded
        assert queue.stats().degrade_transitions == 0

    def test_repeated_snapshot_does_not_advance_the_controller(self) -> None:
        """The caller repeats the latest sample every step; only a strictly
        newer timestamp may advance the history and the regime machine."""
        queue = make_queue(FakePoolView(), degrade_l1_residence_secs=60.0)
        self._degrade(queue)

        for _ in range(10):
            queue.observe_l1_pressure(110.0, CAPACITY, 11 * 200_000)

        assert queue.degraded
        assert queue.stats().degrade_transitions == 1

    def test_counter_regression_rebaselines(self) -> None:
        """A cumulative counter that moved backwards (server restart)
        contributes a zero delta, not a negative rate."""
        queue = make_queue(FakePoolView(), degrade_l1_residence_secs=60.0)
        queue.observe_l1_pressure(0.0, CAPACITY, 1_000_000)
        queue.observe_l1_pressure(10.0, CAPACITY, 0)  # regression

        self._churn(queue, 2, 7, 0)  # t=20..70: post-restart churn

        assert queue.degraded
        assert queue.stats().degrade_trials == 1

    def test_zero_capacity_sample_is_ignored(self) -> None:
        queue = make_queue(FakePoolView(), degrade_l1_residence_secs=60.0)

        queue.observe_l1_pressure(0.0, 0, 0)
        queue.observe_l1_pressure(10.0, 0, 200_000)

        assert not queue.degraded

    # ------------------------------------------------------------------
    # Degraded drain semantics
    # ------------------------------------------------------------------

    def test_degraded_drain_flushes_the_whole_backlog(self) -> None:
        _, queue = self._degraded_backlog({"a": 2, "b": 1})

        result = queue.collect_due()

        assert [(op.request_id, op.prefix_end_tokens) for op in result.to_store] == [
            ("a", 256),
            ("a", 512),
            ("b", 256),
        ]
        assert queue.num_pending_ops() == 0
        stats = queue.stats()
        assert stats.degraded_emitted == 3
        assert stats.degraded_drain_steps == 1
        assert stats.emitted == 3

    def test_admission_while_degraded_emits_on_the_next_drain(self) -> None:
        pool, queue = self._degraded_backlog({})
        seed_blocks(pool, [1], free=False)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))

        result = queue.collect_due()

        assert [op.request_id for op in result.to_store] == ["req"]

    def test_blocked_request_waits(self) -> None:
        _, queue = self._degraded_backlog({"a": 1, "b": 1})

        result = queue.collect_due(blocked_request_ids={"a"})

        assert [op.request_id for op in result.to_store] == ["b"]
        assert queue.num_pending_ops() == 1

    def test_dedup_hole_cuts_the_batch(self) -> None:
        pool, queue = self._degraded_backlog({})
        seed_blocks(pool, [1, 2], free=False)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(
            make_op(
                "req",
                [2],
                pool,
                prefix_end_tokens=1024,
                prefix_start_tokens=768,
            )
        )

        result = queue.collect_due()

        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 1

    def test_stale_snapshot_is_dropped_not_stored(self) -> None:
        pool, queue = self._degraded_backlog({})
        seed_blocks(pool, [1], free=True)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        pool.evict(1)
        queue.observe_step(
            new_blocks_allocated=1,
            est_next_step_blocks=0,
            allocated_block_ids={1},
        )

        result = queue.collect_due()

        assert result.to_store == []
        assert queue.stats().dropped_evicted == 1

    def test_budget_is_shared_with_the_step_cap(self) -> None:
        _, queue = self._degraded_backlog({"a": 1, "b": 1}, max_drain_per_step=1)

        first = queue.collect_due()
        second = queue.collect_due()

        assert [op.request_id for op in first.to_store] == ["a"]
        assert [op.request_id for op in second.to_store] == ["b"]

    def test_pressure_emission_is_not_doubled_by_the_degraded_path(self) -> None:
        """A request the pressure pass emitted has a batch in flight; the
        degraded pass must leave its remaining ops pending."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool, horizon_steps=1.0, degrade_l1_residence_secs=60.0)
        self._degrade(queue)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        queue.observe_step(new_blocks_allocated=1, est_next_step_blocks=0)

        result = queue.collect_due()

        assert [op.prefix_end_tokens for op in result.to_store] == [256]
        assert queue.num_pending_ops() == 1
        stats = queue.stats()
        assert stats.emitted == 1
        assert stats.degraded_emitted == 0

    def test_recovered_queue_defers_again(self) -> None:
        pool, queue = self._degraded_backlog({})
        evicted = 11 * 200_000
        for tick in range(12, 25):  # eviction stops: residence recovers
            queue.observe_l1_pressure(tick * 10.0, CAPACITY, evicted)
        assert not queue.degraded
        seed_blocks(pool, [1], free=False)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))

        result = queue.collect_due()

        assert result.to_store == []
        assert queue.num_pending_ops() == 1

    def test_truncated_chain_below_break_even_is_dropped(self) -> None:
        """The economy backstop holds in the degraded drain: a chain that
        eviction truncated back below break-even is dropped, not stored."""
        pool, queue = self._degraded_backlog({}, min_prefix_tokens=512)
        seed_blocks(pool, [1, 2], free=True)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=512))
        pool.evict(2)
        queue.observe_step(
            new_blocks_allocated=1,
            est_next_step_blocks=0,
            allocated_block_ids={2},
        )

        result = queue.collect_due()

        assert result.to_store == []
        assert queue.stats().rejected_short_prefix >= 1
