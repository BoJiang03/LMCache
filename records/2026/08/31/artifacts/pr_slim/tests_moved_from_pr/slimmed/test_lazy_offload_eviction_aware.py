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
    prefix_start_tokens: int = -1,
    epoch: int = 0,
) -> PendingStoreOp:
    """Build a pending op whose hash snapshot matches the pool's state.

    ``prefix_start_tokens`` defaults to one 256-token chunk before the end,
    so consecutive ops built with 256-spaced ends form a contiguous chain.
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
    max_drain_per_step: int = 64,
    max_deferral_seconds: float = 0.0,
) -> EvictionAwareStoreQueue:
    config = LazyOffloadPolicyConfig(
        horizon_steps=horizon_steps,
        max_drain_per_step=max_drain_per_step,
        max_deferral_seconds=max_deferral_seconds,
    )
    return EvictionAwareStoreQueue(config, pool)


class TestConfigValidation:
    def test_default_horizon_uses_calibrated_value(self) -> None:
        assert LazyOffloadPolicyConfig().horizon_steps == 2.5

    def test_rejects_non_positive_horizon(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(horizon_steps=0)

    def test_rejects_zero_drain_cap(self) -> None:
        with pytest.raises(ValueError):
            LazyOffloadPolicyConfig(max_drain_per_step=0)

    def test_deferral_deadline_defaults_to_disabled(self) -> None:
        assert LazyOffloadPolicyConfig().max_deferral_seconds == 0.0

    def test_rejects_a_negative_deferral_bound(self) -> None:
        with pytest.raises(ValueError, match="max_deferral_seconds"):
            LazyOffloadPolicyConfig(max_deferral_seconds=-1.0)


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
        assert queue.stats().dropped_evicted_tokens == 512  # two 256-token ops

    def test_deferral_is_measured_in_drains_waited(self) -> None:
        """The mean emitted deferral is what the policy buys, so it must
        count the drains an op actually waited, not the drains that ran."""
        pool = FakePoolView()
        seed_blocks(pool, [1], free=False)  # in use: not at risk yet
        queue = make_queue(pool)
        queue.admit(make_op("req", [1], pool, prefix_end_tokens=256))
        for _ in range(4):
            queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
            queue.collect_due()
        pool.free_queue.append(1)  # now in the free queue, and due
        queue.observe_step(new_blocks_allocated=8, est_next_step_blocks=8)
        assert len(queue.collect_due().to_store) == 1
        stats = queue.stats()
        assert stats.emitted == 1
        assert stats.emitted_deferral_drains == 5  # admitted before drain 1
        assert stats.dropped_deferral_drains == 0

    def test_drop_counter_weighs_lost_tokens_not_ops(self) -> None:
        """Ops differ by orders of magnitude in range, so the drop count
        alone cannot say what a loss cost; the token weight can."""
        pool = FakePoolView()
        seed_blocks(pool, [1, 2], free=True)
        queue = make_queue(pool)
        queue.admit(
            make_op("req", [1], pool, prefix_start_tokens=0, prefix_end_tokens=8192)
        )
        queue.admit(make_op("req", [2], pool, prefix_end_tokens=8448))
        pool.evict(1)
        queue.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        queue.collect_due()
        stats = queue.stats()
        assert stats.dropped_evicted == 2
        assert stats.dropped_evicted_tokens == 8192 + 256

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


class TestDeferralDeadline:
    """The wall-clock bound on how long an operation may wait.

    The danger window answers when a block dies on the GPU; the deadline
    answers when the content is needed again. These tests pin the second
    clock's behaviour, including the cases where it must not fire.
    """

    def test_disabled_by_default_leaves_a_far_block_pending(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, list(range(100)), free=True)
        queue = make_queue(pool, horizon_steps=1.0)
        queue.observe_step(1, 1, None, 0.0)
        assert queue.admit(make_op("r1", [99], pool, 256)) is AdmitResult.ADMITTED
        queue.observe_step(1, 1, None, 10_000.0)
        result = queue.collect_due()
        assert result.to_store == []
        assert queue.stats().emitted_overdue == 0

    def test_emits_when_the_bound_is_passed(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, list(range(100)), free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        queue.observe_step(1, 1, None, 100.0)
        assert queue.admit(make_op("r1", [99], pool, 256)) is AdmitResult.ADMITTED
        # Deep in the free queue: the danger window cannot make it due.
        queue.observe_step(1, 1, None, 125.0)
        assert queue.collect_due().to_store == []
        queue.observe_step(1, 1, None, 131.0)
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["r1"]
        assert queue.stats().emitted_overdue == 1
        assert queue.stats().emitted == 1

    def test_window_emission_is_not_counted_as_overdue(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, [7], free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        queue.observe_step(4, 4, None, 100.0)
        assert queue.admit(make_op("r1", [7], pool, 256)) is AdmitResult.ADMITTED
        queue.observe_step(4, 4, None, 101.0)
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["r1"]
        assert queue.stats().emitted_overdue == 0

    def test_overdue_emits_with_a_zero_danger_depth(self) -> None:
        """An idle engine never opens the window; the deadline still fires."""
        pool = FakePoolView()
        seed_blocks(pool, list(range(100)), free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        queue.observe_step(0, 0, None, 100.0)
        assert queue.admit(make_op("r1", [99], pool, 256)) is AdmitResult.ADMITTED
        queue.observe_step(0, 0, None, 200.0)
        result = queue.collect_due()
        assert [op.request_id for op in result.to_store] == ["r1"]
        assert queue.stats().emitted_overdue == 1
        assert pool.blocks_walked == 0

    def test_overdue_respects_the_drain_budget(self) -> None:
        pool = FakePoolView()
        seed_blocks(pool, list(range(100)), free=True)
        queue = make_queue(
            pool,
            horizon_steps=1.0,
            max_deferral_seconds=30.0,
            max_drain_per_step=1,
        )
        queue.observe_step(0, 0, None, 100.0)
        queue.admit(make_op("r1", [97], pool, 256))
        queue.admit(make_op("r1", [98], pool, 512))
        queue.observe_step(0, 0, None, 200.0)
        first = queue.collect_due()
        assert [op.prefix_end_tokens for op in first.to_store] == [256]
        assert first.ops_held_back == 1
        queue.observe_step(0, 0, None, 201.0)
        second = queue.collect_due()
        assert [op.prefix_end_tokens for op in second.to_store] == [512]

    def test_overdue_still_drops_an_evicted_chain(self) -> None:
        """The deadline releases survivors, never data the pool has lost."""
        pool = FakePoolView()
        seed_blocks(pool, list(range(100)), free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        queue.observe_step(0, 0, None, 100.0)
        queue.admit(make_op("r1", [97], pool, 256))
        queue.admit(make_op("r1", [98], pool, 512))
        pool.evict(97)
        queue.observe_step(0, 0, None, 200.0)
        result = queue.collect_due()
        assert result.to_store == []
        assert queue.stats().dropped_evicted == 2

    def test_deadline_measures_the_front_op_not_the_request(self) -> None:
        """A request keeps its urgency from its oldest surviving op."""
        pool = FakePoolView()
        seed_blocks(pool, list(range(100)), free=True)
        queue = make_queue(pool, horizon_steps=1.0, max_deferral_seconds=30.0)
        queue.observe_step(0, 0, None, 100.0)
        queue.admit(make_op("r1", [99], pool, 256))
        queue.observe_step(0, 0, None, 200.0)
        assert len(queue.collect_due().to_store) == 1
        # Fresh op on the same request: the deadline restarts from its own
        # admission, so the next drain leaves it pending.
        queue.observe_step(0, 0, None, 201.0)
        queue.admit(make_op("r1", [98], pool, 512))
        queue.observe_step(0, 0, None, 210.0)
        assert queue.collect_due().to_store == []
