# SPDX-License-Identifier: Apache-2.0
# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_pending_store import (
    AddOutcome,
    FIFOOffloadPolicy,
    LazyOffloadMode,
    LazyOffloadPendingStore,
)

FIFO_CONFIG = {"lmcache.mp.lazy_offload_policy": "FIFO"}


def _make_meta(
    request_id: str = "req-0", num_blocks: int = 1, end: int = 256
) -> MagicMock:
    """Helper to create a mock LMCacheMPRequestMetadata."""
    meta = MagicMock()
    meta.request_id = request_id
    meta.op.flat_block_ids = list(range(num_blocks))
    meta.op.end = end
    return meta


def _make_block_hashes(block_ids: list[int]) -> dict[int, bytes]:
    """Helper to create mock block hashes."""
    return {bid: f"hash-{bid}".encode() for bid in block_ids}


def _make_gpu_pool(num_blocks: int = 10) -> MagicMock:
    """Mock BlockPool: hashed blocks, all sitting in the free queue."""
    gpu_pool = MagicMock()
    gpu_pool.blocks = {
        bid: MagicMock(block_hash=f"hash-{bid}".encode()) for bid in range(num_blocks)
    }
    gpu_pool.free_block_queue.get_all_free_blocks.return_value = [
        SimpleNamespace(block_id=bid) for bid in range(num_blocks)
    ]
    return gpu_pool


# ===========================================================================
# Tests for FIFOOffloadPolicy
# ===========================================================================


class TestFIFOOffloadPolicy:
    def test_init_default_threshold(self):
        policy = FIFOOffloadPolicy()
        assert policy._threshold == 100

    def test_init_custom_threshold(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 50}
        policy = FIFOOffloadPolicy(configs)
        assert policy._threshold == 50

    def test_add_creates_new_item(self):
        policy = FIFOOffloadPolicy()
        meta = _make_meta("req-0")
        hashes = _make_block_hashes([0, 1])
        policy.add(meta, hashes)
        assert "req-0" in policy._pending_items
        assert len(policy._pending_items["req-0"].metadatas) == 1

    def test_add_same_request_appends_metadatas(self):
        policy = FIFOOffloadPolicy()
        meta1 = _make_meta("req-0", num_blocks=1)
        meta2 = _make_meta("req-0", num_blocks=2)
        policy.add(meta1, _make_block_hashes([0]))
        policy.add(meta2, _make_block_hashes([0, 1]))
        assert len(policy._pending_items["req-0"].metadatas) == 2

    def test_should_offload_below_threshold(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        assert policy._finished_requests_count == 1
        assert policy.should_offload() is False

    def test_should_offload_at_threshold(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        for i in range(3):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")
        assert policy.should_offload() is True

    def test_should_offload_above_threshold(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        for i in range(5):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")
        assert policy.should_offload() is True

    def test_mark_req_finished_not_in_pending_returns_false(self):
        # A request may finish without ever producing store metadata; that
        # must not crash the scheduler.
        policy = FIFOOffloadPolicy()
        assert policy.mark_req_finished("nonexistent") is False

    def test_select_items_returns_only_finished(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        # req-1 is not finished

        selected = policy.select_items(10)
        assert len(selected) == 1
        assert selected[0].request_id == "req-0"

    def test_select_items_removes_from_pending(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        policy.mark_req_finished("req-1")

        selected = policy.select_items(10)
        assert len(selected) == 2
        assert len(policy._pending_items) == 0
        assert policy._finished_requests_count == 0

    def test_select_items_skips_unfinished(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        policy.mark_req_finished("req-1")

        selected = policy.select_items(10)
        assert len(selected) == 1
        assert selected[0].request_id == "req-1"
        # req-0 still pending
        assert "req-0" in policy._pending_items

    def test_select_items_count_limits_output(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        for i in range(5):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")

        selected = policy.select_items(2)
        assert len(selected) == 2
        assert len(policy._pending_items) == 3

    def test_select_items_empty(self):
        policy = FIFOOffloadPolicy()
        assert policy.select_items(5) == []

    def test_drop_request_discards_items_and_finished_count(self):
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        policy.add(_make_meta("req-0", num_blocks=1), _make_block_hashes([0]))
        policy.add(_make_meta("req-0", num_blocks=2), _make_block_hashes([0, 1]))
        policy.mark_req_finished("req-0")
        assert policy.should_offload() is True

        assert policy.drop_request("req-0") == 2
        assert policy.should_offload() is False
        assert policy.select_items(10) == []

    def test_drop_request_unknown_is_noop(self):
        policy = FIFOOffloadPolicy()
        assert policy.drop_request("nonexistent") == 0


# ===========================================================================
# Tests for LazyOffloadPendingStore
# ===========================================================================


class TestLazyOffloadPendingStore:
    def _setup_store_with_gpu_pool(self, configs=None):
        store = LazyOffloadPendingStore({**FIFO_CONFIG, **(configs or {})})
        store.bind_gpu_block_pool(_make_gpu_pool())
        return store

    def test_init_default_policy_is_eviction_aware(self):
        store = LazyOffloadPendingStore()
        assert store.mode is LazyOffloadMode.EVICTION_AWARE

    def test_init_fifo_policy_explicit(self):
        store = LazyOffloadPendingStore(dict(FIFO_CONFIG))
        assert store.mode is LazyOffloadMode.FIFO

    def test_init_unknown_policy_raises(self):
        configs = {"lmcache.mp.lazy_offload_policy": "UNKNOWN"}
        with pytest.raises(ValueError, match="Unknown offload policy"):
            LazyOffloadPendingStore(configs)

    def test_init_default_select_count(self):
        store = LazyOffloadPendingStore()
        assert store._select_count == 10

    def test_init_custom_select_count(self):
        configs = {"lmcache.mp.lazy_offload_select_count": 5}
        store = LazyOffloadPendingStore(configs)
        assert store._select_count == 5

    def test_bind_gpu_block_pool(self):
        store = LazyOffloadPendingStore()
        gpu_pool = MagicMock()
        store.bind_gpu_block_pool(gpu_pool)
        assert store._gpu_block_pool is gpu_pool

    def test_add_without_gpu_pool_raises(self):
        store = LazyOffloadPendingStore()
        meta = _make_meta("req-0")
        with pytest.raises(ValueError, match="gpu block pool not bound"):
            store.add(meta)

    def test_add_with_gpu_pool(self):
        store = self._setup_store_with_gpu_pool()
        meta = _make_meta("req-0", num_blocks=2)
        assert store.add(meta) is AddOutcome.BUFFERED
        # Verify block hashes were computed from gpu pool
        pending = store._fifo_policy._pending_items["req-0"]
        assert len(pending.metadatas) == 1
        assert pending.metadatas[0][1] == {0: b"hash-0", 1: b"hash-1"}

    def test_should_offload_delegates_to_policy(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 2}
        store = self._setup_store_with_gpu_pool(configs)

        store.add(_make_meta("req-0"))
        store.mark_req_finished("req-0")
        assert store.should_offload() is False

        store.add(_make_meta("req-1"))
        store.mark_req_finished("req-1")
        assert store.should_offload() is True

    def test_select_items_returns_correct_count(self):
        configs = {
            "lmcache.mp.lazy_offload_threshold": 1,
            "lmcache.mp.lazy_offload_select_count": 3,
        }
        store = self._setup_store_with_gpu_pool(configs)

        for i in range(5):
            store.add(_make_meta(f"req-{i}"))
            store.mark_req_finished(f"req-{i}")

        selected = store.select_items()
        assert len(selected) == 3

    def test_mark_req_finished(self):
        configs = {"lmcache.mp.lazy_offload_threshold": 1}
        store = self._setup_store_with_gpu_pool(configs)
        store.add(_make_meta("req-0"))
        store.mark_req_finished("req-0")
        assert store.should_offload() is True

    def test_update_get_remove_gpu_block_ids(self):
        store = LazyOffloadPendingStore()
        store.update_request_gpu_block_ids("req-0", [1, 2])
        store.update_request_gpu_block_ids("req-0", [3])
        assert store.get_request_gpu_block_ids("req-0") == [1, 2, 3]

        store.remove_request_gpu_block_ids("req-0")
        assert store.get_request_gpu_block_ids("req-0") == []

    def test_get_gpu_block_ids_nonexistent_returns_empty(self):
        store = LazyOffloadPendingStore()
        assert store.get_request_gpu_block_ids("nonexistent") == []

    def test_unknown_request_lookup_does_not_open_receipt_window(self):
        """A read of an unknown id must not create state: were
        has_in_flight_store to flip True, a stale or duplicate receipt
        would unpin blocks that are not pinned and end the session twice."""
        store = LazyOffloadPendingStore()
        store.get_request_gpu_block_ids("ghost")
        assert store.has_in_flight_store("ghost") is False

    def test_end_to_end_flow(self):
        """Test full add -> mark_finished -> should_offload -> select_items."""
        configs = {
            "lmcache.mp.lazy_offload_threshold": 3,
            "lmcache.mp.lazy_offload_select_count": 2,
        }
        store = self._setup_store_with_gpu_pool(configs)

        # Add items and mark them finished
        for i in range(5):
            store.add(_make_meta(f"req-{i}", num_blocks=1))
        for i in range(5):
            store.mark_req_finished(f"req-{i}")

        # Should be over threshold
        assert store.should_offload() is True

        # Select first 2 (select_count=2)
        selected = store.select_items()
        assert len(selected) == 2
        assert selected[0].request_id == "req-0"
        assert selected[1].request_id == "req-1"

    def test_select_items_multiple_batches(self):
        configs = {
            "lmcache.mp.lazy_offload_threshold": 1,
            "lmcache.mp.lazy_offload_select_count": 2,
        }
        store = self._setup_store_with_gpu_pool(configs)

        for i in range(6):
            store.add(_make_meta(f"req-{i}"))
            store.mark_req_finished(f"req-{i}")

        batch1 = store.select_items()
        assert len(batch1) == 2
        assert batch1[0].request_id == "req-0"

        batch2 = store.select_items()
        assert len(batch2) == 2
        assert batch2[0].request_id == "req-2"

        batch3 = store.select_items()
        assert len(batch3) == 2
        assert batch3[0].request_id == "req-4"

        batch4 = store.select_items()
        assert len(batch4) == 0


# ===========================================================================
# Tests for LazyOffloadPendingStore in EVICTION_AWARE mode
# ===========================================================================


class TestEvictionAwareMode:
    def _setup(self, configs=None) -> tuple[LazyOffloadPendingStore, MagicMock]:
        store = LazyOffloadPendingStore(configs)
        gpu_pool = _make_gpu_pool()
        store.bind_gpu_block_pool(gpu_pool)
        return store, gpu_pool

    def test_add_buffers_hashed_op(self):
        store, _ = self._setup()
        assert store.add(_make_meta("req-0", num_blocks=2)) is AddOutcome.BUFFERED

    def test_add_skips_unhashed_op(self):
        store, gpu_pool = self._setup()
        gpu_pool.blocks[1].block_hash = None
        assert store.add(_make_meta("req-0", num_blocks=2)) is (
            AddOutcome.SKIPPED_UNHASHED
        )

    def test_fifo_entry_points_raise(self):
        store, _ = self._setup()
        with pytest.raises(ValueError, match="FIFO policy unavailable"):
            store.should_offload()
        with pytest.raises(ValueError, match="FIFO policy unavailable"):
            store.select_items()

    def test_collect_due_under_pressure_emits_op(self):
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        meta = _make_meta("req-0", num_blocks=2)
        store.add(meta)
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        result = store.collect_due()
        assert [op.store_metadata for op in result.to_store] == [meta]

    def test_collect_due_without_pressure_holds(self):
        store, _ = self._setup()
        store.add(_make_meta("req-0", num_blocks=2))
        store.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert store.collect_due().to_store == []

    def test_session_release_flow(self):
        """finish -> drain -> receipt: teardown allowed only at the receipt."""
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        assert store.mark_req_finished("req-0") is True
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        result = store.collect_due()
        assert len(result.to_store) == 1
        assert result.released_requests == []
        assert store.notify_store_complete("req-0") is True

    def test_mark_req_finished_without_pending_allows_teardown(self):
        store, _ = self._setup()
        assert store.mark_req_finished("req-unknown") is False

    def test_drop_request_discards_buffered_ops(self):
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))
        assert store.drop_request("req-0") == 1
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        assert store.collect_due().to_store == []

    def test_mark_store_failed_drops_buffered_ops(self):
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))
        assert store.mark_store_failed("req-0") == 1
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        assert store.collect_due().to_store == []

    def test_rebind_same_pool_is_idempotent(self):
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))

        store.bind_gpu_block_pool(gpu_pool)

        # Buffered state survived the redundant bind.
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        assert len(store.collect_due().to_store) == 1

    def test_rebind_different_pool_raises(self):
        store, _ = self._setup()
        with pytest.raises(ValueError, match="already bound"):
            store.bind_gpu_block_pool(_make_gpu_pool())
