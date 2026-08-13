# SPDX-License-Identifier: Apache-2.0
"""Facade-level tests for the lazy offload pending store.

The pure policy behind EVICTION_AWARE mode is covered in depth by
test_lazy_offload_policy.py; here the eviction-aware tests only verify the
facade's routing and mode guards. The FIFO sections exercise the legacy
count-triggered placeholder policy kept for compatibility.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.integration.vllm import lazy_offload_pending_store as pending_store_mod
from lmcache.integration.vllm.lazy_offload_pending_store import (
    AddOutcome,
    FIFOOffloadPolicy,
    LazyOffloadMode,
    LazyOffloadPendingStore,
)

FIFO_CONFIG = {"lmcache.mp.lazy_offload_policy": "FIFO"}


def _spy_logger(monkeypatch: pytest.MonkeyPatch, method: str) -> list[str]:
    """Capture the module logger's lines emitted through one method.

    The lmcache logger does not propagate to the root logger
    (``propagate=False``), so pytest's ``caplog`` never sees its records;
    spy on the method instead.

    Args:
        monkeypatch: The fixture used to install the spy.
        method: The logger method to capture, e.g. ``"info"``.

    Returns:
        A list that accumulates the formatted messages as they are logged.
    """
    messages: list[str] = []

    def spy(msg: object, *args: object, **kwargs: object) -> None:
        messages.append(str(msg) % args if args else str(msg))

    monkeypatch.setattr(pending_store_mod.logger, method, spy)
    return messages


def _spy_logger_info(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Capture the module logger's INFO lines.

    Args:
        monkeypatch: The fixture used to install the spy.

    Returns:
        A list that accumulates the formatted messages as they are logged.
    """
    return _spy_logger(monkeypatch, "info")


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
# Tests for FIFOOffloadPolicy (legacy placeholder)
# ===========================================================================


class TestFIFOOffloadPolicy:
    """Legacy count-triggered policy (``lazy_offload_policy = "FIFO"``).

    Kept for compatibility; EVICTION_AWARE is the default mode."""

    def test_init_default_threshold(self) -> None:
        policy = FIFOOffloadPolicy()
        assert policy._threshold == 100

    def test_init_custom_threshold(self) -> None:
        configs = {"lmcache.mp.lazy_offload_threshold": 50}
        policy = FIFOOffloadPolicy(configs)
        assert policy._threshold == 50

    def test_add_creates_new_item(self) -> None:
        policy = FIFOOffloadPolicy()
        meta = _make_meta("req-0")
        hashes = _make_block_hashes([0, 1])
        policy.add(meta, hashes)
        assert "req-0" in policy._pending_items
        assert len(policy._pending_items["req-0"].metadatas) == 1

    def test_add_same_request_appends_metadatas(self) -> None:
        policy = FIFOOffloadPolicy()
        meta1 = _make_meta("req-0", num_blocks=1)
        meta2 = _make_meta("req-0", num_blocks=2)
        policy.add(meta1, _make_block_hashes([0]))
        policy.add(meta2, _make_block_hashes([0, 1]))
        assert len(policy._pending_items["req-0"].metadatas) == 2

    def test_should_offload_below_threshold(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        assert policy._finished_requests_count == 1
        assert policy.should_offload() is False

    def test_should_offload_at_threshold(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 3})
        for i in range(3):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")
        assert policy.should_offload() is True

    def test_should_offload_above_threshold(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        for i in range(5):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")
        assert policy.should_offload() is True

    def test_mark_req_finished_not_in_pending_returns_false(self) -> None:
        # A request may finish without ever producing store metadata; that
        # must not crash the scheduler.
        policy = FIFOOffloadPolicy()
        assert policy.mark_req_finished("nonexistent") is False

    def test_reclaim_finished_request_drops_predecessor_item(self) -> None:
        """A new request reusing a finished predecessor's id must not
        inherit its buffered item; the caller ends the old session."""
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        assert policy.reclaim_finished_request("req-0") is True
        assert policy.should_offload() is False
        # The successor buffers fresh, unconflated state.
        policy.add(_make_meta("req-0"), _make_block_hashes([1]))
        assert len(policy.select_items(10)) == 0  # not finished yet

    def test_reclaim_finished_request_ignores_live_and_unknown_ids(self) -> None:
        policy = FIFOOffloadPolicy()
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        assert policy.reclaim_finished_request("req-0") is False
        assert "req-0" in policy._pending_items
        assert policy.reclaim_finished_request("nonexistent") is False

    def test_select_items_returns_only_finished(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        # req-1 is not finished

        selected = policy.select_items(10)
        assert len(selected) == 1
        assert selected[0].request_id == "req-0"

    def test_select_items_removes_from_pending(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 2})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.mark_req_finished("req-0")
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        policy.mark_req_finished("req-1")

        selected = policy.select_items(10)
        assert len(selected) == 2
        assert len(policy._pending_items) == 0
        assert policy._finished_requests_count == 0

    def test_select_items_skips_unfinished(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        policy.add(_make_meta("req-0"), _make_block_hashes([0]))
        policy.add(_make_meta("req-1"), _make_block_hashes([1]))
        policy.mark_req_finished("req-1")

        selected = policy.select_items(10)
        assert len(selected) == 1
        assert selected[0].request_id == "req-1"
        # req-0 still pending
        assert "req-0" in policy._pending_items

    def test_select_items_count_limits_output(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        for i in range(5):
            policy.add(_make_meta(f"req-{i}"), _make_block_hashes([i]))
            policy.mark_req_finished(f"req-{i}")

        selected = policy.select_items(2)
        assert len(selected) == 2
        assert len(policy._pending_items) == 3

    def test_select_items_empty(self) -> None:
        policy = FIFOOffloadPolicy()
        assert policy.select_items(5) == []

    def test_drop_request_discards_items_and_finished_count(self) -> None:
        policy = FIFOOffloadPolicy({"lmcache.mp.lazy_offload_threshold": 1})
        policy.add(_make_meta("req-0", num_blocks=1), _make_block_hashes([0]))
        policy.add(_make_meta("req-0", num_blocks=2), _make_block_hashes([0, 1]))
        policy.mark_req_finished("req-0")
        assert policy.should_offload() is True

        assert policy.drop_request("req-0") == 2
        assert policy.should_offload() is False
        assert policy.select_items(10) == []

    def test_drop_request_unknown_is_noop(self) -> None:
        policy = FIFOOffloadPolicy()
        assert policy.drop_request("nonexistent") == 0


# ===========================================================================
# Tests for the LazyOffloadPendingStore facade (FIFO mode + shared plumbing)
# ===========================================================================


class TestLazyOffloadPendingStore:
    """Facade construction, mode-independent plumbing (pool binding, block-id
    tracking), and delegation to the legacy FIFO policy."""

    def _setup_store_with_gpu_pool(
        self, configs: dict | None = None
    ) -> LazyOffloadPendingStore:
        store = LazyOffloadPendingStore({**FIFO_CONFIG, **(configs or {})})
        store.bind_gpu_block_pool(_make_gpu_pool())
        return store

    def test_init_default_policy_is_eviction_aware(self) -> None:
        store = LazyOffloadPendingStore()
        assert store.mode is LazyOffloadMode.EVICTION_AWARE

    def test_init_fifo_policy_explicit(self) -> None:
        store = LazyOffloadPendingStore(dict(FIFO_CONFIG))
        assert store.mode is LazyOffloadMode.FIFO

    def test_init_unknown_policy_raises(self) -> None:
        configs = {"lmcache.mp.lazy_offload_policy": "UNKNOWN"}
        with pytest.raises(ValueError, match="Unknown offload policy"):
            LazyOffloadPendingStore(configs)

    def test_init_default_select_count(self) -> None:
        store = LazyOffloadPendingStore()
        assert store._select_count == 10

    def test_init_custom_select_count(self) -> None:
        configs = {"lmcache.mp.lazy_offload_select_count": 5}
        store = LazyOffloadPendingStore(configs)
        assert store._select_count == 5

    def test_bind_gpu_block_pool(self) -> None:
        store = LazyOffloadPendingStore()
        gpu_pool = MagicMock()
        store.bind_gpu_block_pool(gpu_pool)
        assert store._gpu_block_pool is gpu_pool

    def test_add_without_gpu_pool_raises(self) -> None:
        store = LazyOffloadPendingStore()
        meta = _make_meta("req-0")
        with pytest.raises(ValueError, match="gpu block pool not bound"):
            store.add(meta)

    def test_add_with_gpu_pool(self) -> None:
        store = self._setup_store_with_gpu_pool(
            {"lmcache.mp.lazy_offload_threshold": 1}
        )
        meta = _make_meta("req-0", num_blocks=2)
        assert store.add(meta) is AddOutcome.BUFFERED
        store.mark_req_finished("req-0")
        # The buffered item carries hashes snapshotted from the gpu pool.
        (item,) = store.select_items()
        assert item.metadatas == [(meta, {0: b"hash-0", 1: b"hash-1"})]

    def test_should_offload_delegates_to_policy(self) -> None:
        configs = {"lmcache.mp.lazy_offload_threshold": 2}
        store = self._setup_store_with_gpu_pool(configs)

        store.add(_make_meta("req-0"))
        store.mark_req_finished("req-0")
        assert store.should_offload() is False

        store.add(_make_meta("req-1"))
        store.mark_req_finished("req-1")
        assert store.should_offload() is True

    def test_select_items_returns_correct_count(self) -> None:
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

    def test_mark_req_finished(self) -> None:
        configs = {"lmcache.mp.lazy_offload_threshold": 1}
        store = self._setup_store_with_gpu_pool(configs)
        store.add(_make_meta("req-0"))
        store.mark_req_finished("req-0")
        assert store.should_offload() is True

    def test_update_get_remove_gpu_block_ids(self) -> None:
        store = LazyOffloadPendingStore()
        store.update_request_gpu_block_ids("req-0", [1, 2])
        store.update_request_gpu_block_ids("req-0", [3])
        assert store.get_request_gpu_block_ids("req-0") == [1, 2, 3]

        store.remove_request_gpu_block_ids("req-0")
        assert store.get_request_gpu_block_ids("req-0") == []

    def test_get_gpu_block_ids_nonexistent_returns_empty(self) -> None:
        store = LazyOffloadPendingStore()
        assert store.get_request_gpu_block_ids("nonexistent") == []

    def test_unknown_request_lookup_does_not_open_receipt_window(self) -> None:
        """A read of an unknown id must not create state: were
        has_in_flight_store to flip True, a stale or duplicate receipt
        would unpin blocks that are not pinned and end the session twice."""
        store = LazyOffloadPendingStore()
        store.get_request_gpu_block_ids("ghost")
        assert store.has_in_flight_store("ghost") is False

    def test_end_to_end_flow(self) -> None:
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

    def test_select_items_multiple_batches(self) -> None:
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
    """Facade routing to the eviction-aware queue and FIFO entry-point guards.

    Policy semantics (pressure, prefix closure, deduplication, lifecycle) are
    covered by test_lazy_offload_policy.py; these tests stop at the facade."""

    def _setup(
        self, configs: dict | None = None
    ) -> tuple[LazyOffloadPendingStore, MagicMock]:
        store = LazyOffloadPendingStore(configs)
        gpu_pool = _make_gpu_pool()
        store.bind_gpu_block_pool(gpu_pool)
        return store, gpu_pool

    def test_add_buffers_hashed_op(self) -> None:
        store, _ = self._setup()
        assert store.add(_make_meta("req-0", num_blocks=2)) is AddOutcome.BUFFERED

    def test_add_skips_unhashed_op(self) -> None:
        store, gpu_pool = self._setup()
        gpu_pool.blocks[1].block_hash = None
        assert store.add(_make_meta("req-0", num_blocks=2)) is (
            AddOutcome.SKIPPED_UNHASHED
        )

    def test_fifo_entry_points_raise(self) -> None:
        store, _ = self._setup()
        with pytest.raises(ValueError, match="FIFO policy unavailable"):
            store.should_offload()
        with pytest.raises(ValueError, match="FIFO policy unavailable"):
            store.select_items()

    def test_collect_due_under_pressure_emits_op(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        meta = _make_meta("req-0", num_blocks=2)
        store.add(meta)
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        result = store.collect_due()
        assert [op.store_metadata for op in result.to_store] == [meta]

    def test_collect_due_without_pressure_holds(self) -> None:
        store, _ = self._setup()
        store.add(_make_meta("req-0", num_blocks=2))
        store.observe_step(new_blocks_allocated=0, est_next_step_blocks=0)
        assert store.collect_due().to_store == []

    def test_session_release_flow(self) -> None:
        """finish -> drain -> receipt: teardown allowed only at the receipt."""
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        assert store.mark_req_finished("req-0") is True
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        result = store.collect_due()
        assert len(result.to_store) == 1
        assert result.released_requests == []
        assert store.notify_store_complete("req-0") is True

    def test_mark_req_finished_without_pending_allows_teardown(self) -> None:
        store, _ = self._setup()
        assert store.mark_req_finished("req-unknown") is False

    def test_drop_request_discards_buffered_ops(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))
        assert store.drop_request("req-0") == 1
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        assert store.collect_due().to_store == []

    def test_reclaim_finished_request_routes_to_eviction_queue(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))
        assert store.mark_req_finished("req-0") is True
        assert store.reclaim_finished_request("req-0") is True
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        assert store.collect_due().to_store == []

    def test_mark_store_failed_drops_buffered_ops(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))
        assert store.mark_store_failed("req-0") == 1
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        assert store.collect_due().to_store == []

    def test_stats_reports_the_cumulative_counters(self) -> None:
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        store.collect_due()
        stats = store.stats()
        assert stats.admitted == 1
        assert stats.emitted == 1
        assert stats.dropped_evicted == 0

    def test_stats_counts_a_drop_of_evicted_blocks(self) -> None:
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        gpu_pool.blocks[0].block_hash = b"reallocated"
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        assert store.collect_due().to_store == []
        assert store.stats().dropped_evicted == 1

    def test_stats_unavailable_in_fifo_mode_or_before_bind(self) -> None:
        with pytest.raises(ValueError, match="EVICTION_AWARE queue unavailable"):
            LazyOffloadPendingStore().stats()
        fifo = LazyOffloadPendingStore(dict(FIFO_CONFIG))
        fifo.bind_gpu_block_pool(_make_gpu_pool())
        with pytest.raises(ValueError, match="EVICTION_AWARE queue unavailable"):
            fifo.stats()

    def test_evicted_drop_is_logged_at_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """One aggregate INFO line per drain with the drop count: the drop
        ledger must be visible in production logs (which rarely run at
        DEBUG) without flooding the scheduler path when a burst evicts a
        large queue at once."""
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        messages = _spy_logger_info(monkeypatch)
        store.add(_make_meta("req-0", num_blocks=1))
        gpu_pool.blocks[0].block_hash = b"reallocated"
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        store.collect_due()
        (line,) = [m for m in messages if "blocks evicted before drain" in m]
        assert "dropped 1 store op(s)" in line
        assert "req-0" in line

    def test_short_prefix_drop_is_logged_at_info(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A gate-3 drop is cache-quality loss too, so it gets the same
        aggregate INFO line as the eviction path: the counter alone cannot
        say which request lost its prefix."""
        store, _ = self._setup(
            {
                "lmcache.mp.lazy_offload_horizon_steps": 1.0,
                "lmcache.mp.lazy_offload_min_prefix_tokens": 4096,
            }
        )
        messages = _spy_logger_info(monkeypatch)
        store.add(_make_meta("req-0", num_blocks=1, end=256))
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        result = store.collect_due()
        assert len(result.dropped_short_prefix) == 1
        (line,) = [m for m in messages if "below the break-even length" in m]
        assert "dropped 1 store op(s)" in line
        assert "req-0 (prefix 256)" in line

    def test_drop_line_reports_the_ops_it_could_not_name(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A drain that drops more ops than the line samples must say so.

        The line names at most eight ops, and the count of the rest is the
        only thing between "these are the ops that were lost" and a reader
        (or a per-request-id grep) concluding that nothing else was. The
        truncation itself is untested at every other layer: on hardware the
        aggregate lines are asserted never to truncate, precisely so that
        the greps stay sound."""
        store, _ = self._setup(
            {
                "lmcache.mp.lazy_offload_horizon_steps": 1.0,
                "lmcache.mp.lazy_offload_min_prefix_tokens": 4096,
            }
        )
        messages = _spy_logger_info(monkeypatch)
        for block_id in range(10):
            meta = _make_meta(f"req-{block_id}", end=256)
            meta.op.flat_block_ids = [block_id]
            store.add(meta)
        store.observe_step(new_blocks_allocated=40, est_next_step_blocks=0)
        result = store.collect_due()
        assert len(result.dropped_short_prefix) == 10
        (line,) = [m for m in messages if "below the break-even length" in m]
        assert "dropped 10 store op(s)" in line
        assert line.count("(prefix 256)") == 8
        assert "+2 more" in line

    def test_skip_of_a_broken_request_logs_at_debug_only(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """A broken request keeps producing chunks and every one of them is
        rejected; at INFO those would bury the one line that reported the
        cause, so the tail logs at DEBUG."""
        store, gpu_pool = self._setup()
        gpu_pool.blocks[0].block_hash = None
        assert store.add(_make_meta("req-0", num_blocks=1)) is (
            AddOutcome.SKIPPED_UNHASHED
        )
        info = _spy_logger_info(monkeypatch)
        debug = _spy_logger(monkeypatch, "debug")
        assert store.add(_make_meta("req-0", num_blocks=2)) is (
            AddOutcome.SKIPPED_PREFIX_BROKEN
        )
        assert [m for m in info if "req-0" in m] == []
        assert [m for m in debug if "prefix chain is already broken" in m] != []

    def test_drain_logs_the_ledger_periodically(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """collect_due logs a ledger line when the counters changed, at
        most once per throttle interval, and logs AGAIN once the interval
        lapses: the log must converge to the true ledger even when a
        force-killed engine never reaches the shutdown hook."""
        clock = [1000.0]
        monkeypatch.setattr(pending_store_mod.time, "monotonic", lambda: clock[0])
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        messages = _spy_logger_info(monkeypatch)
        store.add(_make_meta("req-0", num_blocks=1))
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        store.collect_due()  # changed since start -> logs
        store.collect_due()  # unchanged -> silent
        store.add(_make_meta("req-1", num_blocks=2))
        store.collect_due()  # changed, but inside the throttle -> silent
        ledgers = [m for m in messages if m.startswith("Lazy offload counters:")]
        assert len(ledgers) == 1
        assert "admitted=1" in ledgers[0]
        assert "emitted=1" in ledgers[0]
        # The depth belongs on the periodic line, not only on the shutdown
        # one: a force-killed engine never reaches log_final_stats, so this
        # is the line an operator (and the GPU harness) actually reads, and
        # without pending it does not close as an equation.
        assert "pending=0" in ledgers[0]
        clock[0] += 6.0
        store.collect_due()  # throttle lapsed, change pending -> logs again
        ledgers = [m for m in messages if m.startswith("Lazy offload counters:")]
        assert len(ledgers) == 2
        assert "admitted=2" in ledgers[1]
        assert "pending=0" in ledgers[1]

    def test_log_final_stats_emits_the_counter_ledger(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        gpu_pool.blocks[0].block_hash = b"reallocated"
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        store.collect_due()
        messages = _spy_logger_info(monkeypatch)
        store.log_final_stats()
        (line,) = [m for m in messages if "final counters" in m]
        assert "admitted=1" in line
        assert "emitted=0" in line
        assert "dropped_evicted=1" in line

    def test_ledger_reports_the_pending_depth(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """The ledger carries the queue depth at the same instant as the
        counters, so the line closes as an equation: an op that left the
        queue without incrementing any outcome counter would show up as
        admitted > pending + outcomes."""
        store, _ = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=1))
        store.add(_make_meta("req-1", num_blocks=1))
        messages = _spy_logger_info(monkeypatch)
        store.log_final_stats()
        (held,) = [m for m in messages if "final counters" in m]
        assert "admitted=2" in held
        assert "emitted=0" in held
        assert "pending=2" in held

        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        assert len(store.collect_due().to_store) == 2
        messages.clear()
        store.log_final_stats()
        (drained,) = [m for m in messages if "final counters" in m]
        assert "admitted=2" in drained
        assert "emitted=2" in drained
        assert "pending=0" in drained

    def test_log_final_stats_is_silent_when_nothing_counted(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        """Unbound or FIFO stores have no counters; shutdown must neither
        raise nor log a bogus ledger."""
        messages = _spy_logger_info(monkeypatch)
        LazyOffloadPendingStore().log_final_stats()
        fifo = LazyOffloadPendingStore(dict(FIFO_CONFIG))
        fifo.bind_gpu_block_pool(_make_gpu_pool())
        fifo.log_final_stats()
        # The constructor's own banner may log; the ledger line must not.
        assert [m for m in messages if "final counters" in m] == []

    def test_rebind_same_pool_is_idempotent(self) -> None:
        store, gpu_pool = self._setup({"lmcache.mp.lazy_offload_horizon_steps": 1.0})
        store.add(_make_meta("req-0", num_blocks=2))

        store.bind_gpu_block_pool(gpu_pool)

        # Buffered state survived the redundant bind.
        store.observe_step(new_blocks_allocated=4, est_next_step_blocks=0)
        assert len(store.collect_due().to_store) == 1

    def test_rebind_different_pool_raises(self) -> None:
        store, _ = self._setup()
        with pytest.raises(ValueError, match="already bound"):
            store.bind_gpu_block_pool(_make_gpu_pool())
