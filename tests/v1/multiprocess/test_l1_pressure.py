# SPDX-License-Identifier: Apache-2.0
"""Tests for the GET_L1_PRESSURE endpoint and its scheduler-side poll.

Covers the three layers separately:

1. Protocol registration -- the request type resolves to an empty payload
   and an ``L1PressureStats`` response.
2. Server handler -- ``ManagementModule.get_l1_pressure`` pairs the storage
   manager's usage tuple with its deletion totals.
3. Scheduler adapter -- ``poll_l1_pressure`` drives a threadless,
   non-blocking poll: submit cycles at most every ``min_interval``, aggregate
   only complete cycles, drop failed or timed-out cycles, and never submit
   while unhealthy.
"""

# Standard
from unittest.mock import MagicMock
import threading

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (
    L1PressureSample,
    LMCacheMPSchedulerAdapter,
)
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.mq import MessageQueueClient
from lmcache.v1.multiprocess.protocol import (
    RequestType,
    get_payload_classes,
    get_response_class,
)
from lmcache.v1.multiprocess.protocols.controller import L1PressureStats

# ============================================================================
# Protocol registration
# ============================================================================


def test_protocol_registration():
    """GET_L1_PRESSURE takes no payload and returns L1PressureStats."""
    assert get_payload_classes(RequestType.GET_L1_PRESSURE) == []
    assert get_response_class(RequestType.GET_L1_PRESSURE) is L1PressureStats


# ============================================================================
# Server handler
# ============================================================================


def test_handler_pairs_usage_with_deletion_totals():
    """The handler snapshots usage and cumulative deletion totals."""
    module = ManagementModule.__new__(ManagementModule)
    ctx = MagicMock()
    ctx.storage_manager.get_l1_usage.return_value = (700, 1000)
    ctx.storage_manager.get_l1_deletion_totals.return_value = (4096, 3)
    module._ctx = ctx

    stats = module.get_l1_pressure()

    assert stats == L1PressureStats(
        total_bytes=1000,
        used_bytes=700,
        evicted_bytes_total=4096,
        evicted_chunks_total=3,
    )


# ============================================================================
# Scheduler adapter poll
# ============================================================================

URL_A = "tcp://a:0"
URL_B = "tcp://b:0"


def _make_adapter(urls: list[str]) -> LMCacheMPSchedulerAdapter:
    """Build a scheduler adapter skeleton with mocked MQ clients.

    Args:
        urls: Server URLs; one healthy mock client is created per URL.

    Returns:
        An adapter with only the state ``poll_l1_pressure`` touches.
    """
    adapter = LMCacheMPSchedulerAdapter.__new__(LMCacheMPSchedulerAdapter)
    adapter._server_urls = list(urls)
    adapter._mq_timeout = 30.0
    adapter._health_events = {}
    adapter.mq_clients = {}
    for url in urls:
        event = threading.Event()
        event.set()
        adapter._health_events[url] = event
        adapter.mq_clients[url] = MagicMock(spec=MessageQueueClient)
    adapter._l1_pressure_futures = {}
    adapter._l1_pressure_partial = {}
    adapter._l1_pressure_sample = None
    adapter._l1_pressure_last_submit = 0.0
    return adapter


def _pending_future() -> MagicMock:
    """A future whose response has not arrived."""
    future = MagicMock()
    future.query.return_value = False
    return future


def _done_future(stats: L1PressureStats) -> MagicMock:
    """A future that resolved to ``stats``."""
    future = MagicMock()
    future.query.return_value = True
    future.result.return_value = stats
    return future


def _failed_future() -> MagicMock:
    """A future whose result raises."""
    future = MagicMock()
    future.query.return_value = True
    future.result.side_effect = TimeoutError("no response")
    return future


class TestPollL1Pressure:
    def test_first_call_submits_and_returns_none(self):
        adapter = _make_adapter([URL_A, URL_B])
        for client in adapter.mq_clients.values():
            client.submit_request.return_value = _pending_future()

        assert adapter.poll_l1_pressure(10.0) is None

        for client in adapter.mq_clients.values():
            client.submit_request.assert_called_once()
            assert client.submit_request.call_args[0][0] == RequestType.GET_L1_PRESSURE

    def test_complete_cycle_aggregates_across_servers(self):
        adapter = _make_adapter([URL_A, URL_B])
        adapter.mq_clients[URL_A].submit_request.return_value = _done_future(
            L1PressureStats(
                total_bytes=100,
                used_bytes=80,
                evicted_bytes_total=7,
                evicted_chunks_total=1,
            )
        )
        adapter.mq_clients[URL_B].submit_request.return_value = _done_future(
            L1PressureStats(
                total_bytes=200,
                used_bytes=110,
                evicted_bytes_total=5,
                evicted_chunks_total=2,
            )
        )

        assert adapter.poll_l1_pressure(10.0) is None  # submit cycle
        sample = adapter.poll_l1_pressure(10.0)  # fold + aggregate

        assert isinstance(sample, L1PressureSample)
        assert sample.total_bytes == 300
        assert sample.used_bytes == 190
        assert sample.evicted_bytes_total == 12
        assert sample.evicted_chunks_total == 3

    def test_min_interval_gates_resubmission(self):
        adapter = _make_adapter([URL_A])
        adapter.mq_clients[URL_A].submit_request.return_value = _done_future(
            L1PressureStats(
                total_bytes=1,
                used_bytes=1,
                evicted_bytes_total=0,
                evicted_chunks_total=0,
            )
        )

        adapter.poll_l1_pressure(3600.0)
        adapter.poll_l1_pressure(3600.0)
        adapter.poll_l1_pressure(3600.0)

        adapter.mq_clients[URL_A].submit_request.assert_called_once()

    def test_failed_response_drops_cycle_and_keeps_last_sample(self):
        adapter = _make_adapter([URL_A])
        previous = L1PressureSample(
            monotonic_time=1.0,
            total_bytes=1,
            used_bytes=1,
            evicted_bytes_total=1,
            evicted_chunks_total=1,
        )
        adapter._l1_pressure_sample = previous
        adapter.mq_clients[URL_A].submit_request.return_value = _failed_future()

        assert adapter.poll_l1_pressure(3600.0) is previous  # submit
        assert adapter.poll_l1_pressure(3600.0) is previous  # failure folds

        assert adapter._l1_pressure_futures == {}
        assert adapter._l1_pressure_partial == {}

    def test_unhealthy_server_blocks_submission(self):
        adapter = _make_adapter([URL_A, URL_B])
        adapter._health_events[URL_B].clear()

        assert adapter.poll_l1_pressure(10.0) is None

        adapter.mq_clients[URL_A].submit_request.assert_not_called()
        adapter.mq_clients[URL_B].submit_request.assert_not_called()

    def test_partial_cycle_waits_for_all_servers(self):
        adapter = _make_adapter([URL_A, URL_B])
        adapter.mq_clients[URL_A].submit_request.return_value = _done_future(
            L1PressureStats(
                total_bytes=100,
                used_bytes=80,
                evicted_bytes_total=7,
                evicted_chunks_total=1,
            )
        )
        adapter.mq_clients[URL_B].submit_request.return_value = _pending_future()

        adapter.poll_l1_pressure(10.0)  # submit
        assert adapter.poll_l1_pressure(10.0) is None  # A done, B pending

    def test_stuck_cycle_is_abandoned_after_mq_timeout(self):
        adapter = _make_adapter([URL_A])
        adapter._mq_timeout = 0.0
        adapter.mq_clients[URL_A].submit_request.return_value = _pending_future()

        adapter.poll_l1_pressure(0.001)  # submit
        adapter._l1_pressure_last_submit = 0.0  # push the cycle into the past
        adapter.poll_l1_pressure(0.001)

        # The stuck cycle was abandoned and a fresh one submitted in its place.
        assert adapter.mq_clients[URL_A].submit_request.call_count == 2

    def test_non_positive_interval_raises(self):
        adapter = _make_adapter([URL_A])
        with pytest.raises(ValueError):
            adapter.poll_l1_pressure(0.0)
