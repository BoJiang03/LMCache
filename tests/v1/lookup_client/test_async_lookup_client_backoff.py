# SPDX-License-Identifier: Apache-2.0
"""Regression tests for the async lookup client's scheduler-side backoff.

``lookup_cache`` is called from vLLM's scheduler thread, which is also the
thread that steps the engine.  It used to ``time.sleep(lookup_backoff_time)``
once per polled request *while holding* ``self.lock``, which meant:

  * the response thread could not record results while the scheduler waited
    for exactly those results, and
  * the stall grew with the number of in-flight lookups, because the scheduler
    re-polls its whole ``skipped_waiting`` queue on every pass.

Both properties are cheap to assert and expensive to rediscover, so they are
pinned here.  The client is built with ``__new__`` and only the attributes the
polling path touches, to keep the test free of ZMQ sockets and worker
processes.
"""

# Standard
from typing import Optional
import threading
import time

# Third Party
import pytest

# First Party
from lmcache.v1.lookup_client.lmcache_async_lookup_client import (
    LMCacheAsyncLookupClient,
)

BACKOFF = 0.01


class _StubConfig:
    lookup_timeout_ms = 240000


def _make_client() -> LMCacheAsyncLookupClient:
    client = LMCacheAsyncLookupClient.__new__(LMCacheAsyncLookupClient)
    client.config = _StubConfig()
    client.lock = threading.Lock()
    client.reqs_status: dict[str, Optional[int]] = {}
    client.first_lookup_time: dict[str, float] = {}
    client.aborted_lookups = set()
    client.lookup_backoff_time = BACKOFF
    client._backoff_lock = threading.Lock()
    client._next_backoff_allowed_at = 0.0
    return client


def _register(client: LMCacheAsyncLookupClient, lookup_id: str) -> None:
    """First poll registers the request and reports 'not found'."""
    assert client.lookup_cache(lookup_id) == -1


def test_pending_poll_does_not_hold_the_lock(monkeypatch):
    """The response thread must be able to record results while we back off.

    Probes the lock from inside the backoff itself rather than racing a helper
    thread against it: a plain ``threading.Lock`` refuses a non-blocking
    re-acquire by its current owner, so this fails deterministically if the
    backoff ever moves back under ``self.lock``.
    """
    client = _make_client()
    _register(client, "req")

    lock_was_free = []
    real_sleep = time.sleep

    def probing_sleep(seconds):
        acquired = client.lock.acquire(blocking=False)
        lock_was_free.append(acquired)
        if acquired:
            client.lock.release()
        real_sleep(seconds)

    monkeypatch.setattr(time, "sleep", probing_sleep)
    assert client.lookup_cache("req") is None  # still in flight -> backs off

    assert lock_was_free, "the pending poll did not back off at all"
    assert all(lock_was_free), (
        "lookup_cache held self.lock across its backoff, so "
        "process_responses_from_workers cannot record the result being waited on"
    )


def test_backoff_is_per_pass_not_per_request():
    """N pending requests in one scheduler pass must not cost N backoffs."""
    client = _make_client()
    num_requests = 50
    for i in range(num_requests):
        _register(client, f"req{i}")

    start = time.monotonic()
    for i in range(num_requests):
        assert client.lookup_cache(f"req{i}") is None
    elapsed = time.monotonic() - start

    # The old code slept once per request: 50 * 10ms = 500ms.  The gate allows
    # at most one backoff per interval of wall clock, so a pass over 50 pending
    # requests costs a small constant number of them.
    assert elapsed < num_requests * BACKOFF / 5, (
        f"polling {num_requests} pending lookups took {elapsed * 1000:.0f}ms; "
        "the backoff is scaling with the number of in-flight requests"
    )


def test_resolved_lookup_is_returned_without_backoff():
    client = _make_client()
    _register(client, "req")
    with client.lock:
        client.reqs_status["req"] = 4096

    start = time.monotonic()
    assert client.lookup_cache("req") == 4096
    assert time.monotonic() - start < BACKOFF


def test_result_landing_during_backoff_is_picked_up_immediately():
    """A response that arrives during the yield is used in the same poll."""
    client = _make_client()
    _register(client, "req")

    def land_result():
        time.sleep(BACKOFF / 4)
        with client.lock:
            client.reqs_status["req"] = 1234

    t = threading.Thread(target=land_result)
    t.start()
    try:
        assert client.lookup_cache("req") == 1234
    finally:
        t.join(timeout=1.0)


def test_timeout_still_reports_zero_hit_tokens():
    """The lookup_timeout_ms escape hatch must survive the restructuring."""
    client = _make_client()
    _register(client, "req")
    client.config = _StubConfig()
    client.config.lookup_timeout_ms = 0  # everything is instantly overdue
    with client.lock:
        client.first_lookup_time["req"] = time.time() - 1.0

    assert client.lookup_cache("req") == 0
    assert "req" in client.aborted_lookups
    assert "req" not in client.first_lookup_time


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
