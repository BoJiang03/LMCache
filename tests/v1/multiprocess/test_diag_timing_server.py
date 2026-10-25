# SPDX-License-Identifier: Apache-2.0
"""DIAG (not for merge): the server-side timing line and its overhead."""

# Standard
import re
import threading
import time

# Third Party
import pytest
import zmq

# First Party
from lmcache.diag_timing import ServerDiag, server_diag
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.request_handler import HandlerType
from lmcache.v1.multiprocess.transport.zmq_impl.mq import (
    MessageQueueClient,
    MessageQueueServer,
)

# Test helpers
from tests.v1.multiprocess.test_exclusive_retrieve import (  # noqa: F401
    SEPARATE_GROUPS,
    _free_after_alloc,
    _look_up,
    _lookup_module,
    _Server,
    server,
)
from tests.v1.multiprocess.test_mq import _unused_tcp_port

pytestmark = pytest.mark.no_shared_allocator


def _stat(line: str, name: str) -> dict[str, float]:
    match = re.search(rf"\b{name}:(\S+)", line)
    assert match is not None, f"{name} not in {line!r}"
    return {k: float(v) for k, v in (kv.split("=") for kv in match.group(1).split(","))}


def test_mq_handlers_record_wall_and_queue_wait() -> None:
    """Serialized and blocking handlers both land in the window's line."""
    url = f"tcp://127.0.0.1:{_unused_tcp_port()}"
    context = zmq.Context.instance()
    srv = MessageQueueServer(url, context)

    def slow_noop() -> str:
        time.sleep(0.01)
        return "NOOP_OK"

    srv.add_handler("noop", HandlerType.BLOCKING, slow_noop)
    srv.add_normal_thread_pool(["noop"], max_workers=1)
    srv.start()
    client = MessageQueueClient(url, context)
    diag = server_diag()
    diag.flush()
    try:
        futures = [client.submit_request("noop", []) for _ in range(4)]
        for future in futures:
            assert isinstance(future, MessagingFuture)
            assert future.result(timeout=5) == "NOOP_OK"
            assert future.done_at > 0.0
    finally:
        client.close()
        srv.close()

    noop = _stat(diag.flush(), "noop")
    assert noop["n"] == 4
    assert 10.0 <= noop["mean_ms"] < 100.0
    # One worker: the last of four queued requests waits for three before it.
    assert noop["qwait_ms"] >= 5.0


def test_lookup_retrieve_and_sessions(server: _Server) -> None:  # noqa: F811
    """Prefetch, retrieve stream completion and session recreation."""
    diag = server_diag()
    diag.flush()
    lookup = _lookup_module(server, SEPARATE_GROUPS)

    assert _look_up(lookup, True, 0) == 3
    _free_after_alloc(lookup, True, 0, held_tokens=0)
    assert server.retrieve(exclusive=True)
    lookup.end_session("req")
    # A store after END_SESSION creates the session again, as a late lazy
    # store does.
    server.store(exclusive=True)
    line = diag.flush()

    assert _stat(line, "prefetch_submit_to_done")["n"] == 1
    assert _stat(line, "prefetch_done_to_consumed")["n"] == 1
    consumed = _stat(line, "prefetch_submit_to_consumed")
    assert consumed["n"] == 1 and consumed["mean_ms"] < 5000.0
    assert _stat(line, "retrieve_start_to_stream_done")["n"] == 1
    assert "session_ended=1/" in line
    assert "session_recreated_after_end=1/" in line
    assert "live_sessions=1" in line
    assert "live_sessions_recreated=1" in line
    assert "prefetch_jobs=0" in line


def test_record_overhead_and_thread_safety() -> None:
    """Microbench: per-record cost; counts from 4 threads add up exactly."""
    diag = ServerDiag(window_s=1e9)
    n = 200_000
    t0 = time.perf_counter()
    for _ in range(n):
        diag.record("op", 1e-4, 1e-5)
    per_record = (time.perf_counter() - t0) / n
    fut: MessagingFuture[int] = MessagingFuture()
    t0 = time.perf_counter()
    for _ in range(n):
        fut.set_result(1)
    per_set_result = (time.perf_counter() - t0) / n
    print(
        f"\nDIAG overhead: ServerDiag.record {per_record * 1e9:.0f} ns, "
        f"MessagingFuture.set_result (with done_at) {per_set_result * 1e9:.0f} ns"
    )
    diag.flush()

    def work() -> None:
        for _ in range(50_000):
            diag.record("mt", 1e-3)

    threads = [threading.Thread(target=work) for _ in range(4)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert _stat(diag.flush(), "mt")["n"] == 200_000
    assert per_record < 5e-6
