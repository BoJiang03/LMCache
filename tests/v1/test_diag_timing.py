# SPDX-License-Identifier: Apache-2.0
"""DIAG (not for merge): the scheduler-side timing line and its overhead."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from unittest.mock import MagicMock
import re
import time

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402

# First Party
from lmcache.diag_timing import (  # noqa: E402
    SchedulerDiag,
    _bucket,
    _bucket_mid,
    sched_diag,
    timed,
    timed_step,
)
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)

# Test helpers
from tests.v1.test_mp_connector_kv_roles import (  # noqa: E402, F401
    _request,
    _schedule,
    connectors,
    mock_io,
)
from tests.v1.test_mp_nonblocking_lookup import (  # noqa: E402, F401
    AdapterFactory,
    make_adapter,
)


@pytest.fixture
def kv_role() -> str:
    """A role that both stores and loads."""
    return "kv_both"


@pytest.fixture
def lazy_offload() -> bool:
    """Lazy offload on, so the drain and admission parts are charged."""
    return True


@pytest.fixture
def diag() -> Iterator[SchedulerDiag]:
    """The process-wide collector, with a fresh window."""
    d = sched_diag()
    d.flush(time.monotonic())
    yield d


def _field(line: str, pattern: str) -> float:
    match = re.search(pattern, line)
    assert match is not None, f"{pattern!r} not in {line!r}"
    return float(match.group(1))


def test_histogram_buckets_bracket_their_values() -> None:
    """A value's bucket midpoint is within one quarter octave of it."""
    for value in (1.5e-6, 3e-5, 1e-3, 0.0123, 1.0, 7.7):
        mid = _bucket_mid(_bucket(value))
        assert value / 1.19 <= mid <= value * 1.19


def test_connector_steps_methods_and_load(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],  # noqa: F811
    mock_io: SimpleNamespace,  # noqa: F811
    diag: SchedulerDiag,
) -> None:
    """A lookup -> retrieve -> finish sequence shows in the window's line."""
    scheduler, _worker = connectors
    request = _request()
    mock_io.scheduler.check_lookup_result.side_effect = [None, 8]
    assert scheduler.get_num_new_matched_tokens(request, 4) == (None, True)
    assert scheduler.get_num_new_matched_tokens(request, 4) == (4, True)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: ([1, 2],)), 4
    )
    metadata = scheduler.build_connector_meta(_schedule())
    assert [m.direction for m in metadata.requests] == ["RETRIEVE"]
    time.sleep(0.02)
    scheduler.update_connector_output(KVConnectorOutput(finished_recving={"request"}))
    scheduler.request_finished(request, [1, 2])
    scheduler.build_connector_meta(_schedule())

    line = diag.flush(time.monotonic())

    assert "steps=2 " in line
    assert "calls gnmt=2 admit=1 usaa=1 bcm=2 drain=2 uco=1 rf=1 onr=0" in line
    assert _field(line, r"interval_ms mean=([\d.]+)") >= 20.0
    assert _field(line, r"load_ms n=1 mean=([\d.]+)") >= 20.0
    assert "loads_inflight=0" in line
    assert "lazy pending_ops=0 inflight_stores=0" in line
    assert _field(line, r"cpu_us/step mean/p99 gnmt=(\d+)/") > 0
    assert _field(line, r"wall_us/step mean/p99 .*?bcm=(\d+)/") > 0


def test_lookup_split_from_the_adapter(
    make_adapter: AdapterFactory,  # noqa: F811
    diag: SchedulerDiag,
) -> None:
    """Submit -> reply and reply -> consume are measured apart."""
    adapter, (client,) = make_adapter()
    adapter.maybe_submit_lookup_request("r", list(range(256)))
    time.sleep(0.01)
    client.ack.set_result(None)
    assert adapter.check_lookup_result("r") is None  # sends the status query
    time.sleep(0.02)
    client.status.set_result(4)
    time.sleep(0.03)
    assert adapter.check_lookup_result("r") == 256

    line = diag.flush(time.monotonic())

    reply = _field(line, r"submit_reply_ms n=1 mean=([\d.]+)")
    consume = _field(line, r"reply_consume_ms n=1 mean=([\d.]+)")
    ack = _field(line, r"ack_ms n=1 mean=([\d.]+)")
    rtt = _field(line, r"status_rtt_ms n=1 mean=([\d.]+)")
    assert 30.0 <= reply < 200.0
    assert 30.0 <= consume < 200.0
    assert 10.0 <= ack < reply
    assert 20.0 <= rtt < reply
    assert "polls_per_lookup=1.00" in line


def test_instrumentation_overhead_per_call() -> None:
    """Microbench: the wrapper's added cost per call and per step close."""
    d = SchedulerDiag(window_s=1e9)

    def noop() -> None:
        return None

    wrapped = timed(d, "gnmt", noop)
    step = timed_step(d, noop)
    n = 200_000
    t0 = time.perf_counter()
    for _ in range(n):
        noop()
    base = time.perf_counter() - t0
    t0 = time.perf_counter()
    for _ in range(n):
        wrapped()
    per_call = (time.perf_counter() - t0 - base) / n
    t0 = time.perf_counter()
    for _ in range(n // 10):
        step()
    per_step = (time.perf_counter() - t0) / (n // 10)
    t0 = time.perf_counter()
    d.flush(time.monotonic())
    flush = time.perf_counter() - t0
    print(
        f"\nDIAG overhead: per wrapped call {per_call * 1e9:.0f} ns, "
        f"per step close {per_step * 1e9:.0f} ns, "
        f"flush of {n // 10} steps {flush * 1e3:.1f} ms"
    )
    assert per_call < 5e-6
    assert per_step < 20e-6
