# SPDX-License-Identifier: Apache-2.0
"""Worker-adapter tests for the lazy offload store-receipt contract.

The scheduler unpins a drained store batch's blocks (and possibly ends the
request's session) only after collecting one store-completion receipt per
worker rank. These tests pin the adapter-side half of that contract: every
submitted store batch yields exactly one receipt from this rank, regardless
of writer role or server health -- a rank that will never produce a store
future must report completion immediately, or the scheduler waits forever
and the blocks stay pinned.

The adapter is built via ``__new__`` with only the attributes the tested
paths read, mirroring the connector-side suite in
``test_mp_connector_lazy_offload.py``.
"""

# Standard
from types import SimpleNamespace
from typing import cast
import threading

# Third Party
import pytest

pytest.importorskip("vllm", reason="the MP adapter imports vLLM at module top")

# First Party
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    HeartbeatThread,
    LMCacheMPWorkerAdapter,
    LoadStoreOp,
)


def _make_worker_adapter(
    healthy: bool = True,
    is_kv_writer: bool = True,
    lazy_offload: bool = True,
) -> LMCacheMPWorkerAdapter:
    """Build an adapter with only the attributes the tested paths read."""
    adapter = LMCacheMPWorkerAdapter.__new__(LMCacheMPWorkerAdapter)
    adapter.lazy_offload = lazy_offload
    adapter.dispatcher = None
    adapter._health_event = threading.Event()
    if healthy:
        adapter._health_event.set()
    # A non-None sentinel makes _ensure_heartbeat_started a no-op.
    adapter._heartbeat = cast(HeartbeatThread, object())
    adapter.parallel_strategy = SimpleNamespace(  # type: ignore[assignment]
        is_kv_writer=is_kv_writer
    )
    adapter.store_futures = {}
    adapter.retrieve_futures = {}
    adapter.store_events = {}
    adapter.retrieve_events = {}
    adapter._dropped_retrieves = set()
    adapter.error_block_ids = set()
    adapter._completed_store_requests = {}
    return adapter


def _make_op() -> LoadStoreOp:
    return LoadStoreOp(token_ids=list(range(32)), block_ids=[[1, 2]], start=0, end=32)


def _submit_store(adapter: LMCacheMPWorkerAdapter, request_id: str = "req") -> None:
    adapter.submit_store_request(request_id, _make_op(), event=None)  # type: ignore[arg-type]


####
# Receipt completeness: submit-time drops must still produce receipts
####


def test_non_writer_rank_reports_completion_at_submit() -> None:
    """MLA TP>1: a non-writer rank never creates a store future, so it must
    report completion immediately -- the scheduler counts one receipt per
    rank of the whole world before unpinning."""
    adapter = _make_worker_adapter(is_kv_writer=False)

    _submit_store(adapter)

    assert adapter.store_futures == {}
    assert adapter.get_completed_store_requests() == {"req": 1}
    # Exactly once: the receipt is not re-reported on later calls.
    assert adapter.get_completed_store_requests() is None


def test_unhealthy_submit_reports_completion_instead_of_silent_drop() -> None:
    """A store dropped at submit time while the server is unhealthy will
    never get a future; without an immediate receipt the pinned blocks and
    the session leak forever."""
    adapter = _make_worker_adapter(healthy=False)

    _submit_store(adapter)

    assert adapter.store_futures == {}
    assert adapter.get_completed_store_requests() == {"req": 1}
    assert adapter.get_completed_store_requests() is None


def test_non_lazy_submit_drops_do_not_accumulate_receipts() -> None:
    """Outside lazy offload nothing drains the receipt dict; submit-time
    drops must not grow it."""
    adapter = _make_worker_adapter(is_kv_writer=False, lazy_offload=False)

    _submit_store(adapter)

    assert adapter.get_completed_store_requests() is None
