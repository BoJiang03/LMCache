# SPDX-License-Identifier: Apache-2.0
"""A client's retrieve runs before its queued stores, over both transports."""

# Standard
from typing import Any, Literal
from unittest.mock import MagicMock
import threading
import time

# Third Party
import pytest
import zmq

# First Party
from lmcache.v1.multiprocess.affinity_pool import AffinityOrder
from lmcache.v1.multiprocess.config import MPServerConfig
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from lmcache.v1.multiprocess.request_handler import (
    HandlerType,
    get_request_handler_options,
    request_handler,
)
from lmcache.v1.multiprocess.transport.factory import RequestClientFactory
from lmcache.v1.multiprocess.transport.server_factory import create_request_server

# Long enough for the client's requests to reach the server's affinity queue.
_ENQUEUE_SETTLE_S = 0.5


def _key(request_id: str) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        "testmodel", 1, 0, [1] * 256, start=0, end=256, request_id=request_id
    )


class _TransferModule:
    """Store/retrieve handlers that record their run order.

    The first store holds the client's affinity thread until ``gate`` is set,
    so the requests sent after it queue up behind it.
    """

    def __init__(self) -> None:
        self.ctx = MagicMock(name="ctx")
        self.gate = threading.Event()
        self.started = threading.Event()
        self.ran: list[str] = []

    @property
    def context(self) -> Any:
        return self.ctx

    def report_status(self) -> dict:
        return {}

    def close(self) -> None:
        self.gate.set()

    @request_handler(HandlerType.BLOCKING, requires_client_affinity=True)
    def store(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
    ) -> tuple[bytes, bool]:
        if not self.ran:
            self.started.set()
            assert self.gate.wait(timeout=5)
        self.ran.append(key.request_id)
        return b"", True

    @request_handler(
        HandlerType.BLOCKING,
        requires_client_affinity=True,
        affinity_order=AffinityOrder.AHEAD,
    )
    def retrieve(
        self,
        key: IPCCacheServerKey,
        instance_id: int,
        gpu_block_ids: list[list[int]],
        event_ipc_handle: bytes,
        skip_first_n_tokens: int = 0,
    ) -> tuple[bytes, bool]:
        self.ran.append(key.request_id)
        return b"", True


def test_lmcache_driven_retrieve_runs_ahead_of_stores() -> None:
    retrieve = get_request_handler_options(LMCacheDrivenTransferModule.retrieve)
    store = get_request_handler_options(LMCacheDrivenTransferModule.store)

    assert retrieve is not None and store is not None
    assert retrieve.requires_client_affinity and store.requires_client_affinity
    assert retrieve.affinity_order is AffinityOrder.AHEAD
    assert store.affinity_order is AffinityOrder.FIFO


def test_affinity_order_requires_client_affinity() -> None:
    with pytest.raises(ValueError, match="requires client affinity"):
        request_handler(HandlerType.BLOCKING, affinity_order=AffinityOrder.AHEAD)


@pytest.mark.parametrize("transport", ["zmq", "grpc"])
def test_retrieve_runs_before_queued_stores(
    transport: Literal["zmq", "grpc"],
) -> None:
    """A retrieve sent after queued stores runs right after the running one."""
    module = _TransferModule()
    config = MPServerConfig(
        transport=transport,
        host="127.0.0.1",
        port=0,
        max_cpu_workers=1,
        max_gpu_workers=2,
        grpc_server_workers=8,
    )
    server: Any = create_request_server([module], config)
    if transport == "grpc":
        url = f"grpc://127.0.0.1:{server.bound_port}"
    else:
        url = server.socket.getsockopt_string(zmq.LAST_ENDPOINT)
    server.start()
    client = RequestClientFactory.create(url, context=zmq.Context.instance())
    try:
        running = client.store(_key("store-0"), 0, [[0]], b"")
        assert module.started.wait(timeout=5)
        queued = [client.store(_key(f"store-{i}"), 0, [[0]], b"") for i in (1, 2)]
        retrieve = client.retrieve(_key("retrieve"), 0, [[0]], b"", 0)
        time.sleep(_ENQUEUE_SETTLE_S)
        module.gate.set()

        for future in (running, *queued, retrieve):
            assert future.result(timeout=5) == (b"", True)
        assert module.ran[:2] == ["store-0", "retrieve"]
        assert sorted(module.ran[2:]) == ["store-1", "store-2"]
    finally:
        module.close()
        client.close()
        server.close()
