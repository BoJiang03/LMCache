# SPDX-License-Identifier: Apache-2.0
"""The Q store must key its content the same way the KV store does.

A STORE ``LoadStoreOp`` carries only ``[start, end)`` with
``token_offset == start``; the server's session already holds the rolling
chunk hashes of everything before it.  The Q store path reuses that same op,
so it has to forward ``token_offset`` when it builds its key.  Dropping it
tells the server the tokens are the request's whole prefix, and the server
then hashes ``token_ids[start:end]`` of a list that only holds the delta --
the wrong tokens, or a ``SessionTokenGapError``.
"""

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest

# First Party
from lmcache.sdk.qringbuffer import QRingBufferAdapter
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey

CHUNK = 256


@pytest.fixture
def q_adapter() -> tuple[QRingBufferAdapter, MagicMock]:
    """A Q ring adapter whose worker adapter and ring are stubbed.

    Returns:
        The adapter under test and the worker adapter it delegates to.
    """
    worker = MagicMock(name="worker_adapter")
    worker.is_healthy = True
    worker.instance_id = 7
    worker.create_key.return_value = IPCCacheServerKey(
        model_name="kv-model",
        world_size=1,
        worker_id=0,
        token_ids=(),
        start=0,
        end=0,
        request_id="req-1",
        num_kv_readers=1,
    )
    adapter = QRingBufferAdapter.__new__(QRingBufferAdapter)
    adapter._adapter = worker
    adapter.q_model_name = "q-model"
    adapter.q_ring = MagicMock(name="q_ring")
    adapter.q_ring.tensors = []
    adapter.q_store_futures = {}
    adapter.q_store_events = {}
    adapter._q_store_seq = 0
    return adapter, worker


def _store_op(start: int, end: int) -> SimpleNamespace:
    """A STORE op shaped as ``GetStoreMetadata`` builds it: delta + offset.

    Args:
        start: Absolute start token index of the stored range.
        end: Absolute end token index of the stored range.

    Returns:
        An op carrying only ``[start, end)`` with ``token_offset == start``.
    """
    return SimpleNamespace(
        token_ids=list(range(start, end)),
        block_ids=[[0, 1]],
        start=start,
        end=end,
        skip_first_n_tokens=0,
        token_offset=start,
    )


def test_q_store_key_forwards_the_op_token_offset(q_adapter) -> None:
    """A delta op's offset reaches the key, so the server hashes its range."""
    adapter, worker = q_adapter
    op = _store_op(CHUNK, 2 * CHUNK)

    adapter.submit_q_store_request(
        "req-1", op, ring_block_ids=[0, 1], event=MagicMock()
    )

    worker.create_key.assert_called_once()
    kwargs = worker.create_key.call_args.kwargs
    assert kwargs["token_offset"] == CHUNK
    args = worker.create_key.call_args.args
    assert args[1] == CHUNK
    assert args[2] == 2 * CHUNK
    assert len(args[0]) == op.end - op.start


def test_q_store_key_offset_matches_a_whole_prefix_op(q_adapter) -> None:
    """A whole-prefix op (offset 0) is forwarded unchanged as 0."""
    adapter, worker = q_adapter
    op = _store_op(0, CHUNK)

    adapter.submit_q_store_request("req-1", op, ring_block_ids=[0], event=MagicMock())

    assert worker.create_key.call_args.kwargs["token_offset"] == 0
