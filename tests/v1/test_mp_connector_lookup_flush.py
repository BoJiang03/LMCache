# SPDX-License-Identifier: Apache-2.0
"""The MP connector defers a lookup while lazy offload flushes its prefix.

The lazy-offload manager's answer is scripted; network I/O is stubbed.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="MP connector imports vLLM at module top")

# Third Party
from vllm.config import KVTransferConfig, VllmConfig  # noqa: E402
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorRole,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.request import Request, RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as connector_mod  # noqa: E402
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
)

pytestmark = pytest.mark.no_shared_allocator

BLOCK_SIZE = 4
NUM_TOKENS = 3 * BLOCK_SIZE


def _config() -> VllmConfig:
    transfer_config = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_both",
        kv_connector_extra_config={
            "lmcache.mp.lazy_offload": True,
            "lmcache.mp.lazy_offload_policy": "FIFO",
        },
    )
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(
                model="test-model", use_mla=False, max_model_len=4096
            ),
            parallel_config=SimpleNamespace(
                world_size=1, rank=0, tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            cache_config=SimpleNamespace(
                block_size=BLOCK_SIZE,
                enable_prefix_caching=True,
                mamba_cache_mode="none",
            ),
            scheduler_config=SimpleNamespace(
                disable_hybrid_kv_cache_manager=False,
                max_num_batched_tokens=NUM_TOKENS,
                long_prefill_token_threshold=0,
            ),
            kv_transfer_config=transfer_config,
        ),
    )


def _kv_cache_config() -> KVCacheConfig:
    attention = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[KVCacheGroupSpec(["attn"], attention)],
    )


def _request() -> Request:
    return cast(
        Request,
        SimpleNamespace(
            request_id="request",
            cache_salt="",
            all_token_ids=list(range(NUM_TOKENS)),
            num_prompt_tokens=NUM_TOKENS,
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            kv_transfer_params=None,
            block_hashes=[bytes([i + 1]) for i in range(NUM_TOKENS // BLOCK_SIZE)],
        ),
    )


@pytest.fixture
def scheduler(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Stub the LMCache server transport; the scheduler logic stays real."""
    scheduler = MagicMock(lmcache_tokens_per_chunk=BLOCK_SIZE)
    monkeypatch.setattr(
        connector_mod, "LMCacheMPSchedulerAdapter", lambda **kw: scheduler
    )
    monkeypatch.setattr(connector_mod, "print_banner_once", MagicMock())
    monkeypatch.setattr(adapter_mod.RequestClientFactory, "create", MagicMock())
    monkeypatch.setattr(
        adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: BLOCK_SIZE
    )
    return scheduler


@pytest.fixture
def connector(scheduler: MagicMock) -> Iterator[LMCacheMPConnector]:
    """A lazy-offload scheduler connector over one attention group."""
    connector = LMCacheMPConnector(
        _config(), KVConnectorRole.SCHEDULER, _kv_cache_config()
    )
    blocks = [
        SimpleNamespace(block_id=i, block_hash=None, is_null=i == 0, ref_cnt=1)
        for i in range(16)
    ]
    connector.bind_kv_cache_manager(
        cast(
            Any,
            SimpleNamespace(
                coordinator=MagicMock(),
                block_pool=MagicMock(null_block=blocks[0], blocks=blocks),
            ),
        )
    )
    try:
        yield connector
    finally:
        connector.shutdown()


def test_lookup_is_submitted_once_the_flushed_stores_settle(
    monkeypatch: pytest.MonkeyPatch,
    scheduler: MagicMock,
    connector: LMCacheMPConnector,
) -> None:
    """While lazy offload stores the request's deferred prefix, the hit is
    reported as not yet known and no lookup goes out; then the lookup is
    submitted and its hit loaded."""
    waits = iter([True, True, False])
    asked: list[tuple[str, int]] = []

    def flush_for_lookup(_manager: object, request_id: str, num_computed: int) -> bool:
        asked.append((request_id, num_computed))
        return next(waits)

    monkeypatch.setattr(
        connector_mod.LazyOffloadManager, "flush_for_lookup", flush_for_lookup
    )
    scheduler.check_lookup_result.return_value = 2 * BLOCK_SIZE
    request = _request()

    assert connector.get_num_new_matched_tokens(request, 0) == (None, True)
    assert connector.get_num_new_matched_tokens(request, 0) == (None, True)
    scheduler.maybe_submit_lookup_request.assert_not_called()
    assert connector.get_num_new_matched_tokens(request, 0) == (2 * BLOCK_SIZE, True)

    scheduler.maybe_submit_lookup_request.assert_called_once()
    assert asked == [("request", 0)] * 3
