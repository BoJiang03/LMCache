# SPDX-License-Identifier: Apache-2.0
"""Uncached recurrent states are never stored, through the public MP connector.

An uncached state is a block of a recurrent (Mamba / GDN) group that is not
vLLM's null block but has no prefix-cache hash. The scheduler connector nulls
it in the store operation, so the server's all-null-chunk skip drops it.
Network and GPU I/O are stubbed.
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
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
    SlidingWindowSpec,
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
NULL_BLOCK_ID = 0
ATTENTION, SLIDING, RECURRENT = 0, 1, 2


def _kv_cache_config() -> KVCacheConfig:
    attention = FullAttentionSpec(
        block_size=BLOCK_SIZE, num_kv_heads=1, head_size=8, dtype=torch.float16
    )
    sliding = SlidingWindowSpec(
        block_size=BLOCK_SIZE,
        num_kv_heads=1,
        head_size=8,
        dtype=torch.float16,
        sliding_window=BLOCK_SIZE,
    )
    recurrent = MambaSpec(
        block_size=BLOCK_SIZE,
        shapes=((8,),),
        dtypes=(torch.float16,),
        mamba_cache_mode="align",
    )
    return KVCacheConfig(
        num_blocks=16,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec(["attn"], attention),
            KVCacheGroupSpec(["swa"], sliding),
            KVCacheGroupSpec(["gdn"], recurrent),
        ],
    )


def _config(
    enable_prefix_caching: bool, extra_config: dict[str, object] | None = None
) -> VllmConfig:
    transfer_config = KVTransferConfig(
        kv_connector="LMCacheMPConnector",
        kv_role="kv_both",
        kv_connector_extra_config=extra_config or {},
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
                enable_prefix_caching=enable_prefix_caching,
                mamba_cache_mode="align",
            ),
            scheduler_config=SimpleNamespace(
                disable_hybrid_kv_cache_manager=False,
                max_num_batched_tokens=NUM_TOKENS,
                long_prefill_token_threshold=0,
            ),
            kv_transfer_config=transfer_config,
        ),
    )


def _request(request_id: str = "request") -> Request:
    return cast(
        Request,
        SimpleNamespace(
            request_id=request_id,
            cache_salt="",
            all_token_ids=list(range(NUM_TOKENS)),
            num_prompt_tokens=NUM_TOKENS,
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            kv_transfer_params=None,
            block_hashes=[
                f"{request_id}-{i}".encode() for i in range(NUM_TOKENS // BLOCK_SIZE)
            ],
        ),
    )


def _schedule(
    request_id: str, num_tokens: int, block_ids: tuple[list[int], ...]
) -> SchedulerOutput:
    """A step that schedules a new request's first ``num_tokens`` tokens."""
    return cast(
        SchedulerOutput,
        SimpleNamespace(
            scheduled_new_reqs=[
                SimpleNamespace(
                    req_id=request_id, block_ids=block_ids, num_computed_tokens=0
                )
            ],
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[],
                new_block_ids=[],
                resumed_req_ids=set(),
                num_computed_tokens=[],
            ),
            num_scheduled_tokens={request_id: num_tokens},
            total_num_scheduled_tokens=num_tokens,
            finished_req_ids=set(),
            preempted_req_ids=None,
        ),
    )


def _pool(unhashed: set[int]) -> MagicMock:
    """A block pool whose blocks are hashed except the null block and ``unhashed``."""
    blocks = [
        SimpleNamespace(
            block_id=block_id,
            block_hash=None
            if block_id == NULL_BLOCK_ID or block_id in unhashed
            else bytes([block_id]),
            is_null=block_id == NULL_BLOCK_ID,
            ref_cnt=1,
        )
        for block_id in range(16)
    ]
    return MagicMock(
        null_block=blocks[NULL_BLOCK_ID], blocks=blocks, hash_block_size=BLOCK_SIZE
    )


@pytest.fixture
def mock_io(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stub the LMCache server transport; the scheduler logic stays real."""
    scheduler = MagicMock(lmcache_tokens_per_chunk=BLOCK_SIZE)
    scheduler.check_lookup_result.return_value = 0
    monkeypatch.setattr(
        connector_mod, "LMCacheMPSchedulerAdapter", lambda **kw: scheduler
    )
    monkeypatch.setattr(connector_mod, "print_banner_once", MagicMock())
    monkeypatch.setattr(adapter_mod.RequestClientFactory, "create", MagicMock())
    monkeypatch.setattr(
        adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: BLOCK_SIZE
    )
    return SimpleNamespace(scheduler=scheduler)


def _connector(enable_prefix_caching: bool = True) -> LMCacheMPConnector:
    return LMCacheMPConnector(
        _config(enable_prefix_caching),
        KVConnectorRole.SCHEDULER,
        _kv_cache_config(),
    )


def _stored_block_ids(
    connector: LMCacheMPConnector, block_ids: tuple[list[int], ...]
) -> list[list[int]]:
    """Schedule a whole prompt in one step and return the stored block ids."""
    request = _request()
    assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
    connector.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: block_ids), 0
    )
    metadata = connector.build_connector_meta(
        _schedule(request.request_id, NUM_TOKENS, block_ids)
    )
    [store] = metadata.requests
    assert (store.direction, store.op.start, store.op.end) == ("STORE", 0, NUM_TOKENS)
    return store.op.block_ids


@pytest.fixture
def connector(mock_io: SimpleNamespace) -> Iterator[LMCacheMPConnector]:
    """An eager scheduler connector over attention, sliding and GDN groups."""
    connector = _connector()
    try:
        yield connector
    finally:
        connector.shutdown()


def test_retained_recurrent_state_is_stored(connector: LMCacheMPConnector) -> None:
    """A hashed recurrent block (a state vLLM keeps) is stored as-is."""
    connector.bind_gpu_block_pool(_pool(unhashed=set()))
    block_ids = ([1, 2, 3], [4, 5, 6], [NULL_BLOCK_ID, NULL_BLOCK_ID, 9])
    assert _stored_block_ids(connector, block_ids)[RECURRENT] == [
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        9,
    ]


def test_uncached_recurrent_state_is_skipped(connector: LMCacheMPConnector) -> None:
    """An unhashed, non-null recurrent block is stored as the null block."""
    connector.bind_gpu_block_pool(_pool(unhashed={8}))
    block_ids = ([1, 2, 3], [4, 5, 6], [NULL_BLOCK_ID, 8, 9])
    assert _stored_block_ids(connector, block_ids)[RECURRENT] == [
        NULL_BLOCK_ID,
        NULL_BLOCK_ID,
        9,
    ]


def test_attention_and_sliding_blocks_are_never_dropped(
    connector: LMCacheMPConnector,
) -> None:
    """Unhashed attention and sliding-window blocks are stored unchanged."""
    connector.bind_gpu_block_pool(_pool(unhashed={1, 2, 3, 4, 5, 6, 8}))
    block_ids = ([1, 2, 3], [4, 5, 6], [7, 8, 9])
    assert _stored_block_ids(connector, block_ids) == [
        [1, 2, 3],
        [4, 5, 6],
        [7, NULL_BLOCK_ID, 9],
    ]


def test_dense_retention_skips_nothing(connector: LMCacheMPConnector) -> None:
    """With every chunk's state hashed (dense retention) nothing is dropped."""
    connector.bind_gpu_block_pool(_pool(unhashed=set()))
    block_ids = ([1, 2, 3], [4, 5, 6], [7, 8, 9])
    assert _stored_block_ids(connector, block_ids) == [
        [1, 2, 3],
        [4, 5, 6],
        [7, 8, 9],
    ]


def test_without_prefix_caching_recurrent_states_are_kept(
    mock_io: SimpleNamespace,
) -> None:
    """Without prefix caching no block is hashed, so a missing hash drops nothing."""
    connector = _connector(enable_prefix_caching=False)
    try:
        connector.bind_gpu_block_pool(_pool(unhashed=set(range(16))))
        block_ids = ([1, 2, 3], [4, 5, 6], [7, 8, 9])
        assert _stored_block_ids(connector, block_ids)[RECURRENT] == [7, 8, 9]
    finally:
        connector.shutdown()


def test_lazy_admission_never_sees_an_uncached_state(
    mock_io: SimpleNamespace,
) -> None:
    """Lazy offload buffers the nulled op, so the uncached state cannot void it.

    Its unhashed block would otherwise void the whole buffered store.
    """
    connector = LMCacheMPConnector(
        _config(
            True,
            {
                "lmcache.mp.lazy_offload": True,
                "lmcache.mp.lazy_offload_policy": "FIFO",
                "lmcache.mp.lazy_offload_threshold": 1,
                "lmcache.mp.lazy_offload_select_count": 1,
            },
        ),
        KVConnectorRole.SCHEDULER,
        _kv_cache_config(),
    )
    try:
        coordinator = MagicMock()
        coordinator.find_longest_cache_hit.return_value = ([], NUM_TOKENS, 0)
        connector.bind_kv_cache_manager(
            cast(
                Any,
                SimpleNamespace(
                    coordinator=coordinator, block_pool=_pool(unhashed={8})
                ),
            )
        )
        request = _request()
        block_ids = ([1, 2, 3], [4, 5, 6], [NULL_BLOCK_ID, 8, 9])
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        connector.update_state_after_alloc(
            request, MagicMock(get_block_ids=lambda: block_ids), 0
        )
        metadata = connector.build_connector_meta(
            _schedule(request.request_id, NUM_TOKENS, block_ids)
        )
        assert metadata.requests == []
        connector.request_finished(request, block_ids[0])
        # A later token-producing step drains the finished request.
        trigger = _request("trigger")
        trigger_blocks = ([13], [14], [15])
        assert connector.get_num_new_matched_tokens(trigger, 0) == (0, False)
        connector.update_state_after_alloc(
            trigger, MagicMock(get_block_ids=lambda: trigger_blocks), 0
        )
        metadata = connector.build_connector_meta(
            _schedule(trigger.request_id, 1, trigger_blocks)
        )
        [store] = [m for m in metadata.requests if m.request_id == "request"]
        assert (store.op.start, store.op.end) == (0, NUM_TOKENS)
        assert store.op.block_ids[RECURRENT] == [NULL_BLOCK_ID, NULL_BLOCK_ID, 9]
    finally:
        connector.shutdown()
