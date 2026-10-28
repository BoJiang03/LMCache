# SPDX-License-Identifier: Apache-2.0
"""Role contracts through public MP connector APIs; network/GPU I/O is stubbed."""

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
from vllm.distributed.kv_events import (  # noqa: E402
    BlockRemoved,
    BlockStored,
    KVCacheEvent,
)
from vllm.distributed.kv_transfer.kv_connector.factory import (  # noqa: E402
    KVConnectorFactory,
)
from vllm.distributed.kv_transfer.kv_connector.v1.base import (  # noqa: E402
    KVConnectorBase_V1,
    KVConnectorRole,
)
from vllm.distributed.kv_transfer.kv_connector.v1.multi_connector import (  # noqa: E402
    MultiConnector,
)
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)
from vllm.v1.core.sched.output import SchedulerOutput  # noqa: E402
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
)
from vllm.v1.outputs import KVConnectorOutput  # noqa: E402
from vllm.v1.request import Request, RequestStatus  # noqa: E402

# First Party
from lmcache.integration.vllm import lmcache_mp_connector as connector_mod  # noqa: E402
from lmcache.integration.vllm import vllm_multi_process_adapter as adapter_mod
from lmcache.integration.vllm.lmcache_mp_connector import (  # noqa: E402
    LMCacheMPConnector,
    LMCacheMPKVEvents,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPConnectorMetadata,
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import LoadStoreOp
from lmcache.utils import CacheStoreEvent  # noqa: E402
from lmcache.v1.multiprocess.custom_types import (  # noqa: E402
    ENGINE_HELD_BLOCKS_HINT,
    ENGINE_HELD_TOKENS_HINT,
    EXCLUSIVE_RETRIEVE,
    EXCLUSIVE_RETRIEVE_REQUEST_CONFIG,
    SHARED_HELD_TOKENS_HINT,
)
from lmcache.v1.multiprocess.token_codec import pack_token_ids

pytestmark = pytest.mark.no_shared_allocator


@pytest.mark.parametrize(
    "batches, expected",
    [
        ([[0, 1], [1, 2], [0]], [0, 1, 2]),
        ([[0], []], [0]),
        ([[], [0]], [0]),
        ([], []),
        ([[], []], []),
        ([[0, 3, 0], []], [0, 3, 0]),
        ([[0, 3, 0], [0, 3, 0]], [0, 3, 0]),
    ],
)
def test_kv_event_aggregation_preserves_worker_transitions(
    batches: list[list[int]], expected: list[int]
) -> None:
    """Keep skewed stores and store/remove/store order across worker merges."""
    events: list[KVCacheEvent] = [
        BlockStored(
            block_hashes=[index],
            parent_block_hash=None,
            token_ids=[index],
            block_size=1,
            lora_id=None,
            medium="CPU",
            lora_name=None,
        )
        for index in range(3)
    ]
    events.append(BlockRemoved([0], "CPU"))
    aggregate = LMCacheMPKVEvents(num_workers=1)
    for index, batch in enumerate(batches):
        worker = LMCacheMPKVEvents(num_workers=1)
        worker.add_events([events[index] for index in batch])
        aggregate.merge(worker)
        if index:
            aggregate.increment_workers(worker.get_number_of_workers())
    assert aggregate.get_number_of_workers() == max(1, len(batches))
    assert aggregate.aggregate().get_all_events() == [events[i] for i in expected]
    assert aggregate.get_number_of_workers() == 1
    aggregate.clear_events()
    assert aggregate.get_all_events() == []


def _config(
    transfer_config: KVTransferConfig,
    max_model_len: int = 4096,
    max_num_batched_tokens: int = 16,
) -> VllmConfig:
    return cast(
        VllmConfig,
        SimpleNamespace(
            model_config=SimpleNamespace(
                model="test-model", use_mla=False, max_model_len=max_model_len
            ),
            parallel_config=SimpleNamespace(
                world_size=1, rank=0, tensor_parallel_size=1, pipeline_parallel_size=1
            ),
            cache_config=SimpleNamespace(block_size=4, enable_prefix_caching=True),
            scheduler_config=SimpleNamespace(
                disable_hybrid_kv_cache_manager=True,
                max_num_batched_tokens=max_num_batched_tokens,
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
            all_token_ids=list(range(12)),
            num_prompt_tokens=12,
            status=RequestStatus.WAITING,
            num_computed_tokens=0,
            kv_transfer_params=None,
            block_hashes=[f"{request_id}-{i}".encode() for i in range(3)],
        ),
    )


def _bind_serving_whole_prompts(connector: LMCacheMPConnector, pool: Any) -> None:
    """Bind a KV cache manager over ``pool`` whose coordinator serves every
    request's 12-token prompt from the GPU prefix cache."""
    coordinator = MagicMock()
    coordinator.find_longest_cache_hit.return_value = ([], 12, 0)
    connector.bind_kv_cache_manager(
        cast(Any, SimpleNamespace(coordinator=coordinator, block_pool=pool))
    )


def _schedule(
    request_id: str = "request",
    num_tokens: int = 0,
    block_ids: tuple[list[int], ...] = ([],),
    *,
    new: bool = False,
    num_computed_tokens: int = 0,
) -> SchedulerOutput:
    # vLLM hands one block-id list per KV-cache group; this config has one.
    return cast(
        SchedulerOutput,
        SimpleNamespace(
            scheduled_new_reqs=(
                [
                    SimpleNamespace(
                        req_id=request_id,
                        block_ids=block_ids,
                        num_computed_tokens=num_computed_tokens,
                    )
                ]
                if new and num_tokens
                else []
            ),
            scheduled_cached_reqs=SimpleNamespace(
                req_ids=[request_id] if not new and num_tokens else [],
                new_block_ids=[block_ids] if not new and num_tokens else [],
                resumed_req_ids=set(),
                num_computed_tokens=(
                    [num_computed_tokens] if not new and num_tokens else []
                ),
            ),
            num_scheduled_tokens={request_id: num_tokens} if num_tokens else {},
            total_num_scheduled_tokens=num_tokens,
            finished_req_ids=set(),
            preempted_req_ids=None,
        ),
    )


@pytest.fixture(params=["kv_consumer", "kv_producer", "kv_both"])
def kv_role(request: pytest.FixtureRequest) -> str:
    """Exercise all roles supported by KVTransferConfig."""
    return request.param


@pytest.fixture(params=[False, True])
def lazy_offload(request: pytest.FixtureRequest) -> bool:
    """Exercise eager storage and actual FIFO lazy draining."""
    return request.param


@pytest.fixture
def mock_io(monkeypatch: pytest.MonkeyPatch) -> SimpleNamespace:
    """Stub transport, telemetry and GPU resources, keeping worker logic real."""
    # A current server: it advertises exclusive retrieve.
    scheduler = MagicMock(lmcache_tokens_per_chunk=4, experimental={EXCLUSIVE_RETRIEVE})
    scheduler.check_lookup_result.return_value = 0
    transfer = MagicMock()
    transfer.create_recorded_event.return_value = None
    for submit in (transfer.submit_store, transfer.submit_retrieve):
        submit.return_value.query.return_value = True
        submit.return_value.result.return_value = True
    telemetry = MagicMock()
    pool = MagicMock(
        blocks=[
            # Owned by the running request, as a store's blocks are.
            SimpleNamespace(block_id=i, block_hash=bytes([i]), is_null=False, ref_cnt=1)
            for i in range(8)
        ],
        # One request block hash per 4-token block, the chunk size.
        hash_block_size=4,
    )
    monkeypatch.setattr(
        connector_mod, "LMCacheMPSchedulerAdapter", lambda **kw: scheduler
    )
    monkeypatch.setattr(connector_mod, "print_banner_once", MagicMock())
    monkeypatch.setattr(connector_mod, "vllm_layout_hints", lambda config: {})
    monkeypatch.setattr(adapter_mod.RequestClientFactory, "create", MagicMock())
    monkeypatch.setattr(adapter_mod, "get_lmcache_chunk_size", lambda *a, **kw: 4)
    monkeypatch.setattr(adapter_mod, "get_experimental", lambda *a, **kw: set())
    monkeypatch.setattr(adapter_mod, "HeartbeatThread", MagicMock())
    monkeypatch.setattr(adapter_mod, "set_ipc_policy", MagicMock())
    monkeypatch.setattr(
        adapter_mod, "create_transfer_context", lambda *a, **kw: transfer
    )
    monkeypatch.setattr(
        adapter_mod.RequestTelemetryFactory, "create", lambda **kw: telemetry
    )
    return SimpleNamespace(
        scheduler=scheduler,
        transfer=transfer,
        telemetry=telemetry,
        pool=pool,
        kv_caches={"layer": torch.empty(2, 8, 4, 1, 8)},
    )


@pytest.fixture
def connectors(
    kv_role: str, lazy_offload: bool, mock_io: SimpleNamespace
) -> Iterator[tuple[LMCacheMPConnector, LMCacheMPConnector]]:
    """Construct real scheduler/worker connectors and a bound lazy manager."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role=kv_role,
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": lazy_offload,
                "lmcache.mp.lazy_offload_policy": "FIFO",
                "lmcache.mp.lazy_offload_threshold": 1,
                "lmcache.mp.lazy_offload_select_count": 1,
            },
        )
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    worker = LMCacheMPConnector(config, KVConnectorRole.WORKER)
    _bind_serving_whole_prompts(scheduler, mock_io.pool)
    worker.register_kv_caches(mock_io.kv_caches)
    try:
        yield scheduler, worker
    finally:
        worker.shutdown()
        scheduler.shutdown()


@pytest.mark.parametrize("save_decode_cache", [None, False, True])
@pytest.mark.parametrize("lazy", [False, True])
def test_decode_store_policy_through_scheduler(
    mock_io: SimpleNamespace, save_decode_cache: bool | None, lazy: bool
) -> None:
    """The configured write policy applies before immediate or lazy storage."""
    options = {"lmcache.mp.lazy_offload": lazy}
    if save_decode_cache is not None:
        options["lmcache.mp.save_decode_cache"] = save_decode_cache
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config=options,
        )
    )
    connector = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    _bind_serving_whole_prompts(connector, mock_io.pool)
    request = _request()
    request.num_prompt_tokens = 6
    try:
        assert connector.get_num_new_matched_tokens(request, 0) == (0, False)
        connector.update_state_after_alloc(
            request, MagicMock(get_block_ids=lambda: ([0, 1, 2],)), 0
        )
        connector.build_connector_meta(_schedule(num_tokens=6, new=True))
        # Decode finishes the partial prompt chunk and one all-output chunk.
        metadata = connector.build_connector_meta(_schedule(num_tokens=6))
        tracker = connector.request_trackers[request.request_id]
        assert tracker.num_stored_tokens == (12 if save_decode_cache else 4)
        if not lazy:
            if save_decode_cache:
                assert [(m.op.start, m.op.end) for m in metadata.requests] == [(4, 12)]
            else:
                assert metadata.requests == []
    finally:
        connector.shutdown()


@pytest.mark.parametrize("bound", [False, True])
def test_lazy_offload_turns_off_without_a_bound_kv_cache_manager(
    mock_io: SimpleNamespace, bound: bool
) -> None:
    """A vLLM that binds only the block pool, never the KV cache manager,
    runs with lazy offload off: stores go out eagerly, the finished request
    waits for them, and the worker reports them to vLLM. Bound, lazy offload
    stays on and the store is buffered."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": True,
                "lmcache.mp.lazy_offload_policy": "FIFO",
            },
        )
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    worker = LMCacheMPConnector(config, KVConnectorRole.WORKER)
    if bound:
        _bind_serving_whole_prompts(scheduler, mock_io.pool)
    else:
        scheduler.bind_gpu_block_pool(mock_io.pool)
    worker.register_kv_caches(mock_io.kv_caches)
    request = _request()
    try:
        assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
        assert scheduler.lazy_offload is bound
        scheduler.update_state_after_alloc(
            request, MagicMock(get_block_ids=lambda: ([1],)), 0
        )
        metadata = scheduler.build_connector_meta(
            _schedule(num_tokens=4, block_ids=([1],), new=True)
        )
        stores = [(m.direction, m.op.start, m.op.end) for m in metadata.requests]
        assert stores == ([] if bound else [("STORE", 0, 4)])
        worker.bind_connector_metadata(metadata)
        worker.wait_for_save()

        assert scheduler.request_finished(request, [1]) == (not bound, None)
        sending, _ = worker.get_finished({"request"})
        assert (sending or set()) == (set() if bound else {"request"})
    finally:
        worker.shutdown()
        scheduler.shutdown()


@pytest.mark.parametrize("value", ["false", "true", 0, 1, None])
def test_decode_store_policy_rejects_non_boolean(
    mock_io: SimpleNamespace, value: object
) -> None:
    """A string 'false' must never silently enable decode storage."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"lmcache.mp.save_decode_cache": value},
        )
    )
    with pytest.raises(ValueError, match="save_decode_cache must be a boolean"):
        LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)


def test_chunked_prefill_stores_and_completion(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],
    kv_role: str,
    lazy_offload: bool,
    mock_io: SimpleNamespace,
) -> None:
    """Only writers store new/cached chunks and wait for real save completions."""
    scheduler, worker = connectors
    can_store = kv_role != "kv_consumer"
    request = _request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: ([1],)), 0
    )
    mock_io.transfer.submit_store.return_value.query.return_value = False
    for step in range(3):
        metadata = scheduler.build_connector_meta(
            _schedule(num_tokens=4, block_ids=([step + 1],), new=step == 0)
        )
        assert isinstance(metadata, LMCacheMPConnectorMetadata)
        tracker = scheduler.request_trackers[request.request_id]
        assert tracker.num_scheduled_tokens == (step + 1) * 4
        assert tracker.allocated_block_ids == {0: list(range(1, step + 2))}
        assert tracker.num_stored_tokens == ((step + 1) * 4 if can_store else 0)
        if can_store and not lazy_offload:
            assert len(metadata.requests) == 1
            meta = metadata.requests[0]
            assert (meta.direction, meta.op.start, meta.op.end, meta.op.block_ids) == (
                "STORE",
                step * 4,
                (step + 1) * 4,
                [[step + 1]],
            )
        else:
            assert metadata.requests == []
        worker.bind_connector_metadata(metadata)
        worker.wait_for_save()
        assert not any(worker.get_finished(set()))

    mock_io.scheduler.cleanup_lookup_result.reset_mock()
    assert scheduler.request_finished(request, [1, 2, 3]) == (
        can_store and not lazy_offload,
        None,
    )
    assert scheduler.request_trackers == {}
    mock_io.scheduler.cleanup_lookup_result.assert_called_once_with("request")

    if lazy_offload:
        # A later token step drains a writer's completed request, never a consumer's.
        trigger = _request("trigger")
        assert scheduler.get_num_new_matched_tokens(trigger, 0) == (0, False)
        scheduler.update_state_after_alloc(
            trigger, MagicMock(get_block_ids=lambda: ([4],)), 0
        )
        metadata = scheduler.build_connector_meta(
            _schedule("trigger", 1, ([4],), new=True)
        )
        assert len(metadata.requests) == int(can_store)
        if can_store:
            mock_io.scheduler.end_session.assert_not_called()
            meta = metadata.requests[0]
            assert (meta.request_id, meta.op.start, meta.op.end, meta.op.block_ids) == (
                "request",
                0,
                12,
                [[1, 2, 3]],
            )
        assert scheduler.has_pending_push_work() is can_store
        worker.bind_connector_metadata(metadata)
        worker.wait_for_save()

    assert not any(worker.get_finished({"request"}))
    mock_io.telemetry.on_request_store_finished.assert_not_called()
    mock_io.transfer.submit_store.return_value.query.return_value = True
    sending, receiving = worker.get_finished({"request"})
    assert (sending or set()) == (
        {"request"} if can_store and not lazy_offload else set()
    )
    assert not receiving
    scheduler.update_connector_output(
        KVConnectorOutput(kv_connector_worker_meta=worker.build_connector_worker_meta())
    )
    assert not scheduler.has_pending_push_work()
    mock_io.scheduler.end_session.assert_called_once_with("request")
    assert mock_io.transfer.submit_store.call_count == (
        (1 if lazy_offload else 3) if can_store else 0
    )
    if can_store:
        mock_io.telemetry.on_request_store_finished.assert_called_once_with(
            request_ids_set={"request"},
            model_name="test-model",
            world_size=1,
            kv_rank=0,
        )
    else:
        mock_io.telemetry.on_request_store_finished.assert_not_called()
    assert not any(worker.get_finished({"request"}))


@pytest.mark.parametrize("kv_role", ["kv_both"])
@pytest.mark.parametrize("lazy_offload", [True])
@pytest.mark.parametrize("schedules_tokens", [False, True])
def test_store_of_a_step_without_forward_is_submitted_once(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],
    schedules_tokens: bool,
) -> None:
    """A lazy drain on a step that schedules nothing still reaches LMCache.

    vLLM 0.18 to 0.30 skip ``wait_for_save`` on such steps but still poll
    ``get_finished``, which then submits the step's stores. With a forward,
    ``wait_for_save`` may come after ``get_finished`` (it waits for the draft
    model), so polling must not submit early; either way, one submission.
    """
    scheduler, worker = connectors
    request = _request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: ([1],)), 0
    )
    metadata = scheduler.build_connector_meta(
        _schedule(num_tokens=4, block_ids=([1],), new=True)
        if schedules_tokens
        else _schedule()
    )
    metadata.add_request_metadata(
        LMCacheMPRequestMetadata(
            request_id="request",
            direction="STORE",
            op=LoadStoreOp(
                token_bytes=pack_token_ids(range(4)), block_ids=[[1]], start=0, end=4
            ),
        )
    )
    worker.bind_connector_metadata(metadata)
    with pytest.MonkeyPatch.context() as patcher:
        submit = MagicMock()
        patcher.setattr(worker.worker_adapter, "batched_submit_store_requests", submit)
        worker.get_finished(set())
        assert submit.call_count == int(not schedules_tokens)
        worker.wait_for_save()
        assert submit.call_count == 1


@pytest.mark.parametrize("retrieve_success", [False, True])
def test_lookup_retrieve_and_cleanup(
    connectors: tuple[LMCacheMPConnector, LMCacheMPConnector],
    kv_role: str,
    lazy_offload: bool,
    mock_io: SimpleNamespace,
    retrieve_success: bool,
) -> None:
    """Every role retains lookup, receive completion/error reporting and stats."""
    scheduler, worker = connectors
    request = _request()
    request.kv_transfer_params = {"cached_token_stats": True}
    mock_io.scheduler.check_lookup_result.side_effect = [None, 8]
    assert scheduler.get_num_new_matched_tokens(request, 4) == (None, True)
    assert scheduler.get_num_new_matched_tokens(request, 4) == (4, True)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: ([1, 2],)), 4
    )
    metadata = scheduler.build_connector_meta(_schedule())
    assert len(metadata.requests) == 1
    meta = metadata.requests[0]
    assert (meta.direction, meta.op.start, meta.op.end, meta.op.block_ids) == (
        "RETRIEVE",
        4,
        8,
        [[2]],
    )
    worker.bind_connector_metadata(metadata)
    future = mock_io.transfer.submit_retrieve.return_value
    future.query.return_value = False
    future.result.return_value = retrieve_success
    worker.start_load_kv(MagicMock())
    with pytest.MonkeyPatch.context() as patcher:
        submit = MagicMock(wraps=worker.worker_adapter.batched_submit_store_requests)
        patcher.setattr(worker.worker_adapter, "batched_submit_store_requests", submit)
        worker.wait_for_save()
        submit.assert_not_called()
    mock_io.transfer.submit_store.assert_not_called()
    mock_io.transfer.submit_retrieve.assert_called_once()
    assert not any(worker.get_finished(set()))
    future.query.return_value = True
    sending, receiving = worker.get_finished(set())
    assert not sending
    assert receiving == {"request"}
    assert worker.get_block_ids_with_load_errors() == (
        set() if retrieve_success else {2}
    )
    assert not any(worker.get_finished(set()))
    mock_io.scheduler.cleanup_lookup_result.reset_mock()
    delay_free, params = scheduler.request_finished(request, [1, 2])
    assert delay_free is (kv_role != "kv_consumer" and not lazy_offload)
    assert params == {
        "cached_token_stats": {
            "num_vllm_cached_tokens": 4,
            "num_lmcache_cached_tokens": 8,
            "num_lmcache_extra_cached_tokens": 4,
        }
    }
    assert scheduler.request_trackers == {}
    mock_io.scheduler.cleanup_lookup_result.assert_called_once_with("request")
    mock_io.scheduler.end_session.assert_called_once_with("request")
    sending, receiving = worker.get_finished({"request"})
    assert (sending or set()) == ({"request"} if delay_free else set())
    assert not receiving


@pytest.mark.parametrize("num_groups,legacy_api", [(1, False), (2, False), (2, True)])
@pytest.mark.parametrize("retrieve_success", [False, True])
def test_receive_failure_reporting(
    mock_io: SimpleNamespace,
    lazy_offload: bool,
    monkeypatch: pytest.MonkeyPatch,
    num_groups: int,
    legacy_api: bool,
    retrieve_success: bool,
) -> None:
    """Hybrid failures use request IDs; single-group and old engines keep blocks."""
    if legacy_api:
        monkeypatch.setattr(
            connector_mod, "KVConnectorTransferResults", None, raising=False
        )
    elif not hasattr(KVConnectorBase_V1, "get_transfer_results"):
        pytest.skip("vLLM does not expose request-level transfer results")
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={"lmcache.mp.lazy_offload": lazy_offload},
        )
    )
    spec = FullAttentionSpec(
        block_size=4, num_kv_heads=1, head_size=8, dtype=torch.float32
    )
    cache_config = KVCacheConfig(
        num_blocks=8,
        kv_cache_tensors=[],
        kv_cache_groups=[
            KVCacheGroupSpec([f"layer{i}"], spec) for i in range(num_groups)
        ],
    )
    worker = LMCacheMPConnector(config, KVConnectorRole.WORKER, cache_config)
    try:
        worker.register_kv_caches(
            {f"layer{i}": mock_io.kv_caches["layer"].clone() for i in range(num_groups)}
        )
        future = mock_io.transfer.submit_retrieve.return_value
        future.query.return_value = False
        future.result.return_value = retrieve_success
        worker.worker_adapter.submit_retrieve_request(
            "request",
            LoadStoreOp(
                token_bytes=pack_token_ids([1, 2, 3, 4]),
                block_ids=[[2]] * num_groups,
                start=0,
                end=4,
            ),
            None,
        )
        assert not any(worker.get_finished(set()))
        future.query.return_value = True
        if legacy_api:
            sending, receiving = worker.get_finished(set())
        else:
            result = worker.get_transfer_results(set())
            sending, receiving = result.finished_sending, result.finished_recving
            assert result.failed_recving == (
                {"request"} if num_groups > 1 and not retrieve_success else set()
            )
        assert not sending
        assert receiving == {"request"}
        assert worker.get_block_ids_with_load_errors() == (
            {2} if not retrieve_success and (legacy_api or num_groups == 1) else set()
        )
        if legacy_api:
            assert not any(worker.get_finished(set()))
        else:
            result = worker.get_transfer_results(set())
            assert not (
                result.finished_sending
                or result.finished_recving
                or result.failed_recving
            )
        assert worker.get_block_ids_with_load_errors() == set()
    finally:
        worker.shutdown()


@pytest.mark.parametrize("mp_role", ["kv_both", "kv_consumer"])
@pytest.mark.parametrize("peer_delays_free", [False, True])
def test_multi_connector_child_role_and_completion(
    mp_role: str,
    peer_delays_free: bool,
    mock_io: SimpleNamespace,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real MultiConnector construction honors child roles and combines completions."""
    scheduler_peer = MagicMock()
    scheduler_peer.get_num_new_matched_tokens.return_value = (0, False)
    scheduler_peer.request_finished.return_value = (
        peer_delays_free,
        {"remote_engine_id": "peer"},
    )
    worker_peer = MagicMock()
    worker_peer.get_finished.side_effect = [
        (None, None),
        ({"request"} if peer_delays_free else None, None),
    ]
    # MultiConnector polls children through get_transfer_results(), whose base
    # implementation wraps get_finished(). A bare MagicMock would answer that
    # call with a mock whose result sets iterate empty, silently dropping the
    # peer's completions, so route it through the real base implementation.
    worker_peer.get_transfer_results.side_effect = lambda finished_req_ids: (
        KVConnectorBase_V1.get_transfer_results(worker_peer, finished_req_ids)
    )
    peer_factory = MagicMock(side_effect=[scheduler_peer, worker_peer])

    def connector_class(config: KVTransferConfig) -> Any:
        return (
            LMCacheMPConnector
            if config.kv_connector == "LMCacheMPConnector"
            else peer_factory
        )

    monkeypatch.setattr(KVConnectorFactory, "get_connector_class", connector_class)
    config = _config(
        KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_consumer",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "NixlConnector", "kv_role": "kv_consumer"},
                    {"kv_connector": "LMCacheMPConnector", "kv_role": mp_role},
                ]
            },
        )
    )
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, None)
    worker = MultiConnector(config, KVConnectorRole.WORKER, None)
    worker.register_kv_caches(mock_io.kv_caches)
    request = _request()
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: ([1],)), 0
    )
    metadata = scheduler.build_connector_meta(
        _schedule(num_tokens=4, block_ids=([1],), new=True)
    )
    can_store = mp_role == "kv_both"
    assert len(metadata.metadata[1].requests) == int(can_store)
    worker.bind_connector_metadata(metadata)
    worker.wait_for_save()
    assert mock_io.transfer.submit_store.call_count == int(can_store)
    assert scheduler.request_finished(request, [1]) == (
        can_store or peer_delays_free,
        {"remote_engine_id": "peer"},
    )
    worker.bind_connector_metadata(scheduler.build_connector_meta(_schedule()))
    sending, receiving = worker.get_finished({"request"})
    assert (sending or set()) == (
        {"request"} if can_store and not peer_delays_free else set()
    )
    assert not receiving
    sending, receiving = worker.get_finished(set())
    assert (sending or set()) == ({"request"} if peer_delays_free else set())
    assert not receiving
    mock_io.scheduler.end_session.assert_called_once_with("request")
    worker.shutdown()
    scheduler.shutdown()


def _hashed_pool(num_gpu_blocks: int) -> tuple[BlockPool, list[int]]:
    """A prefix-caching pool whose every block is allocated and hashed.

    Returns:
        The pool and its block ids in free-queue order: freed in that order,
        they line up from the queue head the same way.
    """
    pool = BlockPool(
        num_gpu_blocks=num_gpu_blocks, enable_caching=True, hash_block_size=4
    )
    blocks = pool.get_new_blocks(pool.get_num_free_blocks())
    for i, block in enumerate(blocks):
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(i.to_bytes(32, "big")), 0)
        )
    return pool, [block.block_id for block in blocks]


def _eviction_aware_connectors(
    mock_io: SimpleNamespace,
    pool: BlockPool,
    max_model_len: int,
    max_num_batched_tokens: int,
) -> tuple[LMCacheMPConnector, LMCacheMPConnector]:
    """Scheduler and worker connectors running the default lazy policy."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"lmcache.mp.lazy_offload": True},
        ),
        max_model_len=max_model_len,
        max_num_batched_tokens=max_num_batched_tokens,
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    worker = LMCacheMPConnector(config, KVConnectorRole.WORKER)
    _bind_serving_whole_prompts(scheduler, pool)
    worker.register_kv_caches(mock_io.kv_caches)
    return scheduler, worker


def _admit_and_schedule(
    scheduler: LMCacheMPConnector,
    request_id: str,
    num_tokens: int,
    num_scheduled: int,
    block_ids: list[int],
) -> LMCacheMPConnectorMetadata:
    """Admit a new request with an LMCache miss and run its first step.

    Returns:
        The connector metadata of that step.
    """
    request = _request(request_id)
    request.all_token_ids = list(range(num_tokens))
    request.num_prompt_tokens = num_tokens
    assert scheduler.get_num_new_matched_tokens(request, 0) == (0, False)
    scheduler.update_state_after_alloc(
        request, MagicMock(get_block_ids=lambda: (block_ids,)), 0
    )
    metadata = scheduler.build_connector_meta(
        _schedule(request_id, num_scheduled, (block_ids,), new=True)
    )
    assert isinstance(metadata, LMCacheMPConnectorMetadata)
    return metadata


def _stored_ranges(
    metadata: LMCacheMPConnectorMetadata, request_id: str
) -> list[tuple[int, int]]:
    return [
        (meta.op.start, meta.op.end)
        for meta in metadata.requests
        if meta.direction == "STORE" and meta.request_id == request_id
    ]


def test_failed_lazy_store_is_submitted_again(mock_io: SimpleNamespace) -> None:
    """A store the server failed keeps its range buffered and is resubmitted.

    The finished request's KV is still on the GPU, so the next drain that
    finds its blocks due stores the same range again instead of losing it.
    """
    # Every free block sits within the danger window of either drain rule.
    pool, ids = _hashed_pool(8)
    mock_io.scheduler.update_pending_store_count.return_value = True
    scheduler, worker = _eviction_aware_connectors(mock_io, pool, 4, 16)
    try:
        victim = ids[:2]
        assert _admit_and_schedule(scheduler, "victim", 8, 8, victim).requests == []
        scheduler.request_finished(_request("victim"), victim)
        pool.free_blocks([pool.blocks[block_id] for block_id in victim])

        metadata = _admit_and_schedule(scheduler, "first", 4, 4, [ids[2]])
        assert _stored_ranges(metadata, "victim") == [(0, 8)]
        mock_io.transfer.submit_store.return_value.result.return_value = False
        worker.bind_connector_metadata(metadata)
        worker.wait_for_save()
        worker.get_finished(set())
        scheduler.update_connector_output(
            KVConnectorOutput(
                kv_connector_worker_meta=worker.build_connector_worker_meta()
            )
        )

        metadata = _admit_and_schedule(scheduler, "second", 4, 4, [ids[3]])
        assert _stored_ranges(metadata, "victim") == [(0, 8)]
    finally:
        worker.shutdown()
        scheduler.shutdown()


# max_model_len and max_num_batched_tokens of the danger-depth test. One
# admission allocates at most 68 blocks: the 257 slots of a full-length
# request with its recomputed token, a partial block at either end, a
# copy-on-write block, and one more for a split between hit and allocation.
DEPTH_TEST_TOKENS = 256
# A buffered block this deep in the free queue lies past one admission's
# reach, but within a 256-token prefill chunk's 63 more blocks.
VICTIM_DEPTH = 100


@pytest.mark.parametrize(
    ("runner_tokens", "stored"),
    [(4, False), (DEPTH_TEST_TOKENS, True)],
    ids=["runner-decoding", "runner-prefilling"],
)
def test_lazy_store_is_emitted_before_the_next_step_can_reach_it(
    mock_io: SimpleNamespace, runner_tokens: int, stored: bool
) -> None:
    """A running prefill's next chunk deepens the danger window to its reach.

    The runner has computed one block. Decoding, its next step allocates at
    most one more block, and the buffered block deep in the free queue
    stays buffered. With its prompt mostly ahead, its next chunk can take
    the whole token budget's blocks from the queue head, which reaches the
    buffered block, so the store is emitted on this step, before that
    allocation can recycle it. This step itself schedules only 4 tokens.
    """
    pool, ids = _hashed_pool(160)
    scheduler, worker = _eviction_aware_connectors(
        mock_io, pool, DEPTH_TEST_TOKENS, DEPTH_TEST_TOKENS
    )
    try:
        victim = ids[VICTIM_DEPTH]
        pool.free_blocks([pool.blocks[block_id] for block_id in ids[:VICTIM_DEPTH]])
        assert _admit_and_schedule(scheduler, "victim", 4, 4, [victim]).requests == []
        scheduler.request_finished(_request("victim"), [victim])
        runner = ids[-1]
        pool.free_blocks([pool.blocks[block_id] for block_id in ids[VICTIM_DEPTH:-1]])

        metadata = _admit_and_schedule(scheduler, "runner", runner_tokens, 4, [runner])

        assert bool(_stored_ranges(metadata, "victim")) is stored
    finally:
        worker.shutdown()
        scheduler.shutdown()


def _kv_event(block_hash: int) -> CacheStoreEvent:
    """An adapter-level completed-store event, as ``get_kv_events`` returns."""
    return CacheStoreEvent(
        block_hashes=[block_hash],
        parent_block_hash=None,
        token_ids=[0, 1, 2, 3],
        lora_id=None,
        block_size=4,
        medium="CPU",
        lora_name=None,
    )


def _stored_hashes(connector: KVConnectorBase_V1) -> list[Any]:
    """Block hashes of the BlockStored events a scheduler connector publishes."""
    return [e.block_hashes[0] for e in connector.take_events()]


@pytest.fixture
def kv_events_queue(monkeypatch: pytest.MonkeyPatch) -> list[list[CacheStoreEvent]]:
    """Per-call results of ``LMCacheMPWorkerAdapter.get_kv_events``."""
    queue: list[list[CacheStoreEvent]] = []
    monkeypatch.setattr(
        adapter_mod.LMCacheMPWorkerAdapter,
        "get_kv_events",
        lambda self: queue.pop(0) if queue else [],
    )
    return queue


def test_kv_events_survive_multi_connector(
    lazy_offload: bool,
    mock_io: SimpleNamespace,
    kv_events_queue: list[list[CacheStoreEvent]],
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """LMCache store events reach a real MultiConnector's take_events (P/D).

    Two steps, so events keep flowing after the first step.
    """
    peer = MagicMock()
    peer.take_events.return_value = []
    peer.get_kv_connector_kv_cache_events.return_value = None
    peer.build_connector_worker_meta.return_value = None
    monkeypatch.setattr(
        KVConnectorFactory,
        "get_connector_class",
        lambda c: (
            LMCacheMPConnector
            if c.kv_connector == "LMCacheMPConnector"
            else MagicMock(return_value=peer)
        ),
    )
    config = _config(
        KVTransferConfig(
            kv_connector="MultiConnector",
            kv_role="kv_producer",
            kv_connector_extra_config={
                "connectors": [
                    {"kv_connector": "NixlConnector", "kv_role": "kv_producer"},
                    {
                        "kv_connector": "LMCacheMPConnector",
                        "kv_role": "kv_both",
                        "kv_connector_extra_config": {
                            "lmcache.mp.lazy_offload": lazy_offload
                        },
                    },
                ]
            },
        )
    )
    cast(Any, config).kv_events_config = SimpleNamespace(enable_kv_cache_events=True)
    scheduler = MultiConnector(config, KVConnectorRole.SCHEDULER, None)
    worker = MultiConnector(config, KVConnectorRole.WORKER, None)
    scheduler.bind_gpu_block_pool(mock_io.pool)
    try:
        kv_events_queue.extend([[_kv_event(11)], [_kv_event(12)]])
        for expected in ([11], [12]):
            # Same calls vLLM's model runner makes after each step.
            scheduler.update_connector_output(
                KVConnectorOutput(
                    kv_cache_events=worker.get_kv_connector_kv_cache_events(),
                    kv_connector_worker_meta=worker.build_connector_worker_meta(),
                )
            )
            assert _stored_hashes(scheduler) == expected
        assert _stored_hashes(scheduler) == []
    finally:
        worker.shutdown()
        scheduler.shutdown()


def test_kv_events_standalone_use_hook_only(
    mock_io: SimpleNamespace, kv_events_queue: list[list[CacheStoreEvent]]
) -> None:
    """Standalone, events leave only via the vLLM hook, never worker metadata.

    The second queue entry stands in for events that arrive between the hook
    and ``build_connector_worker_meta`` in one step; they must wait for the
    next hook call rather than leak into worker metadata.
    """
    config = _config(
        KVTransferConfig(kv_connector="LMCacheMPConnector", kv_role="kv_both")
    )
    cast(Any, config).kv_events_config = SimpleNamespace(enable_kv_cache_events=True)
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    worker = LMCacheMPConnector(config, KVConnectorRole.WORKER)
    try:
        kv_events_queue.extend([[_kv_event(11)], [_kv_event(12)]])
        for expected in ([11], [12]):
            # Same calls vLLM's model runner makes after each step.
            output = KVConnectorOutput(
                kv_cache_events=worker.get_kv_connector_kv_cache_events(),
                kv_connector_worker_meta=worker.build_connector_worker_meta(),
            )
            assert getattr(output.kv_connector_worker_meta, "kv_events", None) is None
            scheduler.update_connector_output(output)
            assert _stored_hashes(scheduler) == expected
    finally:
        worker.shutdown()
        scheduler.shutdown()


def test_kv_events_worker_reports_merge_across_ranks(
    mock_io: SimpleNamespace, kv_events_queue: list[list[CacheStoreEvent]]
) -> None:
    """TP ranks' worker-metadata events are aggregated and each published once.

    The workers never see the vLLM hook here, as under MultiConnector, so their
    events ride in worker metadata. Both ranks report 11 and only rank 0 reports
    12; the scheduler publishes every reported event exactly once.
    """
    config = _config(
        KVTransferConfig(kv_connector="LMCacheMPConnector", kv_role="kv_both")
    )
    cast(Any, config).kv_events_config = SimpleNamespace(enable_kv_cache_events=True)
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    worker = LMCacheMPConnector(config, KVConnectorRole.WORKER)
    try:
        kv_events_queue.extend([[_kv_event(11), _kv_event(12)], [_kv_event(11)]])
        rank0 = worker.build_connector_worker_meta()
        rank1 = worker.build_connector_worker_meta()
        assert rank0 is not None and rank1 is not None
        # What vLLM's KVOutputAggregator does with the ranks' worker metadata.
        merged = rank0.aggregate(rank1)
        assert merged.kv_events is not None
        assert merged.kv_events.get_number_of_workers() == 2
        scheduler.update_connector_output(
            KVConnectorOutput(kv_connector_worker_meta=merged)
        )
        assert _stored_hashes(scheduler) == [11, 12]
        assert _stored_hashes(scheduler) == []
    finally:
        worker.shutdown()
        scheduler.shutdown()


@pytest.mark.parametrize("replay", [0, 128])
def test_full_prompt_hit_stays_chunk_aligned_under_prefix_replay(
    mock_io: SimpleNamespace, replay: int
) -> None:
    """A full-prompt hit must not retrieve past what vLLM allocates.

    With SWA bounded replay vLLM floors the external hit to the scheduler
    block boundary before allocating, so the classic ``ret - 1`` leaves the
    retrieve one block past the allocation. Under replay the connector gives
    back the last chunk instead (retrieve end 8 of 12, lock on [8, 12)
    released); without replay the ``ret - 1`` rule is unchanged.
    """
    config = _config(
        KVTransferConfig(kv_connector="LMCacheMPConnector", kv_role="kv_both")
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    scheduler.bind_gpu_block_pool(mock_io.pool)
    try:
        scheduler._prefix_replay_tokens = replay
        request = _request()  # 12 prompt tokens, chunk 4
        mock_io.scheduler.check_lookup_result.return_value = 12
        need, async_load = scheduler.get_num_new_matched_tokens(request, 0)
        assert need is not None  # the lookup result is ready
        tracker = scheduler.request_trackers[request.request_id]
        if replay:
            assert (need, async_load) == (8, True)
            assert tracker.num_lmcache_hit_tokens == 8
            assert tracker.pending_free_ranges == [(8, 12)]
        else:
            assert (need, async_load) == (11, True)
            assert tracker.num_lmcache_hit_tokens == 12
            assert tracker.pending_free_ranges == []
        # vLLM allocates exactly the blocks it was told about (8 tokens = 2
        # blocks under replay); the retrieve must fit in them.
        n_blocks = 2 if replay else 3
        scheduler.update_state_after_alloc(
            request,
            MagicMock(get_block_ids=lambda: (list(range(1, n_blocks + 1)),)),
            need,
        )
        metadata = scheduler.build_connector_meta(_schedule())
        assert isinstance(metadata, LMCacheMPConnectorMetadata)
        assert len(metadata.requests) == 1
        meta = metadata.requests[0]
        assert meta.direction == "RETRIEVE"
        assert (meta.op.start, meta.op.end) == (0, 8 if replay else 12)
        assert len(meta.op.block_ids[0]) * 4 >= meta.op.end
        if replay:
            kwargs = mock_io.scheduler.free_lookup_locks.call_args.kwargs
            assert kwargs["extra_ranges"] == ((8, 12),)
    finally:
        scheduler.shutdown()


def test_exclusive_retrieve_requires_lazy_offload(mock_io: SimpleNamespace) -> None:
    """Without lazy offload nothing re-stores a demoted chunk: refuse it."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={"lmcache.mp.exclusive_retrieve": True},
        )
    )
    with pytest.raises(ValueError, match="exclusive_retrieve requires"):
        LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)


def test_exclusive_retrieve_turns_off_with_lazy_offload(
    mock_io: SimpleNamespace,
) -> None:
    """Without a bound KV cache manager lazy offload turns off, and exclusive
    retrieve with it: no request asks the server for it and the hit prefix
    is not re-stored."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": True,
                "lmcache.mp.lazy_offload_policy": "FIFO",
                "lmcache.mp.exclusive_retrieve": True,
            },
        )
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    scheduler.bind_gpu_block_pool(mock_io.pool)
    request = _request()
    mock_io.scheduler.check_lookup_result.side_effect = [8, 0]
    try:
        assert scheduler.get_num_new_matched_tokens(request, 4) == (4, True)
        assert not scheduler.exclusive_retrieve
        scheduler.update_state_after_alloc(
            request, MagicMock(get_block_ids=lambda: ([1, 2],)), 4
        )
        assert scheduler.request_trackers[request.request_id].num_stored_tokens == 8
        assert (
            mock_io.scheduler.free_lookup_locks.call_args.kwargs["request_configs"]
            is None
        )
        (retrieve,) = scheduler.build_connector_meta(_schedule()).requests
        assert retrieve.request_configs is None
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("advertised", [False, True])
def test_exclusive_retrieve_needs_a_server_advertising_it(
    mock_io: SimpleNamespace, monkeypatch: pytest.MonkeyPatch, advertised: bool
) -> None:
    """Against a server that does not advertise exclusive retrieve, the
    connector warns and runs lazy offload alone: the hit prefix is not
    re-stored and no request asks the server for exclusive retrieve."""
    mock_io.scheduler.experimental = {EXCLUSIVE_RETRIEVE} if advertised else set()
    warnings: list[str] = []
    # The module logger does not propagate, so caplog cannot see it.
    monkeypatch.setattr(
        connector_mod.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(msg % args),
    )
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": True,
                "lmcache.mp.lazy_offload_policy": "FIFO",
                "lmcache.mp.exclusive_retrieve": True,
            },
        )
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    _bind_serving_whole_prompts(scheduler, mock_io.pool)
    request = _request()
    mock_io.scheduler.check_lookup_result.side_effect = [8, 0]
    try:
        assert scheduler.get_num_new_matched_tokens(request, 4) == (4, True)
        scheduler.update_state_after_alloc(
            request, MagicMock(get_block_ids=lambda: ([1, 2],)), 4
        )
        tracker = scheduler.request_trackers[request.request_id]
        assert tracker.num_stored_tokens == (0 if advertised else 8)
        (retrieve,) = scheduler.build_connector_meta(_schedule()).requests
        assert retrieve.request_configs == (
            {EXCLUSIVE_RETRIEVE_REQUEST_CONFIG: True} if advertised else None
        )
        xr_warnings = [w for w in warnings if "exclusive_retrieve" in w]
        assert len(xr_warnings) == (0 if advertised else 1)
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("exclusive", [False, True])
def test_exclusive_retrieve_restores_the_loaded_chunks(
    mock_io: SimpleNamespace, exclusive: bool
) -> None:
    """Exclusive mode stores from token 0, so the loaded chunks and vLLM's
    local hit become lazy store candidates; the flag and the held prefix reach
    the server only from the connector's own configuration."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": True,
                "lmcache.mp.lazy_offload_policy": "FIFO",
                "lmcache.mp.lazy_offload_threshold": 1,
                "lmcache.mp.lazy_offload_select_count": 1,
                "lmcache.mp.exclusive_retrieve": exclusive,
            },
        )
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    _bind_serving_whole_prompts(scheduler, mock_io.pool)
    request = _request()
    # A request cannot turn the mode on by itself.
    cast(Any, request).sampling_params = SimpleNamespace(
        extra_args={
            "kv_transfer_params": {
                EXCLUSIVE_RETRIEVE_REQUEST_CONFIG: True,
                ENGINE_HELD_TOKENS_HINT: 12,
            }
        }
    )
    expected_configs = {EXCLUSIVE_RETRIEVE_REQUEST_CONFIG: True} if exclusive else None
    mock_io.scheduler.check_lookup_result.side_effect = [8, 0]
    try:
        # vLLM hits 4 tokens, LMCache 8: the retrieve loads chunk [4, 8).
        assert scheduler.get_num_new_matched_tokens(request, 4) == (4, True)
        scheduler.update_state_after_alloc(
            request, MagicMock(get_block_ids=lambda: ([1, 2],)), 4
        )
        tracker = scheduler.request_trackers[request.request_id]
        assert tracker.num_lmcache_hit_tokens == 8
        assert tracker.num_stored_tokens == (0 if exclusive else 8)
        # The FREE after allocation reports the 4 tokens vLLM holds.
        free = mock_io.scheduler.free_lookup_locks.call_args.kwargs
        assert free["end"] == 4
        assert free["request_configs"] == (
            {**expected_configs, ENGINE_HELD_TOKENS_HINT: 4}
            if expected_configs
            else None
        )
        metadata = scheduler.build_connector_meta(_schedule())
        (retrieve,) = metadata.requests
        assert (retrieve.direction, retrieve.op.start, retrieve.op.end) == (
            "RETRIEVE",
            4,
            8,
        )
        assert retrieve.request_configs == expected_configs

        # The load finished; vLLM schedules the rest of the prompt.
        scheduler.update_state_after_alloc(
            request, MagicMock(get_block_ids=lambda: ([1, 2, 3],)), 0
        )
        scheduler.build_connector_meta(
            _schedule(num_tokens=4, block_ids=([3],), new=True, num_computed_tokens=8)
        )
        assert tracker.num_stored_tokens == 12
        scheduler.request_finished(request, [1, 2, 3])

        # A later step drains the finished request's buffered store.
        trigger = _request("trigger")
        assert scheduler.get_num_new_matched_tokens(trigger, 0) == (0, False)
        scheduler.update_state_after_alloc(
            trigger, MagicMock(get_block_ids=lambda: ([4],)), 0
        )
        metadata = scheduler.build_connector_meta(
            _schedule("trigger", 1, ([4],), new=True)
        )
        (store,) = metadata.requests
        assert (store.request_id, store.direction, store.op.start, store.op.end) == (
            "request",
            "STORE",
            0 if exclusive else 8,
            12,
        )
        assert store.op.block_ids == ([[1, 2, 3]] if exclusive else [[3]])
        # The mock pool's blocks stay referenced, so vLLM still holds them.
        assert store.request_configs == (
            {**expected_configs, ENGINE_HELD_BLOCKS_HINT: [1, 2, 3]}
            if expected_configs
            else None
        )
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize("exclusive", [False, True])
def test_exclusive_local_hit_reports_and_restores_the_held_prefix(
    mock_io: SimpleNamespace, exclusive: bool
) -> None:
    """A request LMCache serves nothing past vLLM's local hit: after
    allocation the connector's FREE reports the held prefix (exclusive only),
    and the local-hit chunks become lazy store candidates, so a demoted L1
    copy is re-stored when vLLM evicts its blocks."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": True,
                "lmcache.mp.lazy_offload_policy": "FIFO",
                "lmcache.mp.lazy_offload_threshold": 1,
                "lmcache.mp.lazy_offload_select_count": 1,
                "lmcache.mp.exclusive_retrieve": exclusive,
            },
        )
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    _bind_serving_whole_prompts(scheduler, mock_io.pool)
    request = _request()
    configs = {EXCLUSIVE_RETRIEVE_REQUEST_CONFIG: True} if exclusive else None
    # The lookup counts vLLM's covered 8 tokens as its hit and finds no more.
    mock_io.scheduler.check_lookup_result.side_effect = [8, 0]
    try:
        assert scheduler.get_num_new_matched_tokens(request, 8) == (0, False)
        assert (
            mock_io.scheduler.maybe_submit_lookup_request.call_args.kwargs["start"] == 8
        )
        # No demotion before allocation: vLLM may still evict the hit.
        mock_io.scheduler.free_lookup_locks.assert_not_called()
        scheduler.update_state_after_alloc(
            request, MagicMock(get_block_ids=lambda: ([1, 2, 3],)), 0
        )
        free = mock_io.scheduler.free_lookup_locks.call_args.kwargs
        assert (free["start"], free["end"]) == (0, 8)
        assert free["request_configs"] == (
            {**configs, ENGINE_HELD_TOKENS_HINT: 8} if configs else None
        )
        tracker = scheduler.request_trackers[request.request_id]
        assert tracker.num_stored_tokens == (0 if exclusive else 8)

        scheduler.build_connector_meta(
            _schedule(
                num_tokens=4, block_ids=([1, 2, 3],), new=True, num_computed_tokens=8
            )
        )
        scheduler.request_finished(request, [1, 2, 3])
        # A later step drains the finished request's buffered store.
        trigger = _request("trigger")
        assert scheduler.get_num_new_matched_tokens(trigger, 0) == (0, False)
        scheduler.update_state_after_alloc(
            trigger, MagicMock(get_block_ids=lambda: ([4],)), 0
        )
        metadata = scheduler.build_connector_meta(
            _schedule("trigger", 1, ([4],), new=True)
        )
        (store,) = metadata.requests
        assert (store.request_id, store.direction, store.op.start, store.op.end) == (
            "request",
            "STORE",
            0 if exclusive else 8,
            12,
        )
        assert store.op.block_ids == ([[1, 2, 3]] if exclusive else [[3]])
        # The mock pool's blocks stay referenced, so vLLM still holds them.
        assert store.request_configs == (
            {**configs, ENGINE_HELD_BLOCKS_HINT: [1, 2, 3]} if configs else None
        )
    finally:
        scheduler.shutdown()


@pytest.mark.parametrize(
    "num_prompt_tokens, expected_hint",
    [
        # 2 tokens past the 8-token hit: less than one 4-token block.
        (10, 8),
        # 4 tokens past the hit: a full block, and nothing else holds it.
        (12, None),
    ],
)
def test_exclusive_hint_keeps_the_hit_of_a_request_adding_no_block(
    mock_io: SimpleNamespace, num_prompt_tokens: int, expected_hint: int | None
) -> None:
    """A request that computes less than one scheduler block past its hit
    caches no new state in vLLM, so its RETRIEVE and post-allocation FREE
    carry ``SHARED_HELD_TOKENS_HINT`` up to the hit end; a request that
    computes a full block past it, with no sibling holding its prefix,
    carries no hint."""
    config = _config(
        KVTransferConfig(
            kv_connector="LMCacheMPConnector",
            kv_role="kv_both",
            kv_connector_extra_config={
                "lmcache.mp.lazy_offload": True,
                "lmcache.mp.lazy_offload_policy": "FIFO",
                "lmcache.mp.lazy_offload_threshold": 1,
                "lmcache.mp.lazy_offload_select_count": 1,
                "lmcache.mp.exclusive_retrieve": True,
            },
        )
    )
    scheduler = LMCacheMPConnector(config, KVConnectorRole.SCHEDULER)
    _bind_serving_whole_prompts(scheduler, mock_io.pool)
    request = _request()
    cast(Any, request).all_token_ids = list(range(num_prompt_tokens))
    cast(Any, request).num_prompt_tokens = num_prompt_tokens
    cast(Any, request).block_hashes = [
        f"request-{i}".encode() for i in range(num_prompt_tokens // 4)
    ]
    hint = {} if expected_hint is None else {SHARED_HELD_TOKENS_HINT: expected_hint}
    mock_io.scheduler.check_lookup_result.side_effect = [8, 0]
    try:
        # vLLM hits 4 tokens, LMCache 8: the retrieve loads chunk [4, 8).
        assert scheduler.get_num_new_matched_tokens(request, 4) == (4, True)
        scheduler.update_state_after_alloc(
            request, MagicMock(get_block_ids=lambda: ([1, 2],)), 4
        )
        free = mock_io.scheduler.free_lookup_locks.call_args.kwargs
        assert free["request_configs"] == {
            EXCLUSIVE_RETRIEVE_REQUEST_CONFIG: True,
            ENGINE_HELD_TOKENS_HINT: 4,
            **hint,
        }
        metadata = scheduler.build_connector_meta(_schedule())
        (retrieve,) = metadata.requests
        assert (retrieve.direction, retrieve.op.start, retrieve.op.end) == (
            "RETRIEVE",
            4,
            8,
        )
        assert retrieve.request_configs == {
            EXCLUSIVE_RETRIEVE_REQUEST_CONFIG: True,
            **hint,
        }
    finally:
        scheduler.shutdown()
