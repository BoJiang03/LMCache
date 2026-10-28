# SPDX-License-Identifier: Apache-2.0
"""Store-failure semantics of ``LMCacheDrivenTransferModule.store`` with
``--separate-object-groups``.

Separate object groups store a chunk as one ObjectKey per object group, and
each key is reserved in L1 on its own. ``store`` answers ok only when every
key of every chunk was stored or was already resident. The device copy, the
stream and its host callbacks are replaced so the store runs on CPU against a
real StorageManager.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.memory_management import MemoryObj
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as mod
from lmcache.v1.multiprocess.modules.lmcache_driven_transfer import (
    LMCacheDrivenTransferModule,
)
from tests.v1.distributed.utils import should_use_lazy_alloc

_MB = 1024 * 1024
_L1_BYTES = 64 * _MB

# Object group 0 (e.g. full attention) and object group 1 (e.g. a separated
# sliding-window group) have different object sizes, as their layer counts do.
_GROUP_LAYOUTS = [
    MemoryLayoutDesc(shapes=[torch.Size([256, 1024])], dtypes=[torch.float32]),
    MemoryLayoutDesc(shapes=[torch.Size([2048, 1024])], dtypes=[torch.float32]),
]
# Leaves 4 MB of L1: room for the 1 MB group-0 object, not the 8 MB group-1 one.
_FILLER_LAYOUT = MemoryLayoutDesc(
    shapes=[torch.Size([15360, 1024])], dtypes=[torch.float32]
)


def _key(chunk_hash: int, object_group_id: int) -> ObjectKey:
    """Return the object key of one chunk's object group."""
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(chunk_hash),
        model_name="test_model",
        kv_rank=0,
        object_group_id=object_group_id,
    )


def _make_module(
    monkeypatch: pytest.MonkeyPatch,
    storage_manager: StorageManager,
    obj_keys_per_group: list[list[ObjectKey]],
) -> tuple[LMCacheDrivenTransferModule, list[list[MemoryObj | None]], MagicMock]:
    """Build a store module over ``storage_manager`` with two object groups.

    Args:
        monkeypatch: Replaces the device-side helpers of the module.
        storage_manager: Real storage manager the store reserves L1 in.
        obj_keys_per_group: Keys the request resolves to, per object group.

    Returns:
        The module, the memory objects each object-group copy was handed per
        call (``None`` where nothing is written), and the server event bus.
    """
    copies: list[list[MemoryObj | None]] = []
    cache_context = MagicMock()
    cache_context.kv_layer_groups_manager = SimpleNamespace(
        num_object_groups=len(obj_keys_per_group),
        num_kernel_groups=len(obj_keys_per_group),
        object_groups=[
            ObjectGroupInfo(kernel_group_indices=[g])
            for g in range(len(obj_keys_per_group))
        ],
    )
    cache_context.calculate_num_blocks.return_value = 1
    cache_context.transfer_object_group.side_effect = (
        lambda block_ids, memory_objs, **kwargs: copies.append(list(memory_objs))
    )
    entry = SimpleNamespace(
        cache_context=cache_context, model_name="test_model", event_backend=MagicMock()
    )

    ctx = MagicMock()
    ctx.storage_manager = storage_manager
    ctx.chunk_size = 256
    ctx.null_block_id = -1
    ctx.resolve_obj_keys.return_value = obj_keys_per_group

    def run_host_callback(stream: Any, kind: str, payload: Any) -> None:
        # The stream has no queued work, so its write callback can run inline.
        if kind == "finish_write_by_owner":
            storage_manager.finish_write_by_owner(payload)

    monkeypatch.setattr(mod, "DeviceHostFuncDispatcher", MagicMock())
    monkeypatch.setattr(mod, "torch_dev", MagicMock())
    monkeypatch.setattr(mod, "submit_callback_to_stream", run_host_callback)
    monkeypatch.setattr(mod, "downsample_and_stage_block_ids", lambda cc, ids: ids)
    monkeypatch.setattr(
        mod,
        "get_layout_desc",
        lambda cc, chunk_size, object_group_id: _GROUP_LAYOUTS[object_group_id],
    )
    module = LMCacheDrivenTransferModule(ctx)
    monkeypatch.setattr(
        module, "get_and_touch_context_entry", lambda instance_id: entry
    )
    return module, copies, ctx.event_bus


def _store(module: LMCacheDrivenTransferModule) -> bool:
    """Store the one-chunk request; return the store's ok."""
    _handle, ok = module.store(
        SimpleNamespace(request_id="req", worker_id=1), 1, [[1], [2]], b"producer"
    )
    return ok


def _last_stored_count(event_bus: MagicMock) -> int:
    """Return ``stored_count`` of the last published MP_STORE_END event."""
    ends = [
        c.args[1]
        for c in event_bus.publish_on_stream.call_args_list
        if c.args[1].event_type == EventType.MP_STORE_END
    ]
    return ends[-1].metadata["stored_count"]


@pytest.fixture
def storage_manager() -> Iterator[StorageManager]:
    """A 64 MB single-L1 storage manager that never evicts."""
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=_L1_BYTES,
                    use_lazy=should_use_lazy_alloc(),
                    init_size_in_bytes=_L1_BYTES,
                    align_bytes=0x1000,
                ),
                write_ttl_seconds=600,
                read_ttl_seconds=300,
            ),
            eviction_config=EvictionConfig(eviction_policy="noop"),
        )
    )
    yield manager
    manager.close()


@pytest.mark.no_shared_allocator
def test_store_fails_when_one_object_group_key_has_no_room_then_retry_fills_it(
    monkeypatch: pytest.MonkeyPatch, storage_manager: StorageManager
) -> None:
    """One group's key OOM fails the store; the retry writes only that key."""
    keys = [[_key(7, 0)], [_key(7, 1)]]
    module, copies, event_bus = _make_module(monkeypatch, storage_manager, keys)
    filler = _key(99, 0)
    storage_manager.finish_write(
        list(storage_manager.reserve_write([filler], _FILLER_LAYOUT))
    )

    assert not _store(module)
    # Group 0's key was written; L1 had no room for group 1's key.
    assert [obj is None for obj in copies[0]] == [False]
    assert [obj is None for obj in copies[1]] == [True]
    assert _last_stored_count(event_bus) == 1

    assert storage_manager.delete_l1_keys([filler]) == (1, 0)
    used_before_retry, _ = storage_manager.get_l1_usage()

    assert _store(module)
    # Group 0's key is resident: the retry copies nothing into it and counts
    # only group 1's key as stored.
    assert [obj is None for obj in copies[2]] == [True]
    assert [obj is None for obj in copies[3]] == [False]
    assert _last_stored_count(event_bus) == 1
    group_1_obj = copies[3][0]
    assert group_1_obj is not None
    used_after_retry, _ = storage_manager.get_l1_usage()
    assert used_after_retry == used_before_retry + group_1_obj.get_size()

    # Both keys are resident now, so a third store writes nothing and succeeds.
    assert _store(module)
    assert [obj is None for copy in copies[4:] for obj in copy] == [True, True]
    assert storage_manager.get_l1_usage()[0] == used_after_retry
