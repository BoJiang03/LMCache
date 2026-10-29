# SPDX-License-Identifier: Apache-2.0
"""The l1_snapshot server module writes every L1 object to a file."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Third Party
import msgspec
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
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleSpec,
    build_server_module_router,
    load_server_module_classes,
)
from lmcache.v1.multiprocess.l1_snapshot import L1SnapshotModule
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallRequest
from lmcache.v1.multiprocess.protocols.state_snapshot import (
    FORMAT_VERSION,
    L1_SNAPSHOT_METHOD,
    L1SnapshotReply,
    L1SnapshotRequest,
    read_l1_snapshot,
)
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

OBJECT_BYTES = 4096
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])


def _key(chunk_hash: bytes, object_group_id: int = 0, salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=chunk_hash,
        model_name="m",
        kv_rank=ObjectKey.ComputeKVRank(2, 1, 2, 1),
        object_group_id=object_group_id,
        cache_salt=salt,
    )


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[StorageManager]:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    l1_config = L1ManagerConfig(
        L1MemoryManagerConfig(
            size_in_bytes=32 * OBJECT_BYTES, use_lazy=False, shm_name=""
        )
    )
    manager = StorageManager(StorageManagerConfig(l1_config, EvictionConfig("LRU")))
    yield manager
    manager.close()


def _module(store: StorageManager) -> L1SnapshotModule:
    return L1SnapshotModule(cast(Any, SimpleNamespace(storage_manager=store)))


def _take(
    module: L1SnapshotModule, tmp_path: Path, snapshot_id: int = 3
) -> L1SnapshotReply:
    request = L1SnapshotRequest(
        run_id="run", snapshot_id=snapshot_id, snapshot_dir=str(tmp_path)
    )
    reply = module.take(msgspec.msgpack.encode(request))
    return msgspec.msgpack.decode(reply, type=L1SnapshotReply)


def test_module_loads_as_a_server_module_plugin() -> None:
    specs = [ExtServerModuleSpec(module_path="lmcache.v1.multiprocess.l1_snapshot")]
    assert load_server_module_classes(specs) == [(L1SnapshotModule, {})]

    module = L1SnapshotModule(MagicMock())
    router = build_server_module_router(MagicMock(), [module])
    assert router is not None
    assert router.report_status()["server_module_plugins"]["methods"] == [
        L1_SNAPSHOT_METHOD
    ]
    module.close()


def test_snapshot_lists_committed_and_reserved_objects_in_lru_order(
    store: StorageManager, tmp_path: Path
) -> None:
    a, b, c = _key(b"a"), _key(b"b", object_group_id=1), _key(b"c", salt="s")
    staged = _key(b"d")
    store.reserve_write([a, b, c], LAYOUT)
    store.finish_write([a, b, c])
    store.touch_cached_keys([a])
    store.reserve_write([staged], LAYOUT)

    module = _module(store)
    reply = _take(module, tmp_path)
    module.close()

    assert reply.num_objects == 4
    assert reply.lock_ms >= 0
    header, [section] = read_l1_snapshot(reply.path)
    assert Path(reply.path).parent == tmp_path
    assert header.server_id in Path(reply.path).name
    assert (header.format_version, header.run_id, header.snapshot_id) == (
        FORMAT_VERSION,
        "run",
        3,
    )
    assert header.total_ms >= header.lock_ms >= 0
    assert section.has_eviction_order

    by_hash = {entry.chunk_hash: entry for entry in section.objects}
    assert set(by_hash) == {b"a", b"b", b"c", b"d"}
    assert all(e.size_bytes == OBJECT_BYTES for e in section.objects)
    assert {h: by_hash[h].state for h in by_hash} == {
        b"a": "committed",
        b"b": "committed",
        b"c": "committed",
        b"d": "staging",
    }
    assert (by_hash[b"b"].object_group_id, by_hash[b"c"].cache_salt) == (1, "s")
    assert {(e.model_name, e.kv_rank) for e in section.objects} == {("m", a.kv_rank)}
    # LRU: the reservation is the newest key, the touched key next; of the
    # write batch, the later keys go first.
    ranks = {h: by_hash[h].eviction_rank for h in by_hash}
    assert sorted(ranks, key=ranks.__getitem__) == [b"c", b"b", b"a", b"d"]
    assert sorted(ranks.values()) == [0, 1, 2, 3]
    assert section.num_unlisted_ordered_keys == 0


def test_snapshot_changes_neither_l1_nor_its_order(
    store: StorageManager, tmp_path: Path
) -> None:
    a, b = _key(b"a"), _key(b"b")
    store.reserve_write([a, b], LAYOUT)
    store.finish_write([a, b])
    module = _module(store)

    first = _take(module, tmp_path, snapshot_id=0)
    second = _take(module, tmp_path, snapshot_id=1)
    # close() waits for the writer thread; the files exist only after it.
    module.close()

    assert read_l1_snapshot(first.path)[1] != []
    assert read_l1_snapshot(first.path)[1][0].objects == (
        read_l1_snapshot(second.path)[1][0].objects
    )
    assert store.get_l1_usage()[0] == 2 * OBJECT_BYTES


def test_unknown_method_fails_through_the_router(tmp_path: Path) -> None:
    router = build_server_module_router(MagicMock(), [L1SnapshotModule(MagicMock())])
    assert router is not None
    response = router.server_module_call(
        ServerModuleCallRequest(method="l1_snapshot.missing", payload=b"")
    )
    assert not response.success
