# SPDX-License-Identifier: Apache-2.0
"""The l1_snapshot server module writes every L1 object to a file."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import threading
import time

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
from lmcache.v1.distributed.storage_manager import StorageManager, l1_owners_of
from lmcache.v1.multiprocess import l1_snapshot as l1_snapshot_mod
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
    eviction_ranks,
    read_l1_snapshot,
    write_atomically,
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


def _write(store: StorageManager, keys: list[ObjectKey]) -> None:
    """Reserve and commit ``keys`` in L1."""
    reserved = store.reserve_write(keys, LAYOUT)
    store.finish_write(keys, l1_owners_of(reserved))


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
    """Send ``l1_snapshot.take`` the way a client does: a server-module call."""
    router = build_server_module_router(MagicMock(), [module])
    assert router is not None
    request = L1SnapshotRequest(
        run_id="run", snapshot_id=snapshot_id, snapshot_dir=str(tmp_path)
    )
    response = router.server_module_call(
        ServerModuleCallRequest(
            method=L1_SNAPSHOT_METHOD, payload=msgspec.msgpack.encode(request)
        )
    )
    assert response.success, response.error
    return msgspec.msgpack.decode(response.payload, type=L1SnapshotReply)


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
    _write(store, [a, b, c])
    store.touch_cached_keys([a])
    store.reserve_write([staged], LAYOUT)

    module = _module(store)
    reply = _take(module, tmp_path)
    module.close()

    assert reply.num_objects == 4
    assert reply.list_ms >= 0
    header, [section] = read_l1_snapshot(reply.path)
    assert Path(reply.path).parent == tmp_path
    assert header.server_id in Path(reply.path).name
    assert (header.format_version, header.run_id, header.snapshot_id) == (
        FORMAT_VERSION,
        "run",
        3,
    )
    assert header.collect_ms >= header.list_ms >= 0
    assert header.build_ms >= 0 and header.encode_ms >= 0
    assert section.eviction_order is not None

    objects = section.objects
    rows = {
        chunk_hash: (group_id, salt, size, state)
        for chunk_hash, group_id, salt, size, state in zip(
            objects.chunk_hash,
            objects.object_group_id,
            objects.cache_salt,
            section.size_bytes,
            section.states,
            strict=True,
        )
    }
    assert rows == {
        b"a": (0, "", OBJECT_BYTES, "committed"),
        b"b": (1, "", OBJECT_BYTES, "committed"),
        b"c": (0, "s", OBJECT_BYTES, "committed"),
        b"d": (0, "", OBJECT_BYTES, "staging"),
    }
    assert set(zip(objects.model_name, objects.kv_rank, strict=True)) == {
        ("m", a.kv_rank)
    }
    # LRU: the reservation is the newest key, the touched key next; of the
    # write batch, the later keys go first.
    assert section.eviction_order.chunk_hash == [b"c", b"b", b"a", b"d"]
    ranks, unlisted = eviction_ranks(section)
    by_hash = dict(zip(objects.chunk_hash, ranks, strict=True))
    assert by_hash == {b"c": 0, b"b": 1, b"a": 2, b"d": 3}
    assert unlisted == 0


def test_missing_snapshot_dir_is_created(store: StorageManager, tmp_path: Path) -> None:
    _write(store, [_key(b"a")])
    module = _module(store)
    reply = _take(module, tmp_path / "new" / "dir")
    module.close()

    header, [section] = read_l1_snapshot(reply.path)
    assert Path(reply.path).parent == tmp_path / "new" / "dir"
    assert len(section.objects.chunk_hash) == 1


def test_request_during_a_running_write_is_skipped(
    store: StorageManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(store, [_key(b"a")])
    release = threading.Event()
    write = l1_snapshot_mod.write_atomically

    def slow_write(path: str, data: bytes) -> None:
        release.wait(timeout=30)
        write(path, data)

    monkeypatch.setattr(l1_snapshot_mod, "write_atomically", slow_write)
    module = _module(store)

    first = _take(module, tmp_path, snapshot_id=0)
    second = _take(module, tmp_path, snapshot_id=1)
    release.set()
    module.close()

    assert not first.skipped and Path(first.path).exists()
    assert second.skipped and second.path == ""
    assert [p.name for p in tmp_path.iterdir()] == [Path(first.path).name]


def test_concurrent_requests_collect_l1_once(
    store: StorageManager, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _write(store, [_key(b"a")])
    release_write = threading.Event()
    write = l1_snapshot_mod.write_atomically

    def slow_write(path: str, data: bytes) -> None:
        release_write.wait(timeout=30)
        write(path, data)

    monkeypatch.setattr(l1_snapshot_mod, "write_atomically", slow_write)
    entered = [threading.Event(), threading.Event()]
    release_collect = threading.Event()
    collects = 0

    def gated_snapshot_l1() -> Any:
        nonlocal collects
        entered[collects].set()
        collects += 1
        release_collect.wait(timeout=30)
        return store.snapshot_l1()

    module = L1SnapshotModule(
        cast(
            Any,
            SimpleNamespace(
                storage_manager=SimpleNamespace(snapshot_l1=gated_snapshot_l1)
            ),
        )
    )
    replies: list[L1SnapshotReply] = []

    def take(snapshot_id: int) -> None:
        replies.append(_take(module, tmp_path, snapshot_id=snapshot_id))

    first = threading.Thread(target=take, args=(0,))
    first.start()
    assert entered[0].wait(timeout=30)
    # The second request arrives while the first is still collecting.
    second = threading.Thread(target=take, args=(1,))
    second.start()
    entered[1].wait(timeout=0.5)
    release_collect.set()
    first.join(timeout=30)
    second.join(timeout=30)
    release_write.set()
    module.close()

    assert collects == 1
    assert sorted(reply.skipped for reply in replies) == [False, True]


def test_snapshot_changes_neither_l1_nor_its_order(
    store: StorageManager, tmp_path: Path
) -> None:
    a, b = _key(b"a"), _key(b"b")
    _write(store, [a, b])
    module = _module(store)

    first = _take(module, tmp_path, snapshot_id=0)
    deadline = time.monotonic() + 30
    while not Path(first.path).exists() and time.monotonic() < deadline:
        time.sleep(0.01)
    second = _take(module, tmp_path, snapshot_id=1)
    # close() waits for the writer thread.
    module.close()

    assert not second.skipped
    [first_section] = read_l1_snapshot(first.path)[1]
    [second_section] = read_l1_snapshot(second.path)[1]
    assert first_section.objects == second_section.objects
    assert first_section.eviction_order == second_section.eviction_order
    assert store.get_l1_usage()[0] == 2 * OBJECT_BYTES


def test_unknown_method_fails_through_the_router(tmp_path: Path) -> None:
    router = build_server_module_router(MagicMock(), [L1SnapshotModule(MagicMock())])
    assert router is not None
    response = router.server_module_call(
        ServerModuleCallRequest(method="l1_snapshot.missing", payload=b"")
    )
    assert not response.success


def test_failed_atomic_write_leaves_no_temporary_file(tmp_path: Path) -> None:
    # A directory in the way makes the final rename fail.
    target = tmp_path / "l1.msgpack"
    target.mkdir()
    with pytest.raises(OSError):
        write_atomically(str(target), b"data")
    assert sorted(p.name for p in tmp_path.iterdir()) == ["l1.msgpack"]
