# SPDX-License-Identifier: Apache-2.0
"""Server side of exclusive retrieve, on CPU against a real L1 and LRU policy.

- A successful retrieve of an exclusive request demotes exactly the keys it
  read in L1 (stream-ordered, after the read locks are released); a plain
  request or a failed retrieve demotes nothing.
- ``end_session`` does not touch the demoted keys back to most recently used.
- A store of an exclusive request touches the keys it finds already resident.

Only pinned-memory and device calls are replaced; host callbacks run through
the production dispatcher, drained when the module closes.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    MemoryLayoutDesc,
    ObjectKey,
    ipc_key_to_object_keys,
)
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.kv_layer_groups import ObjectGroupInfo
from lmcache.v1.multiprocess import native_completion
from lmcache.v1.multiprocess.custom_types import (
    EXCLUSIVE_RETRIEVE_REQUEST_CONFIG,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_hasher import TokenHasher
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

CHUNK = 4
NUM_CHUNKS = 3
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.uint8])
# Object group 0 is full attention (reads every chunk); group 1 is a
# recurrent group with a one-chunk window (reads only the last chunk).
WINDOWS = [-1, 1]
RETRIEVE_BLOCKS = [[1, 2, 3], [-1, -1, 5]]
# A key of another request, resident before this one's.
OTHER_KEY = ObjectKey(ObjectKey.IntHash2Bytes(7), "model", 0)


def _ipc_key(exclusive: bool) -> IPCCacheServerKey:
    return IPCCacheServerKey.from_token_ids(
        model_name="model",
        world_size=1,
        worker_id=0,
        token_ids=list(range(CHUNK * NUM_CHUNKS)),
        start=0,
        end=CHUNK * NUM_CHUNKS,
        request_id="req",
        request_configs=(
            {EXCLUSIVE_RETRIEVE_REQUEST_CONFIG: True} if exclusive else None
        ),
    )


class _Server:
    """A transfer module plus the L1, LRU order and sessions behind it."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.queued: list[tuple[str, bytes]] = []
        monkeypatch.setattr(
            native_completion,
            "_device_ops",
            SimpleNamespace(
                record_completion_on_stream=lambda stream, kind, payload: (
                    self.queued.append((kind, payload))
                ),
                drain_recorded_completions=self._drain,
            ),
        )
        dispatchers: list[native_completion.DeviceHostFuncDispatcher] = []

        class _ManualDispatcher(native_completion.DeviceHostFuncDispatcher):
            """Never runs a drain thread; each ``stop`` drains once."""

            def __init__(self) -> None:
                super().__init__()
                dispatchers.append(self)

            def start(self) -> None:
                pass

        monkeypatch.setattr(transfer, "DeviceHostFuncDispatcher", _ManualDispatcher)
        monkeypatch.setattr(transfer, "torch_dev", MagicMock())
        monkeypatch.setattr(
            transfer, "downsample_and_stage_block_ids", lambda cc, ids: ids
        )
        monkeypatch.setattr(transfer, "get_layout_desc", lambda *a, **kw: LAYOUT)

        self.l1 = L1Manager(
            L1ManagerConfig(
                L1MemoryManagerConfig(
                    size_in_bytes=64 * 4096, use_lazy=False, shm_name=""
                )
            )
        )
        self.lru = LRUEvictionPolicy()
        self.l1.register_listener(L1EvictionPolicy(self.lru))
        self.storage = StorageManager(
            StorageManagerConfig(
                l1_manager_config=self.l1.config,
                eviction_config=EvictionConfig(eviction_policy="noop"),
                l2_adapter_config=L2AdaptersConfig(adapters=[]),
            ),
            _l1_managers=(self.l1,),
        )
        self.sessions = SessionManager(
            TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3"),
            cleanup_interval=None,
        )
        event_bus = MagicMock()
        event_bus.has_subscribers.return_value = False
        self.ctx = SimpleNamespace(
            chunk_size=CHUNK,
            null_block_id=-1,
            storage_manager=self.storage,
            session_manager=self.sessions,
            event_bus=event_bus,
            resolve_obj_keys=self._resolve,
            get_read_owners=lambda request_id: None,
        )
        self.module = transfer.LMCacheDrivenTransferModule(cast(Any, self.ctx))
        (self.dispatcher,) = dispatchers
        cache_context = MagicMock()
        cache_context.calculate_num_blocks.return_value = 1
        cache_context.cupy_stream.ptr = 123
        cache_context.hold_imported_event.return_value = 7
        cache_context.kv_layer_groups_manager = SimpleNamespace(
            num_object_groups=2,
            num_kernel_groups=2,
            object_groups=[ObjectGroupInfo(kernel_group_indices=[g]) for g in (0, 1)],
            get_attn_desc=lambda: SimpleNamespace(
                num_chunks_in_sw=WINDOWS, group_kinds=("attention", "recurrent")
            ),
        )
        entry = SimpleNamespace(
            cache_context=cache_context, model_name="model", event_backend=MagicMock()
        )
        monkeypatch.setattr(
            self.module, "get_and_touch_context_entry", lambda instance_id: entry
        )

    def keys(self) -> list[list[ObjectKey]]:
        """The request's object keys per object group."""
        return self._resolve(_ipc_key(False), [0, 1])

    def commit(self, keys: list[ObjectKey]) -> None:
        """Write the keys into L1 in this order."""
        results = self.l1.reserve_write(keys, [False] * len(keys), LAYOUT)
        assert all(err == L1Error.SUCCESS for err, _ in results.values())
        self.l1.finish_write(keys)

    def lock_for_read(self, keys: list[ObjectKey]) -> None:
        """Take the read locks a lookup would have taken."""
        assert all(
            err == L1Error.SUCCESS for err, _ in self.l1.reserve_read(keys).values()
        )

    def victims(self) -> list[ObjectKey]:
        """Every tracked key in eviction order."""
        return [k for a in self.lru.get_eviction_actions(1.0) for k in a.keys]

    def retrieve(self, exclusive: bool) -> bool:
        """Retrieve the request's three chunks and run the host callbacks."""
        _, ok = self.module.retrieve(_ipc_key(exclusive), 1, RETRIEVE_BLOCKS, b"p")
        self.drain_callbacks()
        return ok

    def store(self, exclusive: bool) -> bool:
        """Store the request's three chunks and run the host callbacks."""
        _, ok = self.module.store(_ipc_key(exclusive), 1, RETRIEVE_BLOCKS, b"p")
        self.drain_callbacks()
        return ok

    def drain_callbacks(self) -> None:
        """Dispatch the queued stream callbacks, as the drain thread would."""
        self.dispatcher.stop()
        assert self.dispatcher.handler_exception_counts() == {}

    def close(self) -> None:
        """Close the module, storage manager and sessions."""
        self.module.close()
        self.storage.close()
        self.sessions.close()

    def _drain(self) -> list[tuple[str, bytes]]:
        result = list(self.queued)
        self.queued.clear()
        return result

    def _resolve(
        self, key: IPCCacheServerKey, object_group_ids: list[int]
    ) -> list[list[ObjectKey]]:
        session = self.sessions.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        if session.lookup_ipc_key is None:
            session.lookup_ipc_key = key.no_worker_id_version()
        hashes = [
            TokenHasher.hash_to_bytes(h) for h in session.get_hashes(key.start, key.end)
        ]
        return ipc_key_to_object_keys(key, hashes, object_group_ids)


@pytest.fixture
def server(monkeypatch: pytest.MonkeyPatch) -> Iterator[_Server]:
    """A server whose L1 holds the request's chunks plus one unrelated key."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    srv = _Server(monkeypatch)
    (full, recurrent) = srv.keys()
    srv.commit([OTHER_KEY])
    srv.commit(full + recurrent)
    try:
        yield srv
    finally:
        srv.close()


def _loaded(srv: _Server) -> list[ObjectKey]:
    """The keys a retrieve reads: every full-attention chunk, the last
    recurrent one."""
    full, recurrent = srv.keys()
    return full + recurrent[-1:]


def test_exclusive_retrieve_demotes_exactly_the_read_keys(server: _Server) -> None:
    """A successful exclusive retrieve demotes exactly the keys it read."""
    loaded = _loaded(server)
    server.lock_for_read(loaded)
    before = server.victims()

    assert server.retrieve(exclusive=True)

    after = server.victims()
    # The request's last loaded key is the very next victim, then backwards.
    assert after[: len(loaded)] == loaded[::-1]
    # Everything else keeps its relative order; nothing was deleted.
    assert after[len(loaded) :] == [k for k in before if k not in loaded]
    # The read locks were released before or with the demotion.
    assert all(server.l1.is_key_evictable(k) for k in loaded)
    session = server.sessions.get("req")
    assert session is not None
    assert session.get_demoted_keys() == set(loaded)


def test_plain_retrieve_keeps_the_eviction_order(server: _Server) -> None:
    """A retrieve without the flag leaves the eviction order unchanged."""
    loaded = _loaded(server)
    server.lock_for_read(loaded)
    before = server.victims()

    assert server.retrieve(exclusive=False)

    assert server.victims() == before
    session = server.sessions.get("req")
    assert session is not None
    assert session.get_demoted_keys() == set()


def test_failed_exclusive_retrieve_demotes_nothing(server: _Server) -> None:
    """A failed exclusive retrieve demotes nothing."""
    loaded = _loaded(server)
    # The recurrent chunk was never locked, so the read fails.
    server.lock_for_read(loaded[:-1])
    before = server.victims()

    assert not server.retrieve(exclusive=True)

    assert server.victims() == before


def test_end_session_does_not_promote_demoted_keys(server: _Server) -> None:
    """``end_session`` touches the request's keys except the demoted ones."""
    loaded = _loaded(server)
    server.lock_for_read(loaded)
    assert server.retrieve(exclusive=True)
    _full, recurrent = server.keys()
    lookup_ctx = MagicMock()
    lookup_ctx.session_manager = server.sessions
    lookup_ctx.storage_manager = server.storage
    lookup_ctx.layout_desc_registry.find_attn_desc.return_value = SimpleNamespace(
        num_object_groups=2
    )

    LookupModule(lookup_ctx).end_session("req")

    victims = server.victims()
    # Still the next victims, in demotion order; only the request's keys that
    # were not loaded (the recurrent prefix) were touched.
    assert victims[: len(loaded)] == loaded[::-1]
    assert victims[len(loaded)] == OTHER_KEY
    assert set(victims[len(loaded) + 1 :]) == set(recurrent[:-1])


@pytest.mark.parametrize("exclusive", [False, True])
def test_restore_of_resident_keys_touches_them_only_when_exclusive(
    server: _Server, exclusive: bool
) -> None:
    """A re-store touches resident keys only for an exclusive request."""
    loaded = _loaded(server)
    server.lock_for_read(loaded)
    assert server.retrieve(exclusive=True)
    before = server.victims()
    used_before = server.storage.get_l1_usage()[0]

    assert server.store(exclusive=exclusive)

    # Nothing new was written: every stored chunk was already resident.
    assert server.storage.get_l1_usage()[0] == used_before
    after = server.victims()
    if exclusive:
        assert after[-len(loaded) :] == loaded[::-1]
    else:
        assert after == before
