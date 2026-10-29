# SPDX-License-Identifier: Apache-2.0
"""Server side of exclusive retrieve, on CPU against a real L1 and LRU policy.

- A successful retrieve of an exclusive request demotes exactly the keys it
  read in L1 (stream-ordered, after the read locks are released); a plain
  request or a failed retrieve demotes nothing.
- ``end_session`` does not touch the demoted keys back to most recently used.
- A store of an exclusive request touches the keys it finds already resident.
- An exclusive store whose every chunk is resident submits no device work and
  returns an empty completion handle; a store missing one chunk still writes
  it. Without exclusive retrieve the same store keeps the copy path.
- An exclusive request's lookup does not touch the prefix vLLM's prefix cache
  holds; its FREE_LOOKUP_LOCKS demotes the keys of that prefix the engine
  holds in every layer, per object group, and ``end_session`` keeps them.
- A store's keys whose blocks ``ENGINE_HELD_BLOCKS_HINT`` lists, per object
  group (the engine can serve and still holds them) are committed or
  touched demoted, a new key too when its write finishes after the demote;
  only under exclusive retrieve.
- Under exclusive retrieve ``end_session`` touches no L1 key.
- Chunks below ``SHARED_HELD_TOKENS_HINT`` (another running request holds
  them on the GPU too) are touched instead of demoted, by a retrieve and by
  the post-allocation FREE_LOOKUP_LOCKS.
- The server advertises exclusive retrieve when it serves LMCache-driven
  retrieves.

Only pinned-memory and device calls are replaced; host callbacks run through
the production dispatcher, drained when the module closes.
"""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock
import random

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import (
    AttnWindowDesc,
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
from lmcache.v1.mp_observability.event import EventType
from lmcache.v1.multiprocess import native_completion
from lmcache.v1.multiprocess import server as server_mod
from lmcache.v1.multiprocess.config import CoordinatorConfig, MPServerConfig
from lmcache.v1.multiprocess.custom_types import (
    ENGINE_HELD_BLOCKS_HINT,
    ENGINE_HELD_TOKENS_HINT,
    EXCLUSIVE_RETRIEVE,
    EXCLUSIVE_RETRIEVE_REQUEST_CONFIG,
    SHARED_HELD_TOKENS_HINT,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
from lmcache.v1.multiprocess.modules import lmcache_driven_transfer as transfer
from lmcache.v1.multiprocess.modules.lookup import LookupModule
from lmcache.v1.multiprocess.modules.management import ManagementModule
from lmcache.v1.multiprocess.session import SessionManager
from lmcache.v1.multiprocess.token_codec import unpack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

CHUNK = 4
NUM_CHUNKS = 3
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.uint8])
# Object group 0 is full attention (reads every chunk); group 1 is a
# recurrent group with a one-chunk window (reads only the last chunk).
WINDOWS = [-1, 1]
WINDOW_KINDS = ("attention", "recurrent")
RETRIEVE_BLOCKS = [[1, 2, 3], [-1, -1, 5]]
# A key of another request, resident before this one's.
OTHER_KEY = ObjectKey(ObjectKey.IntHash2Bytes(7), "model", 0)
# A key of another request, written after this one's.
LATER_KEY = ObjectKey(ObjectKey.IntHash2Bytes(8), "model", 0)


def _ipc_key(
    exclusive: bool, held_block_ids: list[int] | None = None
) -> IPCCacheServerKey:
    configs: dict[str, Any] = {}
    if exclusive:
        configs[EXCLUSIVE_RETRIEVE_REQUEST_CONFIG] = True
    if held_block_ids is not None:
        configs[ENGINE_HELD_BLOCKS_HINT] = held_block_ids
    return IPCCacheServerKey.from_token_ids(
        model_name="model",
        world_size=1,
        worker_id=0,
        token_ids=list(range(CHUNK * NUM_CHUNKS)),
        start=0,
        end=CHUNK * NUM_CHUNKS,
        request_id="req",
        request_configs=configs or None,
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

    def store(self, exclusive: bool, held_block_ids: list[int] | None = None) -> bool:
        """Store the request's three chunks and run the host callbacks."""
        _, ok = self.module.store(
            _ipc_key(exclusive, held_block_ids), 1, RETRIEVE_BLOCKS, b"p"
        )
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
        session.set_tokens(key.token_bytes)
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


def test_exclusive_retrieve_touches_the_shared_held_chunks(server: _Server) -> None:
    """Loaded chunks below the shared-held hint go to the most recently used
    end instead of being demoted; the rest of the read keys are demoted and
    recorded as before."""
    full, _ = server.keys()
    loaded = _loaded(server)
    server.lock_for_read(loaded)
    key = _ipc_key(True)
    key = IPCCacheServerKey.from_token_ids(
        model_name=key.model_name,
        world_size=key.world_size,
        worker_id=key.worker_id,
        token_ids=unpack_token_ids(key.token_bytes),
        start=key.start,
        end=key.end,
        request_id=key.request_id,
        request_configs={**(key.request_configs or {}), SHARED_HELD_TOKENS_HINT: CHUNK},
    )

    _, ok = server.module.retrieve(key, 1, RETRIEVE_BLOCKS, b"p")
    server.drain_callbacks()

    assert ok
    demoted = [k for k in loaded if k != full[0]]
    victims = server.victims()
    assert victims[: len(demoted)] == demoted[::-1]
    assert victims[-1] == full[0]
    session = server.sessions.get("req")
    assert session is not None
    assert session.get_demoted_keys() == set(demoted)


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


def _store_event_types(srv: _Server) -> list[EventType]:
    """The MP_STORE_* event types the server published, in order."""
    bus = srv.ctx.event_bus
    events = [c.args[0] for c in bus.publish.call_args_list] + [
        c.args[1] for c in bus.publish_on_stream.call_args_list
    ]
    return [
        e.event_type
        for e in events
        if e.event_type
        in (
            EventType.MP_STORE_SUBMITTED,
            EventType.MP_STORE_START,
            EventType.MP_STORE_END,
        )
    ]


def test_store_of_resident_chunks_submits_no_device_work(server: _Server) -> None:
    """A store whose every chunk is resident touches them as before, but
    neither waits on the producer event, copies, nor publishes store events,
    and its completion handle is empty: nothing reads the blocks."""
    loaded = _loaded(server)
    server.lock_for_read(loaded)
    assert server.retrieve(exclusive=True)
    entry = server.module.get_and_touch_context_entry(1)
    entry.event_backend.reset_mock()
    entry.cache_context.reset_mock()
    server.ctx.event_bus.reset_mock()

    handle, ok = server.module.store(_ipc_key(True), 1, RETRIEVE_BLOCKS, b"p")
    server.drain_callbacks()

    assert ok
    assert handle == b""
    entry.event_backend.import_event.assert_not_called()
    entry.event_backend.wait_event.assert_not_called()
    entry.cache_context.transfer_object_group.assert_not_called()
    assert _store_event_types(server) == []
    assert server.victims()[-len(loaded) :] == loaded[::-1]


def test_store_of_resident_chunks_without_exclusive_retrieve_keeps_the_copy_path(
    server: _Server,
) -> None:
    """Without exclusive retrieve an all-resident store runs as any store:
    behind the producer event, with its store events and completion handle,
    copying nothing."""
    loaded = _loaded(server)
    server.lock_for_read(loaded)
    assert server.retrieve(exclusive=True)
    entry = server.module.get_and_touch_context_entry(1)
    entry.event_backend.reset_mock()
    server.ctx.event_bus.reset_mock()

    handle, ok = server.module.store(_ipc_key(False), 1, RETRIEVE_BLOCKS, b"p")
    server.drain_callbacks()

    assert ok
    assert handle == entry.event_backend.export_event.return_value
    entry.event_backend.wait_event.assert_called_once()
    assert _store_event_types(server) == [
        EventType.MP_STORE_SUBMITTED,
        EventType.MP_STORE_START,
        EventType.MP_STORE_END,
    ]


def test_store_missing_one_chunk_writes_it(server: _Server) -> None:
    """A store with one chunk evicted and the others resident writes that
    chunk behind the producer event and touches the resident ones."""
    full, _ = server.keys()
    loaded = _loaded(server)
    server.lock_for_read(loaded)
    assert server.retrieve(exclusive=True)
    server.storage.delete_l1_keys([full[1]], force=True)
    entry = server.module.get_and_touch_context_entry(1)
    entry.event_backend.reset_mock()
    entry.cache_context.reset_mock()
    server.ctx.event_bus.reset_mock()

    handle, ok = server.module.store(_ipc_key(True), 1, RETRIEVE_BLOCKS, b"p")
    server.drain_callbacks()

    assert ok
    assert handle == entry.event_backend.export_event.return_value
    entry.event_backend.wait_event.assert_called_once()
    written = [
        obj
        for call in entry.cache_context.transfer_object_group.call_args_list
        for obj in call.args[1]
        if obj is not None
    ]
    assert len(written) == 1
    assert full[1] in server.victims()
    assert _store_event_types(server) == [
        EventType.MP_STORE_SUBMITTED,
        EventType.MP_STORE_START,
        EventType.MP_STORE_END,
    ]
    # The resident keys were touched; the written one joined them at the MRU end.
    assert set(server.victims()[-len(loaded) :]) == set(loaded)


# The server's object groups as the transfer module registers them under
# --separate-object-groups (attention, then Mamba), and as one mixed group.
SEPARATE_GROUPS = AttnWindowDesc(
    num_chunks_in_sw=WINDOWS,
    group_kinds=("attention", "recurrent"),
    engine_held_windows=(-1, 0),
)


def _lookup_module(srv: _Server, attn_desc: AttnWindowDesc) -> LookupModule:
    """A lookup module on the server's L1, sessions and object groups."""
    registry = LayoutDescRegistry()
    registry.register("model", 1, LAYOUT, attn_desc)
    event_bus = MagicMock()
    event_bus.has_subscribers.return_value = False
    ctx = SimpleNamespace(
        chunk_size=CHUNK,
        storage_manager=srv.storage,
        session_manager=srv.sessions,
        token_hasher=TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3"),
        layout_desc_registry=registry,
        event_bus=event_bus,
        get_read_owners=lambda request_id: None,
    )
    return LookupModule(cast(Any, ctx))


def _scheduler_key(
    exclusive: bool, start: int, end: int, held_tokens: int = 0
) -> IPCCacheServerKey:
    """A LOOKUP / FREE_LOOKUP_LOCKS key as the scheduler sends it."""
    configs: dict[str, Any] = {}
    if exclusive:
        configs[EXCLUSIVE_RETRIEVE_REQUEST_CONFIG] = True
    if held_tokens:
        configs[ENGINE_HELD_TOKENS_HINT] = held_tokens
    return IPCCacheServerKey.from_token_ids(
        model_name="model",
        world_size=1,
        worker_id=None,
        token_ids=list(range(CHUNK * NUM_CHUNKS)),
        start=start,
        end=end,
        request_id="req",
        request_configs=configs or None,
    )


def _look_up(lookup: LookupModule, exclusive: bool, lookup_start: int) -> int:
    """Look up the prompt past the covered ``lookup_start`` and wait for it.

    Returns:
        The hit, in chunks (the covered prefix counts as hit).
    """
    lookup.lookup(_scheduler_key(exclusive, lookup_start, CHUNK * NUM_CHUNKS), 1)
    hit = lookup.wait_prefetch_status("req", 5.0)
    assert hit is not None
    return hit


def _free_after_alloc(
    lookup: LookupModule, exclusive: bool, free_end: int, held_tokens: int
) -> None:
    """The FREE_LOOKUP_LOCKS the connector sends once vLLM allocated the
    request's blocks; only an exclusive connector reports the held prefix."""
    lookup.free_lookup_locks(
        _scheduler_key(exclusive, 0, free_end, held_tokens if exclusive else 0), 1
    )


def _drop_last_chunk(srv: _Server) -> None:
    """Evict the request's last chunk, so a lookup past chunk 1 misses."""
    full, recurrent = srv.keys()
    srv.storage.delete_l1_keys([full[-1], recurrent[-1]], force=True)


def test_local_hit_demotes_the_held_prefix_instead_of_touching_it(
    server: _Server,
) -> None:
    """An exclusive local hit of every chunk demotes the keys vLLM holds:
    every attention chunk and the Mamba chunk at the hit's end. No lookup or
    ``end_session`` touch moves them back to most recently used."""
    lookup = _lookup_module(server, SEPARATE_GROUPS)
    full, recurrent = server.keys()
    server.commit([LATER_KEY])
    before = server.victims()

    assert _look_up(lookup, True, 12) == NUM_CHUNKS
    # The lookup itself leaves the eviction order alone.
    assert server.victims() == before
    _free_after_alloc(lookup, True, 12, held_tokens=12)

    held = [full[0], full[1], full[2], recurrent[2]]
    victims = server.victims()
    # The held prefix's last chunk is the very next victim, then backwards.
    assert victims[: len(held)] == held[::-1]
    # Mamba states vLLM no longer holds keep their L1 position.
    assert victims[len(held) :] == [k for k in before if k not in held]
    session = server.sessions.get("req")
    assert session is not None
    assert session.get_demoted_keys() == set(held)

    lookup.end_session("req")

    victims = server.victims()
    assert victims[: len(held)] == held[::-1]
    assert victims[len(held)] == OTHER_KEY


def test_local_hit_touches_the_shared_held_chunks(server: _Server) -> None:
    """Chunks of the held prefix below the shared-held hint are touched to
    most recently used instead of demoted; the rest of the held prefix is
    demoted."""
    lookup = _lookup_module(server, SEPARATE_GROUPS)
    full, recurrent = server.keys()
    assert _look_up(lookup, True, 12) == NUM_CHUNKS
    key = _scheduler_key(True, 0, 12, held_tokens=12)
    key = IPCCacheServerKey.from_token_ids(
        model_name=key.model_name,
        world_size=key.world_size,
        worker_id=None,
        token_ids=unpack_token_ids(key.token_bytes),
        start=key.start,
        end=key.end,
        request_id=key.request_id,
        request_configs={
            **(key.request_configs or {}),
            SHARED_HELD_TOKENS_HINT: 2 * CHUNK,
        },
    )

    lookup.free_lookup_locks(key, 1)

    demoted = [full[2], recurrent[2]]
    victims = server.victims()
    assert victims[: len(demoted)] == demoted[::-1]
    assert set(victims[-2:]) == {full[0], full[1]}
    session = server.sessions.get("req")
    assert session is not None
    assert session.get_demoted_keys() == set(demoted)


def test_plain_local_hit_touches_the_covered_prefix(server: _Server) -> None:
    """Without exclusive retrieve the covered prefix is touched and nothing
    is demoted, even if a held-prefix hint arrives."""
    lookup = _lookup_module(server, SEPARATE_GROUPS)
    full, recurrent = server.keys()
    server.commit([LATER_KEY])

    assert _look_up(lookup, False, 12) == NUM_CHUNKS
    # The covered prefix became most recently used.
    after_lookup = server.victims()
    assert after_lookup[:2] == [OTHER_KEY, LATER_KEY]
    assert set(after_lookup[2:]) == set(full + recurrent)
    lookup.free_lookup_locks(_scheduler_key(False, 0, 12, held_tokens=12), 1)

    assert server.victims() == after_lookup
    session = server.sessions.get("req")
    assert session is not None
    assert session.get_demoted_keys() == set()


@pytest.mark.parametrize(
    ("attn_desc", "held_tokens", "lookup_start", "held"),
    [
        # Separate object groups, a hit ending mid-chunk: attention holds
        # chunks 0 and 1; Mamba's state is no chunk's state.
        (SEPARATE_GROUPS, 10, 8, lambda full, rec: full[:2]),
        # Two chunks, chunk-aligned: Mamba holds chunk 1's end state.
        (SEPARATE_GROUPS, 8, 8, lambda full, rec: [full[0], full[1], rec[1]]),
        # A sliding-window attention group of one chunk holds the last one.
        (
            AttnWindowDesc(num_chunks_in_sw=[-1, 1], engine_held_windows=(-1, 1)),
            10,
            8,
            lambda full, rec: [full[0], full[1], rec[1]],
        ),
    ],
    ids=["mamba-mid-chunk", "mamba-at-chunk-end", "sliding-window"],
)
def test_local_hit_demotes_per_object_group(
    server: _Server,
    attn_desc: AttnWindowDesc,
    held_tokens: int,
    lookup_start: int,
    held: Any,
) -> None:
    """Each object group demotes only the chunks the engine holds in it."""
    lookup = _lookup_module(server, attn_desc)
    full, recurrent = server.keys()
    expected = held(full, recurrent)
    _drop_last_chunk(server)
    assert _look_up(lookup, True, lookup_start) == lookup_start // CHUNK
    before = server.victims()

    _free_after_alloc(lookup, True, lookup_start, held_tokens)

    victims = server.victims()
    assert victims[: len(expected)] == expected[::-1]
    assert victims[len(expected) :] == [k for k in before if k not in expected]


@pytest.mark.parametrize("held_tokens", [8, 12])
def test_mixed_object_group_demotes_only_fully_held_chunks(
    server: _Server, held_tokens: int
) -> None:
    """One mixed attention+Mamba object group: vLLM holds every slice only of
    the chunk ending at the hit, so only that key is demoted."""
    lookup = _lookup_module(
        server, AttnWindowDesc(num_chunks_in_sw=[-1], engine_held_windows=(0,))
    )
    full, _recurrent = server.keys()
    if held_tokens < CHUNK * NUM_CHUNKS:
        _drop_last_chunk(server)
    assert _look_up(lookup, True, held_tokens) == held_tokens // CHUNK
    before = server.victims()

    _free_after_alloc(lookup, True, held_tokens, held_tokens)

    last = full[held_tokens // CHUNK - 1]
    assert server.victims() == [last] + [k for k in before if k != last]


def test_local_hit_only_request_demotes_through_its_free(server: _Server) -> None:
    """A request LMCache serves nothing past vLLM's hit still sends its
    FREE_LOOKUP_LOCKS (the covered prefix counts as hit): the held prefix is
    demoted and no lock is released."""
    lookup = _lookup_module(server, SEPARATE_GROUPS)
    full, recurrent = server.keys()
    _drop_last_chunk(server)
    # vLLM holds 8 tokens; the lookup of [8, 12) misses.
    assert _look_up(lookup, True, 8) == 2
    # A concurrent request reading the shared prefix.
    server.lock_for_read([full[0]])
    before = server.victims()

    _free_after_alloc(lookup, True, 8, held_tokens=8)

    held = [full[0], full[1], recurrent[1]]
    victims = server.victims()
    assert victims[: len(held)] == held[::-1]
    assert victims[len(held) :] == [k for k in before if k not in held]
    # The covered prefix was never locked by this lookup: nothing released.
    assert not server.l1.is_key_evictable(full[0])


def test_restore_after_local_hit_demotion_promotes_resident_keys(
    server: _Server,
) -> None:
    """No drop: the lazy re-store of a demoted local-hit prefix finds its L1
    copies resident, writes nothing and touches them back to most recently
    used."""
    lookup = _lookup_module(server, SEPARATE_GROUPS)
    assert _look_up(lookup, True, 12) == NUM_CHUNKS
    _free_after_alloc(lookup, True, 12, held_tokens=12)
    held = server.victims()[:4]
    used_before = server.storage.get_l1_usage()[0]

    assert server.store(exclusive=True)

    assert server.storage.get_l1_usage()[0] == used_before
    assert set(server.victims()[-len(held) :]) == set(held)


def _reset(srv: _Server) -> None:
    """Back to the fixture's state: only the request's chunks and one
    unrelated key in L1, no lock and no session."""
    srv.storage.clear(force=True)
    srv.sessions.remove("req")
    full, recurrent = srv.keys()
    srv.commit([OTHER_KEY])
    srv.commit(full + recurrent)


def _l1_state(
    srv: _Server,
) -> tuple[list[ObjectKey], list[bool], list[bool], set[ObjectKey]]:
    """Eviction order, the recorded demotions, and which keys hold at least
    one and at least two read locks (the second probed by releasing one;
    this ends the server's use)."""
    victims = srv.victims()
    session = srv.sessions.get("req")
    assert session is not None
    locked = [not srv.l1.is_key_evictable(k) for k in victims]
    srv.l1.finish_read(
        [k for k, is_locked in zip(victims, locked, strict=True) if is_locked]
    )
    locked_twice = [not srv.l1.is_key_evictable(k) for k in victims]
    return victims, locked, locked_twice, session.get_demoted_keys()


def test_held_hint_free_equals_a_plain_free_and_a_demotion(
    server: _Server, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Randomized, fixed seed: an exclusive FREE_LOOKUP_LOCKS with a held
    hint leaves L1 exactly as a plain FREE of the same ranges followed by a
    demotion-only FREE does. Lookup starts and held lengths are block-, not
    chunk-aligned, and an abandoned lookup can leave an extra range past the
    FREE's end, so every release range resolves against the shared hash."""
    split_server = _Server(monkeypatch)
    rng = random.Random(0)
    total = CHUNK * NUM_CHUNKS
    try:
        for _ in range(100):
            covered = rng.randrange(0, total + 1, 2)
            relookup = rng.random() < 0.5
            covered_again = rng.randrange(0, covered + 1, 2)
            held = rng.randrange(0, total + 1, 2)
            free_end = rng.choice([held, total, rng.randrange(0, total + 1)])
            states = []
            for srv in (server, split_server):
                _reset(srv)
                lookup = _lookup_module(srv, SEPARATE_GROUPS)
                hit = _look_up(lookup, True, covered)
                extra: tuple[tuple[int, int], ...] = ()
                if relookup:
                    # The first lookup is abandoned with its locks held.
                    extra = ((covered, hit * CHUNK),)
                    _look_up(lookup, True, covered_again)
                if srv is split_server:
                    lookup.free_lookup_locks(
                        _scheduler_key(False, 0, free_end), 1, extra
                    )
                    lookup.free_lookup_locks(_scheduler_key(True, 0, 0, held), 1)
                else:
                    lookup.free_lookup_locks(
                        _scheduler_key(True, 0, free_end, held), 1, extra
                    )
                states.append(_l1_state(srv))
            assert states[0] == states[1], (covered, relookup, held, free_end)
    finally:
        split_server.close()


@pytest.mark.parametrize(
    ("attn_desc", "mid_chunk"),
    [
        # Without engine-held windows the kinds and windows decide.
        (AttnWindowDesc(num_chunks_in_sw=[-1, 1]), range(1, 2)),
        (
            AttnWindowDesc(num_chunks_in_sw=[-1, 1], group_kinds=WINDOW_KINDS),
            range(0),
        ),
        (SEPARATE_GROUPS, range(0)),
    ],
    ids=["window", "recurrent-kind", "engine-held-windows"],
)
def test_engine_held_chunks(attn_desc: AttnWindowDesc, mid_chunk: range) -> None:
    """Full attention holds every chunk of the prefix; a one-chunk window and
    a recurrent state hold its last chunk. Mid-chunk, a recurrent state is no
    chunk's state, while a window still holds the last full chunk."""
    assert attn_desc.engine_held_chunks(0, 12, CHUNK) == range(3)
    assert attn_desc.engine_held_chunks(1, 12, CHUNK) == range(2, 3)
    assert attn_desc.engine_held_chunks(1, 11, CHUNK) == mid_chunk


def _demoted(srv: _Server, keys: list[ObjectKey]) -> list[ObjectKey]:
    """The keys a demote put ahead of ``OTHER_KEY``, which nothing touches."""
    victims = srv.victims()
    ahead = set(victims[: victims.index(OTHER_KEY)])
    return [k for k in keys if k in ahead]


def test_store_of_held_resident_chunks_demotes_them(server: _Server) -> None:
    """Resident chunks the engine still holds are demoted, last chunk first;
    the store's other resident chunks are touched to the MRU end."""
    full, recurrent = server.keys()

    # Blocks 1 and 2 are full-attention chunks 0 and 1.
    assert server.store(exclusive=True, held_block_ids=[1, 2])

    victims = server.victims()
    assert victims[:2] == [full[1], full[0]]
    assert set(victims[-2:]) == {full[2], recurrent[2]}
    assert set(_demoted(server, full + recurrent)) == {full[0], full[1]}


def test_store_of_held_new_chunks_commits_them_demoted(server: _Server) -> None:
    """New keys of held chunks are admitted demoted, although the demote is
    issued before their write finishes."""
    full, recurrent = server.keys()
    server.storage.delete_l1_keys(full + recurrent)

    # Block 5 is the recurrent state of chunk 2.
    assert server.store(exclusive=True, held_block_ids=[1, 2, 3, 5])

    stored = full + recurrent[2:]
    assert set(stored) <= set(server.victims())
    assert set(_demoted(server, stored)) == set(stored)
    assert set(server.victims()[:4]) == set(stored)


@pytest.mark.parametrize(
    ("held_block_ids", "held_chunk_2"),
    [
        # Chunk 2's attention is held, its own Mamba state is not.
        ([1, 2, 3], "full"),
        ([1, 2, 5], "recurrent"),
    ],
    ids=["attention_only", "state_only"],
)
def test_chunk_is_demoted_per_object_group(
    server: _Server, held_block_ids: list[int], held_chunk_2: str
) -> None:
    """Each object group of a chunk is demoted when the hint lists its
    blocks; the chunk's other object group is written MRU."""
    full, recurrent = server.keys()
    server.storage.delete_l1_keys(full + recurrent)

    assert server.store(exclusive=True, held_block_ids=held_block_ids)

    held, unheld = (
        (full[2], recurrent[2]) if held_chunk_2 == "full" else (recurrent[2], full[2])
    )
    assert set(_demoted(server, full + recurrent[2:])) == {full[0], full[1], held}
    assert server.victims()[-1] == unheld


def _mix_object_groups(srv: _Server) -> list[ObjectKey]:
    """Make the server's two kernel groups one mixed object group.

    Returns:
        The request's mixed keys, none of them resident.
    """
    full, recurrent = srv.keys()
    srv.storage.delete_l1_keys(full + recurrent)
    cache_context = srv.module.get_and_touch_context_entry(1).cache_context
    cache_context.kv_layer_groups_manager = SimpleNamespace(
        num_object_groups=1,
        num_kernel_groups=2,
        object_groups=[ObjectGroupInfo(kernel_group_indices=[0, 1])],
    )
    return full


@pytest.mark.parametrize(
    ("held_block_ids", "demoted"),
    [([1, 2, 3], False), ([1, 2, 3, 5], True)],
    ids=["state_free", "state_held"],
)
def test_mixed_chunk_is_demoted_only_with_its_state(
    server: _Server, held_block_ids: list[int], demoted: bool
) -> None:
    """One mixed object group is demoted only when the hint lists its Mamba
    state too. A re-store of the attention alone, the state nulled, skips
    the chunk and leaves its copy resident: nothing is lost."""
    mixed = _mix_object_groups(server)

    assert server.store(exclusive=True, held_block_ids=held_block_ids)
    assert mixed[2] in server.victims()
    assert _demoted(server, mixed) == ([mixed[2]] if demoted else [])
    used = server.storage.get_l1_usage()[0]

    _, ok = server.module.store(_ipc_key(True), 1, [[1, 2, 3], [-1, -1, -1]], b"p")
    server.drain_callbacks()

    assert ok
    assert server.storage.get_l1_usage()[0] == used
    assert mixed[2] in server.victims()


def test_held_hint_without_exclusive_retrieve_changes_nothing(
    server: _Server,
) -> None:
    """A plain store ignores the hint: resident keys keep their order and new
    keys are written MRU."""
    full, recurrent = server.keys()
    before = server.victims()

    assert server.store(exclusive=False, held_block_ids=[1, 2, 3, 5])

    assert server.victims() == before
    server.storage.delete_l1_keys(full + recurrent)
    assert server.store(exclusive=False, held_block_ids=[1, 2, 3, 5])
    assert _demoted(server, full + recurrent) == []


def test_partly_held_chunk_is_not_demoted(server: _Server) -> None:
    """A hint that lists only some blocks of a chunk leaves it unheld."""
    full, recurrent = server.keys()
    server.storage.delete_l1_keys(full + recurrent)

    # A chunk of two blocks per kernel group, only the first one held.
    blocks = [[1, 2, 3, 4, 5, 6], [-1, -1, -1, -1, 7, 8]]
    cache_context = server.module.get_and_touch_context_entry(1).cache_context
    cache_context.calculate_num_blocks.return_value = 2
    _, ok = server.module.store(_ipc_key(True, [1, 2, 3, 5, 7]), 1, blocks, b"p")
    server.drain_callbacks()

    assert ok
    assert set(_demoted(server, full + recurrent)) == {full[0]}


def test_exclusive_end_session_touches_no_l1_key(server: _Server) -> None:
    """Under exclusive retrieve ``end_session`` leaves the L1 order as is: a
    chunk that is not demoted was made MRU by its own store or lookup."""
    session = server.sessions.get("req")
    assert session is not None
    session.lookup_ipc_key = _ipc_key(True).no_worker_id_version()
    server.commit([LATER_KEY])
    before = server.victims()
    lookup_ctx = MagicMock()
    lookup_ctx.session_manager = server.sessions
    lookup_ctx.storage_manager = server.storage
    lookup_ctx.layout_desc_registry.find_attn_desc.return_value = SimpleNamespace(
        num_object_groups=2
    )

    LookupModule(lookup_ctx).end_session("req")

    assert server.victims() == before


@pytest.mark.parametrize(
    ("mode", "advertised"),
    [("lmcache_driven", True), ("auto", True), ("engine_driven", False)],
)
def test_server_advertises_exclusive_retrieve_with_lmcache_driven_transfer(
    stub_module_resources: Any, mode: str, advertised: bool
) -> None:
    """GET_EXPERIMENTAL names exclusive retrieve exactly when the server
    serves LMCache-driven retrieves, whose demotion implements it."""
    modules = server_mod._build_modules(
        MagicMock(name="ctx"),
        MPServerConfig(worker_reap_timeout_seconds=0, supported_transfer_mode=mode),
        CoordinatorConfig(),
    )
    management = next(m for m in modules if isinstance(m, ManagementModule))

    assert (EXCLUSIVE_RETRIEVE in management.get_experimental()) is advertised
