# SPDX-License-Identifier: Apache-2.0
"""Server side of exclusive retrieve, on CPU against a real L1 and LRU policy.

- A successful retrieve of an exclusive request demotes exactly the keys it
  read in L1 (stream-ordered, after the read locks are released); a plain
  request or a failed retrieve demotes nothing.
- ``end_session`` does not touch the demoted keys back to most recently used.
- A store of an exclusive request touches the keys it finds already resident.
- An exclusive request's lookup does not touch the prefix vLLM's prefix cache
  holds; its FREE_LOOKUP_LOCKS demotes the keys of that prefix the engine
  holds in every layer, per object group, and ``end_session`` keeps them.
- Those local-hit demotions count as demoted in the L1 committed usage.
- A store's chunks whose blocks ``ENGINE_HELD_BLOCKS_HINT`` lists in every
  object group (the engine can serve and still holds them) are committed or
  touched demoted, a new key too when its write finishes after the demote;
  only under exclusive retrieve.
- Under exclusive retrieve ``end_session`` touches no L1 key.

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
from lmcache.v1.multiprocess import native_completion
from lmcache.v1.multiprocess.custom_types import (
    ENGINE_HELD_BLOCKS_HINT,
    ENGINE_HELD_TOKENS_HINT,
    EXCLUSIVE_RETRIEVE_REQUEST_CONFIG,
    IPCCacheServerKey,
)
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
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
    """The keys the L1 committed usage reports as demoted."""
    return [
        k
        for k in keys
        if srv.storage.get_l1_committed_usage([k], "model").matched_demoted_objects
    ]


def test_local_hit_demotion_shows_as_demoted_in_committed_usage(
    server: _Server,
) -> None:
    """The keys an exclusive local hit demotes count as demoted in the L1
    committed usage, through ``end_session``; the rest keep their MRU cause,
    and a plain local hit records its touch as ``lookup_touch``."""
    lookup = _lookup_module(server, SEPARATE_GROUPS)
    full, recurrent = server.keys()
    keys = full + recurrent
    assert _demoted(server, keys) == []

    assert _look_up(lookup, True, 12) == NUM_CHUNKS
    _free_after_alloc(lookup, True, 12, held_tokens=12)
    held = [full[0], full[1], full[2], recurrent[2]]
    assert set(_demoted(server, keys)) == set(held)
    usage = server.storage.get_l1_committed_usage(keys, "model")
    assert usage.matched_demoted_objects == len(held)
    assert usage.matched_mru_objects["write"] == len(keys) - len(held)

    lookup.end_session("req")
    assert set(_demoted(server, keys)) == set(held)

    assert _look_up(lookup, False, 12) == NUM_CHUNKS
    usage = server.storage.get_l1_committed_usage(keys, "model")
    assert usage.matched_demoted_objects == 0
    assert usage.matched_mru_objects["lookup_touch"] == len(keys)


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
    assert server.storage.get_l1_committed_usage(stored, "model").matched_objects == 4
    assert set(_demoted(server, stored)) == set(stored)
    assert set(server.victims()[:4]) == set(stored)


def test_chunk_held_in_one_object_group_only_is_not_demoted(
    server: _Server,
) -> None:
    """A chunk is held as a whole: a hint listing chunk 2's recurrent state
    but not its attention block demotes neither; both are written MRU."""
    full, recurrent = server.keys()
    server.storage.delete_l1_keys(full + recurrent)

    assert server.store(exclusive=True, held_block_ids=[1, 2, 5])

    assert set(_demoted(server, full + recurrent[2:])) == {full[0], full[1]}
    victims = server.victims()
    assert set(victims[:2]) == {full[0], full[1]}
    assert set(victims[-2:]) == {full[2], recurrent[2]}


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
