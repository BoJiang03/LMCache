# SPDX-License-Identifier: Apache-2.0
"""Tests for the ``l1_residency.query`` server module."""

# Standard
from collections.abc import Iterator
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock

# Third Party
import msgspec
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import AttnWindowDesc, MemoryLayoutDesc
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from lmcache.v1.multiprocess.engine_context import LayoutDescRegistry
from lmcache.v1.multiprocess.ext_server_module import (
    ExtServerModuleSpec,
    build_server_module_router,
    load_server_module_classes,
)
from lmcache.v1.multiprocess.l1_residency import L1ResidencyModule
from lmcache.v1.multiprocess.protocols.l1_residency import (
    L1_RESIDENCY_QUERY_METHOD,
    GroupHeldChunks,
    GroupWindow,
    L1ResidencyQuery,
    L1ResidencyReply,
    compute_probe_hash,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from tests.v1.distributed.utils import (
    OBJECT_BYTES,
    commit,
    cpu_storage,
    object_key,
    stage,
)

pytestmark = pytest.mark.no_shared_allocator

# A hybrid model's object groups under --separate-object-groups: full
# attention (id 0) and one-chunk recurrent state (id 1).
ATTENTION = GroupWindow(kind="attention", window_chunks=-1)
STATE = GroupWindow(kind="recurrent", window_chunks=1)
HYBRID = AttnWindowDesc(
    num_chunks_in_sw=[-1, 1], group_kinds=("attention", "recurrent")
)


@pytest.fixture
def storage(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[L1Manager, StorageManager]]:
    l1, store = cpu_storage(monkeypatch)
    yield l1, store
    store.close()


def _module(
    store: StorageManager, hasher: TokenHasher | None = None
) -> L1ResidencyModule:
    ctx = SimpleNamespace(
        storage_manager=store,
        token_hasher=hasher or TokenHasher(),
        separate_object_groups=False,
    )
    return L1ResidencyModule(cast(Any, ctx))


def _separated_module(
    store: StorageManager, attn_desc: AttnWindowDesc | None = HYBRID
) -> L1ResidencyModule:
    """A module on a server that separates object groups.

    ``attn_desc`` is registered for model ``m`` at world size 1; ``None``
    registers nothing.
    """
    registry = LayoutDescRegistry()
    if attn_desc is not None:
        layout = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])
        registry.register("m", 1, layout, attn_desc=attn_desc)
    ctx = SimpleNamespace(
        storage_manager=store,
        token_hasher=TokenHasher(),
        separate_object_groups=True,
        full_sw_kv=False,
        layout_desc_registry=registry,
    )
    return L1ResidencyModule(cast(Any, ctx))


def _query(module: L1ResidencyModule, query: L1ResidencyQuery) -> L1ResidencyReply:
    reply = module.query(msgspec.msgpack.encode(query))
    return msgspec.msgpack.decode(reply, type=L1ResidencyReply)


def test_module_loads_as_a_server_module_plugin() -> None:
    specs = [ExtServerModuleSpec(module_path="lmcache.v1.multiprocess.l1_residency")]
    assert load_server_module_classes(specs) == [(L1ResidencyModule, {})]

    ctx = MagicMock(separate_object_groups=False)
    router = build_server_module_router(MagicMock(), [L1ResidencyModule(ctx)])
    assert router is not None
    assert router.report_status()["server_module_plugins"]["methods"] == [
        L1_RESIDENCY_QUERY_METHOD
    ]


def test_rejects_separate_object_groups_with_full_sw_kv() -> None:
    with pytest.raises(ValueError, match="full_sw_kv"):
        L1ResidencyModule(MagicMock(separate_object_groups=True, full_sw_kv=True))


def test_separated_object_groups_count_each_group_its_own_chunks(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    for chunk_hash in (b"a", b"b"):
        commit(
            l1,
            object_key(chunk_hash, object_group_id=0),
            object_key(chunk_hash, object_group_id=1),
        )

    reply = _query(
        _separated_module(store),
        L1ResidencyQuery(
            model_name="m",
            world_size=1,
            # Ignored with several object groups.
            chunk_hashes_by_salt={"": [b"a", b"b"]},
            chunk_hashes_by_group=[
                GroupHeldChunks(window=ATTENTION, chunk_hashes_by_salt={"": [b"a"]}),
                GroupHeldChunks(window=STATE, chunk_hashes_by_salt={"": [b"a", b"b"]}),
            ],
        ),
    )

    assert reply.object_groups == [ATTENTION, STATE]
    assert (reply.matched_objects, reply.matched_bytes) == (3, 3 * OBJECT_BYTES)
    assert (reply.model_objects, reply.model_bytes) == (4, 4 * OBJECT_BYTES)


def test_group_the_server_does_not_have_counts_nothing(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, object_key(b"a", object_group_id=1))
    other = GroupWindow(kind="recurrent", window_chunks=2)

    reply = _query(
        _separated_module(store),
        L1ResidencyQuery(
            model_name="m",
            world_size=1,
            chunk_hashes_by_salt={},
            chunk_hashes_by_group=[
                GroupHeldChunks(window=other, chunk_hashes_by_salt={"": [b"a"]})
            ],
        ),
    )

    assert reply.object_groups == [ATTENTION, STATE]
    assert (reply.matched_objects, reply.model_objects) == (0, 1)


def test_one_object_group_counts_chunk_hashes_by_salt(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, object_key(b"a"), object_key(b"b"))
    query = L1ResidencyQuery(
        model_name="m",
        world_size=1,
        chunk_hashes_by_salt={"": [b"a"]},
        chunk_hashes_by_group=[
            GroupHeldChunks(window=ATTENTION, chunk_hashes_by_salt={"": [b"b"]})
        ],
    )

    # Without separation, and with separation on a model of one object group.
    for module in (
        _module(store),
        _separated_module(store, AttnWindowDesc(num_chunks_in_sw=[-1])),
    ):
        reply = _query(module, query)
        assert len(reply.object_groups) == 1
        assert (reply.matched_objects, reply.model_objects) == (1, 2)


def test_separated_object_groups_need_a_registered_kv_cache(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    query = L1ResidencyQuery(model_name="m", world_size=1, chunk_hashes_by_salt={})
    with pytest.raises(ValueError, match="No attention-window descriptor"):
        _separated_module(storage[1], None).query(msgspec.msgpack.encode(query))


def test_counts_only_committed_objects_of_the_model(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, object_key(b"a"), object_key(b"b"))
    commit(l1, object_key(b"c"), temporary=True)
    stage(l1, object_key(b"d"))
    commit(l1, object_key(b"a", model_name="other"))

    reply = _query(
        _module(store),
        L1ResidencyQuery(
            model_name="m",
            world_size=1,
            chunk_hashes_by_salt={"": [b"a", b"c", b"d", b"e"]},
        ),
    )

    assert (reply.matched_objects, reply.matched_bytes) == (1, OBJECT_BYTES)
    assert (reply.model_objects, reply.model_bytes) == (2, 2 * OBJECT_BYTES)


def test_keys_fan_out_to_every_rank_and_keep_their_salt(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(
        l1,
        object_key(b"a", rank=0, world_size=2),
        object_key(b"a", rank=1, world_size=2),
    )
    commit(l1, object_key(b"b", world_size=2, cache_salt="u"))

    reply = _query(
        _module(store),
        L1ResidencyQuery(
            model_name="m",
            world_size=2,
            chunk_hashes_by_salt={"": [b"a", b"b"], "u": [b"a"]},
        ),
    )

    # b's only object is salted "u", and "u" was asked about a, not b.
    assert (reply.matched_objects, reply.model_objects) == (2, 3)


def test_reply_carries_the_server_probe_hash(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    _, store = storage
    hasher = TokenHasher(hash_algorithm="sha256_cbor")
    reply = _query(
        _module(store, hasher),
        L1ResidencyQuery(model_name="m", world_size=1, chunk_hashes_by_salt={}),
    )
    assert reply.probe_hash == compute_probe_hash(hasher)
    assert reply.probe_hash != compute_probe_hash(TokenHasher())
    assert (reply.matched_objects, reply.model_objects) == (0, 0)


def test_inuse_chunks_count_again_as_matched_inuse(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    for chunk_hash in (b"a", b"b"):
        commit(
            l1,
            object_key(chunk_hash, object_group_id=0),
            object_key(chunk_hash, object_group_id=1),
        )

    merged = _query(
        _module(store),
        L1ResidencyQuery(
            model_name="m",
            world_size=1,
            chunk_hashes_by_salt={"": [b"a", b"b"]},
            inuse_chunk_hashes_by_salt={"": [b"b"]},
        ),
    )
    separated = _query(
        _separated_module(store),
        L1ResidencyQuery(
            model_name="m",
            world_size=1,
            chunk_hashes_by_salt={},
            chunk_hashes_by_group=[
                GroupHeldChunks(window=ATTENTION, chunk_hashes_by_salt={"": [b"a"]}),
                GroupHeldChunks(window=STATE, chunk_hashes_by_salt={"": [b"a", b"b"]}),
            ],
            inuse_chunk_hashes_by_group=[
                GroupHeldChunks(window=ATTENTION, chunk_hashes_by_salt={"": [b"a"]}),
                GroupHeldChunks(window=STATE, chunk_hashes_by_salt={"": [b"b"]}),
            ],
        ),
    )

    assert (merged.matched_objects, merged.matched_inuse_objects) == (2, 1)
    assert merged.matched_inuse_bytes == OBJECT_BYTES
    assert (separated.matched_objects, separated.matched_inuse_objects) == (3, 2)
    assert separated.matched_inuse_bytes == 2 * OBJECT_BYTES


def test_reply_splits_matched_by_demoted_state_and_last_mru_cause(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    keys = {h: object_key(h) for h in (b"a", b"b", b"c", b"d")}
    commit(l1, *keys.values())
    store.demote_l1_keys([(l1.l1_manager_id, [keys[b"a"], keys[b"b"]])])
    store.touch_cached_keys([keys[b"b"]], "lazy_touch")
    store.touch_cached_keys([keys[b"c"]], "lookup_touch")

    reply = _query(
        _module(store),
        L1ResidencyQuery(
            model_name="m",
            world_size=1,
            chunk_hashes_by_salt={"": list(keys)},
        ),
    )

    assert reply.matched_objects == 4
    assert (reply.matched_demoted_objects, reply.matched_demoted_bytes) == (
        1,
        OBJECT_BYTES,
    )
    assert reply.matched_mru_objects == {
        "write": 1,
        "lazy_touch": 1,
        "lookup_touch": 1,
        "end_session_touch": 0,
        "other": 0,
    }
    assert reply.matched_demoted_objects + sum(reply.matched_mru_objects.values()) == 4
    assert reply.matched_demoted_bytes + sum(reply.matched_mru_bytes.values()) == (
        reply.matched_bytes
    )
