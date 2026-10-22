# SPDX-License-Identifier: Apache-2.0
"""Demotion contract: demoted keys stay resident and become the next victims.

Covers each eviction policy's ``on_keys_demoted``, the L1 listener bridge,
``L1Manager.demote_keys`` and ``StorageManager.demote_l1_keys``.
"""

# Standard
from collections.abc import Iterator

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
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.eviction import EvictionPolicy, L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.distributed.eviction_policy.arc import ARCEvictionPolicy
from lmcache.v1.distributed.eviction_policy.isolated_lru import (
    IsolatedLRUEvictionPolicy,
)
from lmcache.v1.distributed.eviction_policy.noop import NoOpEvictionPolicy
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.l2_adapters.config import L2AdaptersConfig
from lmcache.v1.distributed.storage_manager import StorageManager
import lmcache.v1.memory_management as memory_management

pytestmark = pytest.mark.no_shared_allocator

ALIGN = 4096
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])  # 4 KiB


def _key(index: int, cache_salt: str = "") -> ObjectKey:
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(index),
        model_name="m",
        kv_rank=0,
        cache_salt=cache_salt,
    )


def _victims(policy: EvictionPolicy, cache_salt: str | None = None) -> list[int]:
    """Every tracked key in eviction order, as chunk indices."""
    actions = policy.get_eviction_actions(1.0, cache_salt=cache_salt)
    return [ObjectKey.Bytes2IntHash(k.chunk_hash) for a in actions for k in a.keys]


@pytest.fixture
def cpu_memory(monkeypatch: pytest.MonkeyPatch) -> None:
    """Back L1 with plain CPU tensors so no pinned-memory runtime is needed."""
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)


def _l1(tag: str = "l1") -> L1Manager:
    return L1Manager(
        L1ManagerConfig(
            L1MemoryManagerConfig(
                size_in_bytes=32 * ALIGN, use_lazy=False, shm_name=""
            ),
            tag=tag,
        )
    )


def _commit(l1: L1Manager, keys: list[ObjectKey]) -> None:
    results = l1.reserve_write(keys, [False] * len(keys), LAYOUT)
    assert all(err == L1Error.SUCCESS for err, _ in results.values())
    assert all(err == L1Error.SUCCESS for err in l1.finish_write(keys).values())


class TestPolicyDemotion:
    def test_lru_demotes_request_with_its_last_key_first(self) -> None:
        policy = LRUEvictionPolicy()
        policy.on_keys_created([_key(i) for i in range(1, 6)])
        assert _victims(policy) == [5, 4, 3, 2, 1]

        policy.on_keys_demoted([_key(1), _key(2)])

        # Demoted keys go first, the request's suffix (2) before its prefix.
        assert _victims(policy) == [2, 1, 5, 4, 3]
        assert policy.get_num_tracked_keys() == 5

    def test_lru_demote_ignores_untracked_keys(self) -> None:
        policy = LRUEvictionPolicy()
        policy.on_keys_created([_key(1), _key(2)])
        policy.on_keys_demoted([_key(9), _key(1)])
        assert _victims(policy) == [1, 2]
        assert policy.get_num_tracked_keys() == 2

    def test_touch_after_demote_promotes_again(self) -> None:
        policy = LRUEvictionPolicy()
        policy.on_keys_created([_key(1), _key(2), _key(3)])
        policy.on_keys_demoted([_key(1)])
        policy.on_keys_touched([_key(1)])
        assert _victims(policy) == [3, 2, 1]

    def test_isolated_lru_demotes_within_each_salt(self) -> None:
        policy = IsolatedLRUEvictionPolicy()
        policy.on_keys_created([_key(1, "a"), _key(2, "a"), _key(3, "b")])
        policy.on_keys_demoted([_key(1, "a"), _key(3, "b")])
        assert _victims(policy, "a") == [1, 2]
        assert _victims(policy, "b") == [3]

    def test_arc_demotes_frequent_key_to_next_victim(self) -> None:
        policy = ARCEvictionPolicy()
        policy.on_keys_created([_key(1), _key(2), _key(3)])
        policy.on_keys_touched([_key(1)])  # key 1 becomes frequent (T2)

        policy.on_keys_demoted([_key(1)])

        actions = policy.get_eviction_actions(0.0001)
        assert [ObjectKey.Bytes2IntHash(k.chunk_hash) for k in actions[0].keys] == [1]
        assert policy.get_num_tracked_keys() == 3

    def test_noop_and_default_policies_ignore_demotion(self) -> None:
        policy = NoOpEvictionPolicy()
        policy.on_keys_demoted([_key(1)])
        assert policy.get_eviction_actions(1.0) == []


class TestL1Demotion:
    def test_listener_bridge_forwards_demotion(self) -> None:
        lru = LRUEvictionPolicy()
        listener = L1EvictionPolicy(lru)
        lru.on_keys_created([_key(1), _key(2)])
        assert _victims(lru) == [2, 1]
        listener.on_l1_keys_demoted([_key(1)])
        assert _victims(lru) == [1, 2]

    @pytest.mark.usefixtures("cpu_memory")
    def test_l1_manager_demote_keeps_keys_readable(self) -> None:
        l1 = _l1()
        lru = LRUEvictionPolicy()
        l1.register_listener(L1EvictionPolicy(lru))
        try:
            keys = [_key(i) for i in range(1, 4)]
            _commit(l1, keys)
            assert _victims(lru) == [3, 2, 1]

            l1.demote_keys([_key(1)])

            assert _victims(lru) == [1, 3, 2]
            read = l1.reserve_read([_key(1)])
            assert read[_key(1)][0] == L1Error.SUCCESS
            l1.finish_read([_key(1)])
        finally:
            l1.close()


@pytest.fixture
def two_l1s(
    cpu_memory: None,
) -> Iterator[tuple[StorageManager, list[L1Manager], list[LRUEvictionPolicy]]]:
    """A storage manager over two L1s, each with an observable LRU listener."""
    managers = [_l1("l1-0"), _l1("l1-1")]
    policies = [LRUEvictionPolicy() for _ in managers]
    for manager, policy in zip(managers, policies, strict=True):
        manager.register_listener(L1EvictionPolicy(policy))
    storage = StorageManager(
        StorageManagerConfig(
            l1_manager_config=managers[0].config,
            eviction_config=EvictionConfig(eviction_policy="noop"),
            l2_adapter_config=L2AdaptersConfig(adapters=[]),
        ),
        _l1_managers=tuple(managers),
    )
    yield storage, managers, policies
    storage.close()


def test_storage_manager_demotes_each_key_on_its_owner(
    two_l1s: tuple[StorageManager, list[L1Manager], list[LRUEvictionPolicy]],
) -> None:
    storage, managers, policies = two_l1s
    _commit(managers[0], [_key(1), _key(2)])
    _commit(managers[1], [_key(3), _key(4)])

    storage.demote_l1_keys(
        [(managers[0].l1_manager_id, [_key(1)]), (managers[1].l1_manager_id, [_key(3)])]
    )

    assert _victims(policies[0]) == [1, 2]
    assert _victims(policies[1]) == [3, 4]
    # Nothing was deleted.
    assert storage.get_l1_usage()[0] == 4 * ALIGN


def test_storage_manager_demote_rejects_unknown_owner(
    two_l1s: tuple[StorageManager, list[L1Manager], list[LRUEvictionPolicy]],
) -> None:
    storage, _managers, _policies = two_l1s
    with pytest.raises(ValueError, match="registered L1 owner"):
        storage.demote_l1_keys([(-1, [_key(1)])])
