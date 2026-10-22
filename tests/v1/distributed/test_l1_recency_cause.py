# SPDX-License-Identifier: Apache-2.0
"""Per-key demoted state and last MRU cause in L1 committed usage.

``get_committed_usage`` splits the matched keys into demoted ones and, for the
rest, the ``L1RecencyCause`` that last made each one MRU.
"""

# Standard
from collections.abc import Iterator
import threading

# Third Party
import pytest

# First Party
from lmcache.v1.distributed.api import ObjectKey
from lmcache.v1.distributed.eviction import L1EvictionPolicy
from lmcache.v1.distributed.eviction_policy import LRUEvictionPolicy
from lmcache.v1.distributed.internal_api import (
    L1_RECENCY_CAUSES,
    L1CommittedUsage,
    L1RecencyCause,
)
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import (
    OBJECT_BYTES,
    commit,
    cpu_storage,
    object_key,
    stage,
)

pytestmark = pytest.mark.no_shared_allocator

A, B, C = (object_key(h) for h in (b"a", b"b", b"c"))


@pytest.fixture
def storage(
    monkeypatch: pytest.MonkeyPatch,
) -> Iterator[tuple[L1Manager, StorageManager]]:
    l1, store = cpu_storage(monkeypatch)
    yield l1, store
    store.close()


def _state(usage: L1CommittedUsage) -> dict[str, int]:
    """Matched objects per category, dropping empty ones."""
    counts: dict[str, int] = {"demoted": usage.matched_demoted_objects}
    counts.update(usage.matched_mru_objects)
    return {name: n for name, n in counts.items() if n}


def _assert_sums_to_matched(usage: L1CommittedUsage) -> None:
    assert set(usage.matched_mru_objects) == set(L1_RECENCY_CAUSES)
    assert set(usage.matched_mru_bytes) == set(L1_RECENCY_CAUSES)
    assert (
        usage.matched_demoted_objects + sum(usage.matched_mru_objects.values())
        == usage.matched_objects
    )
    assert (
        usage.matched_demoted_bytes + sum(usage.matched_mru_bytes.values())
        == usage.matched_bytes
    )


def _usage(store: StorageManager, *keys: ObjectKey) -> L1CommittedUsage:
    usage = store.get_l1_committed_usage(list(keys), "m")
    _assert_sums_to_matched(usage)
    return usage


def test_store_commit_is_write(storage: tuple[L1Manager, StorageManager]) -> None:
    l1, store = storage
    commit(l1, A, B)
    usage = _usage(store, A, B)
    assert _state(usage) == {"write": 2}
    assert usage.matched_mru_bytes["write"] == 2 * OBJECT_BYTES


def test_demote_marks_keys_demoted(storage: tuple[L1Manager, StorageManager]) -> None:
    l1, store = storage
    commit(l1, A, B)
    store.demote_l1_keys([(l1.l1_manager_id, [A])])
    usage = _usage(store, A, B)
    assert _state(usage) == {"demoted": 1, "write": 1}
    assert usage.matched_demoted_bytes == OBJECT_BYTES


@pytest.mark.parametrize("cause", ["lazy_touch", "lookup_touch", "end_session_touch"])
def test_touch_after_demote_records_its_cause(
    storage: tuple[L1Manager, StorageManager], cause: L1RecencyCause
) -> None:
    l1, store = storage
    commit(l1, A, B)
    store.demote_l1_keys([(l1.l1_manager_id, [A, B])])
    store.touch_cached_keys([A], cause)
    assert _state(_usage(store, A, B)) == {"demoted": 1, cause: 1}


def test_touch_without_cause_is_other(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, A)
    store.touch_cached_keys([A])
    assert _state(_usage(store, A)) == {"other": 1}


def test_latest_touch_wins(storage: tuple[L1Manager, StorageManager]) -> None:
    l1, store = storage
    commit(l1, A)
    store.touch_cached_keys([A], "lookup_touch")
    store.touch_cached_keys([A], "end_session_touch")
    assert _state(_usage(store, A)) == {"end_session_touch": 1}


def test_demote_after_touch_is_demoted(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    commit(l1, A)
    store.touch_cached_keys([A], "lookup_touch")
    store.demote_l1_keys([(l1.l1_manager_id, [A])])
    assert _state(_usage(store, A)) == {"demoted": 1}


def test_eviction_drops_the_state(storage: tuple[L1Manager, StorageManager]) -> None:
    l1, store = storage
    commit(l1, A, B)
    store.demote_l1_keys([(l1.l1_manager_id, [A, B])])
    store.delete_l1_keys([A, B])
    assert _usage(store, A, B).matched_objects == 0

    # A fresh write of an evicted key starts over, not demoted.
    commit(l1, A)
    assert _state(_usage(store, A, B)) == {"write": 1}


def test_untracked_and_staging_keys_get_no_state(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    stage(l1, C)
    store.touch_cached_keys([C], "lookup_touch")
    store.demote_l1_keys([(l1.l1_manager_id, [C])])
    assert _usage(store, C).matched_objects == 0
    # The commit after a touch or demote of the staging object is a write.
    l1.finish_write([C])
    assert _state(_usage(store, C)) == {"write": 1}


def test_l2_prefetch_commit_is_other(
    storage: tuple[L1Manager, StorageManager],
) -> None:
    l1, store = storage
    stage(l1, A)
    l1.finish_write_and_reserve_read([A])
    l1.finish_read([A])
    assert _state(_usage(store, A)) == {"other": 1}


def test_demoted_flag_agrees_with_eviction_order_under_races(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent touches and demotions leave the flag and LRU order agreeing."""
    l1, store = cpu_storage(monkeypatch)
    policy = LRUEvictionPolicy()
    l1.register_listener(L1EvictionPolicy(policy))
    try:
        commit(l1, A, B)
        rounds = 2000

        def demote() -> None:
            for _ in range(rounds):
                l1.demote_keys([A])

        def touch() -> None:
            for _ in range(rounds):
                l1.touch_keys([A], "lazy_touch")

        threads = [threading.Thread(target=demote), threading.Thread(target=touch)]
        for thread in threads:
            thread.start()
        for thread in threads:
            thread.join()

        usage = _usage(store, A)
        order = [k for action in policy.get_eviction_actions(1.0) for k in action.keys]
        if usage.matched_demoted_objects:
            assert order[0] == A
        else:
            assert usage.matched_mru_objects["lazy_touch"] == 1
            assert order[-1] == A
    finally:
        store.close()
