# SPDX-License-Identifier: Apache-2.0
"""Request lifecycle and submitted-batch contracts for lazy offload.

``LazyOffloadRequestRegistry`` answers two questions for the manager: has this
request finished, and does it still have a store batch in flight. The hard
case is a request id outliving its owner: a new request reusing a finished id
must detach the outstanding batch from the request now holding the id, while a
preempted request coming back keeps its batch.
"""

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_policy.base import PendingStoreItem
from lmcache.integration.vllm.lazy_offload_state import LazyOffloadRequestRegistry


def register(registry: LazyOffloadRequestRegistry, block_ids: list[int]) -> None:
    """Put one batch of ``req`` in flight, pinning ``block_ids``."""
    registry.register_batch("req", block_ids, PendingStoreItem(request_id="req"))


def test_arrival_opens_an_active_slot_with_nothing_in_flight() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")

    assert not registry.is_finished("req")
    assert not registry.has_in_flight("req")
    assert registry.finished_request_ids() == set()


def test_unknown_request_is_neither_finished_nor_in_flight() -> None:
    """The manager queries ids it has never seen; that must not create state."""
    registry = LazyOffloadRequestRegistry()

    assert not registry.is_finished("never-seen")
    assert not registry.has_in_flight("never-seen")
    assert not registry.can_end_session("never-seen")


def test_fresh_batch_is_not_orphaned() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    register(registry, [1, 2])

    assert registry.has_in_flight("req")
    assert registry.in_flight_request_ids() == {"req"}
    assert not registry.complete_batch("req").orphaned


def test_preempted_rearrival_keeps_its_batch() -> None:
    """A preempted request re-arrives ACTIVE: its in-flight batch stays its own, so
    the arrival must not orphan it.
    """
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    register(registry, [1])
    registry.arrive("req")

    batch = registry.complete_batch("req")
    assert batch.block_ids == (1,)
    assert not batch.orphaned


def test_finished_id_reuse_orphans_the_outstanding_batch() -> None:
    """A distinct request reusing a finished id detaches the predecessor's batch; the
    batch still closes normally and keeps its block ids.
    """
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    register(registry, [1, 2])
    registry.finish("req")

    registry.arrive("req")

    assert not registry.is_finished("req")
    batch = registry.complete_batch("req")
    assert batch.block_ids == (1, 2)
    assert batch.orphaned
    # The successor is active, so its session must not be releasable.
    assert not registry.can_end_session("req")


def test_session_release_requires_finished_request_and_no_batch() -> None:
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    register(registry, [1])
    registry.finish("req")
    assert registry.is_finished("req")
    assert registry.finished_request_ids() == {"req"}
    assert not registry.can_end_session("req")
    # session_ended on a non-releasable slot is a no-op.
    registry.session_ended("req")
    assert registry.has_in_flight("req")

    # Finishing alone does not detach the batch from its generation.
    assert not registry.complete_batch("req").orphaned
    assert registry.can_end_session("req")
    registry.session_ended("req")
    assert not registry.can_end_session("req")
    assert registry.finished_request_ids() == set()


def test_overlapping_batches_are_rejected() -> None:
    registry = LazyOffloadRequestRegistry()
    register(registry, [1])

    with pytest.raises(RuntimeError, match="already has an in-flight"):
        register(registry, [2])


def test_complete_batch_requires_an_open_batch() -> None:
    registry = LazyOffloadRequestRegistry()
    with pytest.raises(KeyError):
        registry.complete_batch("unknown")
    registry.arrive("req")
    with pytest.raises(KeyError):
        registry.complete_batch("req")


def test_finish_turns_a_later_rearrival_into_an_id_reuse() -> None:
    """After a finish, a rearrival is an id reuse, not the preempted tracker coming
    back: a batch submitted between them (a finished request's op draining late) must
    be orphaned by that arrival.
    """
    registry = LazyOffloadRequestRegistry()
    registry.arrive("req")
    registry.finish("req")
    register(registry, [1])

    registry.arrive("req")

    assert registry.complete_batch("req").orphaned
