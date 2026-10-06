# SPDX-License-Identifier: Apache-2.0
"""Tests for MP-server-side cache-event emission: the event-bus
subscriber's event mapping, batching/ordering, seq/gap semantics on
publish failure, and the HTTP sink end-to-end against a coordinator
app."""

# Standard
from collections.abc import Callable
from dataclasses import asdict
from unittest.mock import MagicMock
import asyncio
import json
import threading

# Third Party
import httpx
import numpy as np
import pytest

# First Party
from lmcache.v1.distributed.api import (
    CapacitySnapshot,
    L1BackendType,
    ModuleMemoryCapacity,
    ObjectKey,
    Tier,
)
from lmcache.v1.distributed.internal_api import L1ObjectMeta
from lmcache.v1.mp_coordinator.api import (
    UNKNOWN_TOKEN_OFFSET,
    BlendNamespace,
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.app import create_app
from lmcache.v1.mp_coordinator.cache_events import (
    EVENTS_TRACE_LIFECYCLE,
    CacheEventPublishError,
    CacheEventSink,
    CacheEventSubscriber,
    HttpCacheEventSink,
    MultiCacheEventSink,
    SpoolingCacheEventSink,
    TraceCacheEventSink,
)
from lmcache.v1.mp_coordinator.config import MPCoordinatorConfig
from lmcache.v1.mp_coordinator.ingest.event_broadcaster import CacheEventBroadcaster
from lmcache.v1.mp_coordinator.ingest.event_gate import EventGate
from lmcache.v1.mp_coordinator.persistence.quiesce import QuiesceLock
from lmcache.v1.mp_coordinator.views.key_directory import KeyDirectory
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import EventBus, EventBusConfig
from lmcache.v1.mp_observability.trace.reader import TraceReader
from lmcache.v1.mp_observability.trace.recorder import EventsTraceRecorder
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    HTTPFrontendConfig,
    MPServerConfig,
)
import lmcache.v1.mp_coordinator.cache_events as cache_events


def _key(hash_byte: int) -> ObjectKey:
    return ObjectKey(chunk_hash=bytes([hash_byte]) * 4, model_name="m", kv_rank=0)


# The namespace ``_key`` stores in; fragment queries must ask from it.
NS = BlendNamespace.from_object_key(_key(0))


def _entry(hash_byte: int, size_bytes: int = 0) -> CacheEventEntry:
    return CacheEventEntry(
        key=_key(hash_byte).to_encoded_object_key(), size_bytes=size_bytes
    )


def _meta(
    size_bytes: int = 0, backend: L1BackendType = L1BackendType.DRAM
) -> L1ObjectMeta:
    return L1ObjectMeta(size_bytes=size_bytes, backend=backend)


class _RecordingSink(CacheEventSink):
    """Sink that records every published list; optionally fails."""

    def __init__(self) -> None:
        self.published: list[list[CacheEventBatch]] = []
        self.fail_next = False
        self.closed = False

    def publish(self, batches: list[CacheEventBatch]) -> None:
        if self.fail_next:
            self.fail_next = False
            raise CacheEventPublishError("injected failure")
        self.published.append(batches)

    def close(self) -> None:
        self.closed = True


def _subscriber(
    sink: CacheEventSink,
    incarnation: int = 7,
    flush_interval: float = 3600.0,
) -> CacheEventSubscriber:
    """Build a subscriber whose default interval never auto-flushes in a
    test, so batching assertions drive ``flush()`` explicitly."""
    return CacheEventSubscriber(
        sink=sink,
        instance_id="node-a",
        incarnation=incarnation,
        flush_interval=flush_interval,
    )


def _dispatch(subscriber: CacheEventSubscriber, *events: Event) -> None:
    """Deliver events to the subscriber the way the bus drain thread does."""
    subscriptions = subscriber.get_subscriptions()
    for event in events:
        subscriptions[event.event_type](event)


# -- Subscriber event mapping -------------------------------------------------


def test_l1_store_access_delete_events_map_to_batches():
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L1_WRITE_FINISHED,
            metadata={"keys": [_key(1), _key(2)], "meta": [_meta(100), _meta(200)]},
        ),
        Event(
            event_type=EventType.L1_WRITE_FINISHED_AND_READ_RESERVED,
            metadata={"keys": [_key(3)], "meta": [_meta(300)]},
        ),
        Event(
            event_type=EventType.L1_KEYS_ACCESSED,
            metadata={"keys": [_key(1)]},
        ),
        Event(
            event_type=EventType.L1_KEYS_ACCESSED,
            metadata={"keys": [_key(2)]},
        ),
        Event(
            event_type=EventType.L1_KEYS_EVICTED,
            metadata={"keys": [_key(3)], "meta": [_meta(300)]},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    # Consecutive same-identity records coalesce across events: the two
    # store events form one batch, the two access events form one batch.
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.ACCESS,
        CacheEventType.DELETE,
    ]
    store, access, delete = batches
    assert [e.size_bytes for e in store.entries] == [100, 200, 300]
    assert len(access.entries) == 2
    assert all(e.size_bytes == 0 for e in access.entries)  # ACCESS never sizes
    assert access.backend == ""  # ACCESS carries no placement identity
    assert delete.entries[0].key == _key(3).to_encoded_object_key()
    assert delete.entries[0].size_bytes == 0
    assert all(b.tier == Tier.L1 for b in batches)
    assert store.backend == "dram" and delete.backend == "dram"
    assert [b.seq for b in batches] == [1, 2, 3]
    assert all(b.instance_id == "node-a" and b.incarnation == 7 for b in batches)


def test_tokens_stored_events_publish_no_batches():
    """Token-binding events only feed the stamping cache; they produce
    no batches of their own."""
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.MP_TOKENS,
            metadata={
                "chunk_hashes": [_key(1).chunk_hash],
                "token_chunks": [[1, 2]],
                "token_offsets": [0],
            },
        ),
    )
    subscriber.flush()

    assert sink.published == []


def test_tokens_event_stamps_store_entries():
    """Bindings arrive before write-finished events (store publishes them
    at submission), so every STORE entry of a known chunk carries the
    chunk's token ids."""
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    rank1_key = ObjectKey(chunk_hash=_key(1).chunk_hash, model_name="m", kv_rank=1)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.MP_TOKENS,
            metadata={
                "chunk_hashes": [_key(1).chunk_hash],
                "token_chunks": [[1, 2]],
                "token_offsets": [0],
            },
        ),
        Event(
            event_type=EventType.L1_WRITE_FINISHED,
            metadata={
                "keys": [_key(1), rank1_key, _key(2)],
                "meta": [_meta(100), _meta(100), _meta(200)],
            },
        ),
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L1_KEYS_EVICTED,
            metadata={"keys": [_key(1)], "meta": [_meta(100)]},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    l1_store, l2_store, delete = batches
    # Every entry of the known chunk is stamped; unknown chunks stay empty.
    assert [e.token_ids for e in l1_store.entries] == [[1, 2], [1, 2], []]
    assert l2_store.entries[0].token_ids == [1, 2]
    # Deletes never carry tokens.
    assert delete.entries[0].token_ids == []
    # The chunk's position rides with its tokens, on every stamped entry.
    # The third key's chunk is unknown to the binding cache, so it carries no
    # position rather than claiming 0.
    assert [e.token_offset for e in l1_store.entries] == [0, 0, UNKNOWN_TOKEN_OFFSET]
    assert l2_store.entries[0].token_offset == 0


def test_tokens_event_stamps_the_chunks_offset():
    """Each chunk carries its own position, so a mid-sequence chunk is
    stamped with its offset rather than the batch's first."""
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    keys = [_key(1), _key(2)]
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.MP_TOKENS,
            metadata={
                "chunk_hashes": [key.chunk_hash for key in keys],
                "token_chunks": [[1, 2], [3, 4]],
                "token_offsets": [256, 512],
            },
        ),
        Event(
            event_type=EventType.L1_WRITE_FINISHED,
            metadata={"keys": keys, "meta": [_meta(100)] * 2},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    store = batches[-1]
    assert [e.token_ids for e in store.entries] == [[1, 2], [3, 4]]
    assert [e.token_offset for e in store.entries] == [256, 512]


def test_token_binding_cache_evicts_oldest_down_to_half(monkeypatch):
    """Passing the bound drops the oldest bindings in one batch, down to
    half the bound, so the newest survive to stamp their STOREs."""
    monkeypatch.setattr(cache_events, "_TOKEN_BINDING_CACHE_SIZE", 2)
    sink = _RecordingSink()
    subscriber = CacheEventSubscriber(
        sink=sink,
        instance_id="node-a",
        incarnation=7,
        flush_interval=3600.0,
    )

    keys = [_key(1), _key(2), _key(3)]
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.MP_TOKENS,
            metadata={
                "chunk_hashes": [key.chunk_hash for key in keys],
                "token_chunks": [[1, 2], [3, 4], [5, 6]],
                "token_offsets": [0, 256, 512],
            },
        ),
        Event(
            event_type=EventType.L1_WRITE_FINISHED,
            metadata={"keys": keys, "meta": [_meta(100)] * 3},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    store = batches[-1]
    # Bound 2 -> evict to 1: only the newest binding is left.
    assert [e.token_ids for e in store.entries] == [[], [], [5, 6]]


def test_token_binding_eviction_warns(monkeypatch):
    """Dropping bindings means later STOREs lose their token ids, so the
    subscriber says so."""
    monkeypatch.setattr(cache_events, "_TOKEN_BINDING_CACHE_SIZE", 2)
    warnings: list[str] = []
    # The module logger does not propagate (see ``lmcache.logging``), so
    # record the call instead of relying on root-handler capture.
    monkeypatch.setattr(
        cache_events.logger,
        "warning",
        lambda msg, *args: warnings.append(msg % args),
    )
    subscriber = CacheEventSubscriber(
        sink=_RecordingSink(),
        instance_id="node-a",
        incarnation=7,
        flush_interval=3600.0,
    )
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.MP_TOKENS,
            metadata={
                "chunk_hashes": [_key(i).chunk_hash for i in (1, 2, 3)],
                "token_chunks": [[1, 2], [3, 4], [5, 6]],
                "token_offsets": [0, 256, 512],
            },
        ),
    )
    assert warnings == [
        "Token binding cache hit its 2-entry bound: evicted the 2 oldest "
        "bindings; STORE events for those chunks carry no token ids "
        "(stores completing far behind their submission)"
    ]


def test_mismatched_token_chunks_raise():
    subscriber = _subscriber(_RecordingSink())
    with pytest.raises(ValueError):
        _dispatch(
            subscriber,
            Event(
                event_type=EventType.MP_TOKENS,
                metadata={
                    "chunk_hashes": [_key(1).chunk_hash, _key(2).chunk_hash],
                    "token_chunks": [[1, 2]],
                    "token_offsets": [0, 256],
                },
            ),
        )


def test_mismatched_token_offsets_raise():
    """Offsets are parallel to the chunks by construction, so a length
    mismatch is a publisher bug, not a degraded binding."""
    subscriber = _subscriber(_RecordingSink())
    with pytest.raises(ValueError):
        _dispatch(
            subscriber,
            Event(
                event_type=EventType.MP_TOKENS,
                metadata={
                    "chunk_hashes": [_key(1).chunk_hash, _key(2).chunk_hash],
                    "token_chunks": [[1, 2], [3, 4]],
                    "token_offsets": [0],
                },
            ),
        )


def test_l1_events_split_batches_by_medium():
    """A hybrid DRAM+DAX store emits one batch per medium, and deletes
    target the same per-medium identity the stores reported."""
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L1_WRITE_FINISHED,
            metadata={
                "keys": [_key(1), _key(2), _key(3)],
                "meta": [
                    _meta(100, L1BackendType.DRAM),
                    _meta(200, L1BackendType.DEVDAX),
                    _meta(300, L1BackendType.DRAM),
                ],
            },
        ),
        Event(
            event_type=EventType.L1_KEYS_EVICTED,
            metadata={"keys": [_key(2)], "meta": [_meta(200, L1BackendType.DEVDAX)]},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    assert [(b.event_type, b.backend) for b in batches] == [
        (CacheEventType.STORE, "dram"),
        (CacheEventType.STORE, "devdax"),
        (CacheEventType.DELETE, "devdax"),
    ]
    dram_store, devdax_store, devdax_delete = batches
    assert [e.size_bytes for e in dram_store.entries] == [100, 300]
    assert [e.size_bytes for e in devdax_store.entries] == [200]
    assert devdax_delete.entries[0].key == _key(2).to_encoded_object_key()


def test_l1_access_batches_have_empty_backend():
    """ACCESS refreshes key-level recency only, so its batches carry no
    placement identity: the backend is empty by contract."""
    sink = _RecordingSink()
    subscriber = _subscriber(sink)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L1_KEYS_ACCESSED,
            metadata={"keys": [_key(1), _key(2)]},
        ),
    )
    subscriber.flush()
    [[batch]] = sink.published
    assert batch.event_type == CacheEventType.ACCESS
    assert batch.tier == Tier.L1
    assert batch.backend == ""
    assert all(e.size_bytes == 0 for e in batch.entries)


def test_read_finished_is_not_consumed():
    """L1_READ_FINISHED is covered by the request-end unified touch
    (L1_KEYS_ACCESSED); consuming both would duplicate ACCESS events."""
    subscriber = _subscriber(_RecordingSink())
    assert EventType.L1_READ_FINISHED not in subscriber.get_subscriptions()


def test_l2_events_map_with_backend_and_sizes():
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1), _key(2)], "sizes": [100, 200], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_ACCESSED,
            metadata={"keys": [_key(1)], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_DELETED,
            metadata={"keys": [_key(2)], "backend": "fs"},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.ACCESS,
        CacheEventType.DELETE,
    ]
    store, access, delete = batches
    assert [e.size_bytes for e in store.entries] == [100, 200]
    assert access.entries[0].key == _key(1).to_encoded_object_key()
    assert delete.entries[0].key == _key(2).to_encoded_object_key()
    assert all(b.tier == Tier.L2 and b.backend == "fs" for b in batches)


def test_l2_shared_flag_rides_the_batch():
    """An adapter mounting a shared pool tags its events; the subscriber
    keeps shared and private runs in distinct batches."""
    sink = _RecordingSink()
    subscriber = _subscriber(sink)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={
                "keys": [_key(1)],
                "sizes": [100],
                "backend": "fs",
                "shared": True,
            },
        ),
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(2)], "sizes": [200], "backend": "fs"},
        ),
    )
    subscriber.flush()
    [batches] = sink.published
    assert [(b.backend, b.shared) for b in batches] == [
        ("fs", True),
        ("fs", False),
    ]


def test_interleaved_events_preserve_total_order():
    # store k1, delete k1, re-store k1: the re-store must not be
    # reordered before the delete, or the directory ends up empty.
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_DELETED,
            metadata={"keys": [_key(1)], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [150], "backend": "fs"},
        ),
    )
    subscriber.flush()

    [batches] = sink.published
    assert [b.event_type for b in batches] == [
        CacheEventType.STORE,
        CacheEventType.DELETE,
        CacheEventType.STORE,
    ]
    assert [b.seq for b in batches] == [1, 2, 3]


def test_mismatched_l1_meta_raises():
    subscriber = _subscriber(_RecordingSink())
    with pytest.raises(ValueError):
        _dispatch(
            subscriber,
            Event(
                event_type=EventType.L1_WRITE_FINISHED,
                metadata={"keys": [_key(1), _key(2)], "meta": [_meta(100)]},
            ),
        )


# -- Emitter flush / seq semantics --------------------------------------------


def test_flush_with_empty_buffer_publishes_nothing():
    sink = _RecordingSink()
    _subscriber(sink).flush()
    assert sink.published == []


def test_publish_failure_drops_batches_and_leaves_a_seq_gap():
    # Failed flushes consume their seq numbers so the ingest gate sees a
    # gap and can flag the instance for replay.
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
        ),
    )
    sink.fail_next = True
    subscriber.flush()
    assert sink.published == []

    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(2)], "sizes": [200], "backend": "fs"},
        ),
    )
    subscriber.flush()
    [[batch]] = sink.published
    assert batch.seq == 2


def test_negative_flush_interval_rejected():
    with pytest.raises(ValueError):
        _subscriber(_RecordingSink(), flush_interval=-1.0)


def test_events_self_pace_flushing():
    """With interval 0 every event flushes; a long interval holds events
    back until an explicit flush."""
    store_event = Event(
        event_type=EventType.L2_KEYS_STORED,
        metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
    )
    eager_sink = _RecordingSink()
    _dispatch(_subscriber(eager_sink, flush_interval=0.0), store_event, store_event)
    assert len(eager_sink.published) == 2

    lazy_sink = _RecordingSink()
    _dispatch(_subscriber(lazy_sink), store_event)
    assert lazy_sink.published == []


def test_eviction_tick_flushes_buffered_tail():
    """A buffered tail (burst-ending events) is flushed by the eviction
    loop's tick once the flush interval elapses, without new cache
    events."""
    # Standard
    import time as _time

    sink = _RecordingSink()
    subscriber = _subscriber(sink, flush_interval=0.05)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
        ),
    )
    assert sink.published == []  # interval not yet elapsed: tail buffered
    _time.sleep(0.06)
    _dispatch(
        subscriber,
        Event(event_type=EventType.L1_EVICTION_LOOP_TICK, metadata={"usage": 0.0}),
    )
    assert len(sink.published) == 1


def test_shutdown_flushes_and_closes_sink():
    sink = _RecordingSink()
    subscriber = _subscriber(sink)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [100], "backend": "fs"},
        ),
    )
    subscriber.shutdown()
    assert len(sink.published) == 1
    assert sink.closed is True


# -- Delivery loss is counted: bus overflow -----------------------------------


def _l2_store(*hash_bytes: int) -> Event:
    return Event(
        event_type=EventType.L2_KEYS_STORED,
        metadata={
            "keys": [_key(h) for h in hash_bytes],
            "sizes": [100] * len(hash_bytes),
            "backend": "fs",
        },
    )


class _GateSink(CacheEventSink):
    """Sink that hands batches straight to a coordinator gate; can be
    told to raise after delivering (a timeout the coordinator survived)."""

    def __init__(self) -> None:
        self.gate = EventGate(CacheEventBroadcaster(), QuiesceLock())
        self.fail_after_delivery = 0
        self.fail_before_delivery = 0
        self.raise_value_error = 0
        self.calls: list[list[int]] = []
        self.published: list[CacheEventBatch] = []

    def publish(self, batches: list[CacheEventBatch]) -> None:
        self.calls.append([b.seq for b in batches])
        if self.raise_value_error:
            self.raise_value_error -= 1
            raise ValueError("injected bad batch")
        if self.fail_before_delivery:
            self.fail_before_delivery -= 1
            raise CacheEventPublishError("injected failure")
        self.published.extend(batches)
        self.gate.ingest_batches(batches)
        if self.fail_after_delivery:
            self.fail_after_delivery -= 1
            raise CacheEventPublishError("injected timeout")

    def loss(self) -> tuple[int, int, int]:
        """The gate's ``(incidents, lost events, admitted events)``."""
        stream = self.gate.stats()["node-a"]
        return (
            stream.loss_incidents_total,
            stream.lost_events_total,
            stream.admitted_events_total,
        )


def _overflowing_bus(subscriber: CacheEventSubscriber) -> EventBus:
    """A bus, not started, whose queue holds one event: every publish
    after the first before a drain is dropped."""
    bus = EventBus(EventBusConfig(enabled=True, max_queue_size=1))
    bus.register_subscriber(subscriber)
    return bus


def test_gate_counts_exactly_the_events_the_bus_dropped():
    """End to end: each key of a dropped store is one lost cache event."""
    sink = _GateSink()
    subscriber = _subscriber(sink)
    bus = _overflowing_bus(subscriber)
    bus.publish(_l2_store(1))
    bus._drain_all()
    subscriber.flush()  # seq 1, nothing lost yet
    bus.publish(_l2_store(2))  # queued
    bus.publish(_l2_store(3, 4))  # dropped: 2 events
    bus.publish(_l2_store(5, 6, 7))  # dropped: 3 events
    bus.stop()  # drains, then the subscriber's shutdown flushes

    assert bus.dropped_events_count() == 2
    assert subscriber.bus_dropped_events_total() == 5
    assert [(b.seq, b.dropped_events) for b in sink.published] == [(1, 0), (2, 5)]
    assert sink.loss() == (1, 5, 2)
    assert sink.gate.stats()["node-a"].gap_detected is True


def test_loss_before_the_first_flush_is_counted():
    sink = _GateSink()
    subscriber = _subscriber(sink)
    bus = _overflowing_bus(subscriber)
    bus.publish(_l2_store(1))
    bus.publish(_l2_store(2, 3))  # dropped
    bus.stop()

    assert [(b.seq, b.dropped_events) for b in sink.published] == [(1, 2)]
    assert sink.loss() == (1, 2, 1)


def test_dropped_events_without_keys_lose_no_cache_event():
    """A lost flush tick loses no cache state."""
    sink = _GateSink()
    subscriber = _subscriber(sink)
    bus = _overflowing_bus(subscriber)
    bus.publish(_l2_store(1))
    bus.publish(
        Event(event_type=EventType.L1_EVICTION_LOOP_TICK, metadata={"usage": 0.0})
    )
    bus.stop()

    assert bus.dropped_events_count() == 1
    assert subscriber.bus_dropped_events_total() == 0
    assert sink.loss() == (0, 0, 1)
    assert sink.gate.stats()["node-a"].gap_detected is False


def test_drops_of_types_the_subscriber_ignores_are_not_counted():
    sink = _GateSink()
    subscriber = _subscriber(sink)
    bus = _overflowing_bus(subscriber)
    bus.publish(_l2_store(1))
    bus.publish(Event(event_type=EventType.L1_READ_FINISHED, metadata={"keys": [1]}))
    bus.stop()

    assert subscriber.bus_dropped_events_total() == 0
    assert sink.loss() == (0, 0, 1)


def test_drops_before_registration_are_not_counted():
    """Events the bus dropped before the subscriber existed were never
    going to reach it."""
    sink = _GateSink()
    bus = EventBus(EventBusConfig(enabled=True, max_queue_size=1))
    bus.publish(_l2_store(1))
    bus.publish(_l2_store(2))  # dropped before registration
    subscriber = _subscriber(sink)
    bus.register_subscriber(subscriber)
    bus.stop()

    assert subscriber.bus_dropped_events_total() == 0
    assert [(b.seq, b.dropped_events) for b in sink.published] == [(1, 0)]
    assert sink.gate.stats()["node-a"].gap_detected is False


def test_events_dropped_gauge_reports_bus_overflow(monkeypatch):
    register = MagicMock()
    monkeypatch.setattr(cache_events, "register_gauge", register)
    subscriber = _subscriber(_RecordingSink())
    bus = _overflowing_bus(subscriber)
    bus.publish(_l2_store(1))
    bus.publish(_l2_store(2, 3))

    cache_events.register_cache_event_metrics(subscriber, None)

    (call,) = register.call_args_list
    assert call.args[1] == "lmcache_mp.cache_events.events_dropped_total"
    assert call.args[3]() == [(2, {"reason": "bus_overflow"})]


# -- Spooled HTTP delivery -----------------------------------------------------


class _Clock:
    def __init__(self) -> None:
        self.now = 1000.0

    def __call__(self) -> float:
        return self.now


def _spool(
    inner: CacheEventSink,
    clock: _Clock,
    max_entries: int = 1000,
    initial_backoff: float = 1.0,
    max_backoff: float = 4.0,
) -> SpoolingCacheEventSink:
    return SpoolingCacheEventSink(
        inner,
        max_entries=max_entries,
        initial_backoff=initial_backoff,
        max_backoff=max_backoff,
        clock=clock,
    )


def test_spool_retries_failed_batches_in_seq_order_without_a_gap():
    gate_sink = _GateSink()
    clock = _Clock()
    spool = _spool(gate_sink, clock)
    subscriber = _subscriber(spool)

    gate_sink.fail_before_delivery = 1
    _dispatch(subscriber, _l2_store(1))
    subscriber.flush()
    assert spool.spool_depth() == 1

    # Backoff not yet elapsed: the new batch queues behind the old one.
    _dispatch(subscriber, _l2_store(2))
    subscriber.flush()
    assert gate_sink.calls == [[1]]
    assert spool.spool_depth() == 2

    clock.now += 1.0
    subscriber.flush()  # nothing new: redelivers the spool
    assert gate_sink.calls == [[1], [1, 2]]
    assert spool.spool_depth() == 0
    assert spool.retries_total() == 1
    stream = gate_sink.gate.stats()["node-a"]
    assert (stream.last_seq, stream.gap_detected) == (2, False)


def test_spool_backoff_doubles_and_caps():
    gate_sink = _GateSink()
    clock = _Clock()
    spool = _spool(gate_sink, clock, initial_backoff=1.0, max_backoff=4.0)
    gate_sink.fail_before_delivery = 100
    spool.publish([_batch_for_spool(1)])
    attempts_at: list[float] = []
    for _ in range(24):
        before = len(gate_sink.calls)
        spool.redeliver()
        if len(gate_sink.calls) > before:
            attempts_at.append(clock.now)
        clock.now += 0.5
    # Failures at t=0 (publish), then 1, 2, 4, 4 seconds apart.
    assert [t - 1000.0 for t in attempts_at] == [1.0, 3.0, 7.0, 11.0]
    assert spool.retries_total() == 4


def test_spool_bound_does_not_cap_a_healthy_flush():
    """The bound limits what is retained during an outage, not how much
    one flush may deliver."""
    gate_sink = _GateSink()
    spool = _spool(gate_sink, _Clock(), max_entries=1)
    spool.publish([_batch_for_spool(1), _batch_for_spool(2)])

    assert gate_sink.calls == [[1, 2]]
    assert spool.overflow_events_total() == 0


def test_spool_backoff_starts_when_the_failed_attempt_ends():
    """A slow failure (a timeout) must not use up the backoff it arms."""

    class _SlowFailingSink(CacheEventSink):
        def __init__(self, clock: _Clock) -> None:
            self.clock = clock
            self.attempts = 0

        def publish(self, batches: list[CacheEventBatch]) -> None:
            self.attempts += 1
            self.clock.now += 2.0  # the request times out after 2 s
            raise CacheEventPublishError("timeout")

    clock = _Clock()
    inner = _SlowFailingSink(clock)
    spool = _spool(inner, clock, initial_backoff=1.0, max_backoff=4.0)
    spool.publish([_batch_for_spool(1)])
    spool.redeliver()

    assert inner.attempts == 1


def test_spool_redelivery_duplicates_are_dropped_by_the_gate():
    """A post that the coordinator applied but that still failed (a
    timeout) is resent; the gate's seq dedup makes that harmless."""
    gate_sink = _GateSink()
    clock = _Clock()
    spool = _spool(gate_sink, clock)
    gate_sink.fail_after_delivery = 1
    spool.publish([_batch_for_spool(1)])
    clock.now += 1.0
    spool.publish([_batch_for_spool(2)])

    assert gate_sink.calls == [[1], [1, 2]]
    stream = gate_sink.gate.stats()["node-a"]
    assert (stream.last_seq, stream.gap_detected) == (2, False)


def test_spool_overflow_drops_oldest_and_shows_up_as_a_gap():
    gate_sink = _GateSink()
    clock = _Clock()
    spool = _spool(gate_sink, clock, max_entries=2)
    gate_sink.fail_before_delivery = 1
    spool.publish([_batch_for_spool(1)])
    spool.publish([_batch_for_spool(2), _batch_for_spool(3)])  # evicts seq 1

    assert spool.overflow_events_total() == 1
    assert spool.spool_depth() == 2
    clock.now += 1.0
    spool.redeliver()
    assert gate_sink.calls[-1] == [2, 3]
    assert gate_sink.gate.stats()["node-a"].gap_detected is True


def test_spool_drains_a_large_backlog_in_bounded_slices(monkeypatch):
    monkeypatch.setattr(cache_events, "_SPOOL_PUBLISH_SLICE_ENTRIES", 2)
    gate_sink = _GateSink()
    clock = _Clock()
    spool = _spool(gate_sink, clock)
    gate_sink.fail_before_delivery = 1
    spool.publish([_batch_for_spool(seq) for seq in range(1, 6)])
    clock.now += 1.0
    spool.redeliver()

    assert gate_sink.calls[1:] == [[1, 2], [3, 4], [5]]
    assert spool.spool_depth() == 0


def test_spool_close_tries_once_more_then_closes_inner():
    inner = _RecordingSink()
    clock = _Clock()
    spool = _spool(inner, clock)
    inner.fail_next = True
    spool.publish([_batch_for_spool(1)])
    spool.close()  # inside the backoff window, but shutdown does not wait

    assert [[b.seq for b in call] for call in inner.published] == [[1]]
    assert inner.closed is True


def _http_spool(
    clock: _Clock, status_for: Callable[[list[int]], int]
) -> tuple[SpoolingCacheEventSink, list[list[int]]]:
    """A spooled HTTP sink whose coordinator answers ``status_for(seqs)``."""
    posts: list[list[int]] = []

    def handler(request: httpx.Request) -> httpx.Response:
        seqs = [b["seq"] for b in json.loads(request.content)["batches"]]
        posts.append(seqs)
        return httpx.Response(status_for(seqs))

    http_sink = HttpCacheEventSink("http://coordinator")
    http_sink._client = httpx.Client(  # noqa: SLF001 — test-only transport swap
        transport=httpx.MockTransport(handler)
    )
    return _spool(http_sink, clock), posts


def test_spool_drops_a_rejected_post_and_keeps_draining():
    clock = _Clock()
    spool, posts = _http_spool(clock, lambda seqs: 422 if 1 in seqs else 200)
    spool.publish([_batch_for_spool(1)])
    spool.publish([_batch_for_spool(2)])

    assert posts == [[1], [2]]
    assert spool.spool_depth() == 0
    assert spool.rejected_events_total() == 1


@pytest.mark.parametrize("status", [408, 429, 503])
def test_spool_retries_transient_http_errors(status):
    clock = _Clock()
    transient = [status]

    def status_for(seqs: list[int]) -> int:
        if 1 in seqs:
            return 422
        return transient.pop() if transient else 200

    spool, posts = _http_spool(clock, status_for)
    spool.publish([_batch_for_spool(1)])  # rejected and dropped
    spool.publish([_batch_for_spool(2)])  # transient failure: retained
    assert spool.spool_depth() == 1

    clock.now += 1.0
    spool.redeliver()
    assert posts == [[1], [2], [2]]
    assert spool.spool_depth() == 0
    assert spool.rejected_events_total() == 1


def test_spool_drops_a_slice_that_raises_unexpectedly():
    gate_sink = _GateSink()
    spool = _spool(gate_sink, _Clock())
    subscriber = _subscriber(spool)
    gate_sink.raise_value_error = 1
    _dispatch(subscriber, _l2_store(1))
    subscriber.flush()  # must not raise
    _dispatch(subscriber, _l2_store(2))
    subscriber.flush()

    assert gate_sink.calls == [[1], [2]]
    assert spool.spool_depth() == 0
    assert spool.rejected_events_total() == 1
    stream = gate_sink.gate.stats()["node-a"]
    assert (stream.last_seq, stream.gap_detected) == (2, True)


def test_gate_counts_exactly_the_entries_the_spool_overflowed():
    """End to end: spool overflow drops whole batches; later batches
    report their entries, so the gate's lost count is exact."""
    gate_sink = _GateSink()
    clock = _Clock()
    spool = _spool(gate_sink, clock, max_entries=3)
    subscriber = _subscriber(spool)
    _dispatch(subscriber, _l2_store(1))
    subscriber.flush()  # seq 1 delivered
    gate_sink.fail_before_delivery = 1
    _dispatch(subscriber, _l2_store(2, 3))
    subscriber.flush()  # seq 2 retained
    _dispatch(subscriber, _l2_store(4, 5, 6))
    subscriber.flush()  # backoff: seq 3 queues; 5 > 3 entries drops seq 2
    assert spool.overflow_events_total() == 2
    assert spool.spool_depth() == 3

    clock.now += 1.0
    _dispatch(subscriber, _l2_store(7))
    subscriber.flush()

    assert [(b.seq, b.dropped_events) for b in gate_sink.published] == [
        (1, 0),
        (3, 0),
        (4, 2),
    ]
    # The gap (seq 3) and the reported count (seq 4) land on different
    # batches, so this one overflow is two incidents.
    assert gate_sink.loss() == (2, 2, 5)


def test_gate_counts_exactly_the_entries_of_a_rejected_slice():
    gate_sink = _GateSink()
    spool = _spool(gate_sink, _Clock())
    subscriber = _subscriber(spool)
    _dispatch(subscriber, _l2_store(1))
    subscriber.flush()  # seq 1 delivered
    gate_sink.raise_value_error = 1
    _dispatch(subscriber, _l2_store(2, 3))
    subscriber.flush()  # seq 2 rejected
    _dispatch(subscriber, _l2_store(4))
    subscriber.flush()

    assert spool.rejected_events_total() == 2
    assert [(b.seq, b.dropped_events) for b in gate_sink.published] == [
        (1, 0),
        (3, 2),
    ]
    assert gate_sink.loss() == (1, 2, 2)


def test_multi_sink_reports_the_sum_of_its_sinks_drops():
    class _Dropping(_RecordingSink):
        def __init__(self, dropped: int) -> None:
            super().__init__()
            self.dropped = dropped

        def dropped_events(self) -> int:
            return self.dropped

    sink = MultiCacheEventSink([_Dropping(2), _RecordingSink(), _Dropping(3)])
    assert sink.dropped_events() == 5


def test_spool_gauges_report_overflow_retries_and_depth(monkeypatch):
    register = MagicMock()
    monkeypatch.setattr(cache_events, "register_gauge", register)
    gate_sink = _GateSink()
    clock = _Clock()
    spool = _spool(gate_sink, clock, max_entries=1)
    gate_sink.fail_before_delivery = 2
    spool.publish([_batch_for_spool(1)])
    spool.publish([_batch_for_spool(2)])  # evicts seq 1
    clock.now += 1.0
    spool.redeliver()  # fails again: one retry

    cache_events.register_cache_event_metrics(_subscriber(spool), spool)

    gauges = {c.args[1]: c.args[3] for c in register.call_args_list}
    assert gauges["lmcache_mp.cache_events.events_dropped_total"]() == [
        (0, {"reason": "bus_overflow"}),
        (1, {"reason": "spool_overflow"}),
        (0, {"reason": "rejected"}),
    ]
    assert gauges["lmcache_mp.cache_events.publish_retries_total"]() == 1
    assert gauges["lmcache_mp.cache_events.spool_depth"]() == 1


def _batch_for_spool(seq: int) -> CacheEventBatch:
    return CacheEventBatch(
        instance_id="node-a",
        incarnation=7,
        seq=seq,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="fs",
        entries=[_entry(seq, size_bytes=100)],
    )


# -- Bus integration -----------------------------------------------------------


def test_subscriber_on_a_real_bus_flushes_on_events():
    """End-to-end through a real EventBus: published events reach the
    sink via the drain thread's event callbacks, with no dedicated
    emission thread."""
    sink = _RecordingSink()
    bus = EventBus(EventBusConfig(enabled=True))
    bus.register_subscriber(_subscriber(sink, flush_interval=0.0))
    bus.start()
    try:
        bus.publish(
            Event(
                event_type=EventType.L1_WRITE_FINISHED,
                metadata={"keys": [_key(1)], "meta": [_meta(100)]},
            )
        )
        waiter = threading.Event()
        for _ in range(100):
            if sink.published:
                break
            waiter.wait(0.05)
        assert sink.published, "event-driven flush never delivered the batch"
        [[batch]] = sink.published
        assert batch.event_type == CacheEventType.STORE
        assert batch.entries[0].size_bytes == 100
    finally:
        bus.stop()
    # Shutdown closed the sink via the subscriber hook.
    assert sink.closed is True


def test_bus_stop_flushes_buffered_events():
    """Events recorded but not yet flushed are delivered by the
    subscriber's shutdown hook during ``EventBus.stop()``."""
    sink = _RecordingSink()
    bus = EventBus(EventBusConfig(enabled=True))
    bus.register_subscriber(_subscriber(sink))
    bus.start()
    bus.publish(
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1)], "sizes": [64], "backend": "fs"},
        )
    )
    bus.stop()
    assert len(sink.published) == 1
    assert sink.closed is True


# -- HTTP sink end-to-end -------------------------------------------------------


class _SyncASGITransport(httpx.BaseTransport):
    """Bridge httpx's sync client onto an in-process ASGI app."""

    def __init__(self, asgi: httpx.ASGITransport) -> None:
        self._asgi = asgi

    def handle_request(self, request: httpx.Request) -> httpx.Response:
        async def _roundtrip() -> tuple[int, httpx.Headers, bytes]:
            response = await self._asgi.handle_async_request(request)
            content = await response.aread()
            return response.status_code, response.headers, content

        status_code, headers, content = asyncio.run(_roundtrip())
        return httpx.Response(status_code=status_code, headers=headers, content=content)


def test_http_sink_feeds_the_directory_end_to_end():
    """Subscriber events -> emitter -> HTTP sink -> coordinator app ->
    directory lookup, all with the synchronous sink."""
    config = MPCoordinatorConfig(health_check_interval=0.0, eviction_check_interval=0.0)
    asgi = httpx.ASGITransport(app=create_app(config))

    sink = HttpCacheEventSink("http://coordinator")
    # Point the sink's client at the in-process app (same public API).
    sink._client = httpx.Client(  # noqa: SLF001 — test-only transport swap
        transport=_SyncASGITransport(asgi), base_url="http://coordinator"
    )
    subscriber = _subscriber(sink, incarnation=3)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1), _key(2)], "sizes": [100, 200], "backend": "fs"},
        ),
        Event(
            event_type=EventType.L2_KEYS_DELETED,
            metadata={"keys": [_key(2)], "backend": "fs"},
        ),
    )
    subscriber.flush()

    async def _verify() -> None:
        async with httpx.AsyncClient(
            transport=asgi, base_url="http://coordinator"
        ) as client:
            resp = await client.post(
                "/directory/lookup",
                json={
                    "keys": [
                        asdict(_key(1).to_encoded_object_key()),
                        asdict(_key(2).to_encoded_object_key()),
                    ]
                },
            )
            resp.raise_for_status()
            results = resp.json()["results"]
            [placement] = results[0]["placements"]
            assert placement["instance_id"] == "node-a"
            assert placement["incarnation"] == 3
            assert placement["tier"] == "l2"
            assert placement["backend"] == "fs"
            assert placement["size_bytes"] == 100
            assert results[1]["placements"] == []

            stats = (await client.get("/directory/stats")).json()
            assert stats["num_keys"] == 1
            assert stats["num_placements"] == 1

    asyncio.run(_verify())


def test_token_bindings_feed_the_key_directory_end_to_end():
    """Token-binding event + store events -> emitter -> HTTP sink ->
    coordinator app -> key directory bindings, with the synchronous sink."""
    config = MPCoordinatorConfig(
        health_check_interval=0.0,
        eviction_check_interval=0.0,
        enable_blend_lookup=True,
        chunk_size=2,
    )
    app = create_app(config)
    asgi = httpx.ASGITransport(app=app)

    sink = HttpCacheEventSink("http://coordinator")
    sink._client = httpx.Client(  # noqa: SLF001 — test-only transport swap
        transport=_SyncASGITransport(asgi), base_url="http://coordinator"
    )
    subscriber = _subscriber(sink)
    _dispatch(
        subscriber,
        Event(
            event_type=EventType.MP_TOKENS,
            metadata={
                "chunk_hashes": [_key(1).chunk_hash, _key(2).chunk_hash],
                "token_chunks": [[1, 2], [3, 4]],
                "token_offsets": [0, 256],
            },
        ),
        Event(
            event_type=EventType.L2_KEYS_STORED,
            metadata={"keys": [_key(1), _key(2)], "sizes": [100, 200], "backend": "fs"},
        ),
    )
    subscriber.flush()

    key_directory = app.state.ctx.views.get(KeyDirectory)
    assert key_directory.get_token_ids([_key(1).chunk_hash, _key(2).chunk_hash]) == [
        (1, 2),
        (3, 4),
    ]
    # The offsets survive the emitter -> HTTP -> directory round trip, and
    # reach a match as the re-RoPE source position.
    (first,) = key_directory.blend_match(np.asarray([1, 2], dtype=np.uint64), NS)
    (second,) = key_directory.blend_match(np.asarray([3, 4], dtype=np.uint64), NS)
    assert (first.old_st, second.old_st) == (0, 256)


def test_http_sink_raises_publish_error_on_http_failure():
    sink = HttpCacheEventSink("http://127.0.0.1:1")  # nothing listens here
    batch = CacheEventBatch(
        instance_id="node-a",
        incarnation=1,
        seq=1,
        event_type=CacheEventType.STORE,
        tier=Tier.L2,
        backend="fs",
        entries=[_entry(1, 100)],
    )
    with pytest.raises(CacheEventPublishError):
        sink.publish([batch])
    sink.close()


# -- Capacity declarations ----------------------------------------------------


def _snapshot(*modules: ModuleMemoryCapacity) -> CapacitySnapshot:
    """A capacity snapshot as StorageManager publishes it."""
    return CapacitySnapshot(modules=tuple(modules))


def _capacity_event(snapshot: CapacitySnapshot) -> Event:
    """The bus event StorageManager emits on a topology change."""
    return Event(
        event_type=EventType.SM_CAPACITY_CHANGED,
        metadata={"snapshot": snapshot},
    )


def test_a_declaration_becomes_one_config_batch_per_compartment():
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        _capacity_event(
            _snapshot(
                ModuleMemoryCapacity(Tier.L1, "dram", 40 * (1 << 30), False),
                ModuleMemoryCapacity(Tier.L2, "s3", 0, True),
            )
        ),
    )
    subscriber.flush()

    batches = sink.published[0]
    assert [b.event_type for b in batches] == [CacheEventType.CONFIG] * 2
    assert [(b.tier, b.backend, b.capacity_bytes, b.shared) for b in batches] == [
        (Tier.L1, "dram", 40 * (1 << 30), False),
        (Tier.L2, "s3", 0, True),
    ]
    # One declaration, so one revision -- that is what lets the coordinator
    # tell a fresh declaration from a continuation.
    assert {b.capacity_revision for b in batches} == {1}
    # A declaration carries no placements.
    assert all(b.entries == [] for b in batches)


def test_config_batches_share_the_seq_space_with_placements():
    # They ride the same stream, so a reused seq would be dropped as a
    # duplicate by the gate.
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        _capacity_event(
            _snapshot(ModuleMemoryCapacity(Tier.L1, "dram", 8 * (1 << 30), False))
        ),
        Event(
            event_type=EventType.L1_WRITE_FINISHED,
            metadata={"keys": [_key(1)], "meta": [_meta(100)]},
        ),
    )
    subscriber.flush()

    batches = sink.published[0]
    assert [b.seq for b in batches] == [1, 2]
    # Declaration first, so the denominator lands before the bytes.
    assert batches[0].event_type == CacheEventType.CONFIG
    assert batches[1].event_type == CacheEventType.STORE


def test_a_newer_declaration_supersedes_an_unflushed_one():
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        _capacity_event(
            _snapshot(ModuleMemoryCapacity(Tier.L1, "dram", 8 * (1 << 30), False))
        ),
        _capacity_event(
            _snapshot(ModuleMemoryCapacity(Tier.L1, "dram", 16 * (1 << 30), False))
        ),
    )
    subscriber.flush()

    batches = sink.published[0]
    assert [b.capacity_bytes for b in batches] == [16 * (1 << 30)]
    # Coalesced into one declaration, so one revision -- not two burnt.
    assert batches[0].capacity_revision == 1


def test_a_declaration_survives_a_publish_failure():
    # The whole topology, so resending repairs it; a byte delta could not.
    sink = _RecordingSink()
    subscriber = _subscriber(sink)

    _dispatch(
        subscriber,
        _capacity_event(
            _snapshot(ModuleMemoryCapacity(Tier.L1, "dram", 8 * (1 << 30), False))
        ),
    )
    sink.fail_next = True
    subscriber.flush()
    assert sink.published == []

    # Re-emitted at a fresh revision; the coordinator takes the newer one.
    subscriber.flush()
    assert [b.capacity_revision for b in sink.published[0]] == [2]


# -- maybe_create_cache_event_subscriber ---------------------------------------------


def _capture_subscriber(monkeypatch: pytest.MonkeyPatch) -> MagicMock:
    """Replace the subscriber class so the test can read what it was built with."""
    factory = MagicMock(return_value=MagicMock())
    monkeypatch.setattr(cache_events, "CacheEventSubscriber", factory)
    return factory


def test_no_destination_builds_no_subscriber(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        cache_events, "get_active_trace_recorder", MagicMock(return_value=None)
    )
    assert (
        cache_events.maybe_create_cache_event_subscriber(
            MPServerConfig(), HTTPFrontendConfig(), CoordinatorConfig()
        )
        is None
    )


def test_events_trace_alone_builds_a_subscriber_with_a_trace_sink(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """No coordinator URL at all: the trace is the only sink, and the file
    opens with a start mark carrying the server's identity."""
    recorder = EventsTraceRecorder(str(tmp_path / "e.lct"), level_meta={})
    monkeypatch.setattr(
        cache_events,
        "get_active_trace_recorder",
        MagicMock(return_value=recorder),
    )
    factory = _capture_subscriber(monkeypatch)
    mp_config = MPServerConfig(instance_id="node-a")

    subscriber = cache_events.maybe_create_cache_event_subscriber(
        mp_config, HTTPFrontendConfig(http_port=8123), CoordinatorConfig()
    )
    recorder.close()

    assert subscriber is factory.return_value
    kwargs = factory.call_args.kwargs
    assert isinstance(kwargs["sink"], TraceCacheEventSink)
    assert kwargs["instance_id"] == "node-a"
    with TraceReader(str(tmp_path / "e.lct")) as reader:
        records = list(reader.records())
    assert [r.qualname for r in records] == [EVENTS_TRACE_LIFECYCLE]
    assert records[0].args["phase"] == "start"
    assert records[0].args["instance_id"] == "node-a"
    assert records[0].args["http_port"] == 8123
    assert records[0].args["incarnation"] == kwargs["incarnation"]


def test_coordinator_and_trace_together_fan_out(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    recorder = EventsTraceRecorder(str(tmp_path / "e.lct"), level_meta={})
    monkeypatch.setattr(
        cache_events,
        "get_active_trace_recorder",
        MagicMock(return_value=recorder),
    )
    coordinator_sink = MagicMock(spec=CacheEventSink)
    monkeypatch.setattr(
        cache_events,
        "create_cache_event_sink",
        MagicMock(return_value=coordinator_sink),
    )
    factory = _capture_subscriber(monkeypatch)

    cache_events.maybe_create_cache_event_subscriber(
        MPServerConfig(),
        HTTPFrontendConfig(),
        CoordinatorConfig(url="http://coordinator:9300", event_reporting=True),
    )
    recorder.close()

    assert isinstance(factory.call_args.kwargs["sink"], MultiCacheEventSink)


def test_reporting_without_http_frontend_records_only_the_trace(
    monkeypatch: pytest.MonkeyPatch, tmp_path
) -> None:
    """Without the HTTP frontend there is nothing to register with the
    coordinator, so its sink is not built even when reporting is on."""
    recorder = EventsTraceRecorder(str(tmp_path / "e.lct"), level_meta={})
    monkeypatch.setattr(
        cache_events,
        "get_active_trace_recorder",
        MagicMock(return_value=recorder),
    )
    create_sink = MagicMock()
    monkeypatch.setattr(cache_events, "create_cache_event_sink", create_sink)
    factory = _capture_subscriber(monkeypatch)

    cache_events.maybe_create_cache_event_subscriber(
        MPServerConfig(),
        None,
        CoordinatorConfig(url="http://coordinator:9300", event_reporting=True),
    )
    recorder.close()

    create_sink.assert_not_called()
    assert isinstance(factory.call_args.kwargs["sink"], TraceCacheEventSink)
