# SPDX-License-Identifier: Apache-2.0
"""MP-server-side cache-event emission for the coordinator key directory.

A :class:`CacheEventSubscriber` on the observability event bus turns the
storage layer's L1/L2 key events (plus the store path's token-binding
events) into ordered :class:`CacheEventBatch`
lists and delivers them through a :class:`CacheEventSink` — the
transport seam (direct HTTP or Kafka). Mapping, batching,
and delivery all run on the bus's drain thread; there is no dedicated
emission thread or task. See
``docs/design/v1/mp_coordinator/cache_events.md``.
"""

# Standard
from abc import ABC, abstractmethod
from collections import OrderedDict, deque
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from enum import Enum
from typing import TYPE_CHECKING
import math
import time

# Third Party
import httpx

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import CapacitySnapshot, L1BackendType, ObjectKey, Tier
from lmcache.v1.distributed.internal_api import L1ObjectMeta
from lmcache.v1.mp_coordinator.api import (
    UNKNOWN_TOKEN_OFFSET,
    CacheEventBatch,
    CacheEventEntry,
    CacheEventType,
)
from lmcache.v1.mp_coordinator.schemas import CacheEventsRequest
from lmcache.v1.mp_observability.event import Event, EventType
from lmcache.v1.mp_observability.event_bus import (
    EventBus,
    EventCallback,
    EventSubscriber,
)
from lmcache.v1.mp_observability.otel_init import register_gauge
from lmcache.v1.mp_observability.trace.lifecycle import get_active_trace_recorder
from lmcache.v1.mp_observability.trace.recorder import EventsTraceRecorder
from lmcache.v1.multiprocess.config import (
    CoordinatorConfig,
    HTTPFrontendConfig,
    KafkaCacheEventSinkConfig,
    MPServerConfig,
)

if TYPE_CHECKING:
    # Third Party
    from confluent_kafka import KafkaError, Message

logger = init_logger(__name__)

_DEFAULT_FLUSH_INTERVAL = 1.0

# Seconds closing the Kafka sink waits for queued records.
_KAFKA_SHUTDOWN_FLUSH_TIMEOUT = 10.0
# Kafka producer buffer cap in KiB (64 MiB).
_KAFKA_MAX_BUFFER_KBYTES = 64 * 1024

# Token-binding cache bound: covers the window between a chunk's
# token-binding event and its last (async L2) store event.
_TOKEN_BINDING_CACHE_SIZE = 65536

# HTTP spool bound, counted in cache-event entries (a batch with no
# entries counts as one). Roughly 150 MB even when every entry carries a
# 256-token chunk's ids.
_SPOOL_MAX_ENTRIES = 16384
# Retry backoff after a failed post: doubles from the initial value up to
# the cap, in seconds.
_SPOOL_RETRY_INITIAL_BACKOFF = 0.5
_SPOOL_RETRY_MAX_BACKOFF = 30.0
# Most entries one spool redelivery sends per publish call, so draining a
# full spool after an outage is several bounded posts rather than one
# post too large to finish within the request timeout.
_SPOOL_PUBLISH_SLICE_ENTRIES = 2048

_METER_NAME = "lmcache.mp_server"


class CacheEventPublishError(Exception):
    """A sink failed to deliver a list of cache-event batches."""


class CacheEventRejectedError(CacheEventPublishError):
    """The receiver refused the batches; resending them cannot succeed."""


# 4xx statuses that are worth retrying: request timeout and rate limiting.
_RETRYABLE_CLIENT_STATUSES = frozenset({408, 429})


class CacheEventSink(ABC):
    """Transport seam for delivering cache-event batches to the directory.

    Successful calls preserve batch order within and across
    :meth:`publish` calls. Retrying an uncertain call is safe because the
    gate deduplicates sequence numbers. Dropping a failed call consumes
    sequence numbers and exposes a gap; a non-durable source such as HTTP
    cannot repair it.
    """

    @abstractmethod
    def publish(self, batches: list[CacheEventBatch]) -> None:
        """Deliver ``batches`` to the directory, in list order.

        A sink may instead retain the batches and return without raising;
        it then delivers them on a later :meth:`publish` or
        :meth:`redeliver` call, or drops them and counts their entries
        in :meth:`dropped_events`.

        Args:
            batches: The batches to deliver.

        Raises:
            CacheEventPublishError: If delivery failed. Retrying is safe;
                dropping may leave the coordinator's view stale. Replay can
                repair only events already retained by a durable source.
        """
        raise NotImplementedError

    def redeliver(self) -> None:  # noqa: B027
        """Retry, best effort, the batches an earlier :meth:`publish`
        retained (see there); never raises. A no-op for sinks that retain
        nothing.
        The subscriber calls it on flushes with nothing new to publish."""
        pass

    def dropped_events(self) -> int:
        """Count the cache events this sink dropped without raising.

        The subscriber adds this count to the ``dropped_events`` it
        stamps on later batches. A sink that reports a failure by
        raising leaves the count alone.

        Returns:
            Cumulative entries of the batches dropped so far; 0 for
            sinks that do not count their drops this way (e.g. Kafka,
            whose dropped records show up as ``seq`` gaps only).
        """
        return 0

    def close(self) -> None:  # noqa: B027
        """Release transport resources. Called once at shutdown."""
        pass


class HttpCacheEventSink(CacheEventSink):
    """Sink that POSTs batches to the coordinator's ``/events``.

    Owns a synchronous HTTP client: publishing happens on the event
    bus's drain thread, so the request timeout bounds how long a flush
    can stall event dispatch.

    Args:
        coordinator_url: Coordinator base URL.
        timeout: Per-request timeout in seconds.
    """

    def __init__(self, coordinator_url: str, timeout: float = 2.0) -> None:
        self._base_url = coordinator_url.rstrip("/")
        self._client = httpx.Client(timeout=timeout)

    def publish(self, batches: list[CacheEventBatch]) -> None:
        """Deliver ``batches`` via one ``POST /events`` request.

        Args:
            batches: The batches to deliver.

        Raises:
            CacheEventRejectedError: If the coordinator answered with a
                4xx status other than 408 or 429.
            CacheEventPublishError: If the request failed or returned
                another non-2xx status.
        """
        body = CacheEventsRequest(batches=batches)
        try:
            resp = self._client.post(
                f"{self._base_url}/events",
                json=body.model_dump(mode="json"),
            )
            resp.raise_for_status()
        except httpx.HTTPStatusError as e:
            status = e.response.status_code
            error = (
                CacheEventRejectedError
                if 400 <= status < 500 and status not in _RETRYABLE_CLIENT_STATUSES
                else CacheEventPublishError
            )
            raise error(
                f"failed to publish {len(batches)} cache-event batches to "
                f"{self._base_url}: {e}"
            ) from e
        except httpx.HTTPError as e:
            raise CacheEventPublishError(
                f"failed to publish {len(batches)} cache-event batches to "
                f"{self._base_url}: {e}"
            ) from e

    def close(self) -> None:
        """Close the HTTP client."""
        self._client.close()


class KafkaCacheEventSink(CacheEventSink):
    """Sink that publishes cache-event batches as keyed Kafka records.

    Each :class:`CacheEventBatch` becomes one JSON record keyed by
    ``instance_id``. Kafka therefore assigns every batch from one emitter to
    the same partition, preserving the per-instance order required by the
    coordinator.

    :meth:`publish` does not wait for the broker; the producer retries in
    order for up to ``delivery_timeout``. A record that does not fit the
    buffer or is never delivered is dropped and counted, leaving a seq gap.

    ``confluent-kafka`` is the optional ``lmcache[kafka]`` extra and is
    imported only here, so deployments on the HTTP transport never load it.

    Args:
        config: Validated Kafka connection and delivery settings.

    Raises:
        ImportError: If ``confluent-kafka`` is not installed.
    """

    def __init__(self, config: KafkaCacheEventSinkConfig) -> None:
        try:
            # Third Party
            from confluent_kafka import KafkaException, Producer
        except ImportError as e:
            raise ImportError(
                "The kafka cache-event transport needs confluent-kafka: "
                "pip install 'lmcache[kafka]'"
            ) from e
        self._kafka_exception: type[Exception] = KafkaException
        self._topic = config.topic
        self._dropped_batches = 0
        self._producer = Producer(
            {
                "bootstrap.servers": config.bootstrap_servers,
                "client.id": "lmcache-cache-events",
                "enable.idempotence": True,
                "acks": "all",
                "message.timeout.ms": math.ceil(config.delivery_timeout * 1000),
                "queue.buffering.max.kbytes": _KAFKA_MAX_BUFFER_KBYTES,
            }
        )

    @property
    def dropped_batches(self) -> int:
        """Return how many batches were dropped so far."""
        return self._dropped_batches

    def publish(self, batches: list[CacheEventBatch]) -> None:
        """Queue batches in list order without waiting for the broker.

        Args:
            batches: Batches to publish. Each becomes one keyed Kafka record.

        Raises:
            CacheEventPublishError: If the producer refused a batch; it and
                the rest of the list are dropped.
        """
        for queued, batch in enumerate(batches):
            payload = CacheEventsRequest(batches=[batch]).model_dump_json().encode()
            try:
                self._producer.produce(
                    topic=self._topic,
                    key=batch.instance_id.encode(),
                    value=payload,
                    on_delivery=self._on_delivery,
                )
            except (BufferError, self._kafka_exception) as e:
                dropped = len(batches) - queued
                self._dropped_batches += dropped
                raise CacheEventPublishError(
                    f"Kafka producer refused {dropped} of {len(batches)} "
                    f"cache-event batches for topic {self._topic!r}: {e}"
                ) from e
            finally:
                self._producer.poll(0)

    def close(self) -> None:
        """Flush queued records, waiting at most 10 seconds."""
        try:
            remaining = self._producer.flush(_KAFKA_SHUTDOWN_FLUSH_TIMEOUT)
        except self._kafka_exception as e:
            logger.warning(
                "Failed to flush Kafka cache-event producer during shutdown: %s",
                e,
            )
            return
        if remaining:
            logger.warning(
                "%d Kafka cache-event record(s) remained queued at shutdown",
                remaining,
            )

    def _on_delivery(self, error: "KafkaError | None", message: "Message") -> None:
        """Count and log a record the producer gave up on.

        Args:
            error: Why delivery failed, or ``None`` on success.
            message: The reported record.
        """
        if error is None:
            return
        self._dropped_batches += 1
        logger.warning(
            "Kafka did not deliver a cache-event record for instance %r "
            "(%d dropped so far): %s",
            message.key(),
            self._dropped_batches,
            error,
        )


#: ``Record.qualname`` of one cache-event batch in an ``events``-level trace.
#: ``args`` is the batch in wire form: one element of
#: ``CacheEventsRequest.batches`` as ``POST /events`` would carry it.
EVENTS_TRACE_BATCH = "events.batch"
#: ``Record.qualname`` of a lifecycle mark in an ``events``-level trace.
#: ``args`` carries ``phase`` (:class:`TraceLifecyclePhase`) and
#: ``instance_id``; at start also the rest of the emitter's identity:
#: ``incarnation``, ``ip``, ``http_port``, ``mq_port``.
EVENTS_TRACE_LIFECYCLE = "events.lifecycle"


class TraceLifecyclePhase(str, Enum):
    """Where in the emitter's life an ``events.lifecycle`` record was written."""

    START = "start"
    """The subscriber began emitting; the record carries its identity."""
    STOP = "stop"
    """The sink closed at shutdown; no batch follows in this file."""


class TraceCacheEventSink(CacheEventSink):
    """Sink that appends every batch to an ``events``-level trace file.

    Each batch becomes one :data:`EVENTS_TRACE_BATCH` record holding the
    exact wire form ``HttpCacheEventSink`` would have posted, so a replayer
    can hand the file's records to a coordinator's ``POST /events`` with no
    conversion. Nothing is needed on the other end: a server with no
    coordinator configured can record what it would have reported.

    Runs on the bus's drain thread like every sink. The recorder's own lock
    serializes the file, and its error handling counts a failed write
    rather than raising, so a full disk never stalls event dispatch.

    Args:
        recorder: The open ``events``-level recorder to write into.
    """

    def __init__(self, recorder: EventsTraceRecorder) -> None:
        self._recorder = recorder
        self._instance_id = ""

    def record_lifecycle(
        self,
        phase: TraceLifecyclePhase,
        instance_id: str = "",
        incarnation: int = 0,
        ip: str = "",
        http_port: int = 0,
        mq_port: int = 0,
    ) -> None:
        """Write one :data:`EVENTS_TRACE_LIFECYCLE` record.

        Every mark names the emitter, so a file's marks stay attributable
        once its records are merged with other servers'. The ``STOP`` mark
        takes the id from the ``START`` mark written before it.

        Args:
            phase: Which mark this is.
            instance_id: The emitter's id (``START`` only).
            incarnation: The emitter's incarnation (``START`` only).
            ip: The IP the emitter advertises to a coordinator, or empty
                when it defers to its outbound address (``START`` only).
            http_port: The emitter's HTTP port (``START`` only).
            mq_port: The emitter's message-queue port, ``0`` when P2P is
                off (``START`` only).
        """
        if phase is TraceLifecyclePhase.START:
            self._instance_id = instance_id
        args: dict[str, object] = {
            "phase": phase.value,
            "instance_id": self._instance_id,
        }
        if phase is TraceLifecyclePhase.START:
            args.update(
                incarnation=incarnation,
                ip=ip,
                http_port=http_port,
                mq_port=mq_port,
            )
        self._recorder.write_record(
            EVENTS_TRACE_LIFECYCLE, args, t_wall=time.time(), t_mono=time.monotonic()
        )

    def publish(self, batches: list[CacheEventBatch]) -> None:
        """Append ``batches`` to the trace, one record each, in list order.

        Args:
            batches: The batches to record.
        """
        t_wall = time.time()
        t_mono = time.monotonic()
        wire = CacheEventsRequest(batches=batches).model_dump(mode="json")
        for batch in wire["batches"]:
            self._recorder.write_record(
                EVENTS_TRACE_BATCH, batch, t_wall=t_wall, t_mono=t_mono
            )

    def close(self) -> None:
        """Mark the end of the stream. The recorder closes with the bus."""
        self.record_lifecycle(TraceLifecyclePhase.STOP)


class MultiCacheEventSink(CacheEventSink):
    """Sink that delivers every batch list to several sinks in turn.

    Used when a server both reports to a coordinator and records an
    ``events`` trace. Every sink is attempted even if an earlier one
    fails, so a coordinator outage does not stop the recording (or the
    reverse); the failures are then raised together.

    Args:
        sinks: The sinks to deliver to, in order.

    Raises:
        ValueError: If ``sinks`` is empty.
    """

    def __init__(self, sinks: Sequence[CacheEventSink]) -> None:
        if not sinks:
            raise ValueError("MultiCacheEventSink needs at least one sink")
        self._sinks = tuple(sinks)

    def publish(self, batches: list[CacheEventBatch]) -> None:
        """Deliver ``batches`` to every sink.

        Args:
            batches: The batches to deliver.

        Raises:
            CacheEventPublishError: If any sink failed, after every sink
                was tried; the message names each failure.
        """
        failures: list[str] = []
        for sink in self._sinks:
            try:
                sink.publish(batches)
            except CacheEventPublishError as e:
                failures.append(f"{type(sink).__name__}: {e}")
        if failures:
            raise CacheEventPublishError(
                f"{len(failures)} of {len(self._sinks)} cache-event sinks failed: "
                + "; ".join(failures)
            )

    def redeliver(self) -> None:
        """Let every sink retry what it retained."""
        for sink in self._sinks:
            sink.redeliver()

    def dropped_events(self) -> int:
        """Sum the sinks' silently dropped cache events.

        Returns:
            The total over every sink.
        """
        return sum(sink.dropped_events() for sink in self._sinks)

    def close(self) -> None:
        """Close every sink, in order."""
        for sink in self._sinks:
            sink.close()


def _spool_cost(batch: CacheEventBatch) -> int:
    """Spool cost of ``batch``: its entry count, at least one."""
    return max(1, len(batch.entries))


class SpoolingCacheEventSink(CacheEventSink):
    """Sink that retains failed batches and redelivers them in seq order.

    Wraps a non-durable transport (HTTP): every published batch joins the
    tail of a bounded spool, and the spool drains head-first into the
    inner sink. A failed attempt keeps the spool and backs off (doubling,
    capped); later :meth:`publish` and :meth:`redeliver` calls retry once
    the backoff has elapsed. Batches never overtake each other, so a
    retried batch reaches the coordinator before any later ``seq``.
    Redelivery after an uncertain failure may duplicate a batch the
    coordinator already applied; the gate's ``seq`` dedup drops it.

    When what a failed attempt left retained exceeds the bound, the
    oldest batches are dropped. A slice the inner sink rejects
    (:class:`CacheEventRejectedError`) or fails on with any exception
    other than :class:`CacheEventPublishError` is dropped too, and
    draining continues. Either way the ``seq`` numbers were assigned
    already, so the coordinator sees a gap, and the dropped entries are
    counted in :meth:`dropped_events` for later batches to report.

    Runs on the bus's drain thread like every sink; not thread-safe.

    Args:
        inner: The transport to deliver through.
        max_entries: Spool bound in entries (a batch with no entries
            counts as one).
        initial_backoff: Seconds to wait after the first failure.
        max_backoff: Cap on the doubling backoff, in seconds.
        clock: Monotonic clock in seconds (injectable for tests).
    """

    def __init__(
        self,
        inner: CacheEventSink,
        max_entries: int = _SPOOL_MAX_ENTRIES,
        initial_backoff: float = _SPOOL_RETRY_INITIAL_BACKOFF,
        max_backoff: float = _SPOOL_RETRY_MAX_BACKOFF,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._inner = inner
        self._max_entries = max_entries
        self._initial_backoff = initial_backoff
        self._max_backoff = max_backoff
        self._clock = clock
        self._spool: deque[CacheEventBatch] = deque()
        self._spooled_entries = 0
        # 0 while healthy; otherwise the backoff after the last failure.
        self._backoff = 0.0
        self._next_attempt = 0.0
        self._retries_total = 0
        self._overflow_events_total = 0
        self._rejected_events_total = 0

    def publish(self, batches: list[CacheEventBatch]) -> None:
        """Spool ``batches`` behind any retained ones and try to drain.

        Never raises: the batches are delivered, retained for a retry, or
        (spool full) dropped as a visible ``seq`` gap.

        Args:
            batches: The batches to deliver, in ``seq`` order.
        """
        for batch in batches:
            self._spool.append(batch)
            self._spooled_entries += _spool_cost(batch)
        self._drain()
        self._trim()

    def redeliver(self) -> None:
        """Retry the retained batches if the backoff has elapsed."""
        self._drain()

    def close(self) -> None:
        """Make one last delivery attempt, then close the inner sink."""
        self._next_attempt = 0.0
        self._drain()
        if self._spool:
            logger.warning(
                "Abandoning %d spooled cache-event batches at shutdown",
                len(self._spool),
            )
        self._inner.close()

    def spool_depth(self) -> int:
        """Measure the spool.

        Returns:
            Entries currently retained for delivery (a batch with no
            entries counts as one).
        """
        return self._spooled_entries

    def retries_total(self) -> int:
        """Count retries.

        Returns:
            Cumulative publish attempts made after a failed one.
        """
        return self._retries_total

    def overflow_events_total(self) -> int:
        """Count overflow drops.

        Returns:
            Cumulative entries dropped because the spool was full.
        """
        return self._overflow_events_total

    def rejected_events_total(self) -> int:
        """Count rejection drops.

        Returns:
            Cumulative entries dropped because a post cannot succeed.
        """
        return self._rejected_events_total

    def dropped_events(self) -> int:
        """Count the entries dropped on overflow or rejection.

        Returns:
            Their cumulative total.
        """
        return self._overflow_events_total + self._rejected_events_total

    def _trim(self) -> None:
        """Drop the oldest batches until the spool is within its bound.
        Called after a drain attempt, so it only trims what an outage left
        retained."""
        dropped = 0
        dropped_entries = 0
        while self._spool and self._spooled_entries > self._max_entries:
            batch = self._spool.popleft()
            self._spooled_entries -= _spool_cost(batch)
            dropped += 1
            dropped_entries += len(batch.entries)
        if dropped:
            self._overflow_events_total += dropped_entries
            logger.warning(
                "Cache-event spool full (%d entries): dropped the %d oldest "
                "batches (%d entries); later batches report them as lost",
                self._max_entries,
                dropped,
                dropped_entries,
            )

    def _drain(self) -> None:
        """Deliver the spool head-first in bounded slices until it is
        empty or an attempt fails retryably (which arms the backoff).
        A slice that cannot succeed is dropped."""
        if not self._spool or self._clock() < self._next_attempt:
            return
        while self._spool:
            batch_slice: list[CacheEventBatch] = []
            units = 0
            for batch in self._spool:
                cost = _spool_cost(batch)
                if batch_slice and units + cost > _SPOOL_PUBLISH_SLICE_ENTRIES:
                    break
                batch_slice.append(batch)
                units += cost
            if self._backoff > 0:
                self._retries_total += 1
            try:
                self._inner.publish(batch_slice)
            except CacheEventRejectedError as e:
                self._reject(batch_slice, e)
            except CacheEventPublishError as e:
                self._backoff = min(
                    self._max_backoff, max(self._initial_backoff, self._backoff * 2)
                )
                # Count the backoff from the end of the failed attempt: a
                # timeout can last longer than the backoff it arms.
                self._next_attempt = self._clock() + self._backoff
                logger.warning(
                    "Cache-event publish failed; %d batches spooled, retrying "
                    "in %.1fs: %s",
                    len(self._spool),
                    self._backoff,
                    e,
                )
                return
            except Exception as e:
                # Not a transport failure (e.g. a batch that fails
                # validation): resending the same slice would fail again.
                self._reject(batch_slice, e)
            for _ in batch_slice:
                self._spool.popleft()
            self._spooled_entries -= units
            self._backoff = 0.0
            self._next_attempt = 0.0

    def _reject(self, batch_slice: list[CacheEventBatch], error: Exception) -> None:
        """Count and log a slice that is dropped because it cannot succeed.

        Args:
            batch_slice: The slice being dropped.
            error: Why it cannot be delivered.
        """
        entries = sum(len(batch.entries) for batch in batch_slice)
        self._rejected_events_total += entries
        logger.warning(
            "Dropping %d cache-event batches (%d entries) that cannot be "
            "delivered; later batches report them as lost: %r",
            len(batch_slice),
            entries,
            error,
        )


@dataclass(frozen=True)
class _ChunkTokens:
    """One chunk's token content, held between its token-binding event
    and the store events that carry it to the directory.

    Attributes:
        token_ids: The chunk's token ids.
        token_offset: Position of its first token in the stored sequence.
    """

    token_ids: tuple[int, ...]
    token_offset: int


# Stands in for a chunk the binding cache does not know, so a STORE entry
# is built the same way whether or not its tokens are still held.
_NO_BINDING = _ChunkTokens(token_ids=(), token_offset=UNKNOWN_TOKEN_OFFSET)


@dataclass
class _PendingBatch:
    """A buffered batch-to-be: entries sharing one ``(event_type, tier,
    backend, shared)`` identity. Becomes exactly one
    :class:`CacheEventBatch` at flush, which stamps ``seq`` and ``ts``."""

    event_type: CacheEventType
    tier: Tier
    backend: str
    shared: bool
    entries: list[CacheEventEntry]


class CacheEventSubscriber(EventSubscriber):
    """Event-bus subscriber that emits the fleet cache-event stream.

    Not thread-safe by design: every method runs on the bus's single
    drain thread (``EventBus.stop()`` invokes :meth:`shutdown` only
    after that thread has been joined), so no locking is needed.

    Args:
        sink: Transport that delivers flushed batches.
        instance_id: This MP server's id (sent with every batch).
        incarnation: This server process's incarnation (its start time);
            fences out placements reported before a restart.
        flush_interval: Minimum seconds between event-driven flushes
            (must be >= 0).

    Raises:
        ValueError: If ``flush_interval`` is negative.
    """

    def __init__(
        self,
        sink: CacheEventSink,
        instance_id: str,
        incarnation: int,
        flush_interval: float = _DEFAULT_FLUSH_INTERVAL,
    ) -> None:
        if flush_interval < 0:
            raise ValueError(f"flush_interval must be >= 0 (got {flush_interval})")
        self._sink = sink
        self._instance_id = instance_id
        self._incarnation = incarnation
        self._flush_interval = flush_interval
        self._last_flush = time.monotonic()
        self._seq = 0
        # Consecutive same-identity entries append to the last pending
        # batch; an identity change starts a new one (order-preserving).
        self._pending_batches: list[_PendingBatch] = []
        # At most one pending declaration: each is the whole topology, so
        # a newer one supersedes rather than queues behind an older.
        self._pending_capacity: CapacitySnapshot | None = None
        # Numbered here, beside _seq, and for the same reason: the bus
        # drains on one thread, so neither counter needs a lock, and the
        # number cannot come apart from the topology it labels. Coalesced
        # publishes therefore share one revision instead of burning several.
        self._capacity_revision = 0
        # Chunk hash → token content from token-binding events (published
        # ahead of the write-finished events), used to stamp STORE
        # entries. LRU-bounded; a miss stamps nothing.
        self._token_bindings: OrderedDict[bytes, _ChunkTokens] = OrderedDict()
        # Bus overflow drops events before they reach us. register() binds
        # this to the bus's count of keys dropped among the types we
        # consume (one cache event per key); flush stamps the growth since
        # registration into every batch as ``dropped_events``.
        self._read_bus_drops: Callable[[], int] = lambda: 0
        self._bus_drops_at_register = 0

    def register(self, bus: EventBus) -> None:
        """Subscribe to ``bus`` and watch its overflow drops of the
        events this subscriber consumes. Drops from before this call are
        not counted: the subscriber would not have received those events.

        Args:
            bus: The bus to subscribe to.
        """
        super().register(bus)
        consumed = list(self.get_subscriptions())
        self._read_bus_drops = lambda: bus.dropped_keys_count_for(consumed)
        self._bus_drops_at_register = self._read_bus_drops()

    def bus_dropped_events_total(self) -> int:
        """Count the cache events lost to bus overflow.

        Returns:
            Keys carried by the consumed events the bus dropped since
            :meth:`register`, one cache event each. Events without keys
            (the flush tick, token bindings, capacity changes) count none.
        """
        return self._read_bus_drops() - self._bus_drops_at_register

    def get_subscriptions(self) -> dict[EventType, EventCallback]:
        """Return the bus events this subscriber consumes."""
        return {
            EventType.L1_WRITE_FINISHED: self._on_l1_store,
            EventType.L1_WRITE_FINISHED_AND_READ_RESERVED: self._on_l1_store,
            EventType.L1_KEYS_EVICTED: self._on_l1_delete,
            EventType.L1_KEYS_ACCESSED: self._on_l1_access,
            EventType.L2_KEYS_STORED: self._on_l2_store,
            EventType.L2_KEYS_DELETED: self._on_l2_delete,
            EventType.L2_KEYS_ACCESSED: self._on_l2_access,
            EventType.MP_TOKENS: self._on_tokens,
            EventType.SM_CAPACITY_CHANGED: self._on_capacity_changed,
            # TODO: decouple the flush tick from the eviction loop (e.g. a
            # bus-owned periodic hook) so cache-event freshness does not
            # silently depend on the eviction loop's cadence.
            EventType.L1_EVICTION_LOOP_TICK: self._on_tick,
        }

    def flush(self) -> None:
        """Drain the buffer and publish one batch per pending batch.

        Every batch carries ``dropped_events``: the cache events lost so
        far, to bus overflow or dropped by the sink
        (:meth:`CacheEventSink.dropped_events`). With nothing to publish, the sink
        retries what it retained instead. A sink that raises loses the
        drained list (its seqs stay consumed); the failure is logged.
        """
        if not self._pending_batches and self._pending_capacity is None:
            self._sink.redeliver()
            return
        pending_batches = self._pending_batches
        self._pending_batches = []
        capacity = self._pending_capacity
        self._pending_capacity = None
        ts = time.time()
        dropped_events = self.bus_dropped_events_total() + self._sink.dropped_events()
        # Declaration first, so a flush that also carries placements gives
        # the coordinator its denominator before the bytes it divides.
        batches = self._capacity_batches(capacity, ts, dropped_events)
        batches += [
            CacheEventBatch(
                instance_id=self._instance_id,
                incarnation=self._incarnation,
                seq=self._seq + len(batches) + offset + 1,
                event_type=pending.event_type,
                tier=pending.tier,
                backend=pending.backend,
                entries=pending.entries,
                shared=pending.shared,
                ts=ts,
                dropped_events=dropped_events,
            )
            for offset, pending in enumerate(pending_batches)
        ]
        self._seq += len(batches)
        try:
            self._sink.publish(batches)
        except CacheEventPublishError as e:
            # Placement batches are lost for good, but a declaration is the
            # whole topology, so restore it for the next flush. A newer one
            # arriving first just supersedes it.
            if capacity is not None and self._pending_capacity is None:
                self._pending_capacity = capacity
            logger.warning(
                "Cache-event publish failed (instance %s); unsent batches "
                "are dropped and leave a seq gap: %s",
                self._instance_id,
                e,
            )

    def _capacity_batches(
        self, capacity: "CapacitySnapshot | None", ts: float, dropped_events: int
    ) -> list[CacheEventBatch]:
        """Expand one declaration into a ``config`` batch per compartment.

        Bumps the revision once and stamps every batch of the declaration
        with it, which is what lets the coordinator tell a fresh declaration
        from a continuation and retire compartments the new one omits.

        Args:
            capacity: The declaration to expand, or ``None`` for no
                declaration this flush.
            ts: Emitter wall-clock seconds to stamp the batches with.
            dropped_events: The lost-event count to stamp the batches with.

        Returns:
            One batch per compartment, seq-numbered from the current
            cursor; empty when there is nothing to declare.
        """
        if capacity is None:
            return []
        self._capacity_revision += 1
        return [
            CacheEventBatch(
                instance_id=self._instance_id,
                incarnation=self._incarnation,
                seq=self._seq + offset + 1,
                event_type=CacheEventType.CONFIG,
                tier=module.tier,
                backend=module.backend,
                shared=module.shared,
                ts=ts,
                capacity_bytes=module.capacity_bytes,
                capacity_revision=self._capacity_revision,
                dropped_events=dropped_events,
            )
            for offset, module in enumerate(capacity.modules)
        ]

    def shutdown(self) -> None:
        """Flush buffered events and close the sink. Called by
        ``EventBus.stop()`` after the final drain."""
        self.flush()
        self._sink.close()

    # -- Event handlers (bus drain thread) ------------------------------------

    def _on_capacity_changed(self, event: Event) -> None:
        """Hold the new declaration for the next flush."""
        snapshot: CapacitySnapshot = event.metadata["snapshot"]
        self._pending_capacity = snapshot
        self._flush_if_due()

    def _on_tick(self, event: Event) -> None:
        self._flush_if_due()

    def _on_l1_store(self, event: Event) -> None:
        self._record_l1_placements(CacheEventType.STORE, event)

    def _on_l1_delete(self, event: Event) -> None:
        self._record_l1_placements(CacheEventType.DELETE, event)

    def _on_l1_access(self, event: Event) -> None:
        keys: list[ObjectKey] = event.metadata["keys"]
        # ACCESS updates key-level recency and access count only; it
        # carries no placement identity, so the backend is empty by contract.
        self._record(
            CacheEventType.ACCESS,
            Tier.L1,
            "",
            [CacheEventEntry(key=key.to_encoded_object_key()) for key in keys],
        )

    def _on_l2_store(self, event: Event) -> None:
        keys: list[ObjectKey] = event.metadata["keys"]
        sizes: list[int] = event.metadata["sizes"]
        self._record(
            CacheEventType.STORE,
            Tier.L2,
            event.metadata["backend"],
            [
                self._store_entry(key, size)
                for key, size in zip(keys, sizes, strict=True)
            ],
            shared=event.metadata.get("shared", False),
        )

    def _on_tokens(self, event: Event) -> None:
        chunk_hashes: list[bytes] = event.metadata["chunk_hashes"]
        token_chunks: list[list[int]] = event.metadata["token_chunks"]
        token_offsets: list[int] = event.metadata["token_offsets"]
        for chunk_hash, chunk, offset in zip(
            chunk_hashes, token_chunks, token_offsets, strict=True
        ):
            self._token_bindings[chunk_hash] = _ChunkTokens(
                token_ids=tuple(chunk), token_offset=offset
            )
            self._token_bindings.move_to_end(chunk_hash)
        if len(self._token_bindings) <= _TOKEN_BINDING_CACHE_SIZE:
            return
        # Evict in one batch down to half the bound: the next eviction is
        # then thousands of stores away, so this stays a rare event (and
        # a rare log line) instead of firing on every store while full.
        evicted = 0
        while len(self._token_bindings) > _TOKEN_BINDING_CACHE_SIZE // 2:
            self._token_bindings.popitem(last=False)
            evicted += 1
        logger.warning(
            "Token binding cache hit its %d-entry bound: evicted the %d oldest "
            "bindings; STORE events for those chunks carry no token ids "
            "(stores completing far behind their submission)",
            _TOKEN_BINDING_CACHE_SIZE,
            evicted,
        )

    def _on_l2_delete(self, event: Event) -> None:
        self._record_l2_keys(CacheEventType.DELETE, event)

    def _on_l2_access(self, event: Event) -> None:
        self._record_l2_keys(CacheEventType.ACCESS, event)

    # -- Internals -------------------------------------------------------------

    def _record_l1_placements(self, event_type: CacheEventType, event: Event) -> None:
        """Record one ``event_type`` batch per medium found in the
        event's ``meta`` list (parallel to ``keys``)."""
        keys: list[ObjectKey] = event.metadata["keys"]
        metadata: list[L1ObjectMeta] = event.metadata["meta"]
        by_backend: dict[L1BackendType, list[CacheEventEntry]] = {}
        is_store = event_type is CacheEventType.STORE
        for key, meta in zip(keys, metadata, strict=True):
            by_backend.setdefault(meta.backend, []).append(
                self._store_entry(key, meta.size_bytes)
                if is_store
                else CacheEventEntry(key=key.to_encoded_object_key())
            )
        for backend, entries in by_backend.items():
            self._record(event_type, Tier.L1, backend.value, entries)

    def _record_l2_keys(self, event_type: CacheEventType, event: Event) -> None:
        """Record a size-less L2 batch (deletes and accesses)."""
        keys: list[ObjectKey] = event.metadata["keys"]
        self._record(
            event_type,
            Tier.L2,
            event.metadata["backend"],
            [CacheEventEntry(key=key.to_encoded_object_key()) for key in keys],
            shared=event.metadata.get("shared", False),
        )

    def _store_entry(self, key: ObjectKey, size_bytes: int) -> CacheEventEntry:
        """Build a STORE entry, stamping the chunk's token content when the
        token-binding cache knows the chunk."""
        binding = self._token_bindings.get(key.chunk_hash, _NO_BINDING)
        return CacheEventEntry(
            key=key.to_encoded_object_key(),
            size_bytes=size_bytes,
            token_ids=list(binding.token_ids),
            token_offset=binding.token_offset,
        )

    def _record(
        self,
        event_type: CacheEventType,
        tier: Tier,
        backend: str,
        entries: list[CacheEventEntry],
        shared: bool = False,
    ) -> None:
        """Buffer ``entries``, then flush if the flush interval elapsed."""
        if not entries:
            return
        last = self._pending_batches[-1] if self._pending_batches else None
        if (
            last is not None
            and last.event_type == event_type
            and last.tier == tier
            and last.backend == backend
            and last.shared == shared
        ):
            last.entries.extend(entries)
        else:
            self._pending_batches.append(
                _PendingBatch(
                    event_type=event_type,
                    tier=tier,
                    backend=backend,
                    shared=shared,
                    entries=list(entries),
                )
            )
        self._flush_if_due()

    def _flush_if_due(self) -> None:
        """Flush once ``flush_interval`` has elapsed since the last flush."""
        now = time.monotonic()
        if now - self._last_flush >= self._flush_interval:
            self._last_flush = now
            self.flush()


def create_cache_event_sink(config: CoordinatorConfig) -> CacheEventSink:
    """Create the configured MP-server cache-event transport.

    Args:
        config: Coordinator connection and event-sink configuration.

    Returns:
        The Kafka sink, or the HTTP sink behind a retrying spool.

    Raises:
        ValueError: If HTTP delivery is selected without a coordinator URL.
    """
    if isinstance(config.event_sink_config, KafkaCacheEventSinkConfig):
        return KafkaCacheEventSink(config.event_sink_config)
    if not config.url:
        raise ValueError("HTTP cache-event reporting requires a coordinator URL")
    return SpoolingCacheEventSink(HttpCacheEventSink(config.url))


def register_cache_event_metrics(
    subscriber: CacheEventSubscriber, spool: SpoolingCacheEventSink | None
) -> None:
    """Register the emitter's delivery-loss gauges.

    Args:
        subscriber: The subscriber whose bus-overflow drops to report.
        spool: The HTTP spool whose overflow and rejection drops,
            retries and depth to report, or ``None`` when the stream has
            no spooling sink.
    """

    def _dropped() -> list[tuple[int | float, dict[str, object]]]:
        observations: list[tuple[int | float, dict[str, object]]] = [
            (subscriber.bus_dropped_events_total(), {"reason": "bus_overflow"})
        ]
        if spool is not None:
            observations.append(
                (spool.overflow_events_total(), {"reason": "spool_overflow"})
            )
            observations.append((spool.rejected_events_total(), {"reason": "rejected"}))
        return observations

    register_gauge(
        _METER_NAME,
        "lmcache_mp.cache_events.events_dropped_total",
        "Cache events (one per key) lost before reaching the coordinator, by reason.",
        _dropped,
    )
    if spool is None:
        return
    register_gauge(
        _METER_NAME,
        "lmcache_mp.cache_events.publish_retries_total",
        "Cache-event publish attempts made after a failed one.",
        spool.retries_total,
    )
    register_gauge(
        _METER_NAME,
        "lmcache_mp.cache_events.spool_depth",
        "Cache-event entries retained for redelivery.",
        spool.spool_depth,
    )


def maybe_create_cache_event_subscriber(
    mp_config: MPServerConfig,
    http_config: HTTPFrontendConfig | None,
    coordinator_config: CoordinatorConfig,
) -> CacheEventSubscriber | None:
    """Create the cache-event subscriber, or ``None`` if nothing wants the stream.

    Destinations, either or both: the coordinator (``--coordinator-url`` with
    ``--coordinator-event-reporting``, when the HTTP frontend exists) and an
    ``events``-level trace file (``--trace-level events``). The trace needs no
    coordinator. The incarnation is the server start time; the trace opens
    with a ``start`` mark carrying it and the server's identity.

    Args:
        mp_config: The server's identity and ports.
        http_config: The HTTP frontend, or ``None`` when it is not running.
        coordinator_config: Coordinator connection and event-sink settings.

    Returns:
        The subscriber to register on the event bus, or ``None``.
    """
    sinks: list[CacheEventSink] = []
    spool: SpoolingCacheEventSink | None = None
    if (
        http_config is not None
        and coordinator_config.url
        and coordinator_config.event_reporting
    ):
        sink = create_cache_event_sink(coordinator_config)
        if isinstance(sink, SpoolingCacheEventSink):
            spool = sink
        sinks.append(sink)
    trace_sink: TraceCacheEventSink | None = None
    recorder = get_active_trace_recorder()
    if isinstance(recorder, EventsTraceRecorder):
        trace_sink = TraceCacheEventSink(recorder)
        sinks.append(trace_sink)
    if not sinks:
        return None

    incarnation = int(time.time())
    if trace_sink is not None:
        trace_sink.record_lifecycle(
            TraceLifecyclePhase.START,
            instance_id=mp_config.instance_id,
            incarnation=incarnation,
            ip=coordinator_config.advertise_ip,
            http_port=http_config.http_port if http_config is not None else 0,
            mq_port=mp_config.port if mp_config.p2p_config.enabled else 0,
        )
    subscriber = CacheEventSubscriber(
        sink=sinks[0] if len(sinks) == 1 else MultiCacheEventSink(sinks),
        instance_id=mp_config.instance_id,
        incarnation=incarnation,
        flush_interval=coordinator_config.event_flush_interval,
    )
    register_cache_event_metrics(subscriber, spool)
    return subscriber
