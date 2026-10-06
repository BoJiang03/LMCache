# Cache-event emission (MP server → key directory)

Module: `lmcache/v1/mp_coordinator/cache_events.py`
Contract vocabulary: `lmcache/v1/mp_coordinator/api.py`
Consumer: `lmcache/v1/mp_coordinator/views/key_directory.py` (see
[key_directory.md](key_directory.md))

This is the emission half of the key directory (M1 of the control-plane
RFC, [issue #4226](https://github.com/LMCache/LMCache/issues/4226)): MP
servers turn storage-listener callbacks into `CacheEventBatch` streams
and deliver them to the coordinator's directory.

## The transport seam

Production deployments may replace direct HTTP push with Kafka or
another message queue. The design isolates that choice behind one
interface so nothing else changes:

```
storage layer ──► EventBus ──► CacheEventSubscriber ──► CacheEventSink ──► directory
 (publishes)      (drain      (event → vocabulary,        (transport)
                  thread)      seq, batching)
```

- **`CacheEventSink`** — `publish(batches)`, preserving list order within
  and across successful calls. Redelivery is safe: the per-instance `seq`
  cursor deduplicates it, and restarts are fenced by `incarnation`.
  Delivery loss surfaces as a `seq` gap. A durable source can replay
  retained events; HTTP cannot repair an event dropped before the
  coordinator accepted it. A sink never needs exactly-once or global
  ordering.
- **`HttpCacheEventSink`** — the first sink: one
  `POST /events` per publish call, batches in list order. Failures
  raise `CacheEventPublishError`. It is always wrapped in a
  **`SpoolingCacheEventSink`**: a failed post stays in a bounded
  in-memory spool (16384 entries) and is retried head-first, in `seq`
  order, on later flushes with capped exponential backoff (0.5 s
  doubling to 30 s), counted from the end of the failed post; the spool
  posts in slices of at most 2048 entries. These are module constants,
  not settings. A flush with nothing new calls the sink's
  `redeliver()` so retries do not wait for new events. Newer
  batches queue behind retained ones, so nothing overtakes a retry.
  A retry that duplicates an applied post is dropped by the gate's
  `seq` dedup. When a failed post leaves more than the bound retained,
  the oldest batches are dropped; their `seq` numbers stay consumed, so
  the loss is a gate gap, and the sink counts their entries in
  `dropped_events()` for later batches to report. A slice that cannot
  succeed is dropped the same way and draining continues: a 4xx answer
  other than 408 or 429 (`CacheEventRejectedError`), or any
  non-transport exception. 5xx, 408, 429 and transport errors are
  retried.
- **`TraceCacheEventSink`** appends each batch, in wire form, to an
  `events`-level trace file (`lmcache server --trace-level events`); it
  needs no coordinator. **`MultiCacheEventSink`** fans one flush out to
  several sinks and raises only after every sink was tried. See
  `docs/design/v1/mp_observability/trace.md` §12.
- **`KafkaCacheEventSink`** produces one JSON record per batch with the
  message key set to `instance_id`, so Kafka assigns one instance's
  records to one partition. The producer enables idempotence and
  requires `acks=all`. The JSON value uses the existing
  `CacheEventsRequest` envelope with exactly one batch, keeping the HTTP
  and Kafka wire vocabulary identical.

  `publish` does not wait for the broker. It hands records to the
  producer's buffer and serves earlier delivery reports. The producer
  retries in order for up to the delivery timeout (5 minutes by
  default), so a broker restart delays events instead of losing them.
  The buffer is capped at 64 MB. A record that does not fit, or that the
  producer gives up on, is dropped and counted. Its `seq` is already
  spent, so the coordinator sees a gap. `close` waits up to 10 seconds
  for what is still queued.

A coordinator started with `--event-transport kafka` consumes the topic
through `KafkaCacheEventSource` (see [ingest.md](ingest.md)) instead of
serving `POST /events`; direct HTTP remains the default end-to-end
transport.

On the coordinator side, transport adapters converge at
`EventGate.ingest_batches`. `HttpCacheEventSource` is non-durable and
advertises no replay capability; `KafkaCacheEventSource` (selected by
`--event-transport kafka` in place of the HTTP source, see
[ingest.md](ingest.md)) polls the topic and advertises `seekable`. Gate cursors (`instance_id` / `incarnation` /
`seq`) remain separate from Kafka's partition offsets, which the consumer
group commits.

## Batching and sequencing (inside the subscriber)

One `CacheEventSubscriber` per MP-server process owns the buffer, the
`seq` counter, and the sink:

- **Order-preserving batching.** The buffer is a list of *pending
  batches*: consecutive records with the same `(event_type, tier,
  backend, shared)` identity append to the last pending batch; an identity
  change starts a new one (never merged backwards). Flushing emits one
  `CacheEventBatch` per pending batch, so the batch sequence preserves
  the total order of recorded events — a store followed by a delete of
  the same key can never be reordered into "delete, then store".
  Alternating identities therefore produce multiple pending batches of
  the same identity; that is intentional (extra batch headers, never
  reordering).
- **`seq` is consumed even when publish fails.** A flush whose sink
  raises (Kafka, trace) drops the drained list but keeps the `seq`
  numbers it assigned; the spooled HTTP sink drops only on spool
  overflow or a rejected slice, with the same effect. The directory
  sees a gap and sets `gap_detected` for the instance when a later
  batch arrives — the honest signal that events were lost. Replay can
  reconcile the gap only if a durable transport retained the dropped
  batch; the HTTP spool keeps nothing it dropped.
  Reusing the seqs instead would hide partial-delivery ambiguity (a
  dropped post the coordinator had in fact applied).
- **`dropped_events` says how much was lost.** A `seq` numbers a
  batch, so a gap shows that something was lost, not how much. Every
  batch also carries `dropped_events`: the cumulative number of cache
  events (entries) this incarnation lost before they reached the
  coordinator, as of when the batch was built. The gate counts its
  increases as lost events (see [ingest.md](ingest.md)). The field is
  optional on the wire (default 0), so older emitters and trace files
  still parse. The subscriber stamps bus overflow (below) plus the
  entries its sink dropped without raising
  (`CacheEventSink.dropped_events()`: the HTTP spool's overflow and
  rejected slices; `MultiCacheEventSink` sums its sinks, so an
  `events` trace recorded beside HTTP carries the HTTP losses too). A
  Kafka record the producer drops stays a `seq` gap only.
- **`incarnation` = server start time** (`int(time.time())` at
  lifespan startup). A restarted server's first batch fences out the
  **L1** placements its previous incarnation reported, matching the
  fact that its memory restarted empty; L2 placements survive because
  the bytes persist on disk (see `key_directory.md`).

## Event flow (the observability bus)

The storage layer already publishes key-level events to the
observability `EventBus` (`mp_observability/event_bus.py`);
cache-event emission rides the same bus instead of adding parallel
listener plumbing or a dedicated flush task:

- **Producers.** `L1Manager` publishes `l1.write.finished`,
  `l1.write_finished_and_read_reserved`, `l1.keys.evicted` (all delete
  paths), and the new `l1.keys.accessed` (`touch_keys` — the MP request
  end's unified touch of a request's retrieved and stored keys; the
  subscriber deliberately does **not** consume `l1.read.finished`,
  which would duplicate those accesses). The placement-bearing events
  (stores and evictions) carry `meta: list[L1ObjectMeta]` — each
  object's `size_bytes` (`MemoryObj.get_size()`) and its
  `L1BackendType` medium from `L1ManagerProtocol.get_backend_type()` (the
  Device-DAX tier resolves it per object via
  `DevDaxMemoryAllocator.is_devdax_obj`, i.e. `MemoryObj.parent()`) —
  so a hybrid DRAM+DAX L1 reports exactly where each object landed,
  and deletes target the same placement identity `(instance, tier,
  backend)` their store reported. The L2 base adapter's
  listener-notify funnel publishes the new `l2.keys.stored`
  (`keys`+`sizes`+`backend`), `l2.keys.accessed`, and `l2.keys.deleted`
  events; the backend name is the registered adapter type and the
  optional `shared` flag comes from the adapter config, both stamped by
  the storage manager via `set_backend_identity` at build time — so
  **runtime-added adapters emit automatically**. Batches carry the
  `shared` flag; for shared batches the backend type name identifies
  the pool fleet-wide (one pool per backend type), so the directory
  deduplicates shared-storage placements across emitters (see
  `key_directory.md` — Shared pools).
  The LMCache-driven store path additionally publishes
  `mp.tokens` (parallel `chunk_hashes` + `token_chunks` +
  `token_offsets`) at
  store submission — ordered ahead of the store's write-finished
  events, built only when the event has a subscriber, so the cost is
  zero with event reporting off (and no hashing anywhere: the directory
  indexes tokens by the chunk hash already in every key). Only
  worker 0 reports: bindings depend on token content alone, so one
  report covers every rank's keys. Other store paths (engine-driven
  transfer, blend pre-computed docs, experimental qstore) do not emit
  bindings yet — the engine-driven path can publish the same event from
  its ``commit_store`` when it needs directory tokens.
- **`CacheEventSubscriber`** maps those events onto the directory
  vocabulary (writes → `STORE`, evictions/deletes → `DELETE`, split per
  actual L1 medium from the event metadata; touches → `ACCESS`). The
  token-binding events produce no batches of their own: the subscriber
  remembers their chunk-hash → (token ids, offset) pairs (LRU cache
  bounded at
  65536; passing the bound evicts the oldest half in one batch, so
  eviction — and its warning — stays rare) and stamps `token_ids` and
  `token_offset` onto
  every L1/L2 `STORE` entry,
  so token bindings ride the store events themselves. Tokens are
  therefore repeated per rank/group/tier placement — an accepted wire
  trade for a self-contained protocol (see
  [key_directory.md](key_directory.md) — Token index).
  `ACCESS` batches carry an **empty backend**: the directory only
  refreshes key-level recency and access count on access, so there is
  no placement identity to name. The vocabulary requires a non-empty
  backend for `store`/`delete` only. The subscriber is single-threaded
  by design - everything runs on the bus's drain thread, so it needs no
  locking.
- **Threading.** The bus dispatches on one drain thread, which is
  exactly the per-instance FIFO the directory needs. The subscriber
  self-paces delivery: recording flushes when `flush_interval` has
  elapsed since the last flush, bounding the sink-publish rate under
  load. There is no timer of its own — the subscriber additionally
  subscribes to `l1.eviction.loop_tick` (published continuously by the
  L1 eviction loop) as a flush pump, so a burst-ending tail (e.g. L2
  store completions) is delivered within one tick of the interval
  elapsing instead of waiting for the next request. The sink posts
  synchronously with a short timeout (a slow coordinator briefly
  stalls the drain, bounded by the timeout). Overflow beyond the bus's
  bounded queue happens before the subscriber assigns `seq`, so it
  leaves no `seq` gap. The bus counts, per event type, the keys its
  dropped events carried; the subscriber adds the growth since it
  registered (its consumed types only) to `dropped_events`, one cache
  event per key. A dropped event without keys (the flush tick, a token
  binding, a capacity change) counts none. Node counters (`lmcache.mp_server`
  gauges):
  `lmcache_mp.cache_events.events_dropped_total{reason=bus_overflow|spool_overflow|rejected}`
  (all in cache events), `publish_retries_total`, and `spool_depth`
  (entries, at least one per batch).
- **Coupling.** The stream requires the bus: enabling
  `--coordinator-event-reporting` together with
  `--disable-observability` is rejected at startup. Bus-level drops under
  overload remain possible; the directory is eventually consistent soft
  state, but the loss is not currently self-healing.

## L1 media

L1 media are a closed set, hence the `L1BackendType` enum
(`distributed/api.py`: `DRAM`, `DEVDAX`, `GDS`); L2 backends stay
strings because adapter types are an open registry (plugins register
new type names).

The `shared` flag is tier-agnostic, and L1 already contains the
shared-capable medium: `DEVDAX` (e.g. CXL-attached memory exposed as a
`/dev/dax` device) can be mapped by several instances, while `DRAM` and
`GDS` are inherently instance-private. Today each instance uses its
DevDAX region privately (its own allocator, its own lifetime), so L1
events emit `shared=False`. When pooled DevDAX lands (allocation
governed by the pool's own controller, reporting still per instance),
only the producer side changes: the subscriber already splits L1
records per `L1BackendType`, so it stamps `shared` on the pooled
backend's runs — the vocabulary, batching identity, and directory
semantics need no change.

## Wiring and configuration

Enabled in the MP HTTP server lifespan when a coordinator URL is set
and `--coordinator-event-reporting` (or
`LMCACHE_COORDINATOR_EVENT_REPORTING`) is on;
`--coordinator-event-flush-interval` paces the subscriber's
event-driven flushes (default 1s).

`--coordinator-event-transport kafka` selects Kafka instead of HTTP and
requires `--coordinator-kafka-bootstrap-servers`. The topic defaults to
`lmcache-cache-events`. `--coordinator-kafka-delivery-timeout` (default
300 s) is how long the producer retries a record before dropping it.
These flags have no
environment-variable fallback. `confluent-kafka` ships as the optional
`lmcache[kafka]` extra and is imported only when the Kafka sink is built,
so HTTP-only deployments never load it.

## Known limitations (follow-ups)

- **Dropped events are lost for good.** Bus overflow, spool overflow,
  a rejected HTTP post and a record the Kafka producer gives up on are
  all visible at the gate, but nothing yet resyncs the instance's slice
  (a durable transport cannot replay an event that never reached its
  producer either). A dropped capacity change is not counted as a lost
  event (it carries no keys).
- **A loss is reported only by a later batch.** `dropped_events` is
  sampled when a flush builds batches, so a spool drop shows up on the
  batches built after it, and a node that goes idle right after
  dropping events does not report them until it emits again.
- **A Kafka declaration dropped after it was queued is not re-sent.** The
  subscriber restores a capacity declaration only when `publish` raises.
  When the producer drops it later, the coordinator lacks that
  instance's capacity until the next declaration.
- **The flush pump is coupled to the eviction loop's tick** — decouple
  it (e.g. a bus-owned periodic hook) so tail freshness does not depend
  on that loop's cadence.
