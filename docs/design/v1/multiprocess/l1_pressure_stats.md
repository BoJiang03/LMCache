# L1 pressure stats endpoint

Status: implemented (`GET_L1_PRESSURE`).

## Problem

Whether the server's L1 is churning -- stored content evicted faster
than the workload re-uses it -- is invisible outside the MP server:
capacity and eviction live entirely there. This endpoint exposes a
snapshot a client can turn into an eviction rate and a residence
estimate. (Its original consumer, the lazy-offload adaptive-degradation
controller, has been removed; the endpoint stays as a generic
observability probe.)

## Interface

New request type `RequestType.GET_L1_PRESSURE` (controller group).

- Payload: none.
- Response: `L1PressureStats` with
  - `total_bytes`: L1 capacity.
  - `used_bytes`: bytes currently resident.
  - `evicted_bytes_total`: cumulative bytes freed by key deletion since
    server start (monotonic).
  - `evicted_chunks_total`: cumulative chunk count for the same
    deletions (monotonic).
- Handler: SYNC (reads two in-memory counters and the memory-manager
  usage tuple; no locks beyond theirs, no I/O).

The response is a snapshot; rate computation is the caller's job
(two samples and their arrival times). One byte counter deliberately
covers *all* key deletions -- watermark eviction, store-failure
cleanup, CLEAR -- because every deletion shortens effective residence;
callers that need eviction-only cadence can use the observability
metrics instead.

## Server side

`L1Manager` accounts the totals at the source: every deletion site
already computes the freed objects' `L1ObjectMeta` list (for the
`L1_KEYS_EVICTED` event), and now also folds their `size_bytes` into
two monotonic counters under the manager lock, read back via
`L1Manager.deletion_totals()`. Deliberately NOT an event-bus
subscriber: the global bus is disabled unless observability is
configured, and a functional signal must not depend on observability
config. `StorageManager.get_l1_deletion_totals()` is the public
pass-through.

The handler lives in the management module next to `GET_CHUNK_SIZE`
and pairs the totals with `StorageManager.get_l1_usage()`; the two
reads are not under a common lock, so a response's usage and totals
can be one deletion apart -- fine for a rate estimator.
