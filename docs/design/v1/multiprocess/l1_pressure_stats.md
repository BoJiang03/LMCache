# L1 pressure stats endpoint

Status: implemented (`GET_L1_PRESSURE`).

## Problem

The vLLM-side lazy-offload policy needs to know whether the server's L1
is churning: when stored content is evicted faster than the workload
re-uses it, deferring stores has no coverage dividend and the policy
should degrade to immediate emission (see
`docs/design/integration/vllm/lazy_offload_policy/eviction_aware.md`,
"Adaptive degradation"). The scheduler process has no view of L1 today;
capacity and eviction live entirely in the MP server.

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

## Scheduler-side polling

`VLLMSchedulerAdapter.poll_l1_pressure(min_interval)` drives a
threadless poll from the step path: each call checks an in-flight
`MessagingFuture` per server (non-blocking), folds finished responses
into the latest aggregate sample, and submits the next request once
`min_interval` has elapsed since the previous submission. Multi-server
deployments are aggregated by summing all four fields -- L1 shards see
symmetric traffic, and a summed rate over summed capacity is the fleet
residence. Returns the latest `L1PressureSample`
(`monotonic_time` + the four sums) or `None` before the first response
or while any server is unhealthy (a stale partial aggregate would bias
the rate).
