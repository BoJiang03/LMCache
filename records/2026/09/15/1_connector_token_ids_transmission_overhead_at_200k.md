# Connector `token_ids` transmission overhead at 200k context

**Date:** 2026-09-15
**Branch:** `delta_token_ids` (worktree `/raid/bo/delta_token_ids`, based on `dev` @ `c40b7807`)
**Commit produced:** `63fa80ee` — `benchmarks: measure the connector's per-op token_ids transmission cost`

## Goal

Create a `delta_token_ids` worktree and measure what it costs the MP connector to
ship `token_ids` on every load/store op at long context (200k class).

## Setup

- Synced the fork first: local `dev` already matched `upstream/dev` (`c40b7807`);
  fast-forwarded `origin/dev` from `6b73fe90` (100 commits behind) so the fork
  matches upstream.
- New worktree placed on `/raid` (1.8T free) rather than `/home` (16G free,
  99% used), matching where the other worktrees already live.
- Ran against `/home/bo/LMCache-worktrees/mtp/.venv` (vllm 0.26.1rc1,
  msgspec 0.21.1), with `PYTHONPATH` pointed at this checkout.

## The finding

`LMCacheMPRequestTracker.get_token_ids()` (`lmcache/integration/vllm/lmcache_mp_metadata.py:157`)
returns the request's **entire** token id list, and that list is stored into
every `LoadStoreOp.token_ids` — for both STORE and RETRIEVE. The op itself only
covers `[start, end)`, but the whole list is copied, pickled, msgpacked and
shipped on every scheduler step, across three process hops:

| hop | carrier | code |
| --- | --- | --- |
| scheduler CPU | `list(ConstantList)` copy | `get_token_ids()` |
| scheduler → worker | pickle into vLLM's shm ring | `shm_broadcast.MessageQueue.enqueue` pickles `SchedulerOutput`, connector metadata inside |
| worker → LMCache server | msgspec msgpack over ZMQ DEALER | `_create_key()` → `IPCCacheServerKey(token_ids=tuple(...))`, once per TP rank |
| server CPU | `Session.set_tokens` + chunk hashing | `EngineContext.resolve_obj_keys` |

## Numbers (200k context, per op, H200 host)

| stage | whose CPU | full | delta | ratio |
| --- | --- | ---: | ---: | ---: |
| A `get_token_ids()` list copy | scheduler | 722 µs | 36 µs | 20x |
| B1 `pickle.dumps` connector metadata | scheduler | 2537 µs | 102 µs | 25x |
| B2 `pickle.loads` | each worker | 4144 µs | 146 µs | 28x |
| C1 `tuple()` + key + msgspec encode | each worker | 2299 µs | 69 µs | 33x |
| C2 msgspec decode | server | 4479 µs | 145 µs | 31x |
| D `Session.set_tokens` + memoized hash | server | 570 µs | 16 µs | 37x |
| E LOOKUP uncached full chunk hash | server | 4859 µs | 216 µs | 23x |

`full` = what ships today. `delta` = `token_ids[start:end]` only.

### Per-request rollup — 200k prefill, 25 scheduler steps, TP=8

```
TOTAL CPU                       2.58 s  ->  0.08 s     (2.50 s saved)
bytes crossing process bounds  170.8 MB ->  7.1 MB
```

Dominated by **C2 server msgspec decode (1047 ms)** and **B2 worker unpickle
(856 ms)**, both multiplied by 200 = 25 steps × 8 ranks, because every TP rank
submits its own STORE and unpickles its own copy of the metadata.

## Things worth remembering

- **Transport is not the bottleneck; deserialization is.** A real cross-process
  ZMQ round trip measured 4.7 ms against 5.2 ms for the decode alone — the wire
  costs almost nothing, the cost is turning 200k msgpack ints into a Python
  tuple.
- **Server-side hashing is already memoized.** `Session._compute_hash`
  (`lmcache/v1/multiprocess/session.py:119`) skips chunks it has already
  processed, so the STORE path is *not* O(n²). What is still paid in full on
  every store, by every rank, is `session.set_tokens(list(key.token_ids))`.
- **The LOOKUP path is not memoized.** `modules/lookup.py:204` calls
  `compute_chunk_hashes(list(key.token_ids))` with no `end`, re-hashing the
  whole sequence. Only ~1x per request today so it barely registers, but it
  would amplify if the async lookup gets polled repeatedly.
- **No headroom below ~8k context.** With `PREFILL_CHUNK_TOKENS=8192` the delta
  range equals the full range at 1k/4k; the gap only opens from 16k up.
- `get_token_ids()` is called unconditionally *before* `maybe_submit_lookup_request`
  in `get_num_new_matched_tokens`, so stage A fires on every poll while a
  request waits on an in-flight lookup, not just once.

## Benchmark mechanics

`benchmarks/microbenchmark/token_ids_transport_benchmark.py`

- Measures the **real production types** loaded from this checkout:
  `LoadStoreOp`, `LMCacheMPConnectorMetadata`, `LMCacheMPRequestMetadata`,
  `IPCCacheServerKey`, `Session`, `TokenHasher`.
- The worker → server hop runs over an actual ZMQ DEALER/ROUTER pair, with the
  ROUTER in a spawned child process decoding keys the way the server does.
- `benchmarks/microbenchmark/_native_stubs.py` supplies import-time stubs for
  `lmcache.lmcache_native` / `lmcache.device_ops`. The worktree has no compiled
  native extension, and that import chain otherwise blocks the connector
  modules. Nothing on the token-id path calls into native code, so the stubs
  do not touch what is being measured.

Run:

```bash
cd /raid/bo/delta_token_ids
PYTHONPATH=$PWD /home/bo/LMCache-worktrees/mtp/.venv/bin/python \
  benchmarks/microbenchmark/token_ids_transport_benchmark.py [--quick]
```

Saved output: `benchmarks/microbenchmark/token_ids_transport_results.txt`

## Caveats

- Pure microbenchmark. No end-to-end vLLM run, so this does not yet state a
  TTFT impact — it states CPU burned and bytes moved.
- The `delta` column is a lower bound on a redesign, not a measured
  implementation. It assumes an op can be keyed from `token_ids[start:end]`
  plus a carried-forward prefix hash; that contract change is not yet designed.

## Next step (not started)

Implement the delta change on this branch, then re-run e2e to convert the
2.5 s of saved CPU into a real TTFT / throughput delta.
