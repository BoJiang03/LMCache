# L0/L1 State Snapshot

Every 10 s of a live vLLM + LMCache MP run, write the full content of the GPU
prefix cache (L0), the lazy-offload buffer, and every server's CPU cache (L1)
to files, for offline analysis of lazy offload: which chunks L1 holds that L0
still has, which L0 blocks lazy offload has not stored, what is about to be
evicted on each side.

Off by default. It is a diagnostic: nothing in the serving path reads the
files.

## Enabling

Scheduler side, in `kv_connector_extra_config`:

```json
{"lmcache.mp.snapshot_dir": "/raid/run-42/snapshots"}
```

Server side, load the server module (it is not in the built-in `modules`
package, so a server without the flag pays nothing):

```bash
lmcache server ... \
  --server-module '{"module_path": "lmcache.v1.multiprocess.l1_snapshot"}'
```

The server writes its L1 file to the `snapshot_dir` the scheduler sends, so
the path must name the same directory on the server's host. A server
without the module answers the request with an error; the scheduler logs a
warning naming the flag and keeps going.

Summarize one snapshot offline (output shape below; the numbers are
illustrative, not measured):

```bash
python -m lmcache.tools.state_snapshot /raid/run-42/snapshots 12
```

```text
run h200-26-123-1761750000 snapshot 12
L0: 51200 GPU blocks, 20480 free
  group 0: 30712 cached block hashes
  lazy offload: 85 buffered stores, 3 batches in flight
L1: 1 server file(s)
  object group 0, committed: 8100 objects, 50960793600 bytes
  object group 0, staging: 12 objects, 75497472 bytes
L1 chunks: 8112 distinct, 7950 mapped, 2210 with any block in L0, 2105 with every block in L0
cost: scheduler collect 9.10 ms, dump 9.22 ms, mapping 0.84 ms, encode 18.40 ms (writer thread)
  server h200-26-4567: lock 3.10 ms, total 160.20 ms, 0 ordered keys unlisted
```

The loader counts only. It does not classify chunks; that belongs to the
analysis reading the files.

## Flow

```text
scheduler step (build_connector_meta, after the lazy drain)
  StateSnapshotter.on_scheduler_step()
    read replies of the last L1 request, if all arrived   (never waits)
    every 10 s (monotonic), unless the previous L0 file is still being
    written (then skipped, logged at debug):
      copy BlockPool + lazy buffer         -> collect_ms
      server_module_call("l1_snapshot.take") to every server, async
      queue L0 file on writer thread       -> dump_ms (whole scheduler cost)

writer thread (scheduler):  encode body (encode_ms), write l0-*.msgpack
                            hash prompts, append requests-*.bin

MP server, l1_snapshot.take handler:
  previous file still being written? reply skipped, collect nothing
  StorageManager.snapshot_l1()                        -> collect_ms
    per L1 manager: list_objects() under its lock     -> list_ms (lock_ms)
                    eviction_order() under policy lock -> order_ms
  reply at once; writer thread builds columns (build_ms), encodes
  (encode_ms) and writes l1-*.msgpack
```

Skipping instead of queueing keeps at most one snapshot in memory per
side: a slow disk or a burst of requests costs missing files, never a
growing backlog of 100k-entry listings.

A server that has not answered when the next snapshot is due is logged and
forgotten; there are no retries. A failed file write is logged by the
writer thread's done callback.

## Files

All files are msgpack (`msgspec`). Schemas live in
`lmcache/v1/multiprocess/protocols/state_snapshot.py` with
`FORMAT_VERSION = 2`; read them with `read_l0_snapshot`,
`read_l1_snapshot` and `iter_records`.

| File | Written | Content |
|---|---|---|
| `l0-<run_id>-<id:06d>.msgpack` | each snapshot, by the scheduler | `L0SnapshotFile` |
| `l1-<run_id>-<id:06d>-<server_id>.msgpack` | each snapshot, by each server | `L1SnapshotFile` |
| `requests-<run_id>.bin` | append-only, once per request | `RequestMappingRecord`s, each `>I` length-prefixed |

`run_id` is `<hostname>-<pid>-<start unix time>` of the scheduler;
`server_id` is `<hostname>-<pid>` of the server. One snapshot id ties an L0
file to its L1 files.

Both snapshot files are columnar: one list per field rather than one row
per entry. Building a row object per entry was most of the cost in format
1 (see Cost); a column is one C-level `map` pass and encodes as one msgpack
array.

### L0 file

Header (`L0SnapshotFile`): `format_version`, `run_id`, `snapshot_id`,
`wall_time`, `monotonic_time`, `model_name`, `kv_world_size`, `chunk_size`,
`hash_block_size`, `hash_algorithm`, `num_gpu_blocks`,
`group_tokens_per_block`, `l1_requested`, the costs `collect_ms`, `dump_ms`,
`mapping_ms`, `encode_ms`, and `body` (nested msgpack, so `encode_ms` can be
stored in the same file). Body (`L0Body`):

- `block_ids`, `block_keys`, `ref_cnts`, `hash_num_tokens`: one entry per
  prefix-cache entry. `block_keys` holds vLLM's `BlockHashWithGroupId` as
  is (block hash, then the group id in 4 bytes big-endian);
  `split_block_key` splits it. `hash_num_tokens` is
  `KVCacheBlock.block_hash_num_tokens` (fewer than a block's tokens for a
  partial Mamba-state entry; null where vLLM records none).
- `num_primary`: the first `num_primary` entries are blocks' own hashes;
  the rest are extra entries vLLM registered for a block
  (`BlockPool.cached_block_hashes_by_block`), with null `hash_num_tokens`.
- `free_queue`: free block ids, the next one vLLM reuses first.
- `lazy` (null when lazy offload is off): `ops`, one `[request_id, start,
  end, block_ids per group, held, expedited]` row per buffered store (held or
  expedited = due); `in_flight`, one `[request_id, [(start, end)...],
  num_pinned_blocks, failed, orphaned]` row per submitted batch awaiting its
  receipt (`failed` = its operations are buffered again once every worker
  reported, the retry state); `finished_request_ids`.

### L1 file

Header (`L1SnapshotFile`): `format_version`, `run_id`, `snapshot_id`,
`server_id`, `wall_time`, `monotonic_time`, the costs `lock_ms`,
`collect_ms`, `build_ms`, `encode_ms`, `total_ms`, and `managers`. One
`L1ManagerSection` per L1 manager: `l1_manager_id`, `tag`, `list_ms`,
`order_ms`, and

- `objects`: `L1KeyColumns` (`chunk_hash`, `model_name`, `kv_rank`,
  `object_group_id`, `cache_salt` lists), one entry per object;
- `size_bytes`, `state`: per object, same order;
- `eviction_order`: `L1KeyColumns` of every key the policy tracks, the
  next victim first, or null when the policy keeps no single order (no-op,
  ARC, isolated LRU). Only LRU implements `EvictionPolicy.eviction_order()`
  today.

`state` is `committed`, `temporary` (committed, deleted after its read), or
`staging` (write-reserved, not committed). `kv_rank` is the packed value
`ObjectKey` stores. `eviction_ranks(section)` returns each object's
position in the order (0 for the next victim, -1 when untracked) and the
number of ordered keys the listing lacks. The order is stored as keys
rather than as ranks because matching 100k keys on the server costs about
as much as everything else together; the loader matches offline.

### Request mapping

`RequestMappingRecord`: `request_id`, `cache_salt`, `num_prompt_tokens`,
`block_hashes` (vLLM `Request.block_hashes` when the connector first tracks
the request: one per full `hash_block_size` block of the prompt),
`chunk_hashes` (the servers' `ObjectKey.chunk_hash` of every full chunk of
the prompt, by `TokenHasher(chunk_size, hash_algorithm)` over the tracker's
token ids, as the server's lookup computes them), `wall_time`,
`monotonic_time`.

Chunk `i` covers block hashes `[i*chunk_size // hash_block_size,
ceil((i+1)*chunk_size / hash_block_size))`. With vLLM's
`BlockHashListWithBlockSize`, a group with larger blocks uses one of these
same hashes, so the block hash part of an L0 `block_keys` entry can be
matched against `block_hashes` directly.

## What is read, and from where

Nothing in vLLM is modified. Attributes read, all public:

| vLLM (705e124c0) | Use |
|---|---|
| `BlockPool.blocks`, `num_gpu_blocks`, `hash_block_size` | block list, header |
| `KVCacheBlock.block_id`, `block_hash`, `block_hash_num_tokens`, `ref_cnt` | body columns |
| `BlockPool.cached_block_hashes_by_block` | extra (non-primary) entries |
| `BlockPool.free_block_queue.get_all_free_blocks()` | `free_queue` |
| `Request.request_id`, `num_prompt_tokens`, `block_hashes` | mapping |

LMCache accessors added for this (public, read-only):
`EvictionPolicy.eviction_order()`, `L1EvictionController.eviction_policy`,
`L1ManagerInterface.list_objects()`, `StorageManager.snapshot_l1()`,
`OffloadPolicy.buffered_ops()`, `LazyOffloadRequestRegistry.in_flight_batches()`,
`LazyOffloadManager.buffer_snapshot()`,
`LMCacheMPSchedulerAdapter.server_module_call()`.

## Cost

Budget at 100k GPU blocks and 100k L1 objects, per snapshot: server
Python time under 100 ms with the L1 lock held under 20 ms; scheduler
`dump_ms + encode_ms` under 30 ms (the writer thread takes the GIL from
the scheduler thread too).

Measured with a CPU-only script that builds a real `BlockPool` (100k
blocks, all cached, half free) and a real LRU `StorageManager` (100k
committed objects) in one process. Snapshots spaced as at the 10 s cadence
(each waits for the previous file); format 1 over 7 snapshots, format 2
over 20. Values are min / median / max, in ms:

| Component | Format 1 | Format 2 |
|---|---|---|
| Scheduler `dump_ms` (scheduler thread) | 15.2 / 15.8 / 18.6 | 15.4 / 15.8 / 18.9 |
| Scheduler `encode_ms` (writer thread) | 43.2 / 45.8 / 49.5 | 1.5 / 1.6 / 5.0 |
| Scheduler total | 58.5 / 61.6 / 68.1 | 16.9 / 17.4 / 22.9 |
| Server `lock_ms` (L1 lock) | 31.5 / 35.4 / 458.1 | 8.3 / 9.8 / 17.5 |
| Server `order_ms` (policy lock) | 32.5 / 36.2 / 41.7 | 25.4 / 26.8 / 31.8 |
| Server `build_ms` + `encode_ms` | (in total) | 24.6 / 28.2 / 36.0 |
| Server `total_ms` | 220.6 / 247.9 / 661.3 | 62.0 / 65.6 / 79.0 |
| Server file write (I/O, not in total) | | 5.6 / 14.8 / 20.7 |

Where format 1 spent it (one profile, 100k objects): about 145 ms building
one msgspec row per object, 27 ms listing under the L1 lock (one
NamedTuple per object; the 458 ms outlier was not root-caused, one
suspect is a garbage collection those allocations trigger under the lock),
30 ms copying the LRU order.
On the scheduler, 31-36 ms of the writer's `encode_ms` built one row per
block. Format 2 lists into per-field lists under the lock, reads key
fields through `ObjectKey.identity` with `attrgetter` maps, and stores the
eviction order as keys.

Copying the LRU order is now the largest server item: iterating an
`OrderedDict` hashes every key, and `ObjectKey.__hash__` is Python code.
It runs under the policy lock, which L1 listener callbacks also take while
holding the L1 lock, so L1 operations can wait up to `order_ms` on it.

Files are 4.5 MB (L0) and 8.9 MB (L1, the order is stored as keys) at
that scale. Between snapshots the scheduler pays one `is not None` check
per step, plus one prompt copy per new request for the mapping
(`mapping_ms`, summed between snapshots).

Back to back, 20 requests sent while the first file was still being
written: 19 were skipped.

## Limits

- L0 and L1 are not one instant. The L1 request leaves after L0 is copied
  and is handled a few ms later; each L1 manager is listed under its own
  lock, then its eviction order is copied under the policy lock. Keys
  evicted or written between those two reads show up as ordered keys the
  listing lacks, or as listed objects with rank -1 (`eviction_ranks`).
- Nothing between snapshots is recorded: a chunk stored and evicted within
  10 s never appears.
- The mapping is taken when the request is first tracked, prompts only. It
  does not apply `request_configs` effects (skipped chunks, max offload
  tokens), so a mapped chunk need not have been stored.
- Every server in the scheduler adapter's list is asked; a TP deployment
  records each object's packed `kv_rank` as stored.
- The interval is fixed at 10 s; a slow server is not waited for, and a
  side still writing its previous file skips the snapshot, so an L0 or L1
  file for a snapshot id may be missing.
