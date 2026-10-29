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
    every 10 s (monotonic):
      copy BlockPool + lazy buffer         -> collect_ms
      server_module_call("l1_snapshot.take") to every server, async
      queue L0 file on writer thread       -> dump_ms (whole scheduler cost)

writer thread (scheduler):  encode body (encode_ms), write l0-*.msgpack
                            hash prompts, append requests-*.bin

MP server, l1_snapshot.take handler:
  StorageManager.snapshot_l1()
    per L1 manager: list_objects() under its lock     -> list_ms (lock_ms)
                    eviction_order() under policy lock -> order_ms
  reply at once; writer thread encodes and writes l1-*.msgpack (total_ms)
```

A server that has not answered when the next snapshot is due is logged and
forgotten; there are no retries. A failed file write is logged by the
writer thread's done callback.

## Files

All files are msgpack (`msgspec`). Schemas live in
`lmcache/v1/multiprocess/protocols/state_snapshot.py` with
`FORMAT_VERSION = 1`; read them with `read_l0_snapshot`,
`read_l1_snapshot` and `iter_records`.

| File | Written | Content |
|---|---|---|
| `l0-<run_id>-<id:06d>.msgpack` | each snapshot, by the scheduler | `L0SnapshotFile` |
| `l1-<run_id>-<id:06d>-<server_id>.msgpack` | each snapshot, by each server | `L1SnapshotFile` |
| `requests-<run_id>.bin` | append-only, once per request | `RequestMappingRecord`s, each `>I` length-prefixed |

`run_id` is `<hostname>-<pid>-<start unix time>` of the scheduler;
`server_id` is `<hostname>-<pid>` of the server. One snapshot id ties an L0
file to its L1 files.

### L0 file

Header (`L0SnapshotFile`): `format_version`, `run_id`, `snapshot_id`,
`wall_time`, `monotonic_time`, `model_name`, `kv_world_size`, `chunk_size`,
`hash_block_size`, `hash_algorithm`, `num_gpu_blocks`,
`group_tokens_per_block`, `l1_requested`, the costs `collect_ms`, `dump_ms`,
`mapping_ms`, `encode_ms`, and `body` (nested msgpack, so `encode_ms` can be
stored in the same file). Body (`L0Body`):

- `cached_blocks`: one `[block_id, group_id, block_hash, ref_cnt,
  hash_num_tokens, primary]` row per prefix-cache entry. `block_hash` is
  vLLM's hash without the group id. `hash_num_tokens` is
  `KVCacheBlock.block_hash_num_tokens` (fewer than a block's tokens for a
  partial Mamba-state entry). Extra entries vLLM registered for a block
  (`BlockPool.cached_block_hashes_by_block`) have `primary = false` and
  `hash_num_tokens = null`.
- `free_queue`: free block ids, the next one vLLM reuses first.
- `lazy` (null when lazy offload is off): `ops`, one `[request_id, start,
  end, block_ids per group, held, expedited]` row per buffered store (held or
  expedited = due); `in_flight`, one `[request_id, [(start, end)...],
  num_pinned_blocks, failed, orphaned]` row per submitted batch awaiting its
  receipt (`failed` = its operations are buffered again once every worker
  reported, the retry state); `finished_request_ids`.

### L1 file

Header (`L1SnapshotFile`): `format_version`, `run_id`, `snapshot_id`,
`server_id`, `wall_time`, `monotonic_time`, `lock_ms`, `total_ms`,
`managers`. One `L1ManagerSection` per L1 manager: `l1_manager_id`, `tag`,
`has_eviction_order`, `num_unlisted_ordered_keys`, `list_ms`, `order_ms`,
and `objects`, one row per object:

```text
[chunk_hash, model_name, kv_rank, object_group_id, cache_salt,
 size_bytes, state, eviction_rank]
```

`state` is `committed`, `temporary` (committed, deleted after its read), or
`staging` (write-reserved, not committed). `kv_rank` is the packed value
`ObjectKey` stores. `eviction_rank` is 0 for the next victim and -1 when the
policy keeps no single order (no-op, ARC, isolated LRU) or did not list the
key. Only LRU implements `EvictionPolicy.eviction_order()` today.

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
same hashes, so `block_hash` in the L0 file can be matched against
`block_hashes` directly.

## What is read, and from where

Nothing in vLLM is modified. Attributes read, all public:

| vLLM (705e124c0) | Use |
|---|---|
| `BlockPool.blocks`, `num_gpu_blocks`, `hash_block_size` | block list, header |
| `KVCacheBlock.block_id`, `block_hash`, `block_hash_num_tokens`, `ref_cnt` | `cached_blocks` rows |
| `BlockPool.cached_block_hashes_by_block` | extra (non-primary) entries |
| `BlockPool.free_block_queue.get_all_free_blocks()` | `free_queue` |
| `kv_cache_utils.get_block_hash`, `get_group_id` | split the stored key |
| `Request.request_id`, `num_prompt_tokens`, `block_hashes` | mapping |

LMCache accessors added for this (public, read-only):
`EvictionPolicy.eviction_order()`, `L1EvictionController.eviction_policy`,
`L1ManagerInterface.list_objects()`, `StorageManager.snapshot_l1()`,
`OffloadPolicy.buffered_ops()`, `LazyOffloadRequestRegistry.in_flight_batches()`,
`LazyOffloadManager.buffer_snapshot()`,
`LMCacheMPSchedulerAdapter.server_module_call()`.

## Cost

Measured once with a CPU-only script (not committed) that builds a real
`BlockPool` and a real LRU `StorageManager` in one process, three snapshots
each; the server requests ran back to back, so the writer thread was busy
and the spread reflects GIL contention:

| Scale | Scheduler `collect_ms` / `dump_ms` | Writer `encode_ms` | Server `lock_ms` | Server `order_ms` | Server `total_ms` |
|---|---|---|---|---|---|
| 100k GPU blocks (half free), 100k L1 objects | 18-35 / 18-35 | 37-80 | 41-77, one 513 | 29-65 | 820-960 |

Files are 4.4 MB (L0) and 5.5 MB (L1) at that scale. The scheduler pays
`dump_ms` once per 10 s and one `is not None` check per step otherwise; the
mapping costs one prompt copy per new request (`mapping_ms`, summed between
snapshots). On the server, `lock_ms` is the time L1 writers and readers of
that manager wait; `total_ms` is mostly Python encoding on the writer
thread, which competes for the GIL with the server's request threads.

## Limits

- L0 and L1 are not one instant. The L1 request leaves after L0 is copied
  and is handled a few ms later; each L1 manager is listed under its own
  lock, then its eviction order is copied under the policy lock. Keys
  evicted or written between those two reads show up in
  `num_unlisted_ordered_keys` or with `eviction_rank = -1`.
- Nothing between snapshots is recorded: a chunk stored and evicted within
  10 s never appears.
- The mapping is taken when the request is first tracked, prompts only. It
  does not apply `request_configs` effects (skipped chunks, max offload
  tokens), so a mapped chunk need not have been stored.
- Every server in the scheduler adapter's list is asked; a TP deployment
  records each object's packed `kv_rank` as stored.
- The interval is fixed at 10 s; a slow server is not waited for, so its
  L1 file for a snapshot may be missing.
