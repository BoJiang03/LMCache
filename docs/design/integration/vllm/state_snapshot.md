# L0/L1 State Snapshot

Every 10 s (by default) of a live vLLM + LMCache MP run, write the full
content of the GPU prefix cache (L0) to a file, for offline analysis: which
blocks vLLM caches, which are free and what it reuses next, and which
LMCache chunks each prompt maps to. Under the same snapshot id, every server writes its CPU cache (L1):
which chunks L1 holds that L0 still has, and what each side evicts next.

Off by default. It is a diagnostic: nothing in the serving path reads the
files.

## Enabling

Scheduler side, in `kv_connector_extra_config`:

```json
{"lmcache.mp.snapshot_dir": "/raid/run-42/snapshots"}
```

`lmcache.mp.snapshot_interval_s` sets the seconds between two snapshots
(default 10).

Server side, load the server module (it is not in the built-in `modules`
package, so a server without the flag pays nothing):

```bash
lmcache server ... \
  --server-module '{"module_path": "lmcache.v1.multiprocess.l1_snapshot"}'
```

The server writes its L1 file to the `snapshot_dir` the scheduler sends,
made absolute against the scheduler's working directory and created by the
server if missing, so the path must name the same directory on the server's
host. A server
without the module answers the request with an error; the scheduler logs a
warning with that error, naming the missing module as a possible cause, and
keeps going.

Summarize one snapshot offline (output shape below; the numbers are
illustrative, not measured):

```bash
python -m lmcache.tools.state_snapshot /raid/run-42/snapshots 12
```

```text
run h200-26-123-1761750000 snapshot 12
L0: 51200 GPU blocks, 20480 free
  group 0: 30712 cached block hashes
requests mapped: 1840
L1: 1 server file(s)
  object group 0, committed: 8100 objects, 50960793600 bytes
  object group 0, staging: 12 objects, 75497472 bytes
L1 chunks: 8112 distinct, 7950 mapped, 2210 with any block in L0, 2105 with every block in L0
cost: scheduler collect 9.10 ms, dump 9.22 ms, mapping 0.84 ms, encode 18.40 ms (writer thread)
  server h200-26-4567: list 3.10 ms, total 160.20 ms, 0 ordered keys unlisted
```

The loader counts only. It does not classify blocks or chunks; that belongs to the
analysis reading the files.

## Flow

```text
scheduler step (end of build_connector_meta)
  StateSnapshotter.on_scheduler_step()
    read replies of the last L1 request, if all arrived   (never waits)
    every interval (monotonic), unless the previous L0 file is still being
    written (then skipped, logged at debug):
      copy BlockPool                       -> collect_ms
      server_module_call("l1_snapshot.take") to every server, async
      queue L0 file on writer thread       -> dump_ms (whole scheduler cost)

writer thread (scheduler):  encode body (encode_ms), write l0-*.msgpack
                            hash prompts, append requests-*.bin

MP server, l1_snapshot.take handler:
  previous file still being written? reply skipped, collect nothing
  StorageManager.snapshot_l1()                        -> collect_ms
    per L1 manager: list_objects() under its lock     -> list_ms
                    eviction_order() under policy lock -> order_ms
  reply at once; writer thread builds columns (build_ms), encodes
  (encode_ms) and writes l1-*.msgpack
```

Skipping instead of queueing keeps at most one snapshot in memory: a slow
disk costs missing files, never a growing backlog of 100k-entry copies.
The server's handler skips the same way.

A server that has not answered when the next snapshot is due is logged and
forgotten; there are no retries. A failed file write is logged by the
writer thread's done callback. A server's L1 file write fails after its
reply, so that failure is logged on the server only, not by the scheduler.

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

The L0 file is columnar: one list per field rather than one row per entry.
A column is one C-level `map` pass and encodes as one msgpack array;
building a row object per entry would dominate the cost at 100k entries.
The L1 file is columnar too.

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

### L1 file

Written by each server's `l1_snapshot` module; its schema, the eviction
order and `eviction_ranks` are in
[`docs/design/v1/multiprocess/l1_snapshot.md`](../../v1/multiprocess/l1_snapshot.md).

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

| vLLM (a3af78384) | Use |
|---|---|
| `BlockPool.blocks`, `num_gpu_blocks`, `hash_block_size` | block list, header |
| `KVCacheBlock.block_id`, `block_hash`, `block_hash_num_tokens`, `ref_cnt` | body columns |
| `BlockPool.cached_block_hashes_by_block` | extra (non-primary) entries |
| `BlockPool.free_block_queue.get_all_free_blocks()` | `free_queue` |
| `Request.request_id`, `num_prompt_tokens`, `block_hashes` | mapping |

LMCache accessors added for this (public, read-only):
`EvictionPolicy.eviction_order()`, `L1EvictionController.eviction_policy`,
`L1ManagerInterface.list_objects()`, `StorageManager.snapshot_l1()`,
`LMCacheMPSchedulerAdapter.server_module_call()` and `.hash_algorithm`.

The snapshotter starts in the connector's `bind_gpu_block_pool`, which vLLM
calls once the scheduler's KV cache manager exists. A vLLM whose
`BlockPool` lacks `cached_block_hashes_by_block` is rejected there with a
`ValueError`.

## Cost

Budget at 100k GPU blocks and 100k L1 objects, per snapshot: server
Python time under 100 ms with the L1 lock held under 20 ms; scheduler
`dump_ms + encode_ms` under 30 ms (the writer thread takes the GIL from
the scheduler thread too).

Measured with a CPU-only script that builds a real `BlockPool` (100k
blocks, all cached, half free) and a real LRU `StorageManager` (100k
committed objects) in one process. Snapshots spaced as at the 10 s cadence
(each waits for the previous file), 20 snapshots. Values are min / median
/ max, in ms:

| Component | ms |
|---|---|
| Scheduler `dump_ms` (scheduler thread) | 15.4 / 15.8 / 18.9 |
| Scheduler `encode_ms` (writer thread) | 1.5 / 1.6 / 5.0 |
| Scheduler total | 16.9 / 17.4 / 22.9 |
| Server `list_ms` (L1 lock) | 8.3 / 9.8 / 17.5 |
| Server `order_ms` (policy lock) | 25.4 / 26.8 / 31.8 |
| Server `build_ms` + `encode_ms` | 24.6 / 28.2 / 36.0 |
| Server `collect_ms + build_ms + encode_ms` | 62.0 / 65.6 / 79.0 |
| Server file write (I/O, not in total) | 5.6 / 14.8 / 20.7 |

Copying the LRU order is the largest server item: iterating an
`OrderedDict` hashes every key, and `ObjectKey.__hash__` is Python code.
It runs under the policy lock, which L1 listener callbacks also take while
holding the L1 lock, so L1 operations can wait up to `order_ms` on it.

The file is 4.5 MB at that scale. Between snapshots the scheduler pays a
clock read and a comparison per step, plus one prompt copy per new request
for the mapping (`mapping_ms`, summed between snapshots). The L1 file is 8.9 MB (the
order is stored as keys).

Back to back, 20 requests sent while the first file was still being
written: 19 were skipped.

## Limits

- L0 and L1 are not one instant. The L1 request leaves after L0 is copied
  and is handled a few ms later; each L1 manager is listed under its own
  lock, then its eviction order is copied under the policy lock. Keys
  evicted or written between those two reads show up as ordered keys the
  listing lacks, or as listed objects with rank -1 (`eviction_ranks`).
- Nothing between snapshots is recorded: a block cached and evicted within
  one interval never appears, nor does a chunk stored and evicted.
- The mapping is taken when the request is first tracked, prompts only. It
  does not apply `request_configs` effects (skipped chunks, max offload
  tokens), so a mapped chunk need not have been stored.
- Every server in the scheduler adapter's list is asked; a TP deployment
  records each object's packed `kv_rank` as stored.
- A snapshot due while the previous file is still being written is
  skipped, so an L0 file for a snapshot id may be missing. Likewise for a server's L1 file; a slow server is not
  waited for.
