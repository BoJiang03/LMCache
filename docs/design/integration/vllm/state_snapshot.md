# L0 State Snapshot

Every 10 s (by default) of a live vLLM + LMCache MP run, write the full
content of the GPU prefix cache (L0) to a file, for offline analysis: which
blocks vLLM caches, which are free and what it reuses next, and which
LMCache chunks each prompt maps to.

Off by default. It is a diagnostic: nothing in the serving path reads the
files.

## Enabling

Scheduler side, in `kv_connector_extra_config`:

```json
{"lmcache.mp.snapshot_dir": "/raid/run-42/snapshots"}
```

`lmcache.mp.snapshot_interval_s` sets the seconds between two snapshots
(default 10).

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
cost: scheduler collect 9.10 ms, dump 9.22 ms, mapping 0.84 ms, encode 18.40 ms (writer thread)
```

The loader counts only. It does not classify blocks; that belongs to the
analysis reading the files.

## Flow

```text
scheduler step (end of build_connector_meta)
  StateSnapshotter.on_scheduler_step()
    every interval (monotonic), unless the previous L0 file is still being
    written (then skipped, logged at debug):
      copy BlockPool                       -> collect_ms
      queue L0 file on writer thread       -> dump_ms (whole scheduler cost)

writer thread (scheduler):  encode body (encode_ms), write l0-*.msgpack
                            hash prompts, append requests-*.bin
```

Skipping instead of queueing keeps at most one snapshot in memory: a slow
disk costs missing files, never a growing backlog of 100k-entry copies.

A failed file write is logged by the writer thread's done callback.

## Files

All files are msgpack (`msgspec`). Schemas live in
`lmcache/v1/multiprocess/protocols/state_snapshot.py` with
`FORMAT_VERSION = 1`; read them with `read_l0_snapshot` and `iter_records`.

| File | Written | Content |
|---|---|---|
| `l0-<run_id>-<id:06d>.msgpack` | each snapshot, by the scheduler | `L0SnapshotFile` |
| `requests-<run_id>.bin` | append-only, once per request | `RequestMappingRecord`s, each `>I` length-prefixed |

`run_id` is `<hostname>-<pid>-<start unix time>` of the scheduler.

The L0 file is columnar: one list per field rather than one row per entry.
A column is one C-level `map` pass and encodes as one msgpack array;
building a row object per entry would dominate the cost at 100k entries.

### L0 file

Header (`L0SnapshotFile`): `format_version`, `run_id`, `snapshot_id`,
`wall_time`, `monotonic_time`, `model_name`, `kv_world_size`, `chunk_size`,
`hash_block_size`, `hash_algorithm`, `num_gpu_blocks`,
`group_tokens_per_block`, the costs `collect_ms`, `dump_ms`, `mapping_ms`,
`encode_ms`, and `body` (nested msgpack, so `encode_ms` can be stored in
the same file). Body (`L0Body`):

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

LMCache accessor added for this (public, read-only):
`LMCacheMPSchedulerAdapter.hash_algorithm`.

The snapshotter starts in the connector's `bind_gpu_block_pool`, which vLLM
calls once the scheduler's KV cache manager exists. A vLLM whose
`BlockPool` lacks `cached_block_hashes_by_block` is rejected there with a
`ValueError`.

## Cost

Budget at 100k GPU blocks, per snapshot: scheduler `dump_ms + encode_ms`
under 30 ms (the writer thread takes the GIL from the scheduler thread
too).

Measured with a CPU-only script that builds a real `BlockPool` (100k
blocks, all cached, half free). Snapshots spaced as at the 10 s cadence
(each waits for the previous file), 20 snapshots. Values are min / median
/ max, in ms:

| Component | ms |
|---|---|
| Scheduler `dump_ms` (scheduler thread) | 15.4 / 15.8 / 18.9 |
| Scheduler `encode_ms` (writer thread) | 1.5 / 1.6 / 5.0 |
| Scheduler total | 16.9 / 17.4 / 22.9 |

The file is 4.5 MB at that scale. Between snapshots the scheduler pays a
clock read and a comparison per step, plus one prompt copy per new request
for the mapping (`mapping_ms`, summed between snapshots).

## Limits

- Nothing between snapshots is recorded: a block cached and evicted within
  one interval never appears.
- The mapping is taken when the request is first tracked, prompts only. It
  does not apply `request_configs` effects (skipped chunks, max offload
  tokens), so a mapped chunk need not have been stored.
- A snapshot due while the previous file is still being written is
  skipped, so an L0 file for a snapshot id may be missing.
