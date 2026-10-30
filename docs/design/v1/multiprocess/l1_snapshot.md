# l1_snapshot server module

`lmcache/v1/multiprocess/l1_snapshot.py` answers the server-module method
`l1_snapshot.take`: it lists every L1 object and the eviction order, replies,
and writes one L1 file per request on its own writer thread. It is a
read-only diagnostic for offline analysis of what L1 holds and what it
evicts next; nothing in the serving path reads the file.

## Enabling

The module is not in the built-in `modules` package, so a server without the
flag pays nothing:

```bash
lmcache server ... \
  --server-module '{"module_path": "lmcache.v1.multiprocess.l1_snapshot"}'
```

Any `RequestClient` (here `req_client`) asks for a file:

```python
request = L1SnapshotRequest(run_id="run-42", snapshot_id=12,
                            snapshot_dir="/raid/run-42/snapshots")
response = req_client.server_module_call(ServerModuleCallRequest(
    method=L1_SNAPSHOT_METHOD, payload=msgspec.msgpack.encode(request)))
reply = msgspec.msgpack.decode(response.result().payload, type=L1SnapshotReply)
```

`snapshot_dir` is a path on the server's host. A server without the module
answers with `success=False`.

## Flow

```text
l1_snapshot.take handler (request thread):
  one request at a time (a lock spans the check, the collect and the submit)
  previous file still being written? reply skipped, collect nothing
  StorageManager.snapshot_l1()                        -> collect_ms
    per L1 manager: list_objects() under its lock     -> list_ms
                    eviction_order() under policy lock -> order_ms
  reply at once (path, num_objects, list_ms summed over managers)
writer thread: build columns (build_ms), encode (encode_ms), write
  l1-<run_id>-<id:06d>-<server_id>.msgpack atomically (tmp + rename)
```

Skipping instead of queueing keeps at most one listing in memory: a slow
disk costs missing files, never a growing backlog of 100k-entry listings. A
failed write is logged by the writer thread's done callback.

## L1 file

msgpack (`msgspec`), schema in
`lmcache/v1/multiprocess/protocols/state_snapshot.py` (`FORMAT_VERSION =
1`); read it with `read_l1_snapshot`. `server_id` is `<hostname>-<pid>`.

Header (`L1SnapshotFile`): `format_version`, `run_id`, `snapshot_id`,
`server_id`, `wall_time`, `monotonic_time`, the costs `list_ms`
(summed over managers, lock waits included), `collect_ms`, `build_ms`,
`encode_ms`, and `managers` (nested
msgpack, so `encode_ms` can be stored in the same file). One
`L1ManagerSection` per L1 manager: `l1_manager_id`, `tag`, `list_ms`,
`order_ms`, and

- `objects`: `L1KeyColumns` (`chunk_hash`, `model_name`, `kv_rank`,
  `object_group_id`, `cache_salt` lists), one entry per object;
- `size_bytes`, `states`: per object, same order;
- `eviction_order`: `L1KeyColumns` of every key the policy tracks, the
  next victim first, or null when the policy keeps no single order (no-op,
  ARC, IsolatedLRU). Only LRU implements `EvictionPolicy.eviction_order()`
  today.

The file is columnar (one list per field): a column is one C-level `map`
pass and encodes as one msgpack array, where one row object per entry
would cost more than everything else together at 100k objects.

`states` holds `committed`, `temporary` (committed, deleted after its read), or
`staging` (write-reserved, not committed). `kv_rank` is the packed value
`ObjectKey` stores. `eviction_ranks(section)` returns each object's
position in the order (0 for the next victim, -1 when untracked) and the
number of ordered keys the listing lacks. The order is stored as keys
rather than as ranks because matching 100k keys on the server costs about
as much as everything else together; the reader matches offline.

## Accessors

Added for this module, public and read-only:
`EvictionPolicy.eviction_order()`, `L1EvictionController.eviction_policy`,
`L1ManagerInterface.list_objects()`, `StorageManager.snapshot_l1()`.

## Cost

Budget at 100k L1 objects: server Python time under 100 ms per request,
with the L1 lock held under 20 ms. Measured with a CPU-only script over a
real LRU `StorageManager` holding 100k committed objects, 20 requests,
min / median / max in ms: `list_ms` 8.3 / 9.8 / 17.5, `order_ms` 25.4 /
26.8 / 31.8, `build_ms + encode_ms` 24.6 / 28.2 / 36.0,
`collect_ms + build_ms + encode_ms` 62.0 / 65.6 / 79.0; the file write
(I/O, not recorded) 5.6 / 14.8 / 20.7.
The file is 8.9 MB.

Copying the LRU order is the largest item: iterating an `OrderedDict`
hashes every key, and `ObjectKey.__hash__` is Python code. It runs under
the policy lock, which L1 listener callbacks also take while holding the L1
lock, so L1 operations can wait up to `order_ms` on it.

## Limits

- Each L1 manager is listed under its own lock, then its eviction order is
  copied under the policy lock. Keys evicted or written between those two
  reads show up as ordered keys the listing lacks, or as listed objects
  with rank -1 (`eviction_ranks`). Nothing is consistent across managers.
- A request that arrives while the previous file is still being written is
  skipped, so a file for a snapshot id may be missing.
- With several schedulers on one server (each sending its own requests),
  a request that arrives while another scheduler's file is being written is
  skipped too: the module keeps one write in flight per server, not per
  scheduler.
- A failed write removes its temporary file and is logged on the server
  only.
