State Snapshots
===============

A state snapshot writes the full content of vLLM's GPU prefix cache (L0)
and of each LMCache server's CPU cache (L1) to files at a fixed interval,
for offline analysis: which blocks vLLM caches and reuses next, which
chunks L1 holds and evicts next, and which LMCache chunks each prompt maps
to, so the two tiers can be compared at one moment. Metrics count events;
a snapshot shows what every cache holds.

Snapshots are off by default and are a diagnostic: nothing in the serving
path reads the files.

Enabling
--------

Set the directory in vLLM's ``kv_connector_extra_config`` for
``LMCacheMPConnector``:

.. code-block:: json

    {"lmcache.mp.snapshot_dir": "/data/snapshots",
     "lmcache.mp.snapshot_interval_s": 10}

.. list-table::
   :header-rows: 1
   :widths: 35 15 50

   * - Option
     - Default
     - Description
   * - ``lmcache.mp.snapshot_dir``
     - *(unset)*
     - Directory the scheduler writes snapshot files to. Unset disables
       snapshots.
   * - ``lmcache.mp.snapshot_interval_s``
     - ``10``
     - Seconds between two snapshots. Must be positive.

For L1, also load the server module on every LMCache server:

.. code-block:: bash

    lmcache server ... \
      --server-module '{"module_path": "lmcache.v1.multiprocess.l1_snapshot"}'

At each snapshot the scheduler asks every server to write its L1 file. The
server writes it to the scheduler's ``lmcache.mp.snapshot_dir`` (made
absolute, and created if missing), so that path must name the same
directory on the server's host. A server without the module answers with
an error, and the scheduler logs a warning and keeps writing L0 files.

A snapshot due while the previous file is still being written is skipped,
so a snapshot id may have no file. A server that has not answered when the
next snapshot is due is skipped for that snapshot.

Files
-----

Each scheduler run (``run_id`` = ``<hostname>-<pid>-<start unix time>``)
writes:

- ``l0-<run_id>-<id>.msgpack``: one per snapshot. Every cached block hash
  with its block id and reference count, and the free queue in reuse order.
- ``l1-<run_id>-<id>-<server_id>.msgpack``: one per snapshot and server
  (``server_id`` = ``<hostname>-<pid>``). Every L1 object with its key,
  size and state, and each L1 manager's eviction order.
- ``requests-<run_id>.bin``: append-only, one record per request, mapping
  its vLLM block hashes to the LMCache chunk hashes the servers store it
  under.

Summarizing a snapshot
----------------------

.. code-block:: bash

    python -m lmcache.tools.state_snapshot /data/snapshots 12

prints the counts of snapshot 12 (cached blocks per KV cache group, free
blocks, requests mapped, L1 objects per state), how many L1 chunks are also
cached in L0, and what the snapshot cost the scheduler and each server. Pass
``--run-id`` when several runs share the directory.

Cost
----

Measured with 100k GPU blocks: one snapshot takes about 16 ms of the
scheduler thread and 2 ms of a writer thread, and the file is 4.5 MB.
Between snapshots the scheduler pays a clock read per step and one prompt
copy per new request for the mapping. With 100k L1 objects, a server spends
about 66 ms per snapshot (L1 lock held about 10 ms, eviction-policy lock
about 27 ms), and its file is 8.9 MB.

The file formats and the full cost breakdown are in the design docs,
``docs/design/integration/vllm/state_snapshot.md`` and
``docs/design/v1/multiprocess/l1_snapshot.md``.
