State Snapshots
===============

A state snapshot writes the full content of vLLM's GPU prefix cache (L0) to
a file at a fixed interval, for offline analysis: which blocks vLLM caches,
which are free and which it reuses next, and which LMCache chunks each
prompt maps to. Metrics count events; a snapshot shows what every cache
holds at one moment.

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

A snapshot due while the previous file is still being written is skipped,
so a snapshot id may have no file.

Files
-----

Each scheduler run (``run_id`` = ``<hostname>-<pid>-<start unix time>``)
writes:

- ``l0-<run_id>-<id>.msgpack``: one per snapshot. Every cached block hash
  with its block id and reference count, and the free queue in reuse order.
- ``requests-<run_id>.bin``: append-only, one record per request, mapping
  its vLLM block hashes to the LMCache chunk hashes the servers store it
  under.

Summarizing a snapshot
----------------------

.. code-block:: bash

    python -m lmcache.tools.state_snapshot /data/snapshots 12

prints the counts of snapshot 12 (cached blocks per KV cache group, free
blocks, requests mapped) and what the snapshot cost the scheduler. Pass
``--run-id`` when several runs share the directory.

Cost
----

Measured with 100k GPU blocks: one snapshot takes about 16 ms of the
scheduler thread and 2 ms of a writer thread, and the file is 4.5 MB.
Between snapshots the scheduler pays a clock read per step and one prompt
copy per new request for the mapping.

The file format and the full cost breakdown are in the design doc,
``docs/design/integration/vllm/state_snapshot.md``.
