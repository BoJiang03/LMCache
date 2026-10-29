# SPDX-License-Identifier: Apache-2.0
"""Write the server's whole L1 to a snapshot file when a scheduler asks.

The vLLM connector's state snapshot (``lmcache.mp.snapshot_dir``) sends
``l1_snapshot.take`` every 10 s. The module lists every L1 manager's objects
under the manager's lock, copies the eviction order under the policy's lock,
answers at once, and encodes and writes the file on its own writer thread.

The module lives outside the built-in ``modules`` package, so a server pays
nothing unless it is loaded. Start the MP server with::

    --server-module '{"module_path": "lmcache.v1.multiprocess.l1_snapshot"}'

Design: ``docs/design/integration/vllm/state_snapshot.md``.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
import os
import socket
import time

# Third Party
import msgspec

# First Party
from lmcache.v1.distributed.internal_api import L1ManagerSnapshot
from lmcache.v1.multiprocess.engine_context import MPCacheServerContext
from lmcache.v1.multiprocess.engine_module import (
    DiscoverableModule,
    EngineModule,
    ModuleBuildContext,
)
from lmcache.v1.multiprocess.ext_server_module import server_module_handler
from lmcache.v1.multiprocess.protocols.state_snapshot import (
    FORMAT_VERSION,
    L1_SNAPSHOT_METHOD,
    L1ManagerSection,
    L1ObjectEntry,
    L1SnapshotFile,
    L1SnapshotReply,
    L1SnapshotRequest,
    l1_snapshot_path,
    log_failed_write,
    write_atomically,
)


def _manager_section(snapshot: L1ManagerSnapshot) -> L1ManagerSection:
    """Turn one L1 manager's snapshot into its file section.

    Args:
        snapshot: The manager's listed objects and eviction order.

    Returns:
        The section, each object ranked by the eviction order.
    """
    order = snapshot.eviction_order or []
    rank = {key: index for index, key in enumerate(order)}
    objects = [
        L1ObjectEntry(
            chunk_hash=record.key.chunk_hash,
            model_name=record.key.model_name,
            kv_rank=record.key.kv_rank,
            object_group_id=record.key.object_group_id,
            cache_salt=record.key.cache_salt,
            size_bytes=record.size_bytes,
            state=record.state,
            eviction_rank=rank.get(record.key, -1),
        )
        for record in snapshot.objects
    ]
    listed = {record.key for record in snapshot.objects}
    return L1ManagerSection(
        l1_manager_id=snapshot.l1_manager_id,
        tag=snapshot.tag,
        has_eviction_order=snapshot.eviction_order is not None,
        num_unlisted_ordered_keys=sum(key not in listed for key in order),
        list_ms=snapshot.list_ms,
        order_ms=snapshot.order_ms,
        objects=objects,
    )


def _write_l1_snapshot(
    path: str,
    request: L1SnapshotRequest,
    server_id: str,
    wall_time: float,
    monotonic_time: float,
    snapshots: list[L1ManagerSnapshot],
    started_at: float,
) -> None:
    """Encode and write one L1 file; runs on the module's writer thread.

    Args:
        path: The file to write.
        request: The scheduler's request.
        server_id: This server's ``<hostname>-<pid>``.
        wall_time: ``time.time()`` when the request was handled.
        monotonic_time: ``time.monotonic()`` at the same moment.
        snapshots: Every L1 manager's snapshot.
        started_at: ``time.perf_counter()`` when the request arrived.

    Raises:
        OSError: If the file cannot be written.
    """
    managers = msgspec.msgpack.encode([_manager_section(s) for s in snapshots])
    header = L1SnapshotFile(
        format_version=FORMAT_VERSION,
        run_id=request.run_id,
        snapshot_id=request.snapshot_id,
        server_id=server_id,
        wall_time=wall_time,
        monotonic_time=monotonic_time,
        lock_ms=sum(s.list_ms for s in snapshots),
        total_ms=(time.perf_counter() - started_at) * 1e3,
        managers=msgspec.Raw(managers),
    )
    write_atomically(path, msgspec.msgpack.encode(header))


class L1SnapshotModule(DiscoverableModule):
    """Serve ``l1_snapshot.take`` from the storage manager's L1 managers.

    Args:
        ctx: Engine context whose storage manager is read.
    """

    module_name = "l1_snapshot"

    @classmethod
    def create(cls, build_ctx: ModuleBuildContext) -> EngineModule | None:
        """Build the module over ``build_ctx``'s engine context.

        Args:
            build_ctx: The server's module build context.

        Returns:
            The module; never ``None``, since it needs no optional resource.
        """
        return cls(build_ctx.engine_context)

    def __init__(self, ctx: MPCacheServerContext) -> None:
        self._ctx = ctx
        self._server_id = "%s-%d" % (socket.gethostname(), os.getpid())
        # One thread: files are written in request order, and encoding a
        # large L1 never occupies a request-handling thread.
        self._writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lmcache-l1-snapshot"
        )

    @property
    def context(self) -> MPCacheServerContext:
        """Return the shared engine context."""
        return self._ctx

    def report_status(self) -> dict:
        """Return no status; the module keeps no state between requests."""
        return {}

    def close(self) -> None:
        """Finish writing every requested file, then stop the writer."""
        self._writer.shutdown(wait=True)

    @server_module_handler(L1_SNAPSHOT_METHOD)
    def take(self, payload: bytes) -> bytes:
        """List L1 for an encoded :class:`L1SnapshotRequest` and queue its file.

        The reply goes out once the objects are listed; the file appears
        later, written atomically, and a failed write is only logged.

        Args:
            payload: The msgpack-encoded request.

        Returns:
            The msgpack-encoded :class:`L1SnapshotReply`.

        Raises:
            msgspec.DecodeError: If the payload is not an encoded request.
        """
        started_at = time.perf_counter()
        request = msgspec.msgpack.decode(payload, type=L1SnapshotRequest)
        wall_time = time.time()
        monotonic_time = time.monotonic()
        snapshots = self._ctx.storage_manager.snapshot_l1()
        path = l1_snapshot_path(
            request.snapshot_dir, request.run_id, request.snapshot_id, self._server_id
        )
        write = self._writer.submit(
            _write_l1_snapshot,
            path,
            request,
            self._server_id,
            wall_time,
            monotonic_time,
            snapshots,
            started_at,
        )
        write.add_done_callback(log_failed_write)
        return msgspec.msgpack.encode(
            L1SnapshotReply(
                path=path,
                num_objects=sum(len(s.objects) for s in snapshots),
                lock_ms=sum(s.list_ms for s in snapshots),
            )
        )
