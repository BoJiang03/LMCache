# SPDX-License-Identifier: Apache-2.0
"""Write the server's whole L1 to a snapshot file when a scheduler asks.

The vLLM connector's state snapshot (``lmcache.mp.snapshot_dir``) sends
``l1_snapshot.take`` every 10 s. The module lists every L1 manager's objects
under the manager's lock, copies the eviction order under the policy's lock,
answers at once, and builds, encodes and writes the file on its own writer
thread. A request that arrives while the previous file is still being
written is skipped.

The module lives outside the built-in ``modules`` package, so a server pays
nothing unless it is loaded. Start the MP server with::

    --server-module '{"module_path": "lmcache.v1.multiprocess.l1_snapshot"}'

Design: ``docs/design/integration/vllm/state_snapshot.md``.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from operator import attrgetter
import os
import socket
import time

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import ObjectKey
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
    L1KeyColumns,
    L1ManagerSection,
    L1SnapshotFile,
    L1SnapshotReply,
    L1SnapshotRequest,
    l1_snapshot_path,
    log_failed_write,
    write_atomically,
)

logger = init_logger(__name__)

_IDENTITY = attrgetter("identity")
_CHUNK_HASH = attrgetter("chunk_hash")
_MODEL_NAME = attrgetter("model_name")
_KV_RANK = attrgetter("kv_rank")
_OBJECT_GROUP_ID = attrgetter("object_group_id")
_CACHE_SALT = attrgetter("cache_salt")


def _key_columns(keys: list[ObjectKey]) -> L1KeyColumns:
    """Split keys into one column per field.

    Reads ``ObjectKey.identity`` once per key and its plain fields with
    C-level ``map`` passes; the ``ObjectKey`` properties would cost a Python
    call per field per key.

    Args:
        keys: The keys.

    Returns:
        Their columns, in key order.
    """
    identities = list(map(_IDENTITY, keys))
    return L1KeyColumns(
        chunk_hash=list(map(_CHUNK_HASH, identities)),
        model_name=list(map(_MODEL_NAME, identities)),
        kv_rank=list(map(_KV_RANK, identities)),
        object_group_id=list(map(_OBJECT_GROUP_ID, identities)),
        cache_salt=list(map(_CACHE_SALT, identities)),
    )


def _manager_section(snapshot: L1ManagerSnapshot) -> L1ManagerSection:
    """Turn one L1 manager's snapshot into its file section.

    Args:
        snapshot: The manager's listed objects and eviction order.

    Returns:
        The section.
    """
    order = snapshot.eviction_order
    return L1ManagerSection(
        l1_manager_id=snapshot.l1_manager_id,
        tag=snapshot.tag,
        list_ms=snapshot.list_ms,
        order_ms=snapshot.order_ms,
        objects=_key_columns(snapshot.objects.keys),
        size_bytes=snapshot.objects.size_bytes,
        state=list(snapshot.objects.states),
        eviction_order=None if order is None else _key_columns(order),
    )


def _write_l1_snapshot(
    path: str,
    request: L1SnapshotRequest,
    server_id: str,
    wall_time: float,
    monotonic_time: float,
    snapshots: list[L1ManagerSnapshot],
    collect_ms: float,
) -> None:
    """Build, encode and write one L1 file; runs on the module's writer thread.

    Args:
        path: The file to write.
        request: The scheduler's request.
        server_id: This server's ``<hostname>-<pid>``.
        wall_time: ``time.time()`` when the request was handled.
        monotonic_time: ``time.monotonic()`` at the same moment.
        snapshots: Every L1 manager's snapshot.
        collect_ms: Request-thread time taking ``snapshots``.

    Raises:
        OSError: If the file cannot be written.
    """
    started_at = time.perf_counter()
    sections = [_manager_section(s) for s in snapshots]
    built_at = time.perf_counter()
    managers = msgspec.msgpack.encode(sections)
    encoded_at = time.perf_counter()
    build_ms = (built_at - started_at) * 1e3
    encode_ms = (encoded_at - built_at) * 1e3
    header = L1SnapshotFile(
        format_version=FORMAT_VERSION,
        run_id=request.run_id,
        snapshot_id=request.snapshot_id,
        server_id=server_id,
        wall_time=wall_time,
        monotonic_time=monotonic_time,
        lock_ms=sum(s.list_ms for s in snapshots),
        collect_ms=collect_ms,
        build_ms=build_ms,
        encode_ms=encode_ms,
        total_ms=collect_ms + build_ms + encode_ms,
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
        # The last queued write; a request arriving before it finishes is
        # skipped, so a slow disk never piles snapshots up in memory.
        self._last_write: Future[None] | None = None

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
        later, written atomically, and a failed write is only logged. While
        the previous file is still being written, nothing is collected and
        the reply says so.

        Args:
            payload: The msgpack-encoded request.

        Returns:
            The msgpack-encoded :class:`L1SnapshotReply`.

        Raises:
            msgspec.DecodeError: If the payload is not an encoded request.
        """
        request = msgspec.msgpack.decode(payload, type=L1SnapshotRequest)
        if self._last_write is not None and not self._last_write.done():
            logger.debug(
                "L1 snapshot %d skipped: the previous file is still being written",
                request.snapshot_id,
            )
            return msgspec.msgpack.encode(
                L1SnapshotReply(skipped=True, path="", num_objects=0, lock_ms=0.0)
            )
        started_at = time.perf_counter()
        wall_time = time.time()
        monotonic_time = time.monotonic()
        snapshots = self._ctx.storage_manager.snapshot_l1()
        collect_ms = (time.perf_counter() - started_at) * 1e3
        path = l1_snapshot_path(
            request.snapshot_dir, request.run_id, request.snapshot_id, self._server_id
        )
        self._last_write = self._writer.submit(
            _write_l1_snapshot,
            path,
            request,
            self._server_id,
            wall_time,
            monotonic_time,
            snapshots,
            collect_ms,
        )
        self._last_write.add_done_callback(log_failed_write)
        return msgspec.msgpack.encode(
            L1SnapshotReply(
                skipped=False,
                path=path,
                num_objects=sum(len(s.objects.keys) for s in snapshots),
                lock_ms=sum(s.list_ms for s in snapshots),
            )
        )
