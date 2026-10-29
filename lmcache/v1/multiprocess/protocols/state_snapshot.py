# SPDX-License-Identifier: Apache-2.0
"""Wire payloads and file formats of the L0/L1 state snapshot.

Every :data:`SNAPSHOT_INTERVAL_S` the vLLM scheduler writes its GPU prefix
cache (L0) and lazy-offload buffer to an L0 file and asks every LMCache MP
server, through the ``l1_snapshot.take`` server-module method, to write its
CPU cache (L1) to an L1 file under the same snapshot id. Each prompt the
scheduler tracks is mapped once, in an append-only request file, from its
vLLM block hashes to the LMCache chunk hashes the servers key it by.

Files are msgpack (``msgspec``). A snapshot file is one object whose bulk
is a nested msgpack object (``msgspec.Raw``), so the cost of encoding it can
be recorded in the same file; :func:`read_l0_snapshot` and
:func:`read_l1_snapshot` decode both levels. The request file is a sequence
of length-prefixed records (:func:`iter_records`).

Design: ``docs/design/integration/vllm/state_snapshot.md``.
"""

# Standard
from collections.abc import Iterator
from concurrent.futures import Future
from typing import IO, Any, TypeVar
import os
import struct

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

L1_SNAPSHOT_METHOD = "l1_snapshot.take"

#: Seconds between two snapshots.
SNAPSHOT_INTERVAL_S = 10.0

#: Bumped whenever a file schema below changes incompatibly.
FORMAT_VERSION = 1

_RECORD_LENGTH = struct.Struct(">I")
_T = TypeVar("_T")


class L1SnapshotRequest(msgspec.Struct):
    """Ask a server to write its L1 to a snapshot file.

    Args:
        run_id: The scheduler run the snapshot belongs to.
        snapshot_id: The snapshot id the scheduler's L0 file carries too.
        snapshot_dir: The directory the server writes into, as the server
            sees it.
    """

    run_id: str
    snapshot_id: int
    snapshot_dir: str


class L1SnapshotReply(msgspec.Struct):
    """What a server collected; its file is written after the reply.

    Args:
        path: The L1 file the server writes.
        num_objects: Objects listed, over every L1 manager.
        lock_ms: Wall time spent listing objects under the L1 manager locks
            (see :class:`L1ManagerSection`), summed over managers.
    """

    path: str
    num_objects: int
    lock_ms: float


class L1ObjectEntry(msgspec.Struct, array_like=True):
    """One L1 object: its ``ObjectKey`` fields, size, state and LRU rank.

    Args:
        chunk_hash: ``ObjectKey.chunk_hash``.
        model_name: ``ObjectKey.model_name``.
        kv_rank: ``ObjectKey.kv_rank`` (packed, as the key stores it).
        object_group_id: ``ObjectKey.object_group_id``.
        cache_salt: ``ObjectKey.cache_salt``.
        size_bytes: Logical size in bytes.
        state: ``committed``, ``temporary`` or ``staging`` (write-reserved,
            not yet committed).
        eviction_rank: Position in the eviction policy's order, 0 for the
            next victim; -1 when the policy keeps no order or does not list
            the key.
    """

    chunk_hash: bytes
    model_name: str
    kv_rank: int
    object_group_id: int
    cache_salt: str
    size_bytes: int
    state: str
    eviction_rank: int


class L1ManagerSection(msgspec.Struct):
    """One L1 manager in an L1 file.

    Args:
        l1_manager_id: The manager's process-local id.
        tag: The manager's configured tag.
        has_eviction_order: Whether its policy keeps an order
            (``eviction_rank`` is meaningful).
        num_unlisted_ordered_keys: Keys in the policy's order that the
            object listing did not have: evicted or written between the two
            reads, or tracked by the policy before their write finished.
        list_ms: Wall time of listing the objects under the manager's lock,
            including the wait for it.
        order_ms: Wall time of copying the policy's order under its lock.
        objects: Every listed object.
    """

    l1_manager_id: int
    tag: str
    has_eviction_order: bool
    num_unlisted_ordered_keys: int
    list_ms: float
    order_ms: float
    objects: list[L1ObjectEntry]


class L1SnapshotFile(msgspec.Struct):
    """An L1 file: one server's L1 at one snapshot.

    Args:
        format_version: :data:`FORMAT_VERSION`.
        run_id: The scheduler run that asked.
        snapshot_id: The snapshot id it asked for.
        server_id: ``<hostname>-<pid>`` of the server.
        wall_time: Server ``time.time()`` when the request was handled.
        monotonic_time: Server ``time.monotonic()`` at the same moment.
        lock_ms: ``list_ms`` summed over managers.
        total_ms: Server wall time from receiving the request to the body's
            encoding finished, including the wait for the writer thread;
            only the final file write is left out.
        managers: Encoded ``list[L1ManagerSection]``.
    """

    format_version: int
    run_id: str
    snapshot_id: int
    server_id: str
    wall_time: float
    monotonic_time: float
    lock_ms: float
    total_ms: float
    managers: msgspec.Raw


class CachedBlockEntry(msgspec.Struct, array_like=True):
    """One prefix-cache entry of a GPU block.

    Args:
        block_id: The block.
        group_id: The KV cache group its hash belongs to.
        block_hash: vLLM's block hash, without the group id.
        ref_cnt: The block's reference count; 0 for a free block.
        hash_num_tokens: The tokens the hash covers (fewer than a block's
            for a partial Mamba-state entry); ``None`` when vLLM does not
            record it, as for a non-primary entry.
        primary: Whether this is the block's own hash, or an extra entry
            vLLM registered for it (``BlockPool.cached_block_hashes_by_block``).
    """

    block_id: int
    group_id: int
    block_hash: bytes
    ref_cnt: int
    hash_num_tokens: int | None
    primary: bool


class BufferedOpEntry(msgspec.Struct, array_like=True):
    """One store operation lazy offload buffers.

    Args:
        request_id: Its request.
        start: First token of its range.
        end: Token its range ends at.
        block_ids: Per KV cache group, the GPU blocks of the range.
        held: Its blocks are pinned while it waits; it is due.
        expedited: A finish or lookup made it due early.
    """

    request_id: str
    start: int
    end: int
    block_ids: list[list[int]]
    held: bool
    expedited: bool


class InFlightBatchEntry(msgspec.Struct, array_like=True):
    """One submitted store batch awaiting its receipt.

    Args:
        request_id: Its request.
        ranges: The ``(start, end)`` token range of each of its operations.
        num_pinned_blocks: GPU blocks it pins.
        failed: A worker reported it failed; its operations are buffered
            again once every worker reported.
        orphaned: A new request took over its finished request's id.
    """

    request_id: str
    ranges: list[tuple[int, int]]
    num_pinned_blocks: int
    failed: bool
    orphaned: bool


class LazyBufferSection(msgspec.Struct):
    """What lazy offload held back from the servers.

    Args:
        ops: Every buffered operation.
        in_flight: Every submitted batch awaiting its receipt.
        finished_request_ids: Finished requests lazy offload still tracks.
    """

    ops: list[BufferedOpEntry]
    in_flight: list[InFlightBatchEntry]
    finished_request_ids: list[str]


class L0Body(msgspec.Struct):
    """The bulk of an L0 file.

    Args:
        cached_blocks: Every prefix-cache entry of every GPU block.
        free_queue: The free blocks' ids, the next to be reused first.
        lazy: The lazy-offload state, or ``None`` when lazy offload is off.
    """

    cached_blocks: list[CachedBlockEntry]
    free_queue: list[int]
    lazy: LazyBufferSection | None


class L0SnapshotFile(msgspec.Struct):
    """An L0 file: the scheduler's GPU prefix cache at one snapshot.

    Args:
        format_version: :data:`FORMAT_VERSION`.
        run_id: ``<hostname>-<pid>-<start time>`` of the scheduler.
        snapshot_id: Counts from 0 within the run.
        wall_time: Scheduler ``time.time()`` when collecting started.
        monotonic_time: Scheduler ``time.monotonic()`` at the same moment.
        model_name: The model name the servers key objects by.
        kv_world_size: The kv world size; L1 holds one object per kv rank.
        chunk_size: LMCache tokens per chunk.
        hash_block_size: Tokens per vLLM block hash.
        hash_algorithm: The servers' token hash algorithm.
        num_gpu_blocks: Blocks in vLLM's pool, the null block included.
        group_tokens_per_block: Tokens per block of each KV cache group; 0
            for a group LMCache does not cache.
        l1_requested: Whether the servers were asked for L1 files (not when
            a server was unhealthy).
        collect_ms: Scheduler-thread time reading vLLM's pool and the lazy
            buffer.
        dump_ms: Scheduler-thread time of the whole snapshot step: collect,
            send the L1 request, hand off to the writer thread.
        mapping_ms: Scheduler-thread time copying prompts for the request
            file since the previous snapshot.
        encode_ms: Writer-thread time encoding the body.
        body: Encoded :class:`L0Body`.
    """

    format_version: int
    run_id: str
    snapshot_id: int
    wall_time: float
    monotonic_time: float
    model_name: str
    kv_world_size: int
    chunk_size: int
    hash_block_size: int
    hash_algorithm: str
    num_gpu_blocks: int
    group_tokens_per_block: list[int]
    l1_requested: bool
    collect_ms: float
    dump_ms: float
    mapping_ms: float
    encode_ms: float
    body: msgspec.Raw


class RequestMappingRecord(msgspec.Struct):
    """One prompt's vLLM block hashes and LMCache chunk hashes.

    Args:
        request_id: The request.
        cache_salt: Its cache salt, as the servers key it.
        num_prompt_tokens: Its prompt length.
        block_hashes: vLLM's ``Request.block_hashes`` when it was first
            tracked: one per full hash block of the prompt.
        chunk_hashes: ``ObjectKey.chunk_hash`` of every full chunk of the
            prompt, by the servers' ``TokenHasher``.
        wall_time: Scheduler ``time.time()`` when it was first tracked.
        monotonic_time: Scheduler ``time.monotonic()`` at the same moment.
    """

    request_id: str
    cache_salt: str
    num_prompt_tokens: int
    block_hashes: list[bytes]
    chunk_hashes: list[bytes]
    wall_time: float
    monotonic_time: float


def l0_snapshot_path(snapshot_dir: str, run_id: str, snapshot_id: int) -> str:
    """Return the L0 file of a snapshot.

    Args:
        snapshot_dir: The snapshot directory.
        run_id: The scheduler run.
        snapshot_id: The snapshot.

    Returns:
        ``<dir>/l0-<run_id>-<snapshot_id>.msgpack``, the id zero-padded.
    """
    return os.path.join(snapshot_dir, "l0-%s-%06d.msgpack" % (run_id, snapshot_id))


def l1_snapshot_path(
    snapshot_dir: str, run_id: str, snapshot_id: int, server_id: str
) -> str:
    """Return one server's L1 file of a snapshot.

    Args:
        snapshot_dir: The snapshot directory.
        run_id: The scheduler run.
        snapshot_id: The snapshot.
        server_id: The server, ``<hostname>-<pid>``.

    Returns:
        ``<dir>/l1-<run_id>-<snapshot_id>-<server_id>.msgpack``.
    """
    return os.path.join(
        snapshot_dir, "l1-%s-%06d-%s.msgpack" % (run_id, snapshot_id, server_id)
    )


def requests_path(snapshot_dir: str, run_id: str) -> str:
    """Return the request-mapping file of a run.

    Args:
        snapshot_dir: The snapshot directory.
        run_id: The scheduler run.

    Returns:
        ``<dir>/requests-<run_id>.bin``.
    """
    return os.path.join(snapshot_dir, "requests-%s.bin" % run_id)


def write_atomically(path: str, data: bytes) -> None:
    """Write ``data`` to ``path`` through a rename, so readers never see a
    partial file.

    Args:
        path: The file to create or replace.
        data: Its content.

    Raises:
        OSError: If the file cannot be written.
    """
    tmp_path = path + ".tmp"
    with open(tmp_path, "wb") as f:
        f.write(data)
    os.replace(tmp_path, path)


def append_record(f: IO[bytes], record: msgspec.Struct) -> None:
    """Append one length-prefixed msgpack record and flush it.

    Args:
        f: A file opened for binary append.
        record: The record to append.

    Raises:
        OSError: If the write fails.
    """
    data = msgspec.msgpack.encode(record)
    f.write(_RECORD_LENGTH.pack(len(data)))
    f.write(data)
    f.flush()


def iter_records(path: str, record_type: type[_T]) -> Iterator[_T]:
    """Read the records :func:`append_record` appended, in order.

    A record cut short by a writer still running is not yielded.

    Args:
        path: The record file.
        record_type: The type to decode every record as.

    Yields:
        Each complete record.

    Raises:
        OSError: If the file cannot be read.
        msgspec.DecodeError: If a complete record does not decode.
    """
    with open(path, "rb") as f:
        data = f.read()
    offset = 0
    while offset + _RECORD_LENGTH.size <= len(data):
        (length,) = _RECORD_LENGTH.unpack_from(data, offset)
        offset += _RECORD_LENGTH.size
        if offset + length > len(data):
            return
        yield msgspec.msgpack.decode(data[offset : offset + length], type=record_type)
        offset += length


def read_l0_snapshot(path: str) -> tuple[L0SnapshotFile, L0Body]:
    """Read an L0 file.

    Args:
        path: The file.

    Returns:
        Its header and its decoded body.

    Raises:
        OSError: If the file cannot be read.
        msgspec.DecodeError: If it is not an L0 file of this format.
    """
    with open(path, "rb") as f:
        header = msgspec.msgpack.decode(f.read(), type=L0SnapshotFile)
    return header, msgspec.msgpack.decode(header.body, type=L0Body)


def read_l1_snapshot(path: str) -> tuple[L1SnapshotFile, list[L1ManagerSection]]:
    """Read an L1 file.

    Args:
        path: The file.

    Returns:
        Its header and its decoded manager sections.

    Raises:
        OSError: If the file cannot be read.
        msgspec.DecodeError: If it is not an L1 file of this format.
    """
    with open(path, "rb") as f:
        header = msgspec.msgpack.decode(f.read(), type=L1SnapshotFile)
    return header, msgspec.msgpack.decode(header.managers, type=list[L1ManagerSection])


def log_failed_write(future: "Future[Any]") -> None:
    """Log the failure of a snapshot write a writer thread ran.

    Meant for ``Future.add_done_callback``: an executor otherwise keeps the
    exception on a future nobody reads.

    Args:
        future: The finished write.
    """
    exc = future.exception()
    if exc is not None:
        logger.error("State snapshot: writing a file failed", exc_info=exc)
