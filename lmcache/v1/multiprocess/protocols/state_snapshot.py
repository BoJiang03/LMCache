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
FORMAT_VERSION = 2

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
        skipped: The server's previous file was still being written, so
            nothing was collected and no file is written.
        path: The L1 file the server writes; empty when skipped.
        num_objects: Objects listed, over every L1 manager.
        lock_ms: Wall time spent listing objects under the L1 manager locks
            (see :class:`L1ManagerSection`), summed over managers.
    """

    skipped: bool
    path: str
    num_objects: int
    lock_ms: float


class L1KeyColumns(msgspec.Struct):
    """``ObjectKey`` fields of many keys, one list per field.

    Columns rather than one row per key: a server builds and encodes each
    column with one C-level pass, which keeps a 100k-object L1 cheap.

    Args:
        chunk_hash: ``ObjectKey.chunk_hash`` of each key.
        model_name: ``ObjectKey.model_name``.
        kv_rank: ``ObjectKey.kv_rank`` (packed, as the key stores it).
        object_group_id: ``ObjectKey.object_group_id``.
        cache_salt: ``ObjectKey.cache_salt``.
    """

    chunk_hash: list[bytes]
    model_name: list[str]
    kv_rank: list[int]
    object_group_id: list[int]
    cache_salt: list[str]


class L1ManagerSection(msgspec.Struct):
    """One L1 manager in an L1 file.

    Object ``i`` is entry ``i`` of every column of ``objects`` and of
    ``size_bytes`` and ``state``. :func:`eviction_ranks` places each object
    in ``eviction_order``.

    Args:
        l1_manager_id: The manager's process-local id.
        tag: The manager's configured tag.
        list_ms: Wall time of listing the objects under the manager's lock,
            including the wait for it.
        order_ms: Wall time of copying the policy's order under its lock.
        objects: Every listed object's key.
        size_bytes: Each object's logical size in bytes.
        state: ``committed``, ``temporary`` or ``staging`` (write-reserved,
            not yet committed).
        eviction_order: Every key the eviction policy tracks, the next
            victim first; ``None`` when the policy keeps no single order.
            Read right after the objects, under another lock, so it can
            hold keys the listing lacks and lack keys it holds.
    """

    l1_manager_id: int
    tag: str
    list_ms: float
    order_ms: float
    objects: L1KeyColumns
    size_bytes: list[int]
    state: list[str]
    eviction_order: L1KeyColumns | None


class L1SnapshotFile(msgspec.Struct):
    """An L1 file: one server's L1 at one snapshot.

    Every ``*_ms`` field is time the server's Python threads spent on this
    snapshot, so their sum bounds what it took from serving (the GIL).

    Args:
        format_version: :data:`FORMAT_VERSION`.
        run_id: The scheduler run that asked.
        snapshot_id: The snapshot id it asked for.
        server_id: ``<hostname>-<pid>`` of the server.
        wall_time: Server ``time.time()`` when the request was handled.
        monotonic_time: Server ``time.monotonic()`` at the same moment.
        lock_ms: ``list_ms`` summed over managers.
        collect_ms: Request thread: listing every manager and copying its
            order (``lock_ms`` and every ``order_ms`` included).
        build_ms: Writer thread: turning the keys into columns.
        encode_ms: Writer thread: encoding the sections.
        total_ms: ``collect_ms + build_ms + encode_ms``. Writing the file
            comes after and is not included; it is file I/O, which runs
            without the GIL.
        managers: Encoded ``list[L1ManagerSection]``.
    """

    format_version: int
    run_id: str
    snapshot_id: int
    server_id: str
    wall_time: float
    monotonic_time: float
    lock_ms: float
    collect_ms: float
    build_ms: float
    encode_ms: float
    total_ms: float
    managers: msgspec.Raw


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
    """The bulk of an L0 file: the GPU prefix cache, one list per field.

    Entry ``i`` of ``block_ids``, ``block_keys``, ``ref_cnts`` and
    ``hash_num_tokens`` is one prefix-cache entry. The first
    ``num_primary`` entries are blocks' own hashes, in block id order; the
    rest are extra entries vLLM registered for a block
    (``BlockPool.cached_block_hashes_by_block``).

    Args:
        block_ids: The entry's block.
        block_keys: vLLM's ``BlockHashWithGroupId``: the block hash followed
            by the KV cache group id, 4 bytes big-endian (see
            :func:`split_block_key`).
        ref_cnts: The block's reference count; 0 for a free block.
        hash_num_tokens: The tokens the hash covers (fewer than a block's
            for a partial Mamba-state entry); ``None`` where vLLM does not
            record it, as for an extra entry.
        num_primary: How many leading entries are blocks' own hashes.
        free_queue: The free blocks' ids, the next to be reused first.
        lazy: The lazy-offload state, or ``None`` when lazy offload is off.
    """

    block_ids: list[int]
    block_keys: list[bytes]
    ref_cnts: list[int]
    hash_num_tokens: list[int | None]
    num_primary: int
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
        encode_ms: Writer-thread time encoding the body. ``dump_ms +
            encode_ms`` is what the snapshot took from the scheduler
            process's GIL; writing the file runs without it.
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


def split_block_key(block_key: bytes) -> tuple[bytes, int]:
    """Split an ``L0Body.block_keys`` entry as vLLM's ``get_block_hash`` and
    ``get_group_id`` do.

    Args:
        block_key: A ``BlockHashWithGroupId``.

    Returns:
        The block hash and the KV cache group id.
    """
    return block_key[:-4], int.from_bytes(block_key[-4:], "big")


def key_tuples(columns: L1KeyColumns) -> list[tuple[bytes, str, int, int, str]]:
    """Zip key columns back into one hashable tuple per key.

    Args:
        columns: The columns.

    Returns:
        ``(chunk_hash, model_name, kv_rank, object_group_id, cache_salt)``
        per key, in column order.
    """
    return list(
        zip(
            columns.chunk_hash,
            columns.model_name,
            columns.kv_rank,
            columns.object_group_id,
            columns.cache_salt,
            strict=True,
        )
    )


def eviction_ranks(section: L1ManagerSection) -> tuple[list[int], int]:
    """Place each listed object in the section's eviction order.

    Args:
        section: An L1 manager section.

    Returns:
        Each object's position in ``eviction_order`` (0 for the next
        victim, -1 when the policy keeps no order or does not track it),
        and how many ordered keys the object listing lacks.
    """
    keys = key_tuples(section.objects)
    if section.eviction_order is None:
        return [-1] * len(keys), 0
    order = key_tuples(section.eviction_order)
    rank = {key: index for index, key in enumerate(order)}
    listed = set(keys)
    unlisted = sum(key not in listed for key in order)
    return [rank.get(key, -1) for key in keys], unlisted


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
