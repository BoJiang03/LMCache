# SPDX-License-Identifier: Apache-2.0
"""Load one L0/L1 state snapshot and count what it holds.

Counts only; deciding why a chunk is where it is is left to the analysis
that reads these files. File formats:
``lmcache.v1.multiprocess.protocols.state_snapshot``.
"""

# Standard
from collections import Counter
from dataclasses import dataclass, field
import glob
import os

# First Party
from lmcache.v1.multiprocess.protocols.state_snapshot import (
    L0Body,
    L0SnapshotFile,
    L1ManagerSection,
    L1SnapshotFile,
    RequestMappingRecord,
    eviction_ranks,
    iter_records,
    l0_snapshot_path,
    l1_snapshot_path,
    read_l0_snapshot,
    read_l1_snapshot,
    requests_path,
    split_block_key,
)

# An L1 chunk across kv ranks and object groups: (chunk hash, cache salt).
_ChunkId = tuple[bytes, str]


def _find_run_id(snapshot_dir: str, snapshot_id: int) -> str:
    """Return the only run with an L0 file for ``snapshot_id``.

    Args:
        snapshot_dir: The snapshot directory.
        snapshot_id: The snapshot.

    Returns:
        The run id.

    Raises:
        ValueError: If no run, or more than one, has that L0 file.
    """
    prefix, suffix = os.path.basename(
        l0_snapshot_path(snapshot_dir, "*", snapshot_id)
    ).split("*")
    paths = glob.glob(l0_snapshot_path(glob.escape(snapshot_dir), "*", snapshot_id))
    run_ids = sorted(os.path.basename(p)[len(prefix) : -len(suffix)] for p in paths)
    if len(run_ids) != 1:
        raise ValueError(
            "Expected one run with snapshot %d in %s, found %s; pass run_id"
            % (snapshot_id, snapshot_dir, run_ids)
        )
    return run_ids[0]


def _chunk_block_hashes(
    records: list[RequestMappingRecord], chunk_size: int, hash_block_size: int
) -> dict[_ChunkId, list[bytes]]:
    """Map each mapped chunk to the vLLM block hashes of its tokens.

    Args:
        records: The run's request mapping.
        chunk_size: LMCache tokens per chunk.
        hash_block_size: Tokens per vLLM block hash.

    Returns:
        Per chunk, the hashes of the hash blocks overlapping its tokens.
    """
    mapping: dict[_ChunkId, list[bytes]] = {}
    for record in records:
        for index, chunk_hash in enumerate(record.chunk_hashes):
            first = index * chunk_size // hash_block_size
            end = -(-(index + 1) * chunk_size // hash_block_size)
            mapping.setdefault(
                (chunk_hash, record.cache_salt), record.block_hashes[first:end]
            )
    return mapping


@dataclass
class L1ServerSummary:
    """One server's L1 file.

    Attributes:
        server_id: ``<hostname>-<pid>`` of the server.
        list_ms: Time listing objects under the L1 manager locks, lock
            waits included (``L1SnapshotFile.list_ms``).
        total_ms: Server Python time of the snapshot: the file's
            ``collect_ms + build_ms + encode_ms``.
        num_unlisted_ordered_keys: Summed over its L1 managers.
    """

    server_id: str
    list_ms: float
    total_ms: float
    num_unlisted_ordered_keys: int


@dataclass
class SnapshotSummary:
    """What one snapshot id holds, over its L0 file and every L1 file.

    Attributes:
        run_id: The scheduler run.
        snapshot_id: The snapshot.
        l0_header: The L0 file's header; its ``*_ms`` fields are the
            scheduler's recorded cost.
        l0_blocks_per_group: Cached GPU blocks per KV cache group, counting
            a block's extra (non-primary) hashes too.
        num_free_blocks: Blocks in vLLM's free queue.
        num_mapped_requests: Records in the run's request file, the run so
            far rather than up to this snapshot.
        l1_servers: One entry per L1 file found.
        l1_objects: L1 objects per ``(object_group_id, state)``.
        l1_bytes: L1 bytes per ``(object_group_id, state)``.
        num_l1_chunks: Distinct ``(chunk_hash, cache_salt)`` in L1.
        num_l1_chunks_mapped: Of those, chunks the request mapping knows.
        num_l1_chunks_any_in_l0: Mapped chunks with at least one of their
            hash blocks cached in L0, in any group.
        num_l1_chunks_all_in_l0: Mapped chunks with every one of their hash
            blocks cached in L0, each in some group.
    """

    run_id: str
    snapshot_id: int
    l0_header: L0SnapshotFile
    l0_blocks_per_group: dict[int, int]
    num_free_blocks: int
    num_mapped_requests: int
    l1_servers: list[L1ServerSummary] = field(default_factory=list)
    l1_objects: dict[tuple[int, str], int] = field(default_factory=dict)
    l1_bytes: dict[tuple[int, str], int] = field(default_factory=dict)
    num_l1_chunks: int = 0
    num_l1_chunks_mapped: int = 0
    num_l1_chunks_any_in_l0: int = 0
    num_l1_chunks_all_in_l0: int = 0


def _read_l1_files(summary: SnapshotSummary, snapshot_dir: str) -> set[_ChunkId]:
    """Fold every L1 file of the snapshot into ``summary``.

    Args:
        summary: The summary to fill; its run and snapshot id select files.
        snapshot_dir: The snapshot directory.

    Returns:
        The distinct chunks the L1 files hold.
    """
    pattern = l1_snapshot_path(
        glob.escape(snapshot_dir),
        glob.escape(summary.run_id),
        summary.snapshot_id,
        "*",
    )
    objects: Counter[tuple[int, str]] = Counter()
    nbytes: Counter[tuple[int, str]] = Counter()
    chunks: set[_ChunkId] = set()
    for path in sorted(glob.glob(pattern)):
        l1_header, sections = read_l1_snapshot(path)
        summary.l1_servers.append(_server_summary(l1_header, sections))
        for section in sections:
            columns = section.objects
            for group_id, state, size in zip(
                columns.object_group_id, section.states, section.size_bytes, strict=True
            ):
                objects[(group_id, state)] += 1
                nbytes[(group_id, state)] += size
            chunks.update(zip(columns.chunk_hash, columns.cache_salt, strict=True))
    summary.l1_objects = dict(objects)
    summary.l1_bytes = dict(nbytes)
    summary.num_l1_chunks = len(chunks)
    return chunks


def _server_summary(
    header: L1SnapshotFile, sections: list[L1ManagerSection]
) -> L1ServerSummary:
    """Summarize one L1 file's header and sections.

    Args:
        header: The file's header.
        sections: Its manager sections.

    Returns:
        The server's summary.
    """
    return L1ServerSummary(
        server_id=header.server_id,
        list_ms=header.list_ms,
        total_ms=header.collect_ms + header.build_ms + header.encode_ms,
        num_unlisted_ordered_keys=sum(eviction_ranks(s)[1] for s in sections),
    )


def _count_chunks_in_l0(
    summary: SnapshotSummary,
    body: L0Body,
    l1_chunks: set[_ChunkId],
    records: list[RequestMappingRecord],
) -> None:
    """Count the L1 chunks whose mapped block hashes L0 caches.

    Args:
        summary: The summary to fill.
        body: The L0 file's body.
        l1_chunks: The distinct chunks in L1.
        records: The run's request mapping.
    """
    header = summary.l0_header
    mapping = _chunk_block_hashes(records, header.chunk_size, header.hash_block_size)
    in_l0 = {split_block_key(key)[0] for key in body.block_keys}
    for chunk in l1_chunks:
        block_hashes = mapping.get(chunk)
        if block_hashes is None:
            continue
        summary.num_l1_chunks_mapped += 1
        cached = [block_hash in in_l0 for block_hash in block_hashes]
        summary.num_l1_chunks_any_in_l0 += any(cached)
        summary.num_l1_chunks_all_in_l0 += bool(cached) and all(cached)


def summarize(
    snapshot_dir: str, snapshot_id: int, run_id: str | None = None
) -> SnapshotSummary:
    """Load a snapshot's L0 file, its L1 files and the request mapping.

    Args:
        snapshot_dir: The snapshot directory.
        snapshot_id: The snapshot.
        run_id: The scheduler run; found from the L0 files when ``None``.

    Returns:
        The snapshot's counts.

    Raises:
        ValueError: If ``run_id`` is ``None`` and not exactly one run has
            the snapshot.
        OSError: If a file cannot be read.
        msgspec.DecodeError: If a file is not of this format.
    """
    if run_id is None:
        run_id = _find_run_id(snapshot_dir, snapshot_id)
    header, body = read_l0_snapshot(l0_snapshot_path(snapshot_dir, run_id, snapshot_id))
    records = list(
        iter_records(requests_path(snapshot_dir, run_id), RequestMappingRecord)
    )
    summary = SnapshotSummary(
        run_id=run_id,
        snapshot_id=snapshot_id,
        l0_header=header,
        l0_blocks_per_group=dict(
            Counter(split_block_key(key)[1] for key in body.block_keys)
        ),
        num_free_blocks=len(body.free_queue),
        num_mapped_requests=len(records),
    )
    l1_chunks = _read_l1_files(summary, snapshot_dir)
    _count_chunks_in_l0(summary, body, l1_chunks, records)
    return summary


def format_summary(summary: SnapshotSummary) -> str:
    """Render a summary as the CLI prints it.

    Args:
        summary: The summary.

    Returns:
        Multi-line text.
    """
    header = summary.l0_header
    lines = [
        "run %s snapshot %d" % (summary.run_id, summary.snapshot_id),
        "L0: %d GPU blocks, %d free" % (header.num_gpu_blocks, summary.num_free_blocks),
    ]
    for group_id, count in sorted(summary.l0_blocks_per_group.items()):
        lines.append("  group %d: %d cached block hashes" % (group_id, count))
    lines.append("requests mapped: %d" % summary.num_mapped_requests)
    lines.append("L1: %d server file(s)" % len(summary.l1_servers))
    for key in sorted(summary.l1_objects):
        lines.append(
            "  object group %d, %s: %d objects, %d bytes"
            % (key[0], key[1], summary.l1_objects[key], summary.l1_bytes[key])
        )
    lines.append(
        "L1 chunks: %d distinct, %d mapped, %d with any block in L0, "
        "%d with every block in L0"
        % (
            summary.num_l1_chunks,
            summary.num_l1_chunks_mapped,
            summary.num_l1_chunks_any_in_l0,
            summary.num_l1_chunks_all_in_l0,
        )
    )
    lines.append(
        "cost: scheduler collect %.2f ms, dump %.2f ms, mapping %.2f ms, "
        "encode %.2f ms (writer thread)"
        % (header.collect_ms, header.dump_ms, header.mapping_ms, header.encode_ms)
    )
    for server in summary.l1_servers:
        lines.append(
            "  server %s: list %.2f ms, total %.2f ms, %d ordered keys unlisted"
            % (
                server.server_id,
                server.list_ms,
                server.total_ms,
                server.num_unlisted_ordered_keys,
            )
        )
    return "\n".join(lines)
