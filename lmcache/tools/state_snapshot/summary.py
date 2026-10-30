# SPDX-License-Identifier: Apache-2.0
"""Load one L0 state snapshot and count what it holds.

Counts only; deciding why a chunk is where it is is left to the analysis
that reads these files. File formats:
``lmcache.v1.multiprocess.protocols.state_snapshot``.
"""

# Standard
from collections import Counter
from dataclasses import dataclass
import glob
import os

# First Party
from lmcache.v1.multiprocess.protocols.state_snapshot import (
    L0SnapshotFile,
    RequestMappingRecord,
    iter_records,
    l0_snapshot_path,
    read_l0_snapshot,
    requests_path,
    split_block_key,
)


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


@dataclass
class SnapshotSummary:
    """What one snapshot id holds, over its L0 file and the request mapping.

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
    """

    run_id: str
    snapshot_id: int
    l0_header: L0SnapshotFile
    l0_blocks_per_group: dict[int, int]
    num_free_blocks: int
    num_mapped_requests: int


def summarize(
    snapshot_dir: str, snapshot_id: int, run_id: str | None = None
) -> SnapshotSummary:
    """Load a snapshot's L0 file and the request mapping.

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
    records = iter_records(requests_path(snapshot_dir, run_id), RequestMappingRecord)
    return SnapshotSummary(
        run_id=run_id,
        snapshot_id=snapshot_id,
        l0_header=header,
        l0_blocks_per_group=dict(
            Counter(split_block_key(key)[1] for key in body.block_keys)
        ),
        num_free_blocks=len(body.free_queue),
        num_mapped_requests=sum(1 for _ in records),
    )


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
    lines.append(
        "cost: scheduler collect %.2f ms, dump %.2f ms, mapping %.2f ms, "
        "encode %.2f ms (writer thread)"
        % (header.collect_ms, header.dump_ms, header.mapping_ms, header.encode_ms)
    )
    return "\n".join(lines)
