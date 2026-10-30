# SPDX-License-Identifier: Apache-2.0
"""StateSnapshotter writes L0 and maps prompts; the loader reads one
snapshot back."""

# Standard
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
import io
import threading

# Third Party
import pytest

pytest.importorskip("vllm", reason="the snapshotter reads a vLLM BlockPool")

# Third Party
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)

# First Party
from lmcache.integration.vllm import state_snapshot as state_snapshot_mod  # noqa: E402
from lmcache.integration.vllm.state_snapshot import StateSnapshotter  # noqa: E402
from lmcache.tools.state_snapshot.summary import (  # noqa: E402
    format_summary,
    summarize,
)
from lmcache.v1.multiprocess.protocols.state_snapshot import (  # noqa: E402
    RequestMappingRecord,
    append_record,
    iter_records,
    l0_snapshot_path,
    read_l0_snapshot,
    requests_path,
    split_block_key,
)
from lmcache.v1.multiprocess.token_codec import pack_token_ids  # noqa: E402
from lmcache.v1.multiprocess.token_hasher import TokenHasher  # noqa: E402

HASH_BLOCK = 4
CHUNK = 8
PROMPT = list(range(100, 116))
INTERVAL_S = 10.0
# Request.block_hashes of PROMPT: one per 4-token block.
BLOCK_HASHES = [BlockHash(bytes([i + 1]) * 32) for i in range(4)]


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Adapter:
    """The scheduler adapter's snapshot surface."""

    lmcache_tokens_per_chunk = CHUNK
    hash_algorithm = "blake3"
    model_name = "m"
    world_size = 1


def _pool_caching(num_cached: int) -> tuple[BlockPool, list[int]]:
    """A pool whose first ``num_cached`` blocks cache the prompt's first
    hashes; the last of them is freed, so it waits in the free queue."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    blocks = pool.get_new_blocks(num_cached)
    for block, block_hash in zip(blocks, BLOCK_HASHES, strict=False):
        block.set_block_hash(make_block_hash_with_group_id(block_hash, 0))
    pool.free_blocks(blocks[-1:])
    return pool, [block.block_id for block in blocks]


def _track(snapshotter: StateSnapshotter, request_id: str = "r1") -> None:
    request = SimpleNamespace(
        request_id=request_id,
        num_prompt_tokens=len(PROMPT),
        block_hashes=list(BLOCK_HASHES),
    )
    tracker = SimpleNamespace(
        cache_salt="",
        get_token_ids_slice=lambda start, end: list(PROMPT[start:end]),
    )
    snapshotter.on_request_tracked(cast(Any, request), cast(Any, tracker))


def _snapshotter(
    tmp_path: Path,
    adapter: _Adapter,
    pool: BlockPool,
    clock: _Clock,
) -> StateSnapshotter:
    return StateSnapshotter(
        str(tmp_path), INTERVAL_S, cast(Any, adapter), pool, [HASH_BLOCK], clock=clock
    )


def test_nothing_is_written_before_the_interval(tmp_path: Path) -> None:
    pool, _ = _pool_caching(2)
    clock = _Clock()
    snapshotter = _snapshotter(tmp_path, _Adapter(), pool, clock)

    clock.now = INTERVAL_S - 0.001
    snapshotter.on_scheduler_step()
    snapshotter.close()

    assert not list(tmp_path.glob("l0-*"))


def test_snapshots_follow_the_given_interval(tmp_path: Path) -> None:
    pool, _ = _pool_caching(1)
    clock = _Clock()
    snapshotter = StateSnapshotter(
        str(tmp_path), 2.0, cast(Any, _Adapter()), pool, [HASH_BLOCK], clock=clock
    )

    clock.now = 1.9
    snapshotter.on_scheduler_step()
    clock.now = 2.0
    snapshotter.on_scheduler_step()
    snapshotter.close()

    [path] = tmp_path.glob("l0-*")
    assert path.name == Path(l0_snapshot_path("", snapshotter.run_id, 0)).name


def test_l0_file_holds_cached_blocks_free_queue_and_costs(tmp_path: Path) -> None:
    pool, block_ids = _pool_caching(3)
    clock = _Clock()
    snapshotter = _snapshotter(tmp_path, _Adapter(), pool, clock)
    expected_free = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]

    clock.now = INTERVAL_S
    snapshotter.on_scheduler_step()
    snapshotter.on_scheduler_step()
    snapshotter.close()

    header, body = read_l0_snapshot(
        l0_snapshot_path(str(tmp_path), snapshotter.run_id, 0)
    )
    assert (header.model_name, header.kv_world_size, header.chunk_size) == (
        "m",
        1,
        CHUNK,
    )
    assert (header.hash_block_size, header.num_gpu_blocks) == (HASH_BLOCK, 16)
    assert header.group_tokens_per_block == [HASH_BLOCK]
    assert header.dump_ms >= header.collect_ms >= 0
    assert header.encode_ms >= 0
    rows = list(
        zip(
            body.block_ids,
            map(split_block_key, body.block_keys),
            body.ref_cnts,
            body.hash_num_tokens,
            strict=True,
        )
    )
    assert rows == [
        # set_block_hash without a token count: vLLM records none.
        (block_ids[0], (BLOCK_HASHES[0], 0), 1, None),
        (block_ids[1], (BLOCK_HASHES[1], 0), 1, None),
        (block_ids[2], (BLOCK_HASHES[2], 0), 0, None),
    ]
    assert body.num_primary == 3
    assert body.free_queue == expected_free


def test_each_request_is_mapped_once_to_the_servers_chunk_hashes(
    tmp_path: Path,
) -> None:
    pool, _ = _pool_caching(1)
    snapshotter = _snapshotter(tmp_path, _Adapter(), pool, _Clock())

    _track(snapshotter)
    _track(snapshotter)
    snapshotter.on_request_finished("r1")
    _track(snapshotter)
    snapshotter.close()

    records = list(
        iter_records(
            requests_path(str(tmp_path), snapshotter.run_id), RequestMappingRecord
        )
    )
    assert len(records) == 2
    record = records[0]
    assert record.request_id == "r1"
    assert record.num_prompt_tokens == len(PROMPT)
    assert record.block_hashes == BLOCK_HASHES
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    assert record.chunk_hashes == hasher.compute_chunk_hashes(PROMPT)
    assert record.chunk_hashes == hasher.compute_packed_chunk_hashes(
        pack_token_ids(PROMPT)
    )
    assert len(record.chunk_hashes) == len(PROMPT) // CHUNK


def test_snapshot_due_during_a_running_write_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool, _ = _pool_caching(1)
    clock = _Clock()
    release = threading.Event()
    write = state_snapshot_mod.write_atomically

    def slow_write(path: str, data: bytes) -> None:
        release.wait(timeout=30)
        write(path, data)

    monkeypatch.setattr(state_snapshot_mod, "write_atomically", slow_write)
    snapshotter = _snapshotter(tmp_path, _Adapter(), pool, clock)

    clock.now = INTERVAL_S
    snapshotter.on_scheduler_step()
    clock.now = 2 * INTERVAL_S
    snapshotter.on_scheduler_step()
    release.set()
    snapshotter.close()

    assert [p.name for p in tmp_path.glob("l0-*")] == [
        Path(l0_snapshot_path(str(tmp_path), snapshotter.run_id, 0)).name
    ]


def test_loader_summarizes_a_snapshot(tmp_path: Path) -> None:
    pool, _ = _pool_caching(3)
    clock = _Clock()
    snapshotter = _snapshotter(tmp_path, _Adapter(), pool, clock)

    _track(snapshotter, "r1")
    _track(snapshotter, "r2")
    clock.now = INTERVAL_S
    snapshotter.on_scheduler_step()
    snapshotter.close()

    summary = summarize(str(tmp_path), 0)
    assert summary.run_id == snapshotter.run_id
    assert summary.l0_blocks_per_group == {0: 3}
    assert summary.num_free_blocks == len(pool.free_block_queue.get_all_free_blocks())
    assert summary.num_mapped_requests == 2
    text = format_summary(summary)
    assert "group 0: 3 cached block hashes" in text
    assert "requests mapped: 2" in text
    assert "dump" in text


def test_loader_needs_a_run_id_when_runs_share_a_directory(tmp_path: Path) -> None:
    for run_id in ("a", "b"):
        (tmp_path / ("l0-%s-000000.msgpack" % run_id)).write_bytes(b"")
    with pytest.raises(ValueError, match="pass run_id"):
        summarize(str(tmp_path), 0)


def test_block_pool_without_extra_hashes_is_rejected_at_start(
    tmp_path: Path,
) -> None:
    # A vLLM BlockPool from before cached_block_hashes_by_block existed.
    old_pool = SimpleNamespace(blocks=[], num_gpu_blocks=1, hash_block_size=4)
    with pytest.raises(ValueError, match="cached_block_hashes_by_block"):
        StateSnapshotter(
            str(tmp_path),
            INTERVAL_S,
            cast(Any, _Adapter()),
            cast(Any, old_pool),
            [HASH_BLOCK],
        )


class _FullDisk(io.BytesIO):
    """A file that refuses any write past ``capacity`` bytes, writing none
    of it."""

    def __init__(self, capacity: int) -> None:
        super().__init__()
        self.capacity = capacity

    def write(self, data: Any) -> int:
        if self.tell() + len(data) > self.capacity:
            raise OSError(28, "No space left on device")
        return super().write(data)


def test_a_failed_append_leaves_the_record_file_readable(tmp_path: Path) -> None:
    def record(request_id: str) -> RequestMappingRecord:
        return RequestMappingRecord(
            request_id=request_id,
            cache_salt="",
            num_prompt_tokens=0,
            block_hashes=[],
            chunk_hashes=[b"x" * 32],
            wall_time=0.0,
            monotonic_time=0.0,
        )

    f = _FullDisk(capacity=1 << 20)
    append_record(f, record("a"))
    f.capacity = f.tell() + 8
    with pytest.raises(OSError):
        append_record(f, record("b"))
    f.capacity = 1 << 20
    append_record(f, record("c"))

    path = tmp_path / "requests.bin"
    path.write_bytes(f.getvalue())
    assert [r.request_id for r in iter_records(str(path), RequestMappingRecord)] == [
        "a",
        "c",
    ]
