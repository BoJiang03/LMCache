# SPDX-License-Identifier: Apache-2.0
"""StateSnapshotter writes L0, asks for L1, and maps prompts; the loader
reads one snapshot back."""

# Standard
from collections.abc import Iterator
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast
from unittest.mock import MagicMock, patch
import os
import threading
import time

# Third Party
import msgspec
import pytest
import torch

pytest.importorskip("vllm", reason="the snapshotter reads a vLLM BlockPool")

# Third Party
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)

# First Party
from lmcache.integration.vllm import state_snapshot as state_snapshot_mod  # noqa: E402
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadManager,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.state_snapshot import StateSnapshotter  # noqa: E402
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)
from lmcache.tools.state_snapshot.summary import (  # noqa: E402
    format_summary,
    summarize,
)
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey  # noqa: E402
from lmcache.v1.distributed.config import (  # noqa: E402
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager  # noqa: E402
from lmcache.v1.multiprocess.ext_server_module import (  # noqa: E402
    build_server_module_router,
)
from lmcache.v1.multiprocess.l1_snapshot import L1SnapshotModule  # noqa: E402
from lmcache.v1.multiprocess.protocols.server_module import (  # noqa: E402
    ServerModuleCallRequest,
    ServerModuleCallResponse,
)
from lmcache.v1.multiprocess.protocols.state_snapshot import (  # noqa: E402
    L1_SNAPSHOT_METHOD,
    SNAPSHOT_INTERVAL_S,
    L1SnapshotRequest,
    RequestMappingRecord,
    iter_records,
    l0_snapshot_path,
    read_l0_snapshot,
    requests_path,
    split_block_key,
)
from lmcache.v1.multiprocess.token_codec import pack_token_ids  # noqa: E402
from lmcache.v1.multiprocess.token_hasher import TokenHasher  # noqa: E402
import lmcache.v1.memory_management as memory_management  # noqa: E402

pytestmark = pytest.mark.no_shared_allocator

HASH_BLOCK = 4
CHUNK = 8
PROMPT = list(range(100, 116))
# Request.block_hashes of PROMPT: one per 4-token block.
BLOCK_HASHES = [BlockHash(bytes([i + 1]) * 32) for i in range(4)]
OBJECT_BYTES = 4096
LAYOUT = MemoryLayoutDesc([torch.Size([1024])], [torch.float32])


class _Clock:
    def __init__(self) -> None:
        self.now = 0.0

    def __call__(self) -> float:
        return self.now


class _Done:
    """A transport future that already holds its response."""

    def __init__(self, response: ServerModuleCallResponse | Exception) -> None:
        self._response = response

    def query(self) -> bool:
        return True

    def result(self) -> ServerModuleCallResponse:
        if isinstance(self._response, Exception):
            raise self._response
        return self._response


class _Pending:
    """A transport future whose server never answers."""

    def query(self) -> bool:
        return False


class _Adapter:
    """The scheduler adapter's snapshot surface, answering through ``answer``."""

    lmcache_tokens_per_chunk = CHUNK
    hash_algorithm = "blake3"
    model_name = "m"
    world_size = 1

    def __init__(self, answer: Any = None, is_healthy: bool = True) -> None:
        self.answer = answer
        self.is_healthy = is_healthy
        self.calls: list[L1SnapshotRequest] = []

    def server_module_call(self, method: str, payload: bytes) -> dict[str, Any]:
        assert method == L1_SNAPSHOT_METHOD
        self.calls.append(msgspec.msgpack.decode(payload, type=L1SnapshotRequest))
        return {"tcp://server": self.answer(method, payload)}


class _NoReceipts:
    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        return False


class _EmptyPrefixCache:
    single_type_managers = (SimpleNamespace(block_size=HASH_BLOCK),)

    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list, ...], int, int]:
        return ([],), 0, 0


@pytest.fixture
def store(monkeypatch: pytest.MonkeyPatch) -> Iterator[StorageManager]:
    monkeypatch.setattr(
        memory_management,
        "_allocate_cpu_memory",
        lambda size, *args, **kwargs: torch.empty(size, dtype=torch.uint8),
    )
    monkeypatch.setattr(memory_management, "_free_cpu_memory", lambda *a, **k: None)
    l1_config = L1ManagerConfig(
        L1MemoryManagerConfig(
            size_in_bytes=32 * OBJECT_BYTES, use_lazy=False, shm_name=""
        )
    )
    manager = StorageManager(StorageManagerConfig(l1_config, EvictionConfig("LRU")))
    yield manager
    manager.close()


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
    lazy: LazyOffloadManager | None = None,
) -> StateSnapshotter:
    return StateSnapshotter(
        str(tmp_path), cast(Any, adapter), pool, lazy, [HASH_BLOCK], clock=clock
    )


def _router_answer(module: L1SnapshotModule) -> Any:
    router = build_server_module_router(MagicMock(), [module])
    assert router is not None
    return lambda method, payload: _Done(
        router.server_module_call(
            ServerModuleCallRequest(method=method, payload=payload)
        )
    )


def _wait_for(path: str) -> None:
    deadline = time.monotonic() + 30
    while not os.path.exists(path) and time.monotonic() < deadline:
        time.sleep(0.01)


def _key(chunk_hash: bytes) -> ObjectKey:
    return ObjectKey(chunk_hash=chunk_hash, model_name="m", kv_rank=0)


def test_nothing_is_written_before_the_interval(tmp_path: Path) -> None:
    pool, _ = _pool_caching(2)
    clock = _Clock()
    adapter = _Adapter(answer=lambda *a: _Pending())
    snapshotter = _snapshotter(tmp_path, adapter, pool, clock)

    clock.now = SNAPSHOT_INTERVAL_S - 0.001
    snapshotter.on_scheduler_step()
    snapshotter.close()

    assert adapter.calls == []
    assert not list(tmp_path.glob("l0-*"))


def test_l0_file_holds_cached_blocks_free_queue_and_costs(tmp_path: Path) -> None:
    pool, block_ids = _pool_caching(3)
    clock = _Clock()
    adapter = _Adapter(answer=lambda *a: _Pending())
    snapshotter = _snapshotter(tmp_path, adapter, pool, clock)
    expected_free = [b.block_id for b in pool.free_block_queue.get_all_free_blocks()]

    clock.now = SNAPSHOT_INTERVAL_S
    snapshotter.on_scheduler_step()
    snapshotter.on_scheduler_step()
    snapshotter.close()

    [request] = adapter.calls
    assert (request.run_id, request.snapshot_id) == (snapshotter.run_id, 0)
    assert request.snapshot_dir == str(tmp_path)
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
    assert header.l1_requested
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
    assert body.lazy is None


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


def test_l0_file_lists_the_lazy_offload_buffer(tmp_path: Path) -> None:
    pool, block_ids = _pool_caching(2)
    lazy = LazyOffloadManager(
        {},
        [HASH_BLOCK],
        _NoReceipts(),
        max_step_tokens=1024,
        num_lookahead_tokens=0,
        max_model_len=HASH_BLOCK,
        chunk_size=HASH_BLOCK,
    )
    lazy.bind_block_pool(pool)
    lazy.bind_cache_coordinator(cast(Any, _EmptyPrefixCache()))
    lazy.on_request_arrived("req", [])
    lazy.add_store_candidate(
        LMCacheMPRequestMetadata(
            request_id="req",
            direction="STORE",
            op=LoadStoreOp(
                token_bytes=pack_token_ids(range(2 * HASH_BLOCK)),
                block_ids=[block_ids],
                start=0,
                end=2 * HASH_BLOCK,
            ),
        )
    )
    clock = _Clock()
    snapshotter = _snapshotter(
        tmp_path, _Adapter(answer=lambda *a: _Pending()), pool, clock, lazy
    )

    clock.now = SNAPSHOT_INTERVAL_S
    snapshotter.on_scheduler_step()
    snapshotter.close()

    _, body = read_l0_snapshot(l0_snapshot_path(str(tmp_path), snapshotter.run_id, 0))
    assert body.lazy is not None
    [op] = body.lazy.ops
    assert (op.request_id, op.start, op.end, op.block_ids) == (
        "req",
        0,
        2 * HASH_BLOCK,
        [block_ids],
    )
    assert body.lazy.in_flight == []


def test_snapshot_due_during_a_running_write_is_skipped(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    pool, _ = _pool_caching(1)
    clock = _Clock()
    adapter = _Adapter(answer=lambda *a: _Pending())
    release = threading.Event()
    write = state_snapshot_mod.write_atomically

    def slow_write(path: str, data: bytes) -> None:
        release.wait(timeout=30)
        write(path, data)

    monkeypatch.setattr(state_snapshot_mod, "write_atomically", slow_write)
    snapshotter = _snapshotter(tmp_path, adapter, pool, clock)

    clock.now = SNAPSHOT_INTERVAL_S
    snapshotter.on_scheduler_step()
    clock.now = 2 * SNAPSHOT_INTERVAL_S
    snapshotter.on_scheduler_step()
    release.set()
    snapshotter.close()

    assert [request.snapshot_id for request in adapter.calls] == [0]
    assert [p.name for p in tmp_path.glob("l0-*")] == [
        Path(l0_snapshot_path(str(tmp_path), snapshotter.run_id, 0)).name
    ]


def test_server_failures_are_logged_and_never_raised(tmp_path: Path) -> None:
    pool, _ = _pool_caching(1)
    clock = _Clock()
    answers = iter(
        [
            _Done(
                ServerModuleCallResponse(success=False, payload=b"", error="no method")
            ),
            _Done(RuntimeError("transport down")),
            _Pending(),
            _Pending(),
        ]
    )
    adapter = _Adapter(answer=lambda *a: next(answers))
    snapshotter = _snapshotter(tmp_path, adapter, pool, clock)

    with (
        patch.object(state_snapshot_mod.logger, "warning") as warning,
        patch.object(state_snapshot_mod.logger, "exception") as exception,
    ):
        for step in range(1, 5):
            clock.now = step * SNAPSHOT_INTERVAL_S
            snapshotter.on_scheduler_step()
            snapshotter.on_scheduler_step()
            # The next snapshot is skipped while this file is being written.
            _wait_for(l0_snapshot_path(str(tmp_path), snapshotter.run_id, step - 1))
    snapshotter.close()

    assert len(adapter.calls) == 4
    messages = [call.args[0] for call in warning.call_args_list]
    assert any("failed the L1 request" in m for m in messages)
    assert any("did not answer" in m for m in messages)
    exception.assert_called_once()
    assert len(list(tmp_path.glob("l0-*.msgpack"))) == 4


def test_unhealthy_server_is_not_asked(tmp_path: Path) -> None:
    pool, _ = _pool_caching(1)
    clock = _Clock()
    adapter = _Adapter(is_healthy=False)
    snapshotter = _snapshotter(tmp_path, adapter, pool, clock)

    clock.now = SNAPSHOT_INTERVAL_S
    snapshotter.on_scheduler_step()
    snapshotter.close()

    assert adapter.calls == []
    header, _ = read_l0_snapshot(l0_snapshot_path(str(tmp_path), snapshotter.run_id, 0))
    assert not header.l1_requested


def test_loader_summarizes_a_snapshot_pair(
    tmp_path: Path, store: StorageManager
) -> None:
    # L0 caches the prompt's first three block hashes: all of chunk 0,
    # half of chunk 1.
    pool, _ = _pool_caching(3)
    hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
    chunk_0, chunk_1 = hasher.compute_chunk_hashes(PROMPT)
    store.reserve_write([_key(chunk_0), _key(chunk_1), _key(b"unmapped")], LAYOUT)
    store.finish_write([_key(chunk_0), _key(chunk_1), _key(b"unmapped")])
    store.reserve_write([_key(b"staged")], LAYOUT)
    module = L1SnapshotModule(cast(Any, SimpleNamespace(storage_manager=store)))
    clock = _Clock()
    snapshotter = _snapshotter(
        tmp_path, _Adapter(answer=_router_answer(module)), pool, clock
    )

    _track(snapshotter)
    clock.now = SNAPSHOT_INTERVAL_S
    snapshotter.on_scheduler_step()
    snapshotter.on_scheduler_step()
    snapshotter.close()
    module.close()

    summary = summarize(str(tmp_path), 0)
    assert summary.run_id == snapshotter.run_id
    assert summary.l0_blocks_per_group == {0: 3}
    assert summary.num_buffered_ops == 0
    assert len(summary.l1_servers) == 1
    assert summary.l1_objects == {(0, "committed"): 3, (0, "staging"): 1}
    assert summary.l1_bytes == {
        (0, "committed"): 3 * OBJECT_BYTES,
        (0, "staging"): OBJECT_BYTES,
    }
    assert summary.num_l1_chunks == 4
    assert summary.num_l1_chunks_mapped == 2
    assert summary.num_l1_chunks_any_in_l0 == 2
    assert summary.num_l1_chunks_all_in_l0 == 1
    text = format_summary(summary)
    assert "group 0: 3 cached block hashes" in text
    assert "4 distinct, 2 mapped, 2 with any block in L0, 1 with every" in text
    assert "dump" in text and "lock" in text


def test_loader_needs_a_run_id_when_runs_share_a_directory(tmp_path: Path) -> None:
    for run_id in ("a", "b"):
        (tmp_path / ("l0-%s-000000.msgpack" % run_id)).write_bytes(b"")
    with pytest.raises(ValueError, match="pass run_id"):
        summarize(str(tmp_path), 0)
