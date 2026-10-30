# SPDX-License-Identifier: Apache-2.0
"""Write periodic snapshots of the GPU prefix cache (L0).

Enabled by ``lmcache.mp.snapshot_dir``. File formats:
``lmcache.v1.multiprocess.protocols.state_snapshot``. Design:
``docs/design/integration/vllm/state_snapshot.md``.
"""

# Standard
from concurrent.futures import Future, ThreadPoolExecutor
from itertools import compress, repeat
from operator import attrgetter, is_not
from typing import TYPE_CHECKING, Callable
import os
import socket
import time

# Third Party
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.protocols.state_snapshot import (
    FORMAT_VERSION,
    L0Body,
    L0SnapshotFile,
    RequestMappingRecord,
    append_record,
    l0_snapshot_path,
    log_failed_write,
    requests_path,
    write_atomically,
)
from lmcache.v1.multiprocess.token_codec import pack_token_ids
from lmcache.v1.multiprocess.token_hasher import TokenHasher

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.request import Request

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

logger = init_logger(__name__)

_BLOCK_HASH = attrgetter("block_hash")
_BLOCK_ID = attrgetter("block_id")
_REF_CNT = attrgetter("ref_cnt")
_HASH_NUM_TOKENS = attrgetter("block_hash_num_tokens")


def _collect_cached_blocks(pool: "BlockPool") -> L0Body:
    """Copy every prefix-cache entry and the free queue of ``pool``.

    One C-level ``map`` pass per field keeps a 100k-block pool cheap on the
    scheduler thread.

    Args:
        pool: vLLM's block pool; only read.

    Returns:
        The body.
    """
    blocks = pool.blocks
    keys = list(map(_BLOCK_HASH, blocks))
    hashed = list(map(is_not, keys, repeat(None)))
    cached = list(compress(blocks, hashed))
    block_ids = list(map(_BLOCK_ID, cached))
    block_keys = list(compress(keys, hashed))
    ref_cnts = list(map(_REF_CNT, cached))
    hash_num_tokens: list[int | None] = list(map(_HASH_NUM_TOKENS, cached))
    num_primary = len(block_ids)
    for block_id, extra_keys in pool.cached_block_hashes_by_block.items():
        for key in extra_keys:
            block_ids.append(block_id)
            block_keys.append(key)
            ref_cnts.append(blocks[block_id].ref_cnt)
            hash_num_tokens.append(None)
    return L0Body(
        block_ids=block_ids,
        block_keys=block_keys,
        ref_cnts=ref_cnts,
        hash_num_tokens=hash_num_tokens,
        num_primary=num_primary,
        free_queue=list(map(_BLOCK_ID, pool.free_block_queue.get_all_free_blocks())),
    )


class StateSnapshotter:
    """Write an L0 file every interval.

    The scheduler thread only copies vLLM's state; a writer thread encodes
    and writes. Call every method on the scheduler thread.

    Args:
        snapshot_dir: The directory every file goes to.
        interval_s: Seconds between two snapshots.
        adapter: The scheduler adapter; read for the model and chunk keying.
        block_pool: vLLM's scheduler block pool; only read.
        group_tokens_per_block: Tokens per block of each KV cache group, 0
            for a group LMCache does not cache.
        clock: Monotonic clock, in seconds, that paces the snapshots.

    Raises:
        ValueError: If ``block_pool`` lacks ``cached_block_hashes_by_block``
            (a vLLM older than this snapshotter reads).
    """

    def __init__(
        self,
        snapshot_dir: str,
        interval_s: float,
        adapter: "LMCacheMPSchedulerAdapter",
        block_pool: "BlockPool",
        group_tokens_per_block: list[int],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if not hasattr(block_pool, "cached_block_hashes_by_block"):
            raise ValueError(
                "lmcache.mp.snapshot_dir needs a vLLM whose BlockPool has "
                "cached_block_hashes_by_block; this vLLM's does not"
            )
        os.makedirs(snapshot_dir, exist_ok=True)
        self._snapshot_dir = snapshot_dir
        self._adapter = adapter
        self._pool = block_pool
        self._group_tokens_per_block = list(group_tokens_per_block)
        self._interval_s = interval_s
        self._clock = clock
        self._run_id = "%s-%d-%d" % (socket.gethostname(), os.getpid(), time.time())
        self._hasher = TokenHasher(
            chunk_size=adapter.lmcache_tokens_per_chunk,
            hash_algorithm=adapter.hash_algorithm,
        )
        # One thread, so the request file and every L0 file are written in
        # the order the scheduler thread queued them.
        self._writer = ThreadPoolExecutor(
            max_workers=1, thread_name_prefix="lmcache-state-snapshot"
        )
        self._requests_file = open(requests_path(snapshot_dir, self._run_id), "ab")
        # Requests already mapped, until they finish: a preempted request's
        # new tracker maps nothing again.
        self._mapped: set[str] = set()
        self._mapping_ms = 0.0
        self._next_snapshot_id = 0
        self._next_snapshot_at = clock() + interval_s
        # The last queued L0 write; a snapshot due before it finishes is
        # skipped rather than queued.
        self._last_write: Future[None] | None = None
        logger.info(
            "State snapshot: writing run %s to %s every %g s",
            self._run_id,
            snapshot_dir,
            interval_s,
        )

    @property
    def run_id(self) -> str:
        """The id every file of this scheduler's run carries."""
        return self._run_id

    def on_request_tracked(
        self, request: "Request", tracker: "LMCacheMPRequestTracker"
    ) -> None:
        """Map a newly tracked request's prompt, once per request.

        The scheduler thread copies the prompt and its block hashes; the
        writer thread hashes the chunks and appends the record.

        Args:
            request: The vLLM request; its ``block_hashes`` are copied.
            tracker: Its tracker, whose token ids key the LMCache chunks.
        """
        if request.request_id in self._mapped:
            return
        started_at = time.perf_counter()
        self._mapped.add(request.request_id)
        num_prompt_tokens = request.num_prompt_tokens
        record = RequestMappingRecord(
            request_id=request.request_id,
            cache_salt=tracker.cache_salt,
            num_prompt_tokens=num_prompt_tokens,
            block_hashes=list(request.block_hashes),
            chunk_hashes=[],
            wall_time=time.time(),
            monotonic_time=time.monotonic(),
        )
        tokens = tracker.get_token_ids_slice(0, num_prompt_tokens)
        self._writer.submit(self._append_mapping, record, tokens).add_done_callback(
            log_failed_write
        )
        self._mapping_ms += (time.perf_counter() - started_at) * 1e3

    def on_request_finished(self, request_id: str) -> None:
        """Forget a finished request, so a later one reusing its id maps.

        Args:
            request_id: The finished request.
        """
        self._mapped.discard(request_id)

    def on_scheduler_step(self) -> None:
        """Take a snapshot once the interval elapsed.

        Call once per step, at its end.
        """
        now = self._clock()
        if now < self._next_snapshot_at:
            return
        self._next_snapshot_at = now + self._interval_s
        if self._last_write is not None and not self._last_write.done():
            logger.debug(
                "State snapshot %d skipped: the previous L0 file is still "
                "being written",
                self._next_snapshot_id,
            )
            return
        self._take_snapshot()

    def close(self) -> None:
        """Finish every queued write and close the request file."""
        self._writer.shutdown(wait=True)
        self._requests_file.close()

    def _take_snapshot(self) -> None:
        """Copy L0, queue the L0 file."""
        started_at = time.perf_counter()
        wall_time = time.time()
        monotonic_time = time.monotonic()
        snapshot_id = self._next_snapshot_id
        self._next_snapshot_id += 1

        body = _collect_cached_blocks(self._pool)
        collected_at = time.perf_counter()

        header = L0SnapshotFile(
            format_version=FORMAT_VERSION,
            run_id=self._run_id,
            snapshot_id=snapshot_id,
            wall_time=wall_time,
            monotonic_time=monotonic_time,
            model_name=self._adapter.model_name,
            kv_world_size=self._adapter.world_size,
            chunk_size=self._hasher.chunk_size,
            hash_block_size=self._pool.hash_block_size,
            hash_algorithm=self._hasher.hash_algorithm_name,
            num_gpu_blocks=self._pool.num_gpu_blocks,
            group_tokens_per_block=self._group_tokens_per_block,
            collect_ms=(collected_at - started_at) * 1e3,
            dump_ms=(time.perf_counter() - started_at) * 1e3,
            mapping_ms=self._mapping_ms,
            encode_ms=0.0,
            body=msgspec.Raw(b""),
        )
        self._mapping_ms = 0.0
        self._last_write = self._writer.submit(self._write_l0, header, body)
        self._last_write.add_done_callback(log_failed_write)

    def _append_mapping(self, record: RequestMappingRecord, tokens: list[int]) -> None:
        """Hash a prompt's chunks and append its record; writer thread only.

        Args:
            record: The record, without chunk hashes yet.
            tokens: The prompt's token ids, as the tracker keys them.

        Raises:
            OSError: If the record cannot be appended.
        """
        record.chunk_hashes = self._hasher.compute_packed_chunk_hashes(
            pack_token_ids(tokens)
        )
        append_record(self._requests_file, record)

    def _write_l0(self, header: L0SnapshotFile, body: L0Body) -> None:
        """Encode and write one L0 file; writer thread only.

        Args:
            header: The file's header, without its body and encode time.
            body: The copied prefix cache and free queue.

        Raises:
            OSError: If the file cannot be written.
        """
        started_at = time.perf_counter()
        header.body = msgspec.Raw(msgspec.msgpack.encode(body))
        header.encode_ms = (time.perf_counter() - started_at) * 1e3
        write_atomically(
            l0_snapshot_path(self._snapshot_dir, header.run_id, header.snapshot_id),
            msgspec.msgpack.encode(header),
        )
