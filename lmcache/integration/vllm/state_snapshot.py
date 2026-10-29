# SPDX-License-Identifier: Apache-2.0
"""Write periodic snapshots of the GPU prefix cache (L0) and of LMCache's L1.

Enabled by ``lmcache.mp.snapshot_dir``; every server must load the
``lmcache.v1.multiprocess.l1_snapshot`` server module to write its L1 file.
File formats: ``lmcache.v1.multiprocess.protocols.state_snapshot``. Design:
``docs/design/integration/vllm/state_snapshot.md``.
"""

# Standard
from concurrent.futures import ThreadPoolExecutor
from typing import TYPE_CHECKING, Callable
import os
import socket
import time

# Third Party
from vllm.v1.core.kv_cache_utils import get_block_hash, get_group_id
import msgspec

# First Party
from lmcache.integration.vllm.lazy_offload_manager import LazyBufferSnapshot
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocols.server_module import ServerModuleCallResponse
from lmcache.v1.multiprocess.protocols.state_snapshot import (
    FORMAT_VERSION,
    L1_SNAPSHOT_METHOD,
    SNAPSHOT_INTERVAL_S,
    BufferedOpEntry,
    CachedBlockEntry,
    InFlightBatchEntry,
    L0Body,
    L0SnapshotFile,
    L1SnapshotReply,
    L1SnapshotRequest,
    LazyBufferSection,
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
    from lmcache.integration.vllm.lazy_offload_manager import LazyOffloadManager
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

logger = init_logger(__name__)

# (block id, BlockHashWithGroupId, ref_cnt, hash token count or None,
# primary): one prefix-cache entry as the scheduler thread copies it.
_RawEntry = tuple[int, bytes, int, "int | None", bool]


def _cached_block_entries(raw: list[_RawEntry]) -> list[CachedBlockEntry]:
    """Split each copied entry's hash into vLLM's block hash and group id.

    Args:
        raw: The entries the scheduler thread copied.

    Returns:
        One file entry per copied entry.
    """
    return [
        CachedBlockEntry(
            block_id=block_id,
            group_id=get_group_id(key),
            block_hash=get_block_hash(key),
            ref_cnt=ref_cnt,
            hash_num_tokens=num_tokens,
            primary=primary,
        )
        for block_id, key, ref_cnt, num_tokens, primary in raw
    ]


def _lazy_section(snapshot: LazyBufferSnapshot) -> LazyBufferSection:
    """Turn the lazy-offload buffer copy into its file section.

    Args:
        snapshot: The copy the scheduler thread took.

    Returns:
        The section.
    """
    return LazyBufferSection(
        ops=[
            BufferedOpEntry(
                request_id=op.request_id,
                start=op.start,
                end=op.end,
                block_ids=[list(group) for group in op.block_ids],
                held=op.held,
                expedited=op.expedited,
            )
            for op in snapshot.ops
        ],
        in_flight=[
            InFlightBatchEntry(
                request_id=request_id,
                ranges=[
                    (meta.op.start, meta.op.end)
                    for meta, _ in batch.operations.metadatas
                ],
                num_pinned_blocks=len(batch.block_ids),
                failed=batch.failed,
                orphaned=batch.orphaned,
            )
            for request_id, batch in snapshot.in_flight.items()
        ],
        finished_request_ids=sorted(snapshot.finished_request_ids),
    )


class StateSnapshotter:
    """Write an L0 file and ask every server for an L1 file, every interval.

    The scheduler thread only copies vLLM's and lazy offload's state; a
    writer thread encodes and writes. A server's reply is read on a later
    step, never waited for. Call every method on the scheduler thread.

    Args:
        snapshot_dir: The directory every file goes to; the servers write
            their L1 files to the same path, so it must name the same
            directory on their hosts.
        adapter: The scheduler adapter that reaches the servers.
        block_pool: vLLM's scheduler block pool; only read.
        lazy_offload_manager: The connector's lazy-offload manager, or
            ``None`` when lazy offload is off.
        group_tokens_per_block: Tokens per block of each KV cache group, 0
            for a group LMCache does not cache.
        clock: Monotonic clock, in seconds, that paces the snapshots.
    """

    def __init__(
        self,
        snapshot_dir: str,
        adapter: "LMCacheMPSchedulerAdapter",
        block_pool: "BlockPool",
        lazy_offload_manager: "LazyOffloadManager | None",
        group_tokens_per_block: list[int],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        os.makedirs(snapshot_dir, exist_ok=True)
        self._snapshot_dir = snapshot_dir
        self._adapter = adapter
        self._pool = block_pool
        self._lazy = lazy_offload_manager
        self._group_tokens_per_block = list(group_tokens_per_block)
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
        self._next_snapshot_at = clock() + SNAPSHOT_INTERVAL_S
        # Response future per server URL of the last L1 request; empty once
        # every server answered.
        self._inflight: dict[str, MessagingFuture[ServerModuleCallResponse]] = {}
        logger.info(
            "State snapshot: writing run %s to %s every %.0f s",
            self._run_id,
            snapshot_dir,
            SNAPSHOT_INTERVAL_S,
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
        """Read server replies; take a snapshot once the interval elapsed.

        Call once per step, at its end. Never waits on a server: a server
        that has not answered when the next snapshot is due, or that fails
        the request, is logged as a warning and nothing else.
        """
        if self._inflight:
            self._collect_replies()
        if self._clock() < self._next_snapshot_at:
            return
        self._next_snapshot_at = self._clock() + SNAPSHOT_INTERVAL_S
        if self._inflight:
            logger.warning(
                "State snapshot: servers %s did not answer L1 snapshot request "
                "%d within %.0f s",
                sorted(self._inflight),
                self._next_snapshot_id - 1,
                SNAPSHOT_INTERVAL_S,
            )
            self._inflight = {}
        self._take_snapshot()

    def close(self) -> None:
        """Finish every queued write and close the request file."""
        self._writer.shutdown(wait=True)
        self._requests_file.close()

    def _take_snapshot(self) -> None:
        """Copy L0 and the lazy buffer, ask the servers, queue the L0 file."""
        started_at = time.perf_counter()
        wall_time = time.time()
        monotonic_time = time.monotonic()
        snapshot_id = self._next_snapshot_id
        self._next_snapshot_id += 1

        pool = self._pool
        raw: list[_RawEntry] = [
            (
                block.block_id,
                block.block_hash,
                block.ref_cnt,
                block.block_hash_num_tokens,
                True,
            )
            for block in pool.blocks
            if block.block_hash is not None
        ]
        raw.extend(
            (block_id, key, pool.blocks[block_id].ref_cnt, None, False)
            for block_id, keys in pool.cached_block_hashes_by_block.items()
            for key in keys
        )
        free_queue = [
            block.block_id for block in pool.free_block_queue.get_all_free_blocks()
        ]
        # Converted here, not on the writer: the in-flight batches are live
        # objects the next step mutates.
        lazy = (
            _lazy_section(self._lazy.buffer_snapshot())
            if self._lazy is not None
            else None
        )
        collected_at = time.perf_counter()

        l1_requested = self._adapter.is_healthy
        if l1_requested:
            self._inflight = self._adapter.server_module_call(
                L1_SNAPSHOT_METHOD,
                msgspec.msgpack.encode(
                    L1SnapshotRequest(
                        run_id=self._run_id,
                        snapshot_id=snapshot_id,
                        snapshot_dir=self._snapshot_dir,
                    )
                ),
            )
        else:
            logger.warning(
                "State snapshot %d: a server is unhealthy; no L1 file", snapshot_id
            )
        header = L0SnapshotFile(
            format_version=FORMAT_VERSION,
            run_id=self._run_id,
            snapshot_id=snapshot_id,
            wall_time=wall_time,
            monotonic_time=monotonic_time,
            model_name=self._adapter.model_name,
            kv_world_size=self._adapter.world_size,
            chunk_size=self._hasher.chunk_size,
            hash_block_size=pool.hash_block_size,
            hash_algorithm=self._hasher.hash_algorithm_name,
            num_gpu_blocks=pool.num_gpu_blocks,
            group_tokens_per_block=self._group_tokens_per_block,
            l1_requested=l1_requested,
            collect_ms=(collected_at - started_at) * 1e3,
            dump_ms=(time.perf_counter() - started_at) * 1e3,
            mapping_ms=self._mapping_ms,
            encode_ms=0.0,
            body=msgspec.Raw(b""),
        )
        self._mapping_ms = 0.0
        self._writer.submit(
            self._write_l0, header, raw, free_queue, lazy
        ).add_done_callback(log_failed_write)

    def _collect_replies(self) -> None:
        """Log the servers' answers once every one has answered."""
        if not all(future.query() for future in self._inflight.values()):
            return
        inflight, self._inflight = self._inflight, {}
        for url, future in inflight.items():
            try:
                response = future.result()
            except Exception:
                # The transport stores whatever failed the call on the future
                # (gRPC status, ZMQ send or decode error); a lost L1 file must
                # not fail the scheduler step.
                logger.exception("State snapshot: L1 request to %s failed", url)
                continue
            if not response.success:
                logger.warning(
                    "State snapshot: server %s failed the L1 request: %s; it "
                    "needs --server-module "
                    '\'{"module_path": "lmcache.v1.multiprocess.l1_snapshot"}\'',
                    url,
                    response.error,
                )
                continue
            reply = msgspec.msgpack.decode(response.payload, type=L1SnapshotReply)
            logger.debug(
                "State snapshot: server %s lists %d L1 objects in %.1f ms into %s",
                url,
                reply.num_objects,
                reply.lock_ms,
                reply.path,
            )

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

    def _write_l0(
        self,
        header: L0SnapshotFile,
        raw: list[_RawEntry],
        free_queue: list[int],
        lazy: LazyBufferSection | None,
    ) -> None:
        """Encode and write one L0 file; writer thread only.

        Args:
            header: The file's header, without its body and encode time.
            raw: The copied prefix-cache entries.
            free_queue: The copied free-queue order.
            lazy: The lazy-offload section, or ``None``.

        Raises:
            OSError: If the file cannot be written.
        """
        started_at = time.perf_counter()
        body = msgspec.msgpack.encode(
            L0Body(
                cached_blocks=_cached_block_entries(raw),
                free_queue=free_queue,
                lazy=lazy,
            )
        )
        header.body = msgspec.Raw(body)
        header.encode_ms = (time.perf_counter() - started_at) * 1e3
        write_atomically(
            l0_snapshot_path(self._snapshot_dir, header.run_id, header.snapshot_id),
            msgspec.msgpack.encode(header),
        )
