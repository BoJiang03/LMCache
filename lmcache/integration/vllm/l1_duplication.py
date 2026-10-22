# SPDX-License-Identifier: Apache-2.0
"""Measure the share of L1 that vLLM's GPU prefix cache (L0) also holds.

See ``docs/design/integration/vllm/l1_duplication.md``.
"""

# Standard
from dataclasses import dataclass
from typing import TYPE_CHECKING, Any, Callable
import time

# Third Party
import grpc
import msgspec

# First Party
from lmcache.logging import init_logger
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocols.l1_residency import (
    L1_RESIDENCY_QUERY_METHOD,
    L1ResidencyQuery,
    L1ResidencyReply,
    compute_probe_hash,
)
from lmcache.v1.multiprocess.protocols.server_module import (
    ServerModuleCallResponse,
    unknown_method_error,
)
from lmcache.v1.multiprocess.token_hasher import TokenHasher

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.request import Request

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import LMCacheMPRequestTracker
    from lmcache.integration.vllm.vllm_multi_process_adapter import (
        LMCacheMPSchedulerAdapter,
    )

logger = init_logger(__name__)

# Also the deadline of a query: one unanswered when the next is due is
# dropped, so a lost reply cannot stop measuring.
SAMPLE_INTERVAL_S = 10.0
# Consecutive unanswered queries after which measuring stops. A ZMQ server
# with no server module never answers, and every query would leave a
# pending future behind.
MAX_MISSED_DEADLINES = 3

_SERVER_MODULE_HINT = (
    "start it with --server-module "
    '\'{"module_path": "lmcache.v1.multiprocess.l1_residency"}\''
)

# (cache salt, LMCache chunk hash): the part of an ObjectKey the scheduler
# varies; model, kv ranks and object group come from the query.
_ChunkKey = tuple[str, bytes]


@dataclass(frozen=True)
class DuplicationSample:
    """L0 against committed L1 at one sample, summed over every server.

    Committed as in :class:`~lmcache.v1.distributed.internal_api.L1CommittedUsage`:
    the write finished and the object is not temporary. Both counts take
    one object per kv rank, and once per L1 manager that holds the key.

    Attributes:
        matched_objects: Committed L1 objects of the chunks L0 holds.
        matched_bytes: Their bytes.
        model_objects: Committed L1 objects of this model.
        model_bytes: Their bytes.
    """

    matched_objects: int
    matched_bytes: int
    model_objects: int
    model_bytes: int

    @property
    def byte_ratio(self) -> float:
        """The share of the model's committed L1 bytes L0 also holds.

        Returns:
            ``matched_bytes / model_bytes``, or 0.0 for an empty L1.
        """
        return self.matched_bytes / self.model_bytes if self.model_bytes else 0.0

    @property
    def object_ratio(self) -> float:
        """The share of the model's committed L1 objects L0 also holds.

        Returns:
            ``matched_objects / model_objects``, or 0.0 for an empty L1.
        """
        return self.matched_objects / self.model_objects if self.model_objects else 0.0


@dataclass
class _Prefix:
    """One token prefix whose leading chunks L0 may hold.

    Attributes:
        cache_salt: The request's cache salt.
        block_hashes: vLLM block hashes of the prefix, one per hash block:
            the request's ``Request.block_hashes``, live while it runs and
            kept whole once it finished. A cut list could hit less than the
            prompt does: EAGLE drops the last hit block, and Mamba hits only
            up to a cached state.
        chunk_hashes: LMCache chunk hashes of chunks ``0..n-1``, the ones
            hashed so far.
        tracker: The request's tracker, whose tokens are hashed, while it
            runs; ``None`` once it finished.
        last_raw_hash: Raw hash of chunk ``n-1``, the chain input of chunk
            ``n``; ``None`` while no chunk is hashed and once finished.
        finish_order: How many requests finished before this one; 0 while
            it runs.
    """

    cache_salt: str
    block_hashes: list[Any]
    chunk_hashes: list[bytes]
    tracker: "LMCacheMPRequestTracker | None"
    last_raw_hash: Any = None
    finish_order: int = 0


class L1DuplicationMeter:
    """Sample the L0/L1 duplication every :data:`SAMPLE_INTERVAL_S`.

    L0 holds chunk ``i`` of a prefix iff vLLM would hit at least
    ``(i + 1) * chunk_size`` tokens of it from the GPU now, by vLLM's own
    side-effect-free ``find_longest_cache_hit``. Call every method on the
    scheduler thread; they share the prefix state unlocked.

    Args:
        adapter: The scheduler adapter that reaches the servers.
        coordinator: The scheduler's ``KVCacheManager.coordinator``.
        hash_block_size: Tokens per vLLM block hash.
        chunk_size: LMCache tokens per chunk.
        hash_algorithm: The token hash algorithm the servers use.
        on_sample: Receives every completed sample.
        clock: Monotonic clock, in seconds.
    """

    def __init__(
        self,
        adapter: "LMCacheMPSchedulerAdapter",
        coordinator: "KVCacheCoordinator",
        hash_block_size: int,
        chunk_size: int,
        hash_algorithm: str,
        on_sample: Callable[[DuplicationSample], None],
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self._adapter = adapter
        self._coordinator = coordinator
        self._hash_block_size = hash_block_size
        self._chunk_size = chunk_size
        self._hasher = TokenHasher(chunk_size=chunk_size, hash_algorithm=hash_algorithm)
        self._probe_hash = compute_probe_hash(self._hasher)
        self._on_sample = on_sample
        self._clock = clock
        # Prefix of every running request, by request id.
        self._running: dict[str, _Prefix] = {}
        # Finished prefixes with at least one chunk in L0 at the last sample,
        # one per last chunk, which bounds them by what L0 holds. Of two that
        # share a last chunk the later-finished one is kept: vLLM evicts freed
        # blocks oldest first, so its blocks past that chunk (which EAGLE and
        # Mamba hits depend on) outlive the other's.
        self._finished: dict[_ChunkKey, _Prefix] = {}
        self._num_finished = 0
        # When the next query is due; also the inflight query's deadline.
        self._next_sample_at = clock() + SAMPLE_INTERVAL_S
        # Response future per server URL of the unanswered query; empty when
        # none is inflight.
        self._inflight: dict[str, MessagingFuture[ServerModuleCallResponse]] = {}
        # Queries dropped unanswered in a row.
        self._missed_deadlines = 0
        self._disabled = False

    def on_request_tracked(
        self, request: "Request", tracker: "LMCacheMPRequestTracker"
    ) -> None:
        """Follow the prefix of a request the connector started tracking.

        A preempted request gets a new tracker; its new prefix replaces the
        old one.

        Args:
            request: The vLLM request, whose live block hashes are read.
            tracker: Its tracker, whose token ids key the LMCache chunks.
        """
        if self._disabled:
            return
        self._running[request.request_id] = _Prefix(
            cache_salt=tracker.cache_salt,
            block_hashes=request.block_hashes,
            chunk_hashes=[],
            tracker=tracker,
        )

    def on_request_finished(self, request_id: str) -> None:
        """Keep only the chunks of a finished request that L0 holds now.

        Its blocks stay in L0 until vLLM reallocates them; its tokens are
        not kept.

        Args:
            request_id: The finished request; one never tracked is ignored.
        """
        prefix = self._running.pop(request_id, None)
        if prefix is None:
            return
        num_held = self._num_held_chunks(prefix)
        prefix.tracker = None
        prefix.last_raw_hash = None
        self._num_finished += 1
        prefix.finish_order = self._num_finished
        if num_held:
            self._keep_finished(prefix, num_held)

    def on_scheduler_step(self) -> None:
        """Collect answers and, every interval, sample L0 and query servers.

        Call once per step. Never waits on a server: an answer is read on a
        later step. Server failures are logged, never raised: an unanswered
        or failed query drops one sample; :data:`MAX_MISSED_DEADLINES`
        unanswered in a row, an unserved method, a hash mismatch or an
        undecodable reply stops the meter.
        """
        if self._disabled:
            return
        if self._inflight:
            self._collect_answers()
        if self._disabled or self._inflight or self._clock() < self._next_sample_at:
            return
        self._next_sample_at = self._clock() + SAMPLE_INTERVAL_S
        # Pruning runs even while no server is healthy, so state stays bounded.
        held = self._held_chunk_keys()
        if not self._adapter.is_healthy:
            return
        chunk_hashes_by_salt: dict[str, list[bytes]] = {}
        for cache_salt, chunk_hash in held:
            chunk_hashes_by_salt.setdefault(cache_salt, []).append(chunk_hash)
        payload = msgspec.msgpack.encode(
            L1ResidencyQuery(
                model_name=self._adapter.model_name,
                world_size=self._adapter.world_size,
                chunk_hashes_by_salt=chunk_hashes_by_salt,
            )
        )
        self._inflight = self._adapter.server_module_call(
            L1_RESIDENCY_QUERY_METHOD, payload
        )

    def _held_chunk_keys(self) -> set[_ChunkKey]:
        """Return the chunks L0 holds, cutting finished prefixes to them.

        A finished prefix whose last chunk is already held, through a longer
        prefix that extends it, adds nothing and is not looked up.

        Returns:
            Every chunk L0 holds, with every chunk before it in its prefix.
        """
        held: set[_ChunkKey] = set()
        for prefix in self._running.values():
            self._add_chunks(held, prefix, self._num_held_chunks(prefix))
        finished = sorted(
            self._finished.values(), key=lambda p: len(p.chunk_hashes), reverse=True
        )
        self._finished = {}
        for prefix in finished:
            num_held = len(prefix.chunk_hashes)
            if (prefix.cache_salt, prefix.chunk_hashes[-1]) not in held:
                num_held = self._num_held_chunks(prefix)
                if not num_held:
                    continue
                self._add_chunks(held, prefix, num_held)
            self._keep_finished(prefix, num_held)
        return held

    def _num_held_chunks(self, prefix: _Prefix) -> int:
        """Return how many leading chunks of ``prefix`` L0 holds.

        Hashes a running prefix's chunks up to that count.

        Args:
            prefix: A running prefix, or a finished one that
                :meth:`_keep_finished` cut.

        Returns:
            ``hit_tokens // chunk_size`` for vLLM's current hit on the prefix,
            at most its hashed chunks once it finished: chunks past them
            belong to a longer prefix, which counts them itself.
        """
        _, hit_tokens, _ = self._coordinator.find_longest_cache_hit(
            prefix.block_hashes, len(prefix.block_hashes) * self._hash_block_size
        )
        num_held = hit_tokens // self._chunk_size
        if prefix.tracker is None:
            return min(num_held, len(prefix.chunk_hashes))
        if num_held > len(prefix.chunk_hashes):
            token_ids = prefix.tracker.get_token_ids()
            for index in range(len(prefix.chunk_hashes), num_held):
                start = index * self._chunk_size
                prefix.last_raw_hash = self._hasher.hash_tokens(
                    token_ids[start : start + self._chunk_size], prefix.last_raw_hash
                )
                prefix.chunk_hashes.append(
                    self._hasher.hash_to_bytes(prefix.last_raw_hash)
                )
        return num_held

    def _keep_finished(self, prefix: _Prefix, num_held: int) -> None:
        """Cut a finished prefix's chunks to the first ``num_held`` and keep it.

        Its block hashes stay whole (see :class:`_Prefix`). If another kept
        prefix ends at the same chunk, the later-finished one stays.

        Args:
            prefix: A finished prefix with at least ``num_held`` chunk hashes.
            num_held: Its leading chunks L0 holds; at least 1.
        """
        del prefix.chunk_hashes[num_held:]
        key = (prefix.cache_salt, prefix.chunk_hashes[-1])
        kept = self._finished.get(key)
        if kept is None or kept.finish_order < prefix.finish_order:
            self._finished[key] = prefix

    @staticmethod
    def _add_chunks(held: set[_ChunkKey], prefix: _Prefix, num_held: int) -> None:
        """Add the first ``num_held`` chunks of ``prefix`` to ``held``."""
        held.update(
            (prefix.cache_salt, chunk_hash)
            for chunk_hash in prefix.chunk_hashes[:num_held]
        )

    def _collect_answers(self) -> None:
        """Sum the answers into one sample once every server has answered.

        A query still unanswered when the next one is due is dropped; after
        :data:`MAX_MISSED_DEADLINES` in a row the meter stops. A server that
        does not serve the query, hashes tokens differently, or sends a reply
        this version cannot decode stops the meter. Any other failure drops
        this sample only; the next interval retries.
        """
        pending = [url for url, f in self._inflight.items() if not f.query()]
        if pending:
            if self._clock() < self._next_sample_at:
                return
            self._inflight = {}
            self._missed_deadlines += 1
            if self._missed_deadlines >= MAX_MISSED_DEADLINES:
                self._stop(
                    "servers %s left %d queries in a row unanswered; a server "
                    "likely has no --server-module loaded (%s) or is stuck",
                    pending,
                    MAX_MISSED_DEADLINES,
                    _SERVER_MODULE_HINT,
                )
                return
            logger.warning(
                "L1 duplication: servers %s did not answer within %.0f s; "
                "dropping the sample",
                pending,
                SAMPLE_INTERVAL_S,
            )
            return
        self._missed_deadlines = 0
        inflight, self._inflight = self._inflight, {}
        replies: list[L1ResidencyReply] = []
        for url, future in inflight.items():
            try:
                response = future.result()
            except Exception as exc:
                # The transport stores whatever failed the call on the future
                # (gRPC status, ZMQ send error, decode error); one lost sample
                # must not fail the scheduler step.
                if (
                    isinstance(exc, grpc.RpcError)
                    and exc.code() == grpc.StatusCode.UNIMPLEMENTED
                ):
                    self._stop_unserved(url)
                    return
                logger.exception(
                    "L1 duplication: query to server %s failed; dropping the sample",
                    url,
                )
                return
            if not response.success:
                if response.error == unknown_method_error(L1_RESIDENCY_QUERY_METHOD):
                    self._stop_unserved(url)
                    return
                logger.warning(
                    "L1 duplication: server %s failed the query: %s",
                    url,
                    response.error,
                )
                return
            try:
                reply = msgspec.msgpack.decode(response.payload, type=L1ResidencyReply)
            except msgspec.DecodeError:  # also msgspec.ValidationError
                logger.exception(
                    "L1 duplication: cannot decode the reply of server %s; it "
                    "likely runs another LMCache version. Measuring stops",
                    url,
                )
                self._stop_quietly()
                return
            if reply.probe_hash != self._probe_hash:
                self._stop(
                    "server %s hashes tokens differently from the scheduler "
                    "(%s); both must set the same lmcache.mp.hash_algorithm "
                    "and, for an algorithm other than blake3, the same "
                    "PYTHONHASHSEED. No chunk can match",
                    url,
                    self._hasher.hash_algorithm_name,
                )
                return
            replies.append(reply)
        self._on_sample(
            DuplicationSample(
                matched_objects=sum(r.matched_objects for r in replies),
                matched_bytes=sum(r.matched_bytes for r in replies),
                model_objects=sum(r.model_objects for r in replies),
                model_bytes=sum(r.model_bytes for r in replies),
            )
        )

    def _stop_unserved(self, url: str) -> None:
        """Stop measuring because server ``url`` does not serve the query."""
        self._stop(
            "server %s does not serve %s; %s",
            url,
            L1_RESIDENCY_QUERY_METHOD,
            _SERVER_MODULE_HINT,
        )

    def _stop(self, reason: str, *args: Any) -> None:
        """Log ``reason % args`` as an error and stop measuring."""
        logger.error("L1 duplication: " + reason + ". Measuring stops", *args)
        self._stop_quietly()

    def _stop_quietly(self) -> None:
        """Stop measuring and drop every prefix; the caller logged why."""
        self._disabled = True
        self._running = {}
        self._finished = {}
