# SPDX-License-Identifier: Apache-2.0
"""Measure the share of L1 that vLLM's GPU prefix cache (L0) also holds.

See ``docs/design/integration/vllm/l1_duplication.md``.
"""

# Standard
from dataclasses import dataclass, field
from itertools import accumulate
from typing import TYPE_CHECKING, Any, Callable, Literal
import time

# Third Party
import grpc
import msgspec

# First Party
from lmcache.integration.vllm.kv_cache_groups import (
    is_cachable_mamba_spec,
    is_scratch_spec,
    is_sliding_window_spec,
    layer_window_tokens,
    representative_spec,
)
from lmcache.logging import init_logger
from lmcache.v1.distributed.api import window_tokens_to_chunks
from lmcache.v1.distributed.internal_api import L1_RECENCY_CAUSES
from lmcache.v1.multiprocess.futures import MessagingFuture
from lmcache.v1.multiprocess.protocols.l1_residency import (
    L1_RESIDENCY_QUERY_METHOD,
    GroupHeldChunks,
    GroupWindow,
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

# How a vLLM KV cache group holds chunk i in L0 under per-object-group rules:
# "prefix" by its own prefix hit, "window" by the blocks of the sliding window
# ending at the chunk's end, "state" by the state block at the chunk's end.
_HoldRule = Literal["prefix", "window", "state"]


def _ceil_div(a: int, b: int) -> int:
    """Return ``a / b`` rounded up, for non-negative ``a``."""
    return -(-a // b)


def _by_salt(keys: set[_ChunkKey]) -> dict[str, list[bytes]]:
    """Group chunk keys into chunk hashes per cache salt.

    Args:
        keys: The chunk keys to group.

    Returns:
        Each cache salt's chunk hashes, as a query carries them.
    """
    chunk_hashes_by_salt: dict[str, list[bytes]] = {}
    for cache_salt, chunk_hash in keys:
        chunk_hashes_by_salt.setdefault(cache_salt, []).append(chunk_hash)
    return chunk_hashes_by_salt


def _is_covered(
    held: set[_ChunkKey], prefix: "_Prefix", kept: list[int], prefix_closed: bool
) -> bool:
    """Return whether ``held`` already has every kept chunk of a finished prefix.

    Args:
        held: One object group's chunks held through longer prefixes.
        prefix: The finished prefix.
        kept: Its kept chunk indices in that object group, ascending.
        prefix_closed: Whether the object group's rule holds every chunk
            before a held one; a chunk hash chains every chunk before it, so
            the last kept chunk then stands for all.

    Returns:
        ``True`` if looking the prefix up again cannot add a chunk.
    """
    if prefix_closed:
        return not kept or (prefix.cache_salt, prefix.chunk_hashes[kept[-1]]) in held
    return all((prefix.cache_salt, prefix.chunk_hashes[i]) in held for i in kept)


@dataclass(frozen=True)
class _VllmGroup:
    """One vLLM KV cache group, as the per-object-group rules read it.

    Attributes:
        group_id: Its index in ``kv_cache_config.kv_cache_groups``.
        rule: How it holds a chunk (see :data:`_HoldRule`).
        block_size: Tokens per block of its spec.
        window_blocks: For ``"window"``, the trailing blocks a hit needs,
            vLLM's ``cdiv(sliding_window - 1, block_size)``; else 0.
        block_pool: The block pool its blocks are cached in.
    """

    group_id: int
    rule: _HoldRule
    block_size: int
    window_blocks: int
    block_pool: Any


@dataclass(frozen=True)
class _ObjectGroupRule:
    """The vLLM KV cache groups whose layers one LMCache object group stores.

    Attributes:
        window: The object group's kind and window, as the server names it.
        vllm_groups: Its vLLM KV cache groups; a chunk of the object group is
            in L0 only if every one of them holds it.
    """

    window: GroupWindow
    vllm_groups: tuple[_VllmGroup, ...]

    @property
    def prefix_closed(self) -> bool:
        """Whether holding chunk ``i`` implies holding every chunk before it."""
        return all(group.rule == "prefix" for group in self.vllm_groups)


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
        matched_inuse_objects: The part of ``matched_objects`` whose chunk a
            running request holds on the GPU: some block the meter read to
            decide the chunk is held has ``ref_cnt > 0``. The rest,
            ``matched_objects - matched_inuse_objects``, sits only in vLLM's
            free queue, still hashed.
        matched_inuse_bytes: Their bytes.
        matched_demoted_objects: The part of ``matched_objects`` at L1's
            evict-first end: demoted by an exclusive retrieve, not touched
            or rewritten since.
        matched_demoted_bytes: Their bytes.
        matched_mru_<cause>_objects: The rest of ``matched_objects``, by the
            event that last made the key MRU (``L1RecencyCause``): a store
            commit (``write``), a store finding it resident
            (``lazy_touch``), a lookup (``lookup_touch``), ``end_session``
            (``end_session_touch``) or anything else (``other``). Demoted
            plus every cause equals ``matched_objects`` (``*_bytes`` alike).
        matched_mru_<cause>_bytes: Their bytes.
    """

    matched_objects: int
    matched_bytes: int
    model_objects: int
    model_bytes: int
    matched_inuse_objects: int
    matched_inuse_bytes: int
    matched_demoted_objects: int = 0
    matched_demoted_bytes: int = 0
    matched_mru_write_objects: int = 0
    matched_mru_write_bytes: int = 0
    matched_mru_lazy_touch_objects: int = 0
    matched_mru_lazy_touch_bytes: int = 0
    matched_mru_lookup_touch_objects: int = 0
    matched_mru_lookup_touch_bytes: int = 0
    matched_mru_end_session_touch_objects: int = 0
    matched_mru_end_session_touch_bytes: int = 0
    matched_mru_other_objects: int = 0
    matched_mru_other_bytes: int = 0

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
    """One token prefix whose chunks L0 may hold.

    Attributes:
        cache_salt: The request's cache salt.
        block_hashes: vLLM block hashes of the prefix, one per hash block:
            the request's ``Request.block_hashes``, live while it runs and
            kept whole once it finished. A cut list could hit less than the
            prompt does: EAGLE drops the last hit block, and Mamba hits only
            up to a cached state.
        chunk_hashes: LMCache chunk hashes of chunks ``0..n-1``, the ones
            hashed so far; once finished, up to its last kept chunk.
        tracker: The request's tracker, whose tokens are hashed, while it
            runs; ``None`` once it finished.
        last_raw_hash: Raw hash of chunk ``n-1``, the chain input of chunk
            ``n``; ``None`` while no chunk is hashed and once finished.
        finish_order: How many requests finished before this one; 0 while
            it runs.
        kept: Once finished, per measured object group (one when the server
            has one), the ascending indices of the chunks kept; empty while
            it runs.
    """

    cache_salt: str
    block_hashes: list[Any]
    chunk_hashes: list[bytes]
    tracker: "LMCacheMPRequestTracker | None"
    last_raw_hash: Any = None
    finish_order: int = 0
    kept: list[list[int]] = field(default_factory=list)


class L1DuplicationMeter:
    """Sample the L0/L1 duplication every :data:`SAMPLE_INTERVAL_S`.

    While the servers have one object group, L0 holds chunk ``i`` of a
    prefix iff vLLM would hit at least ``(i + 1) * chunk_size`` tokens of it
    from the GPU now, by vLLM's own side-effect-free
    ``find_longest_cache_hit``. Once a reply shows several object groups
    (``--separate-object-groups``), L0 holds chunk ``i`` of an object group
    iff each of its vLLM KV cache groups holds it by its own rule (see the
    design doc). Call every method on the scheduler thread; they share the
    prefix state unlocked.

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
        # The rule of every object group once the servers reported several;
        # None while they have one, measured by vLLM's whole hit.
        self._rules: list[_ObjectGroupRule] | None = None
        # Every vLLM KV cache group: while the servers have one object group,
        # a held chunk is in use if a block of any of them is.
        self._all_vllm_groups = tuple(
            group for rule in self._object_group_rules() for group in rule.vllm_groups
        )
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
        held = self._held_chunks(prefix)
        prefix.tracker = None
        prefix.last_raw_hash = None
        self._num_finished += 1
        prefix.finish_order = self._num_finished
        if any(held):
            self._keep_finished(prefix, held)

    def on_scheduler_step(self) -> None:
        """Collect answers and, every interval, sample L0 and query servers.

        Call once per step. Never waits on a server: an answer is read on a
        later step. Server failures are logged, never raised: an unanswered
        or failed query drops one sample; :data:`MAX_MISSED_DEADLINES`
        unanswered in a row, an unserved method, a hash mismatch, an object
        group a server does not have, or an undecodable reply stops the
        meter.
        """
        if self._disabled:
            return
        if self._inflight:
            self._collect_answers()
        if self._disabled or self._inflight or self._clock() < self._next_sample_at:
            return
        self._next_sample_at = self._clock() + SAMPLE_INTERVAL_S
        # Pruning runs even while no server is healthy, so state stays bounded.
        held, inuse = self._held_chunk_keys()
        if not self._adapter.is_healthy:
            return
        if self._rules is None:
            query = L1ResidencyQuery(
                model_name=self._adapter.model_name,
                world_size=self._adapter.world_size,
                chunk_hashes_by_salt=_by_salt(held[0]),
                inuse_chunk_hashes_by_salt=_by_salt(inuse[0]),
            )
        else:
            query = L1ResidencyQuery(
                model_name=self._adapter.model_name,
                world_size=self._adapter.world_size,
                chunk_hashes_by_salt={},
                chunk_hashes_by_group=[
                    GroupHeldChunks(
                        window=rule.window, chunk_hashes_by_salt=_by_salt(keys)
                    )
                    for rule, keys in zip(self._rules, held, strict=True)
                ],
                inuse_chunk_hashes_by_group=[
                    GroupHeldChunks(
                        window=rule.window, chunk_hashes_by_salt=_by_salt(keys)
                    )
                    for rule, keys in zip(self._rules, inuse, strict=True)
                ],
            )
        self._inflight = self._adapter.server_module_call(
            L1_RESIDENCY_QUERY_METHOD, msgspec.msgpack.encode(query)
        )

    def _held_chunk_keys(self) -> tuple[list[set[_ChunkKey]], list[set[_ChunkKey]]]:
        """Return the chunks L0 holds, cutting finished prefixes to them.

        A finished prefix whose kept chunks are all held already, through a
        longer prefix that extends it, adds nothing and is not looked up:
        a chunk hash chains its prefix, so that longer prefix reads the same
        blocks for the in-use split.

        Returns:
            Per measured object group, every chunk L0 holds, and the subset
            of those a running request holds (see :meth:`_chunk_in_use`).
        """
        prefix_closed = self._prefix_closed()
        held: list[set[_ChunkKey]] = [set() for _ in prefix_closed]
        inuse: list[set[_ChunkKey]] = [set() for _ in prefix_closed]
        for prefix in self._running.values():
            self._add_chunks(held, inuse, prefix, self._held_chunks(prefix))
        finished = sorted(
            self._finished.values(), key=lambda p: len(p.chunk_hashes), reverse=True
        )
        self._finished = {}
        for prefix in finished:
            covered = [
                _is_covered(keys, prefix, kept, closed)
                for keys, kept, closed in zip(
                    held, prefix.kept, prefix_closed, strict=True
                )
            ]
            kept = prefix.kept
            if not all(covered):
                now = self._held_chunks(prefix)
                kept = [
                    old if is_covered else new
                    for old, new, is_covered in zip(kept, now, covered, strict=True)
                ]
                if not any(kept):
                    continue
                self._add_chunks(held, inuse, prefix, kept)
            self._keep_finished(prefix, kept)
        return held, inuse

    def _held_chunks(self, prefix: _Prefix) -> list[list[int]]:
        """Return, per measured object group, the chunks of ``prefix`` L0 holds.

        Hashes a running prefix's chunks up to the last one held.

        Args:
            prefix: A running prefix, or a finished one that
                :meth:`_keep_finished` cut.

        Returns:
            Ascending chunk indices per measured object group. A finished
            prefix's are limited to the ones it kept: chunks past them belong
            to a longer prefix, which counts them itself.
        """
        if self._rules is None:
            _, hit_tokens, _ = self._coordinator.find_longest_cache_hit(
                prefix.block_hashes, len(prefix.block_hashes) * self._hash_block_size
            )
            held = [list(range(hit_tokens // self._chunk_size))]
        else:
            held = self._held_per_object_group(prefix.block_hashes, self._rules)
        if prefix.tracker is None:
            return [
                sorted(set(indices).intersection(kept))
                for indices, kept in zip(held, prefix.kept, strict=True)
            ]
        num_chunks = max((indices[-1] + 1 for indices in held if indices), default=0)
        if num_chunks > len(prefix.chunk_hashes):
            token_ids = prefix.tracker.get_token_ids()
            for index in range(len(prefix.chunk_hashes), num_chunks):
                start = index * self._chunk_size
                prefix.last_raw_hash = self._hasher.hash_tokens(
                    token_ids[start : start + self._chunk_size], prefix.last_raw_hash
                )
                prefix.chunk_hashes.append(
                    self._hasher.hash_to_bytes(prefix.last_raw_hash)
                )
        return held

    def _held_per_object_group(
        self, block_hashes: list[Any], rules: list[_ObjectGroupRule]
    ) -> list[list[int]]:
        """Apply the per-object-group rules to one prefix's block hashes.

        Every lookup is read-only: vLLM's ``find_longest_cache_hit`` (one
        KV cache group) or ``find_longest_cache_hit_per_group``, and
        ``BlockPool.get_cached_block``.

        Args:
            block_hashes: The prefix's vLLM block hashes.
            rules: The rule of every measured object group.

        Returns:
            Per rule, the ascending indices of the chunks that every one of
            its vLLM KV cache groups holds.
        """
        num_tokens = len(block_hashes) * self._hash_block_size
        num_chunks = num_tokens // self._chunk_size
        prefix_hits: tuple[int, ...] = ()
        if any(g.rule == "prefix" for rule in rules for g in rule.vllm_groups):
            if len(self._coordinator.kv_cache_config.kv_cache_groups) == 1:
                _, hit_tokens, _ = self._coordinator.find_longest_cache_hit(
                    block_hashes, num_tokens
                )
                prefix_hits = (hit_tokens,)
            else:
                _, prefix_hits = self._coordinator.find_longest_cache_hit_per_group(
                    block_hashes, num_tokens
                )
        held: list[list[int]] = []
        for rule in rules:
            indices = set.intersection(
                *(
                    self._held_by_vllm_group(
                        group, block_hashes, num_chunks, prefix_hits
                    )
                    for group in rule.vllm_groups
                )
            )
            held.append(sorted(indices))
        return held

    def _held_by_vllm_group(
        self,
        group: _VllmGroup,
        block_hashes: list[Any],
        num_chunks: int,
        prefix_hits: tuple[int, ...],
    ) -> set[int]:
        """Return the chunks one vLLM KV cache group holds in L0.

        The hash of the block ending at token ``t`` is
        ``block_hashes[t // hash_block_size - 1]`` at any block size: each
        hash chains its whole prefix, which vLLM's
        ``BlockHashListWithBlockSize`` view also relies on.

        Args:
            group: The vLLM KV cache group.
            block_hashes: The prefix's vLLM block hashes.
            num_chunks: Whole chunks the block hashes cover.
            prefix_hits: Per vLLM KV cache group, its own prefix hit in
                tokens; read for ``"prefix"`` groups only.

        Returns:
            The indices of the chunks the group holds.
        """
        if group.rule == "prefix":
            return set(
                range(min(prefix_hits[group.group_id] // self._chunk_size, num_chunks))
            )
        group_ids = [group.group_id]
        hashes_per_block = group.block_size // self._hash_block_size
        blocks_per_chunk = self._chunk_size // group.block_size
        if group.rule == "state":
            return {
                index
                for index in range(num_chunks)
                if group.block_pool.get_cached_block(
                    block_hashes[(index + 1) * blocks_per_chunk * hashes_per_block - 1],
                    group_ids,
                )
                is not None
            }
        # Running count of cached blocks, so each window costs one subtraction.
        num_cached = [
            0,
            *accumulate(
                group.block_pool.get_cached_block(
                    block_hashes[(block + 1) * hashes_per_block - 1], group_ids
                )
                is not None
                for block in range(num_chunks * blocks_per_chunk)
            ),
        ]
        held: set[int] = set()
        for index in range(num_chunks):
            end = (index + 1) * blocks_per_chunk
            start = max(0, end - group.window_blocks)
            if num_cached[end] - num_cached[start] == end - start:
                held.add(index)
        return held

    def _keep_finished(self, prefix: _Prefix, kept: list[list[int]]) -> None:
        """Keep a finished prefix with only the chunks in ``kept``.

        Its chunk hashes are cut after its last kept chunk; its block hashes
        stay whole (see :class:`_Prefix`). If another kept prefix ends at the
        same chunk, the later-finished one stays.

        Args:
            prefix: A finished prefix with every chunk in ``kept`` hashed.
            kept: Per measured object group, the ascending indices of its
                chunks L0 holds; at least one is not empty.
        """
        prefix.kept = kept
        del prefix.chunk_hashes[max(indices[-1] for indices in kept if indices) + 1 :]
        key = (prefix.cache_salt, prefix.chunk_hashes[-1])
        stored = self._finished.get(key)
        if stored is None or stored.finish_order < prefix.finish_order:
            self._finished[key] = prefix

    def _prefix_closed(self) -> list[bool]:
        """Return, per measured object group, whether its rule is prefix-closed.

        Returns:
            ``[True]`` for vLLM's whole hit, else each rule's
            :attr:`_ObjectGroupRule.prefix_closed`.
        """
        if self._rules is None:
            return [True]
        return [rule.prefix_closed for rule in self._rules]

    def _add_chunks(
        self,
        held: list[set[_ChunkKey]],
        inuse: list[set[_ChunkKey]],
        prefix: _Prefix,
        chunks: list[list[int]],
    ) -> None:
        """Add the chunks of ``prefix`` in ``chunks`` to ``held``, per object group.

        Each one a running request holds (:meth:`_chunk_in_use`) also goes to
        ``inuse``.

        Args:
            held: Per measured object group, the chunks held so far.
            inuse: Per measured object group, the held chunks in use so far.
            prefix: The prefix whose chunks are added.
            chunks: Per measured object group, the indices of its held chunks.
        """
        for object_group, (keys, inuse_keys, indices) in enumerate(
            zip(held, inuse, chunks, strict=True)
        ):
            vllm_groups = (
                self._all_vllm_groups
                if self._rules is None
                else self._rules[object_group].vllm_groups
            )
            for index in indices:
                key = (prefix.cache_salt, prefix.chunk_hashes[index])
                keys.add(key)
                if key not in inuse_keys and self._chunk_in_use(
                    vllm_groups, prefix.block_hashes, index
                ):
                    inuse_keys.add(key)

    def _chunk_in_use(
        self, vllm_groups: tuple[_VllmGroup, ...], block_hashes: list[Any], index: int
    ) -> bool:
        """Return whether a running request holds a block of held chunk ``index``.

        Per vLLM KV cache group, the blocks read are the ones its rule reads
        to decide the chunk is held: the blocks overlapping the chunk for
        ``"prefix"``, the window ending at the chunk's end for ``"window"``,
        the state block at the chunk's end for ``"state"``. A block counts
        when ``BlockPool.get_cached_block`` (read-only) finds it with
        ``ref_cnt > 0``; a block not cached counts as not in use.

        Args:
            vllm_groups: The vLLM KV cache groups of the chunk's object group.
            block_hashes: The prefix's vLLM block hashes.
            index: The held chunk's index in the prefix.

        Returns:
            ``True`` if any such block of any group is in use.
        """
        for group in vllm_groups:
            hashes_per_block = group.block_size // self._hash_block_size
            if group.rule == "prefix":
                blocks = range(
                    index * self._chunk_size // group.block_size,
                    _ceil_div((index + 1) * self._chunk_size, group.block_size),
                )
            else:
                end = (index + 1) * (self._chunk_size // group.block_size)
                start = end - 1 if group.rule == "state" else end - group.window_blocks
                blocks = range(max(0, start), end)
            for block in blocks:
                hash_index = (block + 1) * hashes_per_block - 1
                if hash_index >= len(block_hashes):
                    break
                cached = group.block_pool.get_cached_block(
                    block_hashes[hash_index], [group.group_id]
                )
                if cached is not None and cached[0].ref_cnt > 0:
                    return True
        return False

    def _collect_answers(self) -> None:
        """Sum the answers into one sample once every server has answered.

        A query still unanswered when the next one is due is dropped; after
        :data:`MAX_MISSED_DEADLINES` in a row the meter stops. A server that
        does not serve the query, hashes tokens differently, or sends a reply
        this version cannot decode stops the meter, as do servers reporting
        object groups the meter cannot measure. Any other failure drops this
        sample only; the next interval retries.
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
        if not self._adopt_object_groups(replies):
            return
        by_cause: dict[str, int] = {}
        for cause in L1_RECENCY_CAUSES:
            by_cause[f"matched_mru_{cause}_objects"] = sum(
                r.matched_mru_objects.get(cause, 0) for r in replies
            )
            by_cause[f"matched_mru_{cause}_bytes"] = sum(
                r.matched_mru_bytes.get(cause, 0) for r in replies
            )
        self._on_sample(
            DuplicationSample(
                matched_objects=sum(r.matched_objects for r in replies),
                matched_bytes=sum(r.matched_bytes for r in replies),
                model_objects=sum(r.model_objects for r in replies),
                model_bytes=sum(r.model_bytes for r in replies),
                matched_inuse_objects=sum(r.matched_inuse_objects for r in replies),
                matched_inuse_bytes=sum(r.matched_inuse_bytes for r in replies),
                matched_demoted_objects=sum(r.matched_demoted_objects for r in replies),
                matched_demoted_bytes=sum(r.matched_demoted_bytes for r in replies),
                **by_cause,
            )
        )

    def _adopt_object_groups(self, replies: list[L1ResidencyReply]) -> bool:
        """Check that the replies counted what this query measured.

        The first reply showing several object groups switches the meter to
        per-object-group rules (and one showing one switches it back); that
        sample counted the wrong field and is dropped. Finished prefixes keep
        their chunks so far in every object group.

        Args:
            replies: Every server's decoded reply to one query.

        Returns:
            ``True`` if the sample is valid; ``False`` if it is dropped, the
            meter switched rules or stopped.
        """
        object_groups = replies[0].object_groups
        if any(reply.object_groups != object_groups for reply in replies[1:]):
            self._stop(
                "servers report different object groups: %s",
                [reply.object_groups for reply in replies],
            )
            return False
        if len(object_groups) == 1:
            if self._rules is None:
                return True
            self._use_rules(None)
            logger.info("L1 duplication: servers merge object groups again")
            return False
        rules = self._rules if self._rules is not None else self._object_group_rules()
        missing = [rule.window for rule in rules if rule.window not in object_groups]
        if missing:
            self._stop(
                "the scheduler's KV cache groups need object groups %s, which "
                "the servers do not have (%s); scheduler and servers must "
                "share lmcache.mp chunk size and KV cache layout",
                missing,
                object_groups,
            )
            return False
        if self._rules is not None:
            return True
        if not self._coordinator.block_pool.enable_caching:
            self._stop(
                "measuring separated object groups needs vLLM prefix caching, "
                "which is disabled"
            )
            return False
        self._use_rules(rules)
        logger.info(
            "L1 duplication: servers separate object groups %s; measuring "
            "each by its KV cache groups from the next sample",
            object_groups,
        )
        return False

    def _use_rules(self, rules: list[_ObjectGroupRule] | None) -> None:
        """Switch to ``rules``; finished prefixes keep every hashed chunk.

        Args:
            rules: The per-object-group rules, or ``None`` for vLLM's whole
                hit.
        """
        self._rules = rules
        num_measured = len(self._prefix_closed())
        for prefix in self._finished.values():
            prefix.kept = [
                list(range(len(prefix.chunk_hashes))) for _ in range(num_measured)
            ]

    def _object_group_rules(self) -> list[_ObjectGroupRule]:
        """Group vLLM's KV cache groups into the server's object groups.

        A group's object group is its kind and window in chunks, by the same
        :func:`layer_window_tokens` and :func:`window_tokens_to_chunks` the
        server buckets its groups by; scratch groups store nothing.

        Returns:
            One rule per object group, in first-group order.
        """
        groups: dict[GroupWindow, list[_VllmGroup]] = {}
        for group_id, kv_cache_group in enumerate(
            self._coordinator.kv_cache_config.kv_cache_groups
        ):
            if is_scratch_spec(kv_cache_group.kv_cache_spec):
                continue
            spec = representative_spec(kv_cache_group.kv_cache_spec)
            recurrent = is_cachable_mamba_spec(spec)
            window = GroupWindow(
                kind="recurrent" if recurrent else "attention",
                window_chunks=window_tokens_to_chunks(
                    layer_window_tokens(spec), self._chunk_size
                ),
            )
            rule: _HoldRule = "prefix"
            window_blocks = 0
            if is_sliding_window_spec(spec):
                rule = "window"
                window_blocks = _ceil_div(spec.sliding_window - 1, spec.block_size)
            elif recurrent:
                rule = "state"
            groups.setdefault(window, []).append(
                _VllmGroup(
                    group_id=group_id,
                    rule=rule,
                    block_size=spec.block_size,
                    window_blocks=window_blocks,
                    block_pool=self._coordinator.single_type_managers[
                        group_id
                    ].block_pool,
                )
            )
        return [
            _ObjectGroupRule(window=window, vllm_groups=tuple(vllm_groups))
            for window, vllm_groups in groups.items()
        ]

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
