# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side orchestration for lazy cache offload.

The boundary between ``LMCacheMPConnector`` and the lazy-offload policies: it
owns policy dispatch, GPU block pinning, store-batch coalescing, completion
handling and deferred session release. The connector only forwards events.

Terms used here and in ``lazy_offload_state``, on top of the policy-facing
ones defined in ``lazy_offload_policy.base``:

- **Request generation**: one use of a request id. vLLM recreates a tracker
  under the same id when it resumes a preempted request, and a later,
  unrelated request may reuse the id of a finished one, so one id can
  outlive several generations.
- **Receipt**: a worker's report that a submitted store batch ended, in
  success or in failure. The batch settles once every worker has reported.
- **Pin**: the reference this class takes on a GPU block so that vLLM cannot
  recycle it while the worker is still reading it. The receipt releases it.
  A held operation (see ``lazy_offload_policy.base``) is pinned before its
  submission; the submitted batch takes that pin over.
- **Orphaned**: said of a batch whose request finished and whose id a new
  request took over before its receipt arrived. Its pins are still
  released, but its failure is not retried: the new request's tokens are
  not the batch's. A preempted request keeps its id and its tokens, so its
  batch is not orphaned.
- **Token ledger**: the one token-id list a request's buffered operations
  share. The tracker hands out a fresh copy of the whole sequence with every
  operation, which deferral would otherwise retain once per operation.
"""

# Standard
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Protocol

# First Party
from lmcache.integration.vllm.lazy_offload_policy import create_offload_policy
from lmcache.integration.vllm.lazy_offload_policy.base import (
    BlockHashes,
    ConfigValue,
    DrainSignals,
    OffloadPolicy,
    PendingStoreItem,
)
from lmcache.integration.vllm.lazy_offload_state import LazyOffloadRequestRegistry
from lmcache.integration.vllm.lmcache_mp_metadata import (
    LMCacheMPRequestMetadata,
    LoadStoreOp,
)
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_coordinator import KVCacheCoordinator
    from vllm.v1.core.kv_cache_utils import BlockHash
    from vllm.v1.core.sched.output import SchedulerOutput

logger = init_logger(__name__)


class StoreCompletionTracker(Protocol):
    """Aggregate per-worker completion counts for one submitted store."""

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        """Record ``count`` new worker receipts for ``request_id``.

        The parameters are positional-only: the implementer names the first
        one ``req_id``, so a keyword call would not bind.

        Args:
            request_id: The request whose in-flight store batch the receipts
                belong to.
            count: Worker completions newly reported for that batch, added
                to the ones already counted.

        Returns:
            True once the submitted batch has every expected completion.
        """
        ...


@dataclass
class LazyOffloadActions:
    """Explicit connector effects produced by one lazy-offload event.

    Attributes:
        stores_to_submit: What to store now, one coalesced operation per
            request, whose GPU blocks this class has already pinned. The
            caller submits each and the receipt unpins them.
        sessions_to_end: Requests whose LMCache session the caller may
            release: either they have settled (finished, nothing buffered,
            no batch in flight), or a new request is taking over the id and
            the previous holder's deferred teardown has to happen first.
            The caller must apply these before opening a session of its
            own for the same id.
    """

    stores_to_submit: list[LMCacheMPRequestMetadata] = field(default_factory=list)
    sessions_to_end: list[str] = field(default_factory=list)


def _coalesce_store_metadata(
    request_metas: list[LMCacheMPRequestMetadata],
    group_tokens_per_block: list[int],
    null_block_id: int,
) -> LMCacheMPRequestMetadata:
    """Merge one request's STORE metadata, in prefix order, into one.

    The worker tracks one in-flight store future per request, so a drained
    batch must be submitted as a single operation. A segment the policy
    stored from the middle of the buffer leaves a range between two
    operations unbuffered, since it was stored already; that range gets the
    null block, whose chunks the server never stores.

    Args:
        request_metas: One request's buffered STORE metadata, in prefix
            order, each starting at or after the end of the one before it.
        group_tokens_per_block: Tokens per block of each cache group, which
            sizes a gap's null blocks; scratch groups report ``0``.
        null_block_id: vLLM's null block, which fills a gap.

    Returns:
        A single STORE metadata spanning the whole range, carrying the
        concatenated block ids of every cache group.

    Raises:
        ValueError: If the input is empty, overlapping, or changes cache
            group count mid-batch.
    """
    if not request_metas:
        raise ValueError("cannot coalesce an empty store batch")
    if len(request_metas) == 1:
        return request_metas[0]
    first = request_metas[0]
    last = request_metas[-1]
    merged_block_ids: list[list[int]] = [list(group) for group in first.op.block_ids]
    expected_start = first.op.end
    for meta in request_metas[1:]:
        if meta.op.start < expected_start:
            raise ValueError(
                f"overlapping store ops for request {first.request_id}: "
                f"expected start {expected_start} or later, got {meta.op.start}"
            )
        if len(meta.op.block_ids) != len(merged_block_ids):
            raise ValueError(
                f"cache-group count changed within store batch for request "
                f"{first.request_id}"
            )
        gap = meta.op.start - expected_start
        for group_idx, tokens_per_block in enumerate(group_tokens_per_block):
            if tokens_per_block:
                merged_block_ids[group_idx].extend(
                    [null_block_id] * (gap // tokens_per_block)
                )
        expected_start = meta.op.end
        for group_idx, group_ids in enumerate(meta.op.block_ids):
            merged_block_ids[group_idx].extend(group_ids)
    merged_op = LoadStoreOp(
        token_bytes=last.op.token_bytes,
        block_ids=merged_block_ids,
        start=first.op.start,
        end=last.op.end,
    )
    return LMCacheMPRequestMetadata(
        request_id=first.request_id,
        direction="STORE",
        op=merged_op,
        cache_salt=first.cache_salt,
        request_configs=first.request_configs,
    )


class LazyOffloadManager:
    """Own scheduler-side lazy-offload integration and side effects.

    The only lazy-offload object exposed to the connector: it turns scheduler
    events into policy signals, pins and unpins vLLM GPU blocks, and returns
    connector actions. Every method other than :meth:`bind_block_pool`,
    :meth:`bind_cache_coordinator` and :meth:`log_final_stats` raises
    ``ValueError`` until the pool is bound; finishing a request and looking
    a waiting request up also need the coordinator. Scheduler thread only.
    """

    def __init__(
        self,
        configs: dict[str, ConfigValue] | None,
        group_tokens_per_block: list[int],
        completion_tracker: StoreCompletionTracker,
        max_step_tokens: int,
        num_lookahead_tokens: int,
        max_model_len: int,
        chunk_size: int,
        mamba_groups: dict[int, int] | None = None,
        prefill_alignment: int = 0,
    ) -> None:
        """Create an unbound scheduler-side manager.

        Args:
            configs: Connector extra config, read by the policy at bind time.
            group_tokens_per_block: Token capacity per KV-cache group, used
                to bound how many blocks an allocation can take. Scratch
                groups report ``0`` and are left out of every bound.
            completion_tracker: Adapter view aggregating per-worker receipts.
            max_step_tokens: vLLM's ``max_num_batched_tokens``: the most
                tokens one step schedules over all requests together, and
                so for any one of them.
            num_lookahead_tokens: Slots vLLM allocates past the scheduled
                tokens for speculative decoding, which is also the number of
                draft tokens a decode step schedules.
            max_model_len: vLLM's ``max_model_len``: the most tokens one
                request holds, which bounds what one admission allocates.
            chunk_size: LMCache's tokens per chunk, a multiple of every
                group's tokens per block: the unit the server stores, by
                which a lookup finds the buffered ops covering its prefix.
            mamba_groups: Align-mode Mamba groups -> the blocks a running
                request's step can take beyond its one state block: a
                prefill checkpoint block and a partial-hit copy, where the
                model has them. Such a group sizes its state by the
                scheduled tokens alone, without lookahead slots, and nulls
                the positions a step skips rather than allocating them.
            prefill_alignment: Tokens a prefill chunk that stops short of
                the prompt's end ends on a multiple of, as vLLM splits
                chunks of Mamba-hybrid models; 0 when a chunk may end
                anywhere.
        """
        self._configs = dict(configs or {})
        self._group_tokens_per_block = list(group_tokens_per_block)
        self._chunk_size = chunk_size
        self._max_step_tokens = max_step_tokens
        self._num_lookahead_tokens = num_lookahead_tokens
        self._mamba_groups = dict(mamba_groups or {})
        self._prefill_alignment = prefill_alignment
        self._completion_tracker = completion_tracker
        # Both are set by bind_block_pool: the policy reads the pool.
        self._gpu_block_pool: "BlockPool | None" = None
        self._policy: OffloadPolicy | None = None
        # Set by bind_cache_coordinator: vLLM's hit decides what a finished
        # request's and a waiting request's prefix cache serves.
        self._coordinator: "KVCacheCoordinator | None" = None
        # Per request, vLLM's live block-hash chain of its tokens, from its
        # arrival until its session ends: what its servable prefix is
        # looked up by, also after it finished.
        self._block_hash_chains: dict[str, list["BlockHash"]] = {}
        self._requests = LazyOffloadRequestRegistry()
        # One token ledger per request whose operations are buffered.
        self._token_ledgers: dict[str, bytearray] = {}
        # Blocks pinned for held operations, per request, until submission.
        self._held_block_ids: dict[str, set[int]] = {}
        # Blocks the drain or a protection pinned while they sat in the free
        # queue: unpinning returns them to its head, where eviction order had
        # them.
        self._pinned_while_free: set[int] = set()
        # Token counts of admitted requests and, once scheduled, the tokens
        # they will have computed after their last scheduled step: what
        # bounds the blocks they can allocate before the next drain.
        self._num_tokens: dict[str, int] = {}
        self._num_computed: dict[str, int] = {}
        # Requests loading asynchronously -> tokens left after the load.
        self._load_remaining: dict[str, int] = {}
        # Requests whose load finished but that are not yet scheduled -> the
        # blocks their first chunk allocates. vLLM schedules them only when
        # their chunk fits, which can be steps later.
        self._resuming: dict[str, int] = {}
        # The last drain's bound, which the next schedule's running
        # requests allocate before a resumed load does.
        self._next_step_blocks = 0
        self._admission_blocks = self._admission_bound(max_model_len)
        # Per group, block ids each scheduled request has been handed, to
        # tell what a step allocated from blocks vLLM only moved within its
        # table, and its table length, which counts the lookahead slots the
        # last allocation covered too.
        self._handed_blocks: dict[str, list[set[int]]] = {}
        self._table_lengths: dict[str, list[int]] = {}
        # Per chunk, by the block hash ending it, the requests that offered
        # an op storing it, and per such request those hashes: what finds
        # the buffered ops covering a waiting request's prefix. An entry can
        # outlive its op; the policy only flushes ops it still buffers.
        self._chunk_offerers: dict["BlockHash", set[str]] = {}
        self._offered_chunks: dict[str, set["BlockHash"]] = {}
        # Waiting requests whose lookup flushed buffered ops -> the token
        # vLLM's prefix cache serves them up to and, per request owning the
        # flushed ops, the token the flushed range ends at.
        self._lookup_flushes: dict[str, tuple[int, dict[str, int]]] = {}
        # Requests whose lookup was checked for buffered ops, once per
        # generation: a lookup submitted after it would not see a flush.
        self._lookup_checked: set[str] = set()

    @property
    def admission_blocks(self) -> int:
        """Free-queue head each drain keeps clear of buffered blocks for one
        admission: the most blocks one admission can allocate, from
        ``max_model_len``. It deepens the danger window beyond the running
        requests' bound, so a step's first admission recycles no buffered
        block.

        Returns:
            The bound, in blocks summed over the cacheable groups.
        """
        return self._admission_blocks

    def bind_block_pool(self, gpu_block_pool: "BlockPool") -> None:
        """Bind the scheduler's GPU block pool and build the policy.

        Idempotent for the same pool; rebinding a different one would
        silently invalidate every buffered hash snapshot.

        Args:
            gpu_block_pool: The scheduler's block pool, from which the
                manager reads block hashes and which it pins into.

        Raises:
            ValueError: If a different pool is bound, or the policy name is
                unknown.
        """
        if self._gpu_block_pool is gpu_block_pool:
            return
        if self._gpu_block_pool is not None:
            raise ValueError(
                "a different GPU block pool is already bound; rebinding "
                "would discard the buffered store operations"
            )
        self._gpu_block_pool = gpu_block_pool
        self._policy = create_offload_policy(
            self._configs, gpu_block_pool, frozenset(self._mamba_groups)
        )

    def bind_cache_coordinator(self, coordinator: "KVCacheCoordinator") -> None:
        """Bind the scheduler's KV cache coordinator, whose hit decides what
        a finished request's or a waiting request's prefix cache serves.

        Args:
            coordinator: The scheduler's ``KVCacheManager.coordinator``, over
                the bound block pool. Only its read-only
                ``find_longest_cache_hit`` is called.
        """
        self._coordinator = coordinator

    def add_store_candidate(self, metadata: LMCacheMPRequestMetadata) -> None:
        """Buffer one STORE metadata produced by the request tracker.

        Lazy offload stores a block at the latest moment before vLLM reuses
        it. vLLM frees hash-less blocks to the head of the free queue,
        reused first, and hashed blocks to the tail in LRU order
        (``BlockPool.free_blocks``); an allocation takes blocks from the
        head (``get_new_blocks``). A hashed block is reused only after the
        blocks ahead of it, so lazy offload can track it as it moves toward
        the head and store it before reuse: it has slack. A hash-less block
        is taken by the next allocation: it has no slack. Pinning it to
        store it later would make vLLM take the next free block instead,
        evicting a hashed, still reusable block early, which trades a
        reusable block for one vLLM does not intend to reuse. So lazy
        offload covers only hashed blocks, which are exactly the ones vLLM
        retains past the step: it hashes them in ``allocate_slots``, before
        this call. A hash-less block (an align-mode recurrent state sparse
        retention does not keep) is treated as absent: replaced by the null
        block in the buffered copy, like the null block hybrid models put
        at positions holding no data, and it never breaks the prefix chain,
        which is defined over hashed blocks only. The server skips a chunk
        that is null in every group. Full attention blocks in a store op
        are always hashed, since an op covers whole computed blocks and
        lazy offload requires prefix caching. The hashes of the remaining
        blocks are read here, at admission, so that the policy holds the
        snapshot the data was stored under.

        One request's operations must be offered in token order, each
        starting where the previous one ended: they are submitted as one
        coalesced store, which a gap makes impossible.

        The operation is rebound to the request's token ledger before it is
        buffered, so deferring N operations of one request costs one copy of
        the token sequence rather than N.

        Args:
            metadata: The store operation offered by the tracker: the token
                range and the GPU blocks holding its KV. Its ``op`` is not
                mutated; a rebound copy is buffered instead.
        """
        pool = self._require_block_pool()
        null_block_id = pool.null_block.block_id
        protected_block_ids = [
            [
                block_id
                if pool.blocks[block_id].block_hash is not None
                else null_block_id
                for block_id in group_block_ids
            ]
            for group_block_ids in metadata.op.block_ids
        ]
        block_hashes: BlockHashes = {
            block_id: pool.blocks[block_id].block_hash
            for group_block_ids in protected_block_ids
            for block_id in group_block_ids
            if block_id != null_block_id
        }
        rebound = self._rebind_tokens(metadata)
        rebound = replace(
            rebound, op=replace(rebound.op, block_ids=protected_block_ids)
        )
        superseded = self._require_policy().add(rebound, block_hashes)
        self._release_superseded(superseded)
        self._index_offered_chunks(
            metadata.request_id, metadata.op.start, metadata.op.end
        )

    def on_scheduler_step(
        self, scheduler_output: "SchedulerOutput"
    ) -> LazyOffloadActions:
        """Drain stores made due by one scheduler step.

        A step that schedules no tokens drains too: vLLM's no-forward path
        still calls ``wait_for_save``, which submits the stores. It happens
        when blocks run out for every waiting request, and the pins a
        squeezed admission left are then what must be stored to free them.

        Args:
            scheduler_output: The step's schedule, supplying the block
                pressure the drain decision reads.

        Returns:
            Stores to submit and sessions made releasable by the drain.

        Raises:
            ValueError: If no GPU block pool has been bound, or if a
                request's buffered operations are not contiguous.
            RuntimeError: If the policy emitted for a request that
                already has a store batch in flight.
        """
        return self._drain(scheduler_output, self._require_block_pool())

    def flush_for_lookup(self, request_id: str, num_computed_tokens: int) -> bool:
        """Store the buffered ops a waiting request's lookup needs first.

        Lazy offload defers a chunk while vLLM's prefix cache can serve it.
        A request whose local hit stops short of chunks other requests
        still buffer (in a hybrid model, a hit resumes only at a cached
        state) can be served neither from the GPU nor from L1, and would
        recompute them. So on its first call the ops covering its prefix
        past the local hit are held and stored by the next drain, as eager
        offload would have stored them, together with the ops before them
        back to the previous Mamba state (see
        ``EvictionAwareStoreQueue.hold_past``). Until their receipts settle,
        the caller defers the lookup, which then hits L1. Found through the
        chunk hashes of the request's chain, so the cost is linear in its
        chunks. The receipt frees the flushed blocks to the free-queue tail,
        not the head, so the flush does not evict the request's own
        hit-point state block.

        Args:
            request_id: A waiting request whose lookup was not yet
                submitted; it arrived and its session has not ended.
            num_computed_tokens: Its local prefix hit, in tokens.

        Returns:
            True while the lookup must wait for flushed stores; False once
            they settled, or when nothing needed flushing.
        """
        if request_id not in self._lookup_checked:
            self._lookup_checked.add(request_id)
            self._flush_offered_prefix(request_id, num_computed_tokens)
        flush = self._lookup_flushes.get(request_id)
        if flush is None:
            return False
        start, ends = flush
        policy = self._require_policy()
        if any(
            policy.is_flushing(owner, start, end)
            or self._requests.in_flight_overlaps(owner, start, end)
            for owner, end in ends.items()
        ):
            return True
        del self._lookup_flushes[request_id]
        return False

    def on_admission(
        self,
        request_id: str,
        num_tokens: int,
        num_computed_tokens: int,
        num_external_tokens: int,
        load_async: bool,
    ) -> None:
        """Protect buffered blocks an admission is about to recycle.

        vLLM allocates an admitted request's blocks right after asking the
        connector for its external hit, evicting from the free-queue head
        before this step's drain can emit anything there. A long hit
        allocates its whole prefix at once, which no danger depth predicts.
        Pinning those blocks would shrink the free queue the allocation
        draws from and could refuse it, so they move to the tail instead,
        still allocatable and still buffered. Only when the allocation would
        reach the tail too are they pinned now.

        Args:
            request_id: The request being admitted.
            num_tokens: Its tokens, prompt and any output so far.
            num_computed_tokens: Its local prefix hit, in tokens.
            num_external_tokens: Its external hit, in tokens.
            load_async: Whether the hit loads asynchronously. vLLM then
                allocates for the hit alone and schedules the rest once the
                load finishes, which :meth:`on_loads_finished` covers.
        """
        self._num_tokens[request_id] = num_tokens
        remaining = num_tokens - num_computed_tokens - num_external_tokens
        if load_async:
            self._load_remaining[request_id] = remaining
            num_new_tokens = num_external_tokens
        else:
            num_new_tokens = num_external_tokens + self._chunk_tokens(remaining)
        # vLLM touches the local hit before allocating, which takes its free
        # blocks out of the queue, so the allocation reaches that much further.
        # An align-mode Mamba hit is null blocks but its last state block, and
        # an external hit gets one state block of its own before the
        # allocation, outside the allocation's cap.
        num_hit_blocks = sum(
            min(1, num_computed_tokens) + min(1, num_external_tokens)
            if group_idx in self._mamba_groups
            else -(-num_computed_tokens // tokens_per_block)
            for group_idx, tokens_per_block in enumerate(self._group_tokens_per_block)
            if tokens_per_block > 0
        )
        self._protect(self._span_blocks(num_new_tokens) + num_hit_blocks)

    def on_loads_finished(self, request_ids: set[str]) -> None:
        """Protect buffered blocks that resuming finished loads will recycle.

        A request whose asynchronous load finished is scheduled again
        without asking the connector for a hit, so this is the last call
        before it allocates for the rest of its tokens. The next schedule's
        running requests allocate first, so their bound counts too. A full
        cache can keep the request waiting for several steps, while the
        free-queue head moves past the protected blocks; each drain until it
        is scheduled counts its allocation in the danger depth.

        Args:
            request_ids: Requests whose load the workers reported finished.
                Ids without an asynchronous load are ignored.
        """
        resumed = {
            request_id: self._span_blocks(
                self._chunk_tokens(self._load_remaining.pop(request_id)),
                first_allocation=False,
            )
            for request_id in request_ids
            if request_id in self._load_remaining
        }
        if not resumed:
            return
        self._resuming.update(resumed)
        self._protect(self._next_step_blocks + sum(resumed.values()))

    def on_store_results(
        self,
        failed_request_ids: set[str],
        completed_store_counts: dict[str, int],
    ) -> LazyOffloadActions:
        """Apply failed stores and fully aggregated completion receipts.

        A failure only marks the in-flight batch: other workers may still be
        reading its blocks. Once every worker has reported, a failed batch
        that is not orphaned goes back to the policy to be retried, and its
        blocks move to the free-queue tail instead of the head. Stale
        receipts in ``completed_store_counts`` are filtered here.

        Args:
            failed_request_ids: Requests whose submitted store failed on at
                least one worker.
            completed_store_counts: Worker completions newly reported this
                round, keyed by request id. A batch settles once its count
                reaches the number of workers.

        Returns:
            Sessions made releasable by completed batches.
        """
        for request_id in failed_request_ids:
            if self._requests.has_in_flight(request_id):
                self._requests.fail_in_flight(request_id)
        actions = LazyOffloadActions()
        for request_id, count in completed_store_counts.items():
            if not self._requests.has_in_flight(request_id):
                logger.warning(
                    "Ignoring store-completion receipt for request %s with "
                    "no in-flight store batch",
                    request_id,
                )
                continue
            if not self._completion_tracker.update_pending_store_count(
                request_id, count
            ):
                continue
            batch = self._requests.complete_batch(request_id)
            self._unpin(list(batch.block_ids))
            if batch.failed and not batch.orphaned:
                # The data is still on the GPU; an id reuse would have
                # detached the batch from the request it belongs to.
                logger.warning(
                    "Store failed for request %s; retrying its %d store op(s)",
                    request_id,
                    len(batch.operations.metadatas),
                )
                self._require_policy().requeue_failed(batch.operations)
                self._move_to_tail([batch.operations])
            if not self._require_policy().has_pending_request(
                request_id
            ) and self._requests.can_end_session(request_id):
                actions.sessions_to_end.append(request_id)
                self._release_session(request_id)
        return actions

    def on_request_finished(self, request_id: str) -> LazyOffloadActions:
        """Record request completion and decide whether its session can end.

        Its buffered ops ending past the prefix vLLM's hit over its own
        chain serves are held now and stored by the next drain with budget,
        as eager offload would have stored them: no later request hits them
        on the GPU. Under speculative decoding that is at least the chunk
        holding the request's last cached block, which the hit drops. vLLM
        calls this before it frees the request's blocks, so the pin keeps
        them out of the free queue, and the receipt frees them to its tail
        as vLLM would have. That holds too for a block vLLM already freed,
        such as an align-mode Mamba state block: the hold does not send it
        to the head on release, where the next allocation would evict the
        state the next turn's hit resumes at.

        Args:
            request_id: The request vLLM reported as finished.

        Returns:
            A session-release action only when no store is pending or in
            flight; otherwise an empty action.
        """
        self._requests.finish(request_id)
        self._forget_progress(request_id)
        self._forget_lookup(request_id)
        policy = self._require_policy()
        if policy.has_pending_request(request_id):
            held = policy.hold_past(request_id, self._hit_tokens(request_id))
            if held.metadatas:
                self._pin_holds([held], release_to_head=False)
        if policy.has_pending_request(request_id):
            return LazyOffloadActions()
        if self._requests.has_in_flight(request_id):
            return LazyOffloadActions()
        self._release_session(request_id)
        return LazyOffloadActions(sessions_to_end=[request_id])

    def on_request_reset(self, request_id: str) -> None:
        """Restart a preempted request, keeping what it buffered.

        vLLM freed the request's blocks when it preempted it: the hashed
        ones joined the free queue with their hashes, so they are
        prefix-cache victims like any other, and the request's buffered and
        held operations keep protecting them. The restarted request offers
        its tokens again; each offer supersedes the buffered operations it
        overlaps. A batch already in flight is the request's own, retried if
        it fails. Only what describes the request's progress is forgotten.

        Args:
            request_id: The preempted request.
        """
        self._forget_progress(request_id)
        self._forget_lookup(request_id)
        self._require_policy().restart_request(request_id)

    def on_request_arrived(
        self, request_id: str, block_hashes: list["BlockHash"]
    ) -> LazyOffloadActions:
        """Reclaim residual state if a new request reuses a finished id.

        Args:
            request_id: The id whose tracker vLLM just created, which may be
                a first arrival, a preempted request coming back, or a new
                request taking over a finished id.
            block_hashes: The request's ``Request.block_hashes``, which vLLM
                extends as tokens arrive; kept by reference until the
                request's session ends, to look its servable prefix up.

        Returns:
            A session-release action for the predecessor when the id is
            reused and nothing of the predecessor is in flight. With a batch
            still in flight the release is skipped and never happens on its
            own: a session is keyed by request id, so ending it now would
            end the successor's too. The successor's own teardown covers it.
        """
        reused_finished_id = self._requests.is_finished(request_id)
        predecessor_in_flight = self._requests.has_in_flight(request_id)
        self._requests.arrive(request_id)
        self._block_hash_chains[request_id] = block_hashes
        if not reused_finished_id:
            return LazyOffloadActions()
        self._require_policy().discard_for_reuse(request_id)
        self._forget_lookup(request_id)
        self._unindex_offered_chunks(request_id)
        self._release_holds(request_id)
        self._token_ledgers.pop(request_id, None)
        if predecessor_in_flight:
            return LazyOffloadActions()
        logger.info(
            "Lazy offload: request id %s reused while its predecessor's "
            "teardown was deferred; released the predecessor's session",
            request_id,
        )
        return LazyOffloadActions(sessions_to_end=[request_id])

    def has_inflight_store_work(self) -> bool:
        """Whether submitted stores are still holding GPU blocks.

        Buffered operations are deliberately excluded. They are emitted only
        by a step that schedules model tokens, so a connector-only step
        cannot advance them and keeping the engine awake for them would spin.
        A submitted batch is different: its receipt arrives from the worker
        without any further scheduling, and the blocks it pins stay pinned
        until it does.

        Returns:
            True while at least one submitted batch awaits its receipt.
        """
        return bool(self._requests.in_flight_request_ids())

    def log_final_stats(self) -> None:
        """Write the policy's final counter ledger, when one was built."""
        if self._policy is not None:
            self._policy.log_final_stats()

    def _drain(
        self,
        scheduler_output: "SchedulerOutput",
        pool: "BlockPool",
    ) -> LazyOffloadActions:
        """Apply one policy-neutral drain plan and its GPU side effects.

        Released holds are unpinned and new holds pinned first. Each emitted
        operation is re-validated against the hash snapshot the policy kept,
        pinned unless it was held, and coalesced into one submission per
        request. A request whose snapshot no longer matches loses that
        operation and every later one of the same drain.

        Args:
            scheduler_output: The step's schedule, converted into the
                block-pressure signals the policy reads.
            pool: The bound GPU block pool, touched and pinned here.

        Returns:
            The stores to submit and the sessions the drain made releasable.

        Raises:
            RuntimeError: If the policy emitted for a request that already
                has a store batch in flight, which would make the worker's
                receipts ambiguous.
        """
        running_blocks_allocated = self._running_blocks_allocated(scheduler_output)
        self._next_step_blocks = self._running_blocks_bound(scheduler_output)
        for request_id in scheduler_output.num_scheduled_tokens:
            self._resuming.pop(request_id, None)
        drain = self._require_policy().drain(
            DrainSignals(
                next_step_blocks=self._next_step_blocks,
                resuming_blocks=sum(self._resuming.values()),
                admission_blocks=self._admission_blocks,
                finished_request_ids=self._requests.finished_request_ids(),
                blocked_request_ids=self._requests.in_flight_request_ids(),
                running_blocks_allocated=running_blocks_allocated,
            )
        )
        for request_id in drain.releases:
            self._release_holds(request_id)
        self._pin_holds(drain.holds, release_to_head=True)
        actions = LazyOffloadActions()
        for item in drain.items:
            if self._requests.has_in_flight(item.request_id):
                raise RuntimeError(
                    f"request {item.request_id!r} emitted while a store batch "
                    "is still in flight"
                )
            held = self._held_block_ids.pop(item.request_id, set())
            valid = PendingStoreItem(request_id=item.request_id)
            valid_block_ids: list[int] = []
            for metadata, old_block_hashes in item.metadatas:
                gpu_block_ids = list(old_block_hashes)
                self._pin(
                    [block_id for block_id in gpu_block_ids if block_id not in held]
                )
                held.difference_update(gpu_block_ids)
                new_block_hashes = {
                    block_id: pool.blocks[block_id].block_hash
                    for block_id in gpu_block_ids
                }
                if (
                    any(block_hash is None for block_hash in new_block_hashes.values())
                    or old_block_hashes != new_block_hashes
                ):
                    logger.warning(
                        "Block hashes missing or mismatched for request %s, "
                        "dropping its remaining store operations",
                        item.request_id,
                    )
                    self._unpin(gpu_block_ids)
                    break
                valid.metadatas.append((metadata, old_block_hashes))
                valid_block_ids.extend(gpu_block_ids)
            if held:
                # Held ops the drain budget left buffered keep their pins.
                self._held_block_ids[item.request_id] = held
            if not valid.metadatas:
                continue
            actions.stores_to_submit.append(
                _coalesce_store_metadata(
                    [meta for meta, _ in valid.metadatas],
                    self._group_tokens_per_block,
                    pool.null_block.block_id,
                )
            )
            self._requests.register_batch(item.request_id, valid_block_ids, valid)
        for request_id in drain.emptied_request_ids:
            if self._requests.can_end_session(request_id):
                actions.sessions_to_end.append(request_id)
                self._release_session(request_id)
        return actions

    def _rebind_tokens(
        self, metadata: LMCacheMPRequestMetadata
    ) -> LMCacheMPRequestMetadata:
        """Point one operation at its request's token ledger.

        The tracker builds a fresh buffer of the request's whole token
        sequence for every operation it produces. Under lazy offload those
        buffers are retained until the operation is submitted, so a long
        request would hold one copy per buffered operation. Each buffer is a
        prefix of the next -- vLLM only appends -- so the ledger absorbs the
        new tail and every buffered operation of the request shares it.

        The ledger is a ``bytearray`` so it can grow in place and the
        already-buffered operations see the growth.

        Args:
            metadata: The store operation as the tracker produced it.

        Returns:
            A copy whose ``op.token_bytes`` is the shared ledger. The
            caller's metadata is left untouched.
        """
        tokens = metadata.op.token_bytes
        ledger = self._token_ledgers.get(metadata.request_id)
        if ledger is None:
            ledger = bytearray(tokens)
            self._token_ledgers[metadata.request_id] = ledger
        elif len(tokens) > len(ledger):
            ledger += tokens[len(ledger) :]
        return replace(metadata, op=replace(metadata.op, token_bytes=ledger))

    def _running_blocks_allocated(self, scheduler_output: "SchedulerOutput") -> int:
        """Count the blocks this step allocated to requests already running.

        Each scheduled request's table is tracked per group on the way: a
        new or resumed request's is replaced, a running one's grows by the
        block ids this step handed it. Admissions and preempted requests
        resuming are left out of the count: the bound covers neither. A
        step's new block ids also carry null blocks and Mamba speculative
        blocks vLLM moved to the table's end, which come from no free queue.
        A block the request freed and was handed back later is missed, which
        can only hide an excess, never invent one.

        Args:
            scheduler_output: This step's schedule.

        Returns:
            The blocks allocated, over every KV-cache group.
        """
        null_block = {self._require_block_pool().null_block.block_id}
        num_groups = len(self._group_tokens_per_block)
        for new_request in scheduler_output.scheduled_new_reqs:
            self._handed_blocks[new_request.req_id] = [
                set(group) for group in new_request.block_ids
            ]
            self._table_lengths[new_request.req_id] = [
                len(group) for group in new_request.block_ids
            ]
        cached = scheduler_output.scheduled_cached_reqs
        allocated = 0
        for request_id, new_block_ids in zip(
            cached.req_ids, cached.new_block_ids, strict=True
        ):
            groups = new_block_ids or ([],) * num_groups
            if request_id in cached.resumed_req_ids:
                self._handed_blocks[request_id] = [set(group) for group in groups]
                self._table_lengths[request_id] = [len(group) for group in groups]
                continue
            # vLLM moves a request into its running queue only by scheduling
            # it new or resumed, so its table is already recorded.
            known = self._handed_blocks[request_id]
            lengths = self._table_lengths[request_id]
            for group_idx, group in enumerate(groups):
                # Most steps hand a decoding request nothing new in a group.
                if not group:
                    continue
                handed = set(group)
                allocated += len(handed - known[group_idx] - null_block)
                known[group_idx] |= handed
                lengths[group_idx] += len(group)
        return allocated

    def _running_blocks_bound(self, scheduler_output: "SchedulerOutput") -> int:
        """Bound the blocks running requests can allocate in the next step.

        Each request scheduled this step has computed its scheduled tokens
        by the next one (spec-decode rejections only lower that), and vLLM
        allocated their slots already. The next step schedules each request
        at most one chunk, or a decode token and its drafts, and all of
        them together at most ``max_step_tokens``; slots run ``lookahead``
        past what a request schedules. Per group, the most blocks those
        slots can cross bounds what vLLM allocates.

        Args:
            scheduler_output: This step's schedule.

        Returns:
            The bound, over every running request and KV-cache group.
        """
        for request_id in scheduler_output.preempted_req_ids or ():
            self._forget_progress(request_id)
        scheduled = scheduler_output.num_scheduled_tokens
        for new_request in scheduler_output.scheduled_new_reqs:
            self._num_computed[new_request.req_id] = (
                new_request.num_computed_tokens + scheduled[new_request.req_id]
            )
        cached = scheduler_output.scheduled_cached_reqs
        for request_id, num_computed in zip(
            cached.req_ids, cached.num_computed_tokens, strict=True
        ):
            self._num_computed[request_id] = num_computed + scheduled[request_id]
        running: list[tuple[int, int, int, list[int]]] = []
        for request_id, num_computed in self._num_computed.items():
            remaining = max(self._num_tokens[request_id] - num_computed, 0)
            max_scheduled = (
                min(remaining, self._max_step_tokens)
                if remaining
                else 1 + self._num_lookahead_tokens
            )
            running.append(
                (
                    num_computed,
                    max_scheduled,
                    remaining,
                    self._table_lengths[request_id],
                )
            )
        return sum(
            self._budgeted_blocks(running, group_idx)
            for group_idx, tokens_per_block in enumerate(self._group_tokens_per_block)
            if tokens_per_block > 0
        )

    def _budgeted_blocks(
        self, running: list[tuple[int, int, int, list[int]]], group_idx: int
    ) -> int:
        """Bound one group's new blocks within the next step's token budget.

        A request's first new block costs the tokens that fill the blocks
        its table holds, less the lookahead slots; each further one costs a
        whole block. The table already holds the blocks the last step's
        lookahead slots reached. A prefill chunk stopping short of the
        prompt's end ends on an aligned position, which can make the first
        crossing cost up to a whole block too. An align-mode Mamba group
        takes one state block per step, plus its extra blocks.

        The bound is the fractional knapsack over every crossing with the
        step's token budget: the cheapest blocks per token are taken first,
        and the crossing the budget ends in counts in full. Dropping the
        rule that a request's further crossings need its first one only
        raises the optimum, so the bound is at least what any schedule
        allocates; the crossing taken in part costs at most one crossing
        more, and only on a step the budget binds.

        Args:
            running: Per running request: its computed tokens, the most
                tokens the next step can schedule for it, its prompt tokens
                not yet computed (zero once decoding), and its table length
                per group.
            group_idx: The KV-cache group.

        Returns:
            The bound for this group.
        """
        tokens_per_block = self._group_tokens_per_block[group_idx]
        mamba_extra = self._mamba_groups.get(group_idx)
        lookahead = 0 if mamba_extra is not None else self._num_lookahead_tokens
        alignment = self._prefill_alignment
        # (token cost, blocks yielded, count) per kind of crossing.
        crossings: list[tuple[int, int, int]] = []
        num_further = 0
        for num_computed, max_scheduled, remaining, lengths in running:
            # Mamba tables also hold the state blocks reserved per draft
            # token, so their length says nothing about the tokens covered.
            num_allocated = (
                -(-num_computed // tokens_per_block)
                if mamba_extra is not None
                else lengths[group_idx]
            )
            reachable = (
                -(-(num_computed + max_scheduled + lookahead) // tokens_per_block)
                - num_allocated
            )
            if reachable <= 0:
                continue
            room = num_allocated * tokens_per_block - num_computed
            cost = max(1, room + 1 - lookahead)
            if remaining and alignment:
                aligned_end = -(-(num_computed + cost) // alignment) * alignment
                cost = min(aligned_end - num_computed, remaining)
                if cost > max_scheduled:
                    continue
            if mamba_extra is not None:
                crossings.append((cost, 1 + mamba_extra, 1))
                continue
            blocks = (
                -(-(num_computed + cost + lookahead) // tokens_per_block)
                - num_allocated
            )
            crossings.append((cost, blocks, 1))
            num_further += reachable - blocks
        crossings.append((tokens_per_block, 1, num_further))
        budget = self._max_step_tokens
        bound = 0
        for cost, blocks, count in sorted(
            crossings, key=lambda item: item[0] / item[1]
        ):
            taken = min(count, budget // cost)
            bound += taken * blocks
            budget -= taken * cost
            if taken < count:
                bound += -(-blocks * budget // cost)
                break
        return bound

    def _chunk_tokens(self, remaining: int) -> int:
        """Bound the slots one scheduled chunk of a request allocates.

        Args:
            remaining: The request's tokens not yet computed. A full hit
                leaves none, and vLLM then recomputes the last token.

        Returns:
            One step's worth of those tokens, plus the lookahead slots.
        """
        chunk = min(max(remaining, 1), self._max_step_tokens)
        return chunk + self._num_lookahead_tokens

    def _span_blocks(self, num_tokens: int, first_allocation: bool = True) -> int:
        """Bound the blocks an allocation of ``num_tokens`` slots takes.

        Args:
            num_tokens: The slots allocated, starting anywhere in a block.
            first_allocation: Whether the request holds no blocks yet. An
                asynchronous load's admission allocated for its hit, so its
                resumed allocation is not the first.

        Returns:
            Per group, summed over the groups: the blocks the slots fill, a
            partial one at the start, one for a partial hit's copy-on-write
            block or a Mamba checkpoint, and the Mamba state blocks a
            request's first allocation reserves, one per draft token. An
            align-mode Mamba group nulls the blocks the slots fill but its
            state block, so its allocation is capped whatever their number,
            and only a first allocation's cap adds the draft-token blocks.
        """
        extra = 2 + self._num_lookahead_tokens
        mamba_draft_blocks = self._num_lookahead_tokens if first_allocation else 0
        return sum(
            1 + self._mamba_groups[group_idx] + mamba_draft_blocks
            if group_idx in self._mamba_groups
            else -(-num_tokens // tokens_per_block) + extra
            for group_idx, tokens_per_block in enumerate(self._group_tokens_per_block)
            if tokens_per_block > 0
        )

    def _admission_bound(self, max_model_len: int) -> int:
        """Bound what :meth:`on_admission` protects for any one admission.

        An admission's local hit plus the slots it allocates hold at most
        ``max_model_len`` tokens, plus the recomputed last token of a full
        hit and the lookahead slots. Splitting them between the hit and the
        allocation costs a group at most one block more than allocating them
        all: two partial blocks instead of one. An align-mode Mamba group's
        hit is at most its local and its external state block.

        Args:
            max_model_len: The most tokens one request holds.

        Returns:
            The bound, in blocks summed over the cacheable groups.
        """
        split_blocks = sum(
            2 if group_idx in self._mamba_groups else 1
            for group_idx, tokens_per_block in enumerate(self._group_tokens_per_block)
            if tokens_per_block > 0
        )
        return (
            self._span_blocks(max_model_len + 1 + self._num_lookahead_tokens)
            + split_blocks
        )

    def _protect(self, num_blocks: int) -> None:
        """Move or pin the buffered blocks an allocation would recycle.

        Args:
            num_blocks: Upper bound on the blocks the allocation consumes.
        """
        protection = self._require_policy().protect(num_blocks)
        self._move_to_tail(protection.to_tail)
        self._pin_holds(protection.holds, release_to_head=True)

    def _move_to_tail(self, items: list[PendingStoreItem]) -> None:
        """Move buffered blocks to the free-queue tail, unpinned.

        A cached block is freed to the tail with its hash intact; a block a
        request still owns stays out of the queue.

        Args:
            items: The operations whose blocks move, in prefix order.
        """
        pool = self._require_block_pool()
        for item in items:
            for _, block_hashes in item.metadatas:
                blocks = [pool.blocks[block_id] for block_id in block_hashes]
                pool.touch(blocks)
                pool.free_blocks(blocks)

    def _forget_progress(self, request_id: str) -> None:
        """Stop counting a request toward the running requests' bound.

        Args:
            request_id: A request that finished or lost its blocks.
        """
        self._num_tokens.pop(request_id, None)
        self._num_computed.pop(request_id, None)
        self._load_remaining.pop(request_id, None)
        self._resuming.pop(request_id, None)
        self._handed_blocks.pop(request_id, None)
        self._table_lengths.pop(request_id, None)

    def _pin_holds(
        self, holds: list[PendingStoreItem], *, release_to_head: bool
    ) -> None:
        """Pin the blocks of operations the policy just marked held.

        The policy checked these snapshots in the same call, and nothing
        allocates between its check and this pin.

        Args:
            holds: The policy's new holds, one entry per request.
            release_to_head: Whether the unpin returns blocks pinned while
                free to the free-queue head. Only for holds the danger
                window met near the head; a hold taken to store a chunk
                eager offload would have stored (a finished request's tail,
                a lookup flush) can sit anywhere in the queue, and at the
                head the next allocation would evict a block vLLM still
                serves hits from, such as a Mamba state block vLLM freed
                before the request finished.
        """
        for hold in holds:
            held = self._held_block_ids.setdefault(hold.request_id, set())
            for _, block_hashes in hold.metadatas:
                self._pin(list(block_hashes), release_to_head=release_to_head)
                held.update(block_hashes)

    def _release_holds(self, request_id: str) -> None:
        """Unpin the blocks of a request's held operations.

        The policy releases holds it dropped and squeeze holds it no longer
        needs; either way the operations stay buffered or are gone.

        Args:
            request_id: The request whose holds end.
        """
        self._unpin(list(self._held_block_ids.pop(request_id, set())))

    def _release_superseded(self, superseded: PendingStoreItem) -> None:
        """Unpin the held blocks of operations a newer offer superseded.

        Args:
            superseded: One request's operations the policy dropped for a
                newer offer of the same tokens.
        """
        held = self._held_block_ids.get(superseded.request_id)
        if not held:
            return
        released = [
            block_id
            for _, block_hashes in superseded.metadatas
            for block_id in block_hashes
            if block_id in held
        ]
        held.difference_update(released)
        self._unpin(released)

    def _hit_tokens(self, request_id: str) -> int:
        """Return the length of vLLM's hit over a request's chain.

        vLLM's own read-only ``find_longest_cache_hit``, with its rules for
        what a later request extending the chain could hit: the
        speculative-decoding drop of the last block and, in a hybrid model,
        a cached state at the hit's end.

        Args:
            request_id: A request that arrived and whose session has not
                ended.

        Returns:
            The hit length in tokens.
        """
        chain = self._block_hash_chains[request_id]
        _, hit_tokens, _ = self._require_coordinator().find_longest_cache_hit(
            chain, len(chain) * self._require_block_pool().hash_block_size
        )
        return hit_tokens

    def _pin(self, block_ids: list[int], *, release_to_head: bool = True) -> None:
        """Pin blocks, remembering which ones were waiting for eviction.

        Args:
            block_ids: The blocks to pin, in eviction order.
            release_to_head: Whether :meth:`_unpin` returns the ones that
                were free to the queue head. False leaves them where
                ``BlockPool.free_blocks`` puts them, the tail.
        """
        pool = self._require_block_pool()
        blocks = [pool.blocks[block_id] for block_id in block_ids]
        if release_to_head:
            self._pinned_while_free.update(
                block_id
                for block_id, block in zip(block_ids, blocks, strict=True)
                if block.ref_cnt == 0
            )
        pool.touch(blocks)

    def _unpin(self, block_ids: list[int]) -> None:
        """Release pins, returning formerly free blocks to the queue head.

        ``BlockPool.free_blocks`` appends a cached block to the tail as if
        just used; a block the drain or a protection pinned out of the free
        queue was near the head, so the head is closer to its place in
        eviction order. Blocks pinned with ``release_to_head=False`` stay
        at the tail.
        A block another request still owns stays wherever vLLM puts it, and
        so does one a buffered operation still covers: a request sharing the
        prefix buffered it too, and at the head it would sit inside the danger
        window after the drain walked the queue, so the next allocation would
        recycle it unseen.

        Args:
            block_ids: The pinned blocks to release, in eviction order.
        """
        pool = self._require_block_pool()
        blocks = [pool.blocks[block_id] for block_id in block_ids]
        pool.free_blocks(blocks)
        was_free = [
            (block_id, block)
            for block_id, block in zip(block_ids, blocks, strict=True)
            if block_id in self._pinned_while_free
        ]
        if not was_free:
            return
        self._pinned_while_free.difference_update(block_id for block_id, _ in was_free)
        buffered = self._require_policy().buffered_block_ids()
        returned = [
            block
            for block_id, block in was_free
            if block.ref_cnt == 0 and block_id not in buffered
        ]
        for block in returned:
            pool.free_block_queue.remove(block)
        pool.free_block_queue.prepend_n(returned)

    def _release_session(self, request_id: str) -> None:
        """Clear policy and registry state for a settled request.

        Args:
            request_id: The request whose session the caller is ending.
        """
        self._require_policy().release_request(request_id)
        self._requests.session_ended(request_id)
        self._token_ledgers.pop(request_id, None)
        self._block_hash_chains.pop(request_id, None)
        self._unindex_offered_chunks(request_id)

    def _chunk_hash_index(self, num_chunks: int) -> int:
        """Index in a block-hash chain of the hash ending a chunk prefix.

        Args:
            num_chunks: Chunks in the prefix, at least one.

        Returns:
            The index of the hash of the block holding the prefix's last
            token, the one that identifies the prefix.
        """
        return (
            num_chunks * self._chunk_size // self._require_block_pool().hash_block_size
            - 1
        )

    def _index_offered_chunks(self, request_id: str, start: int, end: int) -> None:
        """Record a request as offering the whole chunks ending in a range.

        Args:
            request_id: The request whose op was just offered.
            start: First token of the op.
            end: Token the op ends at. Chunks past the request's hash chain,
                which hashes full blocks only, are left out.
        """
        chain = self._block_hash_chains[request_id]
        offered = self._offered_chunks.setdefault(request_id, set())
        num_chunks = start // self._chunk_size + 1
        while num_chunks * self._chunk_size <= end and (
            index := self._chunk_hash_index(num_chunks)
        ) < len(chain):
            self._chunk_offerers.setdefault(chain[index], set()).add(request_id)
            offered.add(chain[index])
            num_chunks += 1

    def _unindex_offered_chunks(self, request_id: str) -> None:
        """Forget the chunks a request offered.

        Args:
            request_id: A request whose ops were dropped or whose session
                ended.
        """
        for chunk_hash in self._offered_chunks.pop(request_id, set()):
            offerers = self._chunk_offerers[chunk_hash]
            offerers.discard(request_id)
            if not offerers:
                del self._chunk_offerers[chunk_hash]

    def _flush_offered_prefix(self, request_id: str, start: int) -> None:
        """Hold the buffered ops covering a request's prefix past ``start``.

        Args:
            request_id: The waiting request.
            start: Its local prefix hit, in tokens.
        """
        chain = self._block_hash_chains[request_id]
        ends: dict[str, int] = {}
        num_chunks = start // self._chunk_size + 1
        while (index := self._chunk_hash_index(num_chunks)) < len(chain):
            for owner in self._chunk_offerers.get(chain[index], ()):
                if owner != request_id:
                    ends[owner] = num_chunks * self._chunk_size
            num_chunks += 1
        if not ends:
            return
        flush = self._require_policy().flush_for_lookup(start, ends)
        if not flush.ends:
            return
        self._pin_holds(flush.holds, release_to_head=False)
        self._lookup_flushes[request_id] = (start, flush.ends)

    def _forget_lookup(self, request_id: str) -> None:
        """Forget a request's lookup flush, so a new generation checks again.

        Args:
            request_id: A request that finished, was reset, or whose id a
                new request took over.
        """
        self._lookup_checked.discard(request_id)
        self._lookup_flushes.pop(request_id, None)

    def _require_policy(self) -> OffloadPolicy:
        """Return the bound policy or reject an invalid lifecycle call.

        Returns:
            The policy built at bind time.

        Raises:
            ValueError: If no GPU block pool has been bound yet.
        """
        if self._policy is None:
            raise ValueError("lazy offload GPU block pool is not bound")
        return self._policy

    def _require_coordinator(self) -> "KVCacheCoordinator":
        """Return the bound coordinator or reject a lookup of vLLM's hit.

        Returns:
            The coordinator :meth:`bind_cache_coordinator` bound.

        Raises:
            ValueError: If no coordinator has been bound yet.
        """
        if self._coordinator is None:
            raise ValueError("lazy offload KV cache coordinator is not bound")
        return self._coordinator

    def _require_block_pool(self) -> "BlockPool":
        """Return the bound block pool or reject an invalid lifecycle call.

        Returns:
            The scheduler's GPU block pool.

        Raises:
            ValueError: If no GPU block pool has been bound yet.
        """
        if self._gpu_block_pool is None:
            raise ValueError("lazy offload GPU block pool is not bound")
        return self._gpu_block_pool
