# SPDX-License-Identifier: Apache-2.0
"""LazyOffloadManager admission of store operations on hybrid block tables."""

# Standard
from dataclasses import replace
from types import SimpleNamespace
from typing import cast
from unittest.mock import patch

# Third Party
import pytest
import torch

pytest.importorskip("vllm", reason="the manager reads a vLLM BlockPool")

# Third Party
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_coordinator import (  # noqa: E402
    KVCacheCoordinator,
    get_kv_cache_coordinator,
)
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    KVCacheBlock,
    make_block_hash_with_group_id,
)
from vllm.v1.kv_cache_interface import (  # noqa: E402
    FullAttentionSpec,
    KVCacheConfig,
    KVCacheGroupSpec,
    MambaSpec,
)

# First Party
from lmcache.integration.vllm import lazy_offload_policy  # noqa: E402
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadManager,
)
from lmcache.integration.vllm.lazy_offload_policy import (  # noqa: E402
    eviction_aware,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)
from lmcache.v1.multiprocess.custom_types import (  # noqa: E402
    STORE_DUE_HINT,
)
from lmcache.v1.multiprocess.token_codec import pack_token_ids  # noqa: E402

TOKENS_PER_BLOCK = 4
# Large enough that every admission here is scheduled in one chunk.
MAX_STEP_TOKENS = 1024
# One drain stores at most this many buffered ops.
BUDGET = eviction_aware.MAX_DRAIN_PER_STEP
# The free-queue head every drain keeps clear for one admission, with the
# test manager's max_model_len of one block and no draft tokens. Per group:
# the five slots of that block plus a recomputed token span two blocks, plus
# a partial block at the start and a copy-on-write block; plus one block per
# group for a split between a hit and an allocation. Admissions here are
# longer on purpose: they reach past the reserve, as a step's second
# admission does.
ADMISSION_RESERVE = 2 * (2 + 2) + 2
# The same over two align-mode Mamba groups: per group, the state block, plus
# a local and an external hit state block.
MAMBA_RESERVE = 2 * (1 + 2)


class _NoReceipts:
    """Receipts sink that never confirms a store."""

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        return False


def _step(
    scheduled: dict[str, int] | None = None,
    preempted: set[str] | None = None,
    tables: dict[str, tuple[list[int], ...]] | None = None,
) -> SimpleNamespace:
    """A scheduler step running newly admitted requests from token zero.

    Args:
        scheduled: Request id -> tokens scheduled. None schedules one token
            of a request the manager never saw admitted, so no running
            request bounds the danger window.
        preempted: Requests the step preempted.
        tables: Request id -> its block table per group, for requests
            whose table differs from the one holding just their tokens.
    """
    return SimpleNamespace(
        total_num_scheduled_tokens=sum((scheduled or {}).values()) or 1,
        num_scheduled_tokens=dict(scheduled or {}),
        scheduled_new_reqs=[
            SimpleNamespace(
                req_id=request_id,
                num_computed_tokens=0,
                # A table holding the scheduled tokens' blocks, null ones:
                # the null block is block 0 of a BlockPool.
                block_ids=(tables or {}).get(
                    request_id, ([0] * -(-num_tokens // TOKENS_PER_BLOCK),) * 2
                ),
            )
            for request_id, num_tokens in (scheduled or {}).items()
        ],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[],
            num_computed_tokens=[],
            new_block_ids=[],
            resumed_req_ids=set(),
        ),
        finished_req_ids=set(),
        preempted_req_ids=preempted,
    )


class _EmptyPrefixCache:
    """A two-group KV cache coordinator whose GPU prefix cache serves no hit."""

    single_type_managers = (
        SimpleNamespace(block_size=TOKENS_PER_BLOCK),
        SimpleNamespace(block_size=TOKENS_PER_BLOCK),
    )

    def find_longest_cache_hit(
        self, block_hashes: list[BlockHash], max_cache_hit_length: int
    ) -> tuple[tuple[list, ...], int, int]:
        return ([], []), 0, 0


def _manager(
    pool: BlockPool,
    configs: dict | None = None,
    receipts: object | None = None,
    max_step_tokens: int = MAX_STEP_TOKENS,
    mamba_groups: dict[int, int] | None = None,
    prefill_alignment: int = 0,
    num_lookahead_tokens: int = 0,
    max_model_len: int = TOKENS_PER_BLOCK,
) -> LazyOffloadManager:
    """Build a two-group manager bound to ``pool``.

    Request "req" has arrived with no hash chain yet, under a prefix cache
    that serves nothing.
    """
    manager = LazyOffloadManager(
        configs or {},
        [TOKENS_PER_BLOCK, TOKENS_PER_BLOCK],
        receipts or _NoReceipts(),
        max_step_tokens=max_step_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        max_model_len=max_model_len,
        chunk_size=TOKENS_PER_BLOCK,
        mamba_groups=mamba_groups,
        prefill_alignment=prefill_alignment,
    )
    manager.bind_block_pool(pool)
    manager.bind_cache_coordinator(cast(KVCacheCoordinator, _EmptyPrefixCache()))
    manager.on_request_arrived("req", [])
    return manager


def _store(block_ids: list[list[int]]) -> LMCacheMPRequestMetadata:
    n_tokens = TOKENS_PER_BLOCK * len(block_ids[0])
    return LMCacheMPRequestMetadata(
        request_id="req",
        direction="STORE",
        op=LoadStoreOp(
            token_bytes=pack_token_ids(range(n_tokens)),
            block_ids=block_ids,
            start=0,
            end=n_tokens,
        ),
    )


def _hashed_blocks(pool: BlockPool, n: int) -> list[int]:
    blocks = pool.get_new_blocks(n)
    for i, block in enumerate(blocks):
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(bytes([i + 1]) * 32), 0)
        )
    return [block.block_id for block in blocks]


def _free_to_head(pool: BlockPool, block_ids: list[int]) -> None:
    """Free the in-use blocks ``block_ids`` and move them to the queue head.

    vLLM queues a freed cached block at the tail, and allocations bring it
    to the head. Here the blocks ahead of it are taken and freed again
    behind it, hashed so that vLLM queues them at the tail too: the next
    drain finds the blocks within the admission reserve, with free blocks
    behind them. A buffered op on a block ahead of them loses its data.
    """
    pool.free_blocks([pool.blocks[block_id] for block_id in block_ids])
    queue = [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
    ahead = pool.get_new_blocks(min(queue.index(block_id) for block_id in block_ids))
    for i, block in enumerate(ahead):
        if block.block_hash is None:
            block.set_block_hash(
                make_block_hash_with_group_id(
                    BlockHash(b"\xfe" * 30 + i.to_bytes(2, "big")), 0
                )
            )
    pool.free_blocks(ahead)


def _flush(manager: LazyOffloadManager, request_id: str = "req") -> SimpleNamespace:
    """Finish ``request_id`` and drain once.

    Its buffered ops past what vLLM's prefix cache serves it (all of them
    under ``_EmptyPrefixCache``) are held and stored by the drain, unless
    their blocks were recycled.

    Returns:
        The drain's actions.
    """
    manager.on_request_finished(request_id)
    return manager.on_scheduler_step(_step())


def test_null_blocks_do_not_block_the_store() -> None:
    """A mamba group's null blocks (align mode) leave the store to its live blocks."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    attention = _hashed_blocks(pool, 2)
    state = _hashed_blocks(pool, 1)
    null = pool.null_block.block_id
    manager = _manager(pool)

    manager.add_store_candidate(_store([attention, [null, state[0]]]))
    _free_to_head(pool, attention + state)
    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1
    stored = actions.stores_to_submit[0]
    assert stored.request_id == "req"
    assert stored.op.block_ids == [attention, [null, state[0]]]


def test_hash_less_block_is_not_protected_and_keeps_the_chain() -> None:
    """A block without a prefix-cache hash is never a vLLM victim.

    It is stored as the null block, and the request's later ops are still
    buffered and stored: the hash-less block does not break the chain.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    attention = _hashed_blocks(pool, 2)
    # A step-end recurrent state vLLM does not keep: allocated, never hashed.
    dropped_state = pool.get_new_blocks(1)[0].block_id
    null = pool.null_block.block_id
    manager = _manager(pool)
    whole = _store([attention, [dropped_state, null]])
    first = replace(
        whole,
        op=replace(
            whole.op, block_ids=[attention[:1], [dropped_state]], end=TOKENS_PER_BLOCK
        ),
    )
    second = replace(
        whole,
        op=replace(whole.op, block_ids=[attention[1:], [null]], start=TOKENS_PER_BLOCK),
    )

    manager.add_store_candidate(first)
    manager.add_store_candidate(second)
    _free_to_head(pool, attention)
    stores = manager.on_scheduler_step(_step()).stores_to_submit

    assert len(stores) == 1
    assert stores[0].op.start == 0
    assert stores[0].op.end == 2 * TOKENS_PER_BLOCK
    assert stores[0].op.block_ids == [attention, [null, null]]
    # The caller's op is left as the tracker built it.
    assert first.op.block_ids == [attention[:1], [dropped_state]]


def test_emitted_store_is_due() -> None:
    """Every store lazy offload emits is marked due, so L1 must not skip a
    chunk of it for lack of room."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    attention = _hashed_blocks(pool, 1)
    manager = _manager(pool)
    manager.add_store_candidate(_store([attention, [pool.null_block.block_id]]))
    _free_to_head(pool, attention)

    [store] = manager.on_scheduler_step(_step()).stores_to_submit

    assert (store.request_configs or {}).get(STORE_DUE_HINT) is True


@pytest.mark.parametrize(
    ("num_tokens", "stored"),
    [(2 * TOKENS_PER_BLOCK - 1, False), (2 * TOKENS_PER_BLOCK, True)],
    ids=["inside-block", "at-boundary"],
)
def test_decode_crossing_a_block_boundary_deepens_the_window(
    num_tokens: int, stored: bool
) -> None:
    """Only a running request that can need a new block puts one at risk.

    The request has computed its whole prompt, so its next step decodes one
    token. With a full last block that token takes a new block per group,
    which can be the buffered one at the free-queue head; with room left it
    takes none, and the block stays buffered.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    manager = _manager(pool, configs={})
    manager.on_admission("running", num_tokens, 0, 0, load_async=False)
    block = ids[ADMISSION_RESERVE]
    manager.add_store_candidate(_store([[block], [pool.null_block.block_id]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(_step({"running": num_tokens}))

    assert bool(actions.stores_to_submit) is stored


class _Receipts:
    """Every receipt settles its batch, as with a single worker."""

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        return True


def _op(start: int, block_ids: list[list[int]]) -> LMCacheMPRequestMetadata:
    end = start + TOKENS_PER_BLOCK * len(block_ids[0])
    return LMCacheMPRequestMetadata(
        request_id="req",
        direction="STORE",
        op=LoadStoreOp(
            token_bytes=pack_token_ids(range(end)),
            block_ids=block_ids,
            start=start,
            end=end,
        ),
    )


def test_coalesced_store_keeps_the_request_configs() -> None:
    """Ops of one request stored in one drain carry the request's configs."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    null = pool.null_block.block_id
    ids = _hashed_blocks(pool, 2)
    configs = {"lmcache.tag.tenant": "a"}
    manager = _manager(pool)
    for i, block in enumerate(ids):
        op = _op(i * TOKENS_PER_BLOCK, [[block], [null]])
        op.request_configs = configs
        manager.add_store_candidate(op)
    _free_to_head(pool, ids)

    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1
    stored = actions.stores_to_submit[0]
    assert stored.op.block_ids[0] == ids
    assert stored.request_configs == {**configs, STORE_DUE_HINT: True}


def _blocked_with_freed_successor(
    pool: BlockPool,
) -> tuple[LazyOffloadManager, int]:
    """A request whose first store is in flight and whose next op is due.

    The next op's block is free at the queue head, where a running request
    that has just filled its last block can recycle it next step.

    Returns:
        The manager after that step, and the next op's block id.
    """
    null = pool.null_block.block_id
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    first, second = ids[0], ids[1]
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[first], [null]]))
    # The only free block, so in reach.
    pool.free_blocks([pool.blocks[first]])
    assert len(manager.on_scheduler_step(_step()).stores_to_submit) == 1
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    manager.add_store_candidate(_op(TOKENS_PER_BLOCK, [[second], [null]]))
    # Like a mamba state retired mid-request: free while the request runs.
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[1:]])
    actions = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    assert actions.stores_to_submit == []
    return manager, second


def test_blocked_request_pins_only_blocks_in_reach() -> None:
    """An in-flight store leaves the request's other free blocks unpinned.

    No request runs, so only an admission can allocate before the next
    drain, and the next op's block sits past its reserve: pinning it would
    only cost a free block.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    null = pool.null_block.block_id
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[ids[0]], [null]]))
    # The only free block, so in reach.
    pool.free_blocks([pool.blocks[ids[0]]])
    assert len(manager.on_scheduler_step(_step()).stores_to_submit) == 1
    next_block = ids[1 + ADMISSION_RESERVE]
    manager.add_store_candidate(_op(TOKENS_PER_BLOCK, [[next_block], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[1:]])
    num_free = pool.get_num_free_blocks()

    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    assert pool.get_num_free_blocks() == num_free


def test_freed_block_of_blocked_request_survives_until_its_batch_settles() -> None:
    """A slow in-flight store does not cost the request its next op.

    The whole free queue is allocated while the first batch is in flight;
    the next op is stored once the receipt arrives, and every pin is
    released by the time its own receipt does.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, second = _blocked_with_freed_successor(pool)

    pool.get_new_blocks(pool.get_num_free_blocks())
    manager.on_store_results(set(), {"req": 1})
    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1
    assert actions.stores_to_submit[0].op.block_ids[0] == [second]
    manager.on_store_results(set(), {"req": 1})
    assert pool.blocks[second].ref_cnt == 0


# An admission of this many tokens allocates at most this many blocks over
# the manager's two groups: per group, a partial block at each end and one
# for a copy-on-write or checkpoint block (no draft tokens here). It reaches
# past the admission reserve.
ADMISSION_TOKENS = 8 * TOKENS_PER_BLOCK
ADMISSION_BLOCKS = 2 * (8 + 2)


def _buffered_block_in_queue(
    pool: BlockPool,
    position: int,
    running_tokens: int = 0,
    mamba_groups: dict[int, int] | None = None,
    num_lookahead_tokens: int = 0,
) -> tuple[LazyOffloadManager, int, list[int]]:
    """A buffered block at ``position`` of a free queue of cached blocks.

    No request runs, so the danger depth is the admission reserve, and the
    drain leaves a block past it buffered.

    Args:
        pool: The block pool, with every block free.
        position: Where the buffered block sits in the free queue.
        running_tokens: When set, a request ``"running"`` of that many
            tokens is admitted first, to be scheduled by the caller.
        mamba_groups: The manager's align-mode Mamba groups.
        num_lookahead_tokens: The manager's draft tokens per step.

    Returns:
        The manager, the buffered block's id, and every other free block.
    """
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    block = ids[position]
    manager = _manager(
        pool,
        configs={},
        receipts=_Receipts(),
        mamba_groups=mamba_groups,
        num_lookahead_tokens=num_lookahead_tokens,
    )
    if running_tokens:
        manager.on_admission("running", running_tokens, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[block], [pool.null_block.block_id]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])
    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    return manager, block, [block_id for block_id in ids if block_id != block]


@pytest.mark.parametrize("admitted", [False, True], ids=["no-admission", "admission"])
def test_admission_keeps_buffered_block_its_allocation_would_recycle(
    admitted: bool,
) -> None:
    """An admission recycling past a buffered block does not cost a store.

    The block sits past the admission reserve, where an admission longer
    than ``max_model_len`` (as a step's second admission) still reaches.
    Only the admission notice can save the block, and the notice must not
    shrink the free queue the admission allocates from. Moved to the tail,
    the block survives the allocation, still buffered: finishing its
    request stores it intact.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, _, _ = _buffered_block_in_queue(pool, position=ADMISSION_RESERVE + 5)
    num_free = pool.get_num_free_blocks()

    if admitted:
        manager.on_admission("new", ADMISSION_TOKENS, 0, 0, load_async=False)
        assert pool.get_num_free_blocks() == num_free
    pool.get_new_blocks(ADMISSION_BLOCKS)
    actions = _flush(manager, "req")

    assert bool(actions.stores_to_submit) is admitted


def test_protected_op_waits_at_the_tail_until_the_head_reaches_it() -> None:
    """A protected op is stored when it is about to be recycled, not before.

    The admission notice moves the buffered block to the tail. The drain
    leaves it buffered there, so one large protection is not stored in a
    single burst; once allocations bring it within the reserve of the
    head, the drain stores it intact.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, _, _ = _buffered_block_in_queue(pool, position=ADMISSION_RESERVE + 5)
    manager.on_admission("new", ADMISSION_TOKENS, 0, 0, load_async=False)

    waiting = manager.on_scheduler_step(_step()).stores_to_submit
    pool.get_new_blocks(pool.get_num_free_blocks() - ADMISSION_RESERVE)
    stored = manager.on_scheduler_step(_step()).stores_to_submit

    assert waiting == []
    assert stored != []


def test_one_admission_after_a_drain_moves_no_buffered_block() -> None:
    """The drain keeps one ``max_model_len`` admission clear of buffered blocks.

    The buffered block sits right past the reserve, and the admission is as
    long as the model allows: it neither moves nor pins the block, and its
    allocation leaves the block where it was, buffered and intact.
    """
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool, configs={}, receipts=_Receipts(), max_model_len=ADMISSION_TOKENS
    )
    block = ids[manager.admission_blocks]
    manager.add_store_candidate(_op(0, [[block], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])
    assert manager.on_scheduler_step(_step()).stores_to_submit == []

    manager.on_admission("new", ADMISSION_TOKENS, 0, 0, load_async=False)
    allocated = pool.get_new_blocks(ADMISSION_BLOCKS)
    actions = _flush(manager, "req")

    assert block not in {allocated_block.block_id for allocated_block in allocated}
    assert [store.request_id for store in actions.stores_to_submit] == ["req"]


@pytest.mark.parametrize(
    ("mamba_groups", "pinned"),
    [(None, True), ({0: 0, 1: 0}, False)],
    ids=["attention", "mamba"],
)
def test_long_admission_of_mamba_groups_takes_a_few_blocks(
    mamba_groups: dict[int, int] | None, pinned: bool
) -> None:
    """An align-mode Mamba group allocates its state block, not the prompt.

    The admission's ten blocks of prompt per group would reach the buffered
    block at the queue tail, which a squeeze pins. Mamba groups null all
    but the state block, so the allocation stays at the head and the block
    is left in the queue, buffered.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    num_free = pool.get_num_free_blocks()
    manager, _, _ = _buffered_block_in_queue(
        pool, position=num_free - 1, mamba_groups=mamba_groups
    )

    manager.on_admission("new", 10 * TOKENS_PER_BLOCK, 0, 0, load_async=False)

    assert (pool.get_num_free_blocks() < num_free) is pinned


def test_external_hit_of_mamba_groups_takes_a_state_block_of_its_own() -> None:
    """vLLM gives an external hit a Mamba state block before allocating.

    Each Mamba group takes one block for the external hit and one state
    block for the chunk, four in all. An earlier admission of the step has
    taken the reserve, which leaves the buffered block third in the queue,
    in reach. Saved by the admission notice, it is still intact when its
    request's finish stores it.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, _, _ = _buffered_block_in_queue(
        pool, position=MAMBA_RESERVE + 2, mamba_groups={0: 0, 1: 0}
    )
    pool.get_new_blocks(MAMBA_RESERVE)

    manager.on_admission(
        "new", 20 * TOKENS_PER_BLOCK, 0, 10 * TOKENS_PER_BLOCK, load_async=False
    )
    pool.get_new_blocks(4)
    actions = _flush(manager, "req")

    assert actions.stores_to_submit != []


def test_resumed_load_of_mamba_groups_reserves_no_draft_blocks_again() -> None:
    """A resumed load's Mamba state takes one block per group, no drafts.

    The admission's allocation for the hit reserved the draft-token state
    blocks, so vLLM caps the resumed allocation at the state block alone:
    two blocks in all, which stop short of the buffered block third in the
    queue. Counting the drafts again would reach the tail and pin it.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    num_free = pool.get_num_free_blocks()
    manager, _, _ = _buffered_block_in_queue(
        pool,
        position=num_free - 1,
        mamba_groups={0: 0, 1: 0},
        num_lookahead_tokens=3,
    )
    manager.on_admission(
        "new", TOKENS_PER_BLOCK + ADMISSION_TOKENS, 0, TOKENS_PER_BLOCK, load_async=True
    )
    pool.get_new_blocks(num_free - 3)
    num_free = pool.get_num_free_blocks()

    manager.on_loads_finished({"new"})

    assert pool.get_num_free_blocks() == num_free


def test_squeezed_admission_pins_rather_than_loses_the_store() -> None:
    """An admission that needs every free block cannot take a buffered one.

    The tail is no refuge when the allocation reaches it, so the block is
    pinned: vLLM gets the rest of the queue, and the next drain stores it.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, _, others = _buffered_block_in_queue(pool, position=ADMISSION_RESERVE + 5)

    manager.on_admission(
        "new", TOKENS_PER_BLOCK * (len(others) + 1), 0, 0, load_async=False
    )
    pool.get_new_blocks(pool.get_num_free_blocks())
    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1


def test_step_scheduling_nothing_stores_what_a_squeezed_admission_pinned() -> None:
    """A pin that defers the admission is stored without a forward pass.

    With the buffered block pinned the admission does not fit, so if no
    request runs, the next steps schedule nothing. Waiting for a step that
    schedules tokens would keep the pin, and the admission, waiting forever.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, _, others = _buffered_block_in_queue(pool, position=ADMISSION_RESERVE + 5)
    manager.on_admission(
        "new", TOKENS_PER_BLOCK * (len(others) + 1), 0, 0, load_async=False
    )
    nothing_scheduled = SimpleNamespace(
        total_num_scheduled_tokens=0,
        num_scheduled_tokens={},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=[], num_computed_tokens=[], new_block_ids=[], resumed_req_ids=set()
        ),
        finished_req_ids=set(),
        preempted_req_ids=None,
    )

    actions = manager.on_scheduler_step(nothing_scheduled)

    assert len(actions.stores_to_submit) == 1


@pytest.mark.parametrize("notified", [False, True], ids=["no-notice", "notice"])
def test_resumed_load_keeps_buffered_block_its_allocation_would_recycle(
    notified: bool,
) -> None:
    """A finished asynchronous load does not cost a store when it resumes.

    vLLM allocates only for the hit at admission, and the rest of the
    request's blocks once the load finishes, without asking for a hit
    again: the load-finished notice is all that can save the block.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    hit_blocks = 2 * (1 + 2)
    manager, _, _ = _buffered_block_in_queue(
        pool, position=ADMISSION_RESERVE + hit_blocks + 2
    )

    manager.on_admission(
        "new", TOKENS_PER_BLOCK + ADMISSION_TOKENS, 0, TOKENS_PER_BLOCK, load_async=True
    )
    pool.get_new_blocks(hit_blocks)
    if notified:
        manager.on_loads_finished({"new"})
    pool.get_new_blocks(ADMISSION_BLOCKS)
    actions = _flush(manager, "req")

    assert bool(actions.stores_to_submit) is notified


def test_resumed_load_left_waiting_keeps_its_protection() -> None:
    """A finished load vLLM schedules steps later still cannot cost a store.

    The load-finished notice moves the buffered block to the tail, but no
    step schedules the request yet. Another admission takes the queue head
    and a finished request's cached blocks queue up behind the block, so
    when the resumed request finally allocates, its chunk reaches the block.
    The drain in between must store it.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, _, _ = _buffered_block_in_queue(pool, position=10)
    manager.on_admission(
        "new", TOKENS_PER_BLOCK + ADMISSION_TOKENS, 0, TOKENS_PER_BLOCK, load_async=True
    )
    pool.get_new_blocks(2 * (1 + 2))
    manager.on_loads_finished({"new"})

    other_tokens = 6 * TOKENS_PER_BLOCK
    manager.on_admission("other", other_tokens, 0, 0, load_async=False)
    finished = pool.get_new_blocks(2 * (6 + 2))
    for i, block in enumerate(finished):
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(bytes([100 + i]) * 32), 0)
        )
    pool.free_blocks(finished)
    stores = manager.on_scheduler_step(_step()).stores_to_submit
    pool.get_new_blocks(ADMISSION_BLOCKS)
    stores += _flush(manager, "req").stores_to_submit

    assert stores != []


def test_store_over_the_drain_budget_survives_until_the_next_drain() -> None:
    """A due op the budget defers is not recycled while it waits.

    One request more than a drain stores has a block at the free-queue head
    and a running request is about to take two blocks; one drain stores all
    but the last op. That one is held, so the next drain stores it, intact
    though the running request has allocated and nothing presses on it.
    """
    pool = BlockPool(
        num_gpu_blocks=2 * BUDGET + 16, enable_caching=True, hash_block_size=4
    )
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, receipts=_Receipts())
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    request_ids = [f"r{i}" for i in range(BUDGET + 1)]
    for request_id, block_id in zip(request_ids, ids, strict=False):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.on_request_arrived(request_id, [])
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.get_new_blocks(2)
    second = manager.on_scheduler_step(_step())

    assert [store.request_id for store in first.stores_to_submit] == request_ids[:-1]
    assert [store.request_id for store in second.stores_to_submit] == request_ids[-1:]


@pytest.mark.parametrize(
    ("num_ops", "counts"),
    [(BUDGET, [0, 0, 0]), (2 * BUDGET + 1, [BUDGET, 1, 0])],
    ids=["one-drain", "past-one-drain"],
)
def test_backlog_the_budget_cannot_keep_up_with_starts_early(
    num_ops: int, counts: list[int]
) -> None:
    """Ops just past the window are stored early only if the budget is short.

    The ops sit right behind the blocks one admission can take, so the next
    drain's allocations reach them all. More than one drain stores cannot
    all go in that one drain, so the drains before it store a budget's worth
    each, in queue order, until the rest fits the drain that reaches them;
    as many as one drain stores stay deferred.
    """
    pool = BlockPool(
        num_gpu_blocks=num_ops + 32, enable_caching=True, hash_block_size=4
    )
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, receipts=_Receipts(), max_step_tokens=TOKENS_PER_BLOCK)
    request_ids = [f"r{i}" for i in range(num_ops)]
    # One admission reaches 10 blocks with this model length and two groups.
    for request_id, block_id in zip(request_ids, ids[10:], strict=False):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.on_request_arrived(request_id, [])
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    drains = [manager.on_scheduler_step(_step()) for _ in range(3)]

    expected: list[list[str]] = []
    for count in counts:
        start = sum(map(len, expected))
        expected.append(request_ids[start : start + count])
    assert [
        [store.request_id for store in drain.stores_to_submit] for drain in drains
    ] == expected


def test_stored_block_returns_to_queue_head() -> None:
    """Storing a free block does not refresh it in vLLM's eviction order.

    The block was next to be recycled when the store pinned it; once the
    store settles it is next again, rather than the most recently used.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    running = 2 * TOKENS_PER_BLOCK
    manager, block, _ = _buffered_block_in_queue(
        pool, position=ADMISSION_RESERVE, running_tokens=running
    )
    stores = manager.on_scheduler_step(_step({"running": running}))
    assert len(stores.stores_to_submit) == 1

    manager.on_store_results(set(), {"req": 1})

    assert pool.get_new_blocks(1)[0].block_id == block


def test_stored_block_another_request_buffered_is_not_recycled_next() -> None:
    """A shared prefix block stays out of reach after one request stores it.

    Two requests buffered the same block, behind a budget's worth of other
    requests' blocks less one at the free-queue head; one drain stores the
    first of the two but not the second. Once that store settles, the block
    is still buffered by the second, so the next allocation takes another
    block and the second request's op is stored intact.
    """
    pool = BlockPool(
        num_gpu_blocks=2 * BUDGET + 16, enable_caching=True, hash_block_size=4
    )
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, receipts=_Receipts())
    ahead = [f"r{i}" for i in range(BUDGET - 1)]
    shared = ids[len(ahead)]
    for request_id, block_id in [
        *zip(ahead, ids, strict=False),
        ("first", shared),
        ("second", shared),
    ]:
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.on_request_arrived(request_id, [])
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    first = manager.on_scheduler_step(_step())
    manager.on_store_results(set(), {"first": 1})
    allocated = pool.get_new_blocks(1)[0].block_id
    second = manager.on_scheduler_step(_step())

    assert [store.request_id for store in first.stores_to_submit] == [
        *ahead,
        "first",
    ]
    assert allocated != shared
    assert [store.request_id for store in second.stores_to_submit] == ["second"]


def test_preemption_keeps_held_blocks_pinned() -> None:
    """A preempted request's held op still protects its block.

    The block stays pinned through the reset, and the op is stored once
    the batch in flight settles.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, second = _blocked_with_freed_successor(pool)
    free_while_held = pool.get_num_free_blocks()

    manager.on_request_reset("req")
    manager.on_request_arrived("req", [])
    pool.get_new_blocks(pool.get_num_free_blocks())
    manager.on_store_results(set(), {"req": 1})
    (store,) = manager.on_scheduler_step(_step()).stores_to_submit

    assert free_while_held > 0
    assert store.op.block_ids[0] == [second]


def test_window_reaches_past_blocks_the_drain_takes_out() -> None:
    """Pinning a due block does not bring the next one into reach unseen.

    The running request can take two blocks next step. The first buffered
    block is at the head; once it is pinned, the second, two places behind
    it, is among the first two free blocks, so it is stored too.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, configs={}, receipts=_Receipts())
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    for request_id, block_id in (("first", ids[0]), ("second", ids[2])):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.on_request_arrived(request_id, [])
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.get_new_blocks(2)
    second = manager.on_scheduler_step(_step())

    stored = first.stores_to_submit + second.stores_to_submit
    assert {store.request_id for store in stored} == {"first", "second"}


def test_preempted_request_buffer_waits_in_eviction_order() -> None:
    """Preemption alone stores nothing; the op is stored when it comes due.

    vLLM keeps the hashes of a preempted request's freed blocks, so its
    buffered ops are victims like any other.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, block, _ = _buffered_block_in_queue(pool, position=ADMISSION_RESERVE + 5)

    at_preemption = manager.on_scheduler_step(_step(preempted={"req"}))
    manager.on_request_reset("req")
    manager.on_request_arrived("req", [])
    _bring_to_head(pool, block)
    due = manager.on_scheduler_step(_step())

    assert at_preemption.stores_to_submit == []
    (store,) = due.stores_to_submit
    assert store.op.block_ids[0] == [block]


def test_restarted_request_offer_supersedes_its_buffered_op() -> None:
    """The restart's offer of the same tokens replaces the buffered op.

    One store follows, from the new offer's blocks, and the superseded op's
    hold is released.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, block, others = _buffered_block_in_queue(
        pool, position=ADMISSION_RESERVE + 5
    )
    null = pool.null_block.block_id
    manager.on_request_reset("req")
    manager.on_request_arrived("req", [])
    (recomputed,) = _hashed_blocks(pool, 1)
    manager.add_store_candidate(_op(0, [[recomputed], [null]]))
    pool.free_blocks([pool.blocks[recomputed]])
    # Every block ahead of the new one is allocated, the superseded op's
    # too; only the new op is in reach.
    _bring_to_head(pool, recomputed)
    stores = manager.on_scheduler_step(_step()).stores_to_submit

    assert block not in others
    assert [store.op.block_ids[0] for store in stores] == [[recomputed]]
    assert _counts(manager)["dropped_superseded"] == 1


def test_admission_counts_the_local_hit_it_takes_from_the_queue() -> None:
    """A local prefix hit does not pull a buffered block into the allocation.

    vLLM touches the hit blocks, taking them out of the free queue, before
    it allocates from the head: every hit block ahead of the buffered one
    brings it one place closer.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    hit = 2 * TOKENS_PER_BLOCK
    new_blocks = 2 * (2 + 2)
    block = ids[new_blocks + 1]
    manager = _manager(pool, configs={}, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[block], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    manager.on_admission("new", hit + 2 * TOKENS_PER_BLOCK, hit, 0, load_async=False)
    pool.touch([pool.blocks[block_id] for block_id in ids[:2]])
    pool.get_new_blocks(new_blocks)
    actions = _flush(manager, "req")

    assert [store.request_id for store in actions.stores_to_submit] == ["req"]


def test_token_budget_caps_the_window() -> None:
    """Prefilling requests share one step's tokens, so only so many advance.

    Four requests each need a full block's worth of tokens to cross into a
    new one, and the step has room for two: at most two crossings per
    group. A buffered block past those four is not in reach, however much
    each request could take alone.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool, configs={}, receipts=_Receipts(), max_step_tokens=2 * TOKENS_PER_BLOCK
    )
    running = [f"running-{index}" for index in range(4)]
    for request_id in running:
        manager.on_admission(request_id, 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[ADMISSION_RESERVE + 4]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step({request_id: TOKENS_PER_BLOCK + 1 for request_id in running})
    )

    assert actions.stores_to_submit == []


@pytest.mark.parametrize(
    ("position", "stored"), [(1, True), (2, False)], ids=["in-reach", "past-reach"]
)
def test_mamba_group_takes_one_block_per_step(position: int, stored: bool) -> None:
    """An align-mode Mamba group allocates one state block per step.

    The prefilling request could cross many blocks of an attention group
    in one chunk, but each Mamba group takes a single block: two groups
    reach two blocks past the admission reserve, and no further.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool, configs={}, receipts=_Receipts(), mamba_groups={0: 0, 1: 0}
    )
    manager.on_admission("running", 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[MAMBA_RESERVE + position]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(_step({"running": TOKENS_PER_BLOCK}))

    assert (actions.stores_to_submit != []) is stored


@pytest.mark.parametrize(
    ("position", "stored"), [(1, True), (2, False)], ids=["in-reach", "past-reach"]
)
def test_scratch_group_adds_no_reach(position: int, stored: bool) -> None:
    """A scratch group (``0`` tokens per block) is left out of every bound.

    It holds one block per request for the request's lifetime, so a step
    never grows it: with a scratch group between two Mamba groups, the
    window reaches the same two blocks as without it.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = LazyOffloadManager(
        {},
        [TOKENS_PER_BLOCK, 0, TOKENS_PER_BLOCK],
        _Receipts(),
        max_step_tokens=MAX_STEP_TOKENS,
        num_lookahead_tokens=0,
        max_model_len=TOKENS_PER_BLOCK,
        chunk_size=TOKENS_PER_BLOCK,
        mamba_groups={0: 0, 2: 0},
    )
    manager.bind_block_pool(pool)
    manager.on_request_arrived("req", [])
    manager.on_admission("running", 100, 0, 0, load_async=False)
    manager.add_store_candidate(
        _op(0, [[ids[MAMBA_RESERVE + position]], [null], [null]])
    )
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step({"running": TOKENS_PER_BLOCK}, tables={"running": ([null],) * 3})
    )

    assert (actions.stores_to_submit != []) is stored


@pytest.mark.parametrize(
    ("prefill_alignment", "stored"), [(0, True), (TOKENS_PER_BLOCK, False)]
)
def test_aligned_prefill_crossing_costs_a_whole_block(
    prefill_alignment: int, stored: bool
) -> None:
    """A chunk short of the prompt's end ends on an aligned position.

    Four requests sit on a block boundary, so one token each would cross
    into a new block, and the eight-token step reaches five blocks per
    group. Aligned chunks spend a whole block's tokens per crossing, so the
    step reaches two per group, short of the buffered block.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool,
        configs={},
        receipts=_Receipts(),
        max_step_tokens=2 * TOKENS_PER_BLOCK,
        prefill_alignment=prefill_alignment,
    )
    running = [f"running-{index}" for index in range(4)]
    for request_id in running:
        manager.on_admission(request_id, 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[ADMISSION_RESERVE + 4]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step({request_id: TOKENS_PER_BLOCK for request_id in running})
    )

    assert (actions.stores_to_submit != []) is stored


def test_leftover_tokens_short_of_a_crossing_reach_at_most_one_block() -> None:
    """Tokens short of a crossing buy at most one more block per group.

    Four requests each need a full block's worth of tokens to cross into a
    new one, and the step has room for two crossings and one token more.
    No schedule takes a third block per group with that token, but the
    bound takes its crossings fractionally and rounds the last one up, so
    it counts three per group: the fifth and sixth queue positions past
    the admission reserve are the slack, and the seventh is out of reach.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool,
        configs={},
        receipts=_Receipts(),
        max_step_tokens=2 * TOKENS_PER_BLOCK + 1,
    )
    running = [f"running-{index}" for index in range(4)]
    for request_id in running:
        manager.on_admission(request_id, 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[ADMISSION_RESERVE + 6]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step({request_id: TOKENS_PER_BLOCK + 1 for request_id in running})
    )

    assert actions.stores_to_submit == []


@pytest.mark.parametrize(
    ("position", "stored"), [(5, True), (6, False)], ids=["in-reach", "past-reach"]
)
def test_lookahead_slots_count_within_the_token_budget(
    position: int, stored: bool
) -> None:
    """Draft-token slots let a few scheduled tokens cross a block.

    Two prefilling requests have computed one block and, with four
    lookahead slots, hold a second. One token each puts their slots into a
    third block; the six-token step then has four tokens left, one block's
    worth, for one more crossing. That is three blocks per group: past
    the admission reserve, the sixth queue position is in reach, the
    seventh is not. Counting the scheduled tokens alone would reach two per
    group. The reserve itself spans the draft slots too, four per group.
    """
    pool = BlockPool(num_gpu_blocks=48, enable_caching=True, hash_block_size=4)
    reserve = ADMISSION_RESERVE + 2 * (1 + TOKENS_PER_BLOCK)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(
        pool,
        configs={},
        receipts=_Receipts(),
        max_step_tokens=TOKENS_PER_BLOCK + 2,
        num_lookahead_tokens=TOKENS_PER_BLOCK,
    )
    running = [f"running-{index}" for index in range(2)]
    for request_id in running:
        manager.on_admission(request_id, 100, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[reserve + position]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step(
            {request_id: TOKENS_PER_BLOCK for request_id in running},
            tables={request_id: ([null] * 2,) * 2 for request_id in running},
        )
    )

    assert (actions.stores_to_submit != []) is stored


@pytest.mark.parametrize(
    ("table_length", "stored"), [(2, True), (3, False)], ids=["filled", "ahead"]
)
def test_block_already_in_the_table_is_not_allocated_again(
    table_length: int, stored: bool
) -> None:
    """A decode token landing in a block the table holds takes no new one.

    The request has computed two blocks' worth of tokens, so its next token
    crosses into a third block, unless the table already holds it (as
    lookahead slots of an earlier step leave it). Then the step allocates
    nothing, and the buffered block right past the admission reserve is out
    of reach.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, configs={}, receipts=_Receipts())
    num_tokens = 2 * TOKENS_PER_BLOCK
    manager.on_admission("running", num_tokens, 0, 0, load_async=False)
    manager.add_store_candidate(_op(0, [[ids[ADMISSION_RESERVE]], [null]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    actions = manager.on_scheduler_step(
        _step(
            {"running": num_tokens},
            tables={"running": ([null] * table_length,) * 2},
        )
    )

    assert (actions.stores_to_submit != []) is stored


def test_deferred_op_is_pinned_when_the_next_step_can_take_the_whole_queue() -> None:
    """The tail is no refuge for a deferred op when the window reaches it.

    Only the buffered blocks and one more are free, and the running request
    can take two: past the drain budget, the last op is pinned rather than
    moved to the tail, so allocating the whole queue leaves it to store
    next drain.
    """
    pool = BlockPool(
        num_gpu_blocks=2 * BUDGET + 16, enable_caching=True, hash_block_size=4
    )
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, receipts=_Receipts())
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    request_ids = [f"r{i}" for i in range(BUDGET + 1)]
    for request_id, block_id in zip(request_ids, ids, strict=False):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.on_request_arrived(request_id, [])
        manager.add_store_candidate(store)
    pool.free_blocks(
        [pool.blocks[block_id] for block_id in ids[: len(request_ids) + 1]]
    )

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.get_new_blocks(pool.get_num_free_blocks())
    second = manager.on_scheduler_step(_step())

    assert [store.request_id for store in first.stores_to_submit] == request_ids[:-1]
    assert [store.request_id for store in second.stores_to_submit] == request_ids[-1:]


def test_squeezed_drain_pins_a_block_its_request_still_owns() -> None:
    """A window reaching the tail also covers blocks not yet in the queue.

    vLLM frees a running request's older blocks, as align-mode Mamba drops
    past states, before the next step allocates; they join a tail already
    in reach. The op on such a block is not due, but it is pinned while its
    request still owns it, so it survives the whole queue being allocated,
    and its request's finish stores it intact.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, receipts=_Receipts())
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    for request_id, block_id in (("first", ids[0]), ("owned", ids[5])):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.on_request_arrived(request_id, [])
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[:2]])

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.free_blocks([pool.blocks[ids[5]]])
    allocated = pool.get_new_blocks(pool.get_num_free_blocks())
    pool.free_blocks(allocated + [pool.blocks[block_id] for block_id in ids[6:]])
    finished = _flush(manager, "owned")

    assert [store.request_id for store in first.stores_to_submit] == ["first"]
    assert [store.request_id for store in finished.stores_to_submit] == ["owned"]


def test_lasting_squeeze_stores_what_it_pinned() -> None:
    """A squeeze that outlasts a drain makes its own pins due.

    The pinned blocks are out of the free queue, so no window can reach
    them; a squeeze they help keep up would otherwise pin the pool forever
    without storing any of it.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    null = pool.null_block.block_id
    manager = _manager(pool, receipts=_Receipts())
    manager.on_admission("running", 2 * TOKENS_PER_BLOCK, 0, 0, load_async=False)
    for request_id, block_id in (("first", ids[0]), ("owned", ids[5])):
        store = _store([[block_id], [null]])
        store.request_id = request_id
        manager.on_request_arrived(request_id, [])
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[:2]])

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.free_blocks([pool.blocks[ids[5]]])
    pool.get_new_blocks(pool.get_num_free_blocks())
    still_squeezed = manager.on_scheduler_step(_step())

    assert [store.request_id for store in first.stores_to_submit] == ["first"]
    assert [store.request_id for store in still_squeezed.stores_to_submit] == ["owned"]


def test_squeezed_admission_pins_a_block_its_request_still_owns() -> None:
    """An allocation reaching the tail takes blocks freed before it runs.

    A resumed load allocates after the running requests, which may free a
    buffered block first. The walk met no buffered block, yet reached the
    tail, so the op is pinned while its request still owns the block.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    ids = _hashed_blocks(pool, pool.get_num_free_blocks())
    manager = _manager(pool, configs={}, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[ids[0]], [pool.null_block.block_id]]))
    pool.free_blocks([pool.blocks[block_id] for block_id in ids[1:]])
    assert manager.on_scheduler_step(_step()).stores_to_submit == []

    manager.on_admission(
        "new",
        TOKENS_PER_BLOCK * (pool.get_num_free_blocks() + 1),
        0,
        0,
        load_async=False,
    )
    pool.free_blocks([pool.blocks[ids[0]]])
    pool.get_new_blocks(pool.get_num_free_blocks())
    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1


def _decode_step(
    num_computed_tokens: int, new_block_ids: tuple[list[int], ...]
) -> SimpleNamespace:
    """A step decoding ``"running"``, which was scheduled before."""
    return SimpleNamespace(
        total_num_scheduled_tokens=1,
        num_scheduled_tokens={"running": 1},
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(
            req_ids=["running"],
            num_computed_tokens=[num_computed_tokens],
            new_block_ids=[new_block_ids],
            resumed_req_ids=set(),
        ),
        finished_req_ids=set(),
        preempted_req_ids=None,
    )


def _ledger_lines(mock_info: object) -> list[str]:
    """The counter ledgers a patched ``logger.info`` received."""
    return [
        call.args[1]
        for call in mock_info.call_args_list  # type: ignore[attr-defined]
        if call.args[0] == "Lazy offload counters: %s"
    ]


def test_block_another_group_held_counts_as_allocated() -> None:
    """Allocated blocks are counted per group.

    Groups share one block pool, so a block one group of the request freed
    can come back to another group of the same request; it still comes off
    the free queue, and the ledger counts it. LMCache loggers do not
    propagate, so the module logger is patched.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, configs={})
    num_tokens = TOKENS_PER_BLOCK
    manager.on_admission("running", num_tokens, 0, 0, load_async=False)
    manager.on_scheduler_step(
        _step({"running": num_tokens}, tables={"running": ([1], [2])})
    )

    with patch.object(eviction_aware.logger, "info") as mock_info:
        manager.on_scheduler_step(_decode_step(num_tokens, ([3], [1])))

    assert " bound_allocated_blocks=2 " in _ledger_lines(mock_info)[-1]


@pytest.mark.parametrize(
    ("new_block_ids", "exceeded"),
    [(([1], [2]), False), (([1, 2], [3]), True)],
    ids=["within-bound", "over-bound"],
)
def test_allocation_over_the_bound_is_reported(
    new_block_ids: tuple[list[int], ...], exceeded: bool
) -> None:
    """A running request allocating past the danger depth is logged.

    The decoding request has just filled its last block, so the bound is
    one new block per group; a third block means the bound does not hold.
    Either way the ledger sums the bound and the allocation, whose ratio is
    the bound's slack. LMCache loggers do not propagate, so the module
    logger is patched.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, configs={})
    num_tokens = 2 * TOKENS_PER_BLOCK
    manager.on_admission("running", num_tokens, 0, 0, load_async=False)
    manager.on_scheduler_step(_step({"running": num_tokens}))

    with patch.object(eviction_aware.logger, "info") as mock_info:
        manager.on_scheduler_step(_decode_step(num_tokens, new_block_ids))

    reported = any(
        "over the previous drain's bound" in call.args[0]
        for call in mock_info.call_args_list
    )
    allocated = sum(len(group) for group in new_block_ids)
    assert reported is exceeded
    assert (
        f"bound_blocks=2 bound_allocated_blocks={allocated} "
        in _ledger_lines(mock_info)[-1]
    )


class _TwoWorkerReceipts:
    """A batch settles on its second receipt, as with two workers."""

    def __init__(self) -> None:
        self._counts: dict[str, int] = {}

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        """Add ``count`` receipts; True once the request has two."""
        self._counts[request_id] = self._counts.get(request_id, 0) + count
        if self._counts[request_id] < 2:
            return False
        del self._counts[request_id]
        return True


def test_failed_store_is_submitted_again_ahead_of_later_ops() -> None:
    """A store the worker rejected is retried with the request's next op.

    The data is still on the GPU, so the failed range is buffered again in
    front of the op that arrived while it was in flight, and the request's
    prefix chain stays whole.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    null = pool.null_block.block_id
    first, second = _hashed_blocks(pool, 2)
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[first], [null]]))
    _free_to_head(pool, [first, second])
    assert len(manager.on_scheduler_step(_step()).stores_to_submit) == 1
    manager.add_store_candidate(_op(TOKENS_PER_BLOCK, [[second], [null]]))

    manager.on_store_results({"req"}, {"req": 1})
    retried = manager.on_scheduler_step(_step()).stores_to_submit

    assert len(retried) == 1
    assert (retried[0].op.start, retried[0].op.end) == (0, 2 * TOKENS_PER_BLOCK)
    assert retried[0].op.block_ids[0] == [first, second]


def test_failed_store_waits_for_every_worker_before_its_retry() -> None:
    """One worker's failure does not resubmit while another still reads.

    The worker accepts one store per request, and the other worker may
    still be reading the pinned blocks, so the retry waits for the batch to
    settle; the settled retry leaves no pin behind.
    """
    # Smaller than the admission reserve: a buffered op on any free block is
    # due, so a retry submitted early would show.
    pool = BlockPool(
        num_gpu_blocks=ADMISSION_RESERVE, enable_caching=True, hash_block_size=4
    )
    null = pool.null_block.block_id
    (block,) = _hashed_blocks(pool, 1)
    manager = _manager(pool, receipts=_TwoWorkerReceipts())
    manager.add_store_candidate(_op(0, [[block], [null]]))
    pool.free_blocks([pool.blocks[block]])
    assert len(manager.on_scheduler_step(_step()).stores_to_submit) == 1

    manager.on_store_results({"req"}, {"req": 1})
    unsettled = manager.on_scheduler_step(_step()).stores_to_submit
    manager.on_store_results(set(), {"req": 1})
    retried = manager.on_scheduler_step(_step()).stores_to_submit
    manager.on_store_results(set(), {"req": 2})

    assert unsettled == []
    assert len(retried) == 1
    assert retried[0].op.block_ids[0] == [block]
    assert pool.blocks[block].ref_cnt == 0


def test_restarted_request_loss_before_its_offer_keeps_the_chain() -> None:
    """A preempted request's op lost before the restart offers its range
    again does not break the chain: the restart computes it again."""
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, block, _ = _buffered_block_in_queue(pool, position=ADMISSION_RESERVE + 5)
    null = pool.null_block.block_id
    manager.on_request_reset("req")
    manager.on_request_arrived("req", [])
    _bring_to_head(pool, block)
    pool.get_new_blocks(1)
    assert manager.on_scheduler_step(_step()).stores_to_submit == []

    (recomputed,) = _hashed_blocks(pool, 1)
    manager.add_store_candidate(_op(0, [[recomputed], [null]]))
    stores = _flush(manager).stores_to_submit

    assert _counts(manager)["dropped_evicted"] == 1
    assert _counts(manager)["rejected_prefix_broken"] == 0
    assert [store.op.block_ids[0] for store in stores] == [[recomputed]]


def test_failed_store_of_a_preempted_request_is_retried() -> None:
    """A preempted request keeps its batch: a failure is retried.

    The batch's blocks keep their hashes, so their data is still valid.
    """
    # Smaller than the admission reserve: a buffered op on any free block is
    # due, so a retry would show.
    pool = BlockPool(
        num_gpu_blocks=ADMISSION_RESERVE, enable_caching=True, hash_block_size=4
    )
    null = pool.null_block.block_id
    (block,) = _hashed_blocks(pool, 1)
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[block], [null]]))
    pool.free_blocks([pool.blocks[block]])
    assert len(manager.on_scheduler_step(_step()).stores_to_submit) == 1

    manager.on_request_reset("req")
    manager.on_request_arrived("req", [])
    manager.on_store_results({"req"}, {"req": 1})
    (retried,) = manager.on_scheduler_step(_step()).stores_to_submit

    assert retried.op.block_ids[0] == [block]


def _coordinator(
    mamba: bool = False, num_blocks: int = 16, use_eagle: bool = False
) -> KVCacheCoordinator:
    """vLLM's own coordinator over blocks of one chunk each.

    Args:
        mamba: Whether an align-mode Mamba group (group 1) follows the full
            attention group (group 0).
        num_blocks: Blocks of its pool, the null block included.
        use_eagle: Whether speculative decoding drops the last block of a
            full-attention hit.
    """
    groups = [
        KVCacheGroupSpec(
            ["attention"],
            FullAttentionSpec(
                block_size=TOKENS_PER_BLOCK,
                num_kv_heads=1,
                head_size=1,
                dtype=torch.float32,
            ),
        )
    ]
    if mamba:
        groups.append(
            KVCacheGroupSpec(
                ["mamba"],
                MambaSpec(
                    block_size=TOKENS_PER_BLOCK,
                    shapes=((1,),),
                    dtypes=(torch.float32,),
                    mamba_cache_mode="align",
                ),
            )
        )
    return get_kv_cache_coordinator(
        KVCacheConfig(
            num_blocks=num_blocks, kv_cache_tensors=[], kv_cache_groups=groups
        ),
        max_model_len=16 * TOKENS_PER_BLOCK,
        max_in_flight_tokens=MAX_STEP_TOKENS,
        use_eagle=use_eagle,
        enable_caching=True,
        enable_kv_cache_events=False,
        dcp_world_size=1,
        pcp_world_size=1,
        scheduler_block_size=TOKENS_PER_BLOCK,
        hash_block_size=TOKENS_PER_BLOCK,
    )


def _chain(num_blocks: int) -> list[BlockHash]:
    """vLLM block hashes of a request's first ``num_blocks`` blocks."""
    return [BlockHash(bytes([i + 1]) * 32) for i in range(num_blocks)]


def _cache(pool: BlockPool, chain: list[BlockHash], block_ids: list[list[int]]) -> None:
    """Cache each group's blocks under ``chain``, as vLLM does once they fill.

    Null blocks are skipped, as vLLM skips them.
    """
    request = SimpleNamespace(block_hashes=chain)
    for group_id, group_block_ids in enumerate(block_ids):
        blocks = [pool.blocks[block_id] for block_id in group_block_ids]
        pool.cache_full_blocks(
            request, blocks, 0, len(blocks), TOKENS_PER_BLOCK, group_id
        )


def _held_manager(
    coordinator: KVCacheCoordinator, num_blocks: int, configs: dict | None = None
) -> tuple[LazyOffloadManager, list[int]]:
    """A manager over ``coordinator``, and request "req"'s cached blocks.

    Args:
        coordinator: An attention-only :func:`_coordinator`.
        num_blocks: Attention blocks to allocate and cache as the request's
            first chunks; still in use.
        configs: The manager's configs; None for none.

    Returns:
        The manager, which knows the request's hash chain, and the blocks.
    """
    pool = coordinator.block_pool
    block_ids = [block.block_id for block in pool.get_new_blocks(num_blocks)]
    chain = _chain(num_blocks)
    _cache(pool, chain, [block_ids])
    manager = _manager(pool, configs=configs, receipts=_Receipts())
    manager.bind_cache_coordinator(coordinator)
    manager.on_request_arrived("req", chain)
    return manager, block_ids


def _counts(manager: LazyOffloadManager) -> dict[str, int]:
    """The manager's final ledger line, parsed into its counts."""
    with patch.object(eviction_aware.logger, "info") as mock_info:
        manager.log_final_stats()
    (line,) = [call.args[1] for call in mock_info.call_args_list]
    return {
        name: int(value) for name, value in (field.split("=") for field in line.split())
    }


@pytest.mark.parametrize("use_eagle", [False, True])
def test_chunk_past_the_hit_is_stored_when_its_request_finishes(
    use_eagle: bool,
) -> None:
    """A finished request's chunk that vLLM's hit over its chain does not
    serve is stored by the next drain, without eviction pressure, together
    with the prefix the hit serves, so L1 ages them as one. Under speculative
    decoding vLLM drops the last block of the hit, so that is the last chunk.
    Its blocks stay out of the free queue until the store settles, then join
    the tail."""
    coordinator = _coordinator(use_eagle=use_eagle)
    pool = coordinator.block_pool
    null = pool.null_block.block_id
    manager, (first, second) = _held_manager(coordinator, 2)
    manager.add_store_candidate(_op(0, [[first], [null]]))
    manager.add_store_candidate(_op(TOKENS_PER_BLOCK, [[second], [null]]))
    assert manager.on_scheduler_step(_step()).stores_to_submit == []

    manager.on_request_finished("req")
    # vLLM frees a finished request's blocks right after notifying the
    # connector, last block first.
    pool.free_blocks([pool.blocks[second], pool.blocks[first]])
    stores = manager.on_scheduler_step(_step()).stores_to_submit

    if not use_eagle:
        assert stores == []
        return
    (store,) = stores
    assert (store.op.start, store.op.end) == (0, 2 * TOKENS_PER_BLOCK)
    assert store.op.block_ids[0] == [first, second]
    queue = [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
    assert second not in queue
    manager.on_store_results(set(), {"req": 1})
    queue = [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
    assert queue[-2:] == [first, second]
    assert manager.on_scheduler_step(_step()).stores_to_submit == []


# Chunks of the anchored request below whose op holds a Mamba state.
ANCHORS = (1, 4)
NUM_CHUNKS = 5


def _filler(pool: BlockPool) -> list[KVCacheBlock]:
    """Take every free block, hashed apart from any request's chain.

    vLLM frees a block without a hash to the queue head, so only hashed
    blocks keep the order they are freed in.
    """
    blocks = pool.get_new_blocks(pool.get_num_free_blocks())
    for i, block in enumerate(blocks):
        block.set_block_hash(
            make_block_hash_with_group_id(BlockHash(b"\xff" * 30 + bytes([0, i])), 0)
        )
    return blocks


def _anchored_request(
    pool: BlockPool, manager: LazyOffloadManager
) -> tuple[list[int], list[int]]:
    """Buffer one op per chunk of "req", Mamba states (group 1) at ``ANCHORS``.

    Attention blocks stay in use; the states are freed, as align mode frees
    a running request's older ones: chunk 4's at the queue head, chunk 1's
    past the admission reserve.

    Returns:
        Per chunk, the attention block and the Mamba block (null for none).
    """
    null = pool.null_block.block_id
    attention = _hashed_blocks(pool, NUM_CHUNKS)
    mamba = [null] * NUM_CHUNKS
    for chunk in ANCHORS:
        (mamba[chunk],) = _hashed_blocks(pool, 1)
    for chunk in range(NUM_CHUNKS):
        manager.add_store_candidate(
            _op(chunk * TOKENS_PER_BLOCK, [[attention[chunk]], [mamba[chunk]]])
        )
    filler = _filler(pool)
    pool.free_blocks([pool.blocks[mamba[4]], *filler[:20], pool.blocks[mamba[1]]])
    pool.free_blocks(filler[20:])
    return attention, mamba


def _bring_to_head(pool: BlockPool, block_id: int) -> None:
    """Allocate every free block ahead of ``block_id``."""
    queue = [block.block_id for block in pool.free_block_queue.get_all_free_blocks()]
    pool.get_new_blocks(queue.index(block_id))


def _span(store: LMCacheMPRequestMetadata) -> tuple[int, int]:
    """The chunks a store covers, as ``(first, end)``."""
    return store.op.start // TOKENS_PER_BLOCK, store.op.end // TOKENS_PER_BLOCK


def test_due_anchor_stores_only_the_chunks_past_the_earlier_anchor() -> None:
    """vLLM serves chunks 0..1 from the GPU through chunk 1's state, so the
    due state of chunk 4 stores chunks 2..4 only."""
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, configs={}, receipts=_Receipts(), mamba_groups={1: 0})
    attention, mamba = _anchored_request(pool, manager)

    (store,) = manager.on_scheduler_step(_step()).stores_to_submit

    assert _span(store) == (2, NUM_CHUNKS)
    assert store.op.block_ids == [attention[2:], mamba[2:]]


def test_earlier_segment_is_stored_once_its_anchor_is_due() -> None:
    """Chunks 0..1 stay buffered until chunk 1's state is due, then go; every
    op is stored once and none is dropped."""
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, configs={}, receipts=_Receipts(), mamba_groups={1: 0})
    attention, mamba = _anchored_request(pool, manager)
    (later,) = manager.on_scheduler_step(_step()).stores_to_submit
    manager.on_store_results(set(), {"req": 1})
    assert manager.on_scheduler_step(_step()).stores_to_submit == []

    _bring_to_head(pool, mamba[1])
    (earlier,) = manager.on_scheduler_step(_step()).stores_to_submit

    assert _span(earlier) == (0, 2)
    assert earlier.op.block_ids == [attention[:2], mamba[:2]]
    assert sorted([_span(later), _span(earlier)]) == [(0, 2), (2, NUM_CHUNKS)]
    with patch.object(eviction_aware.logger, "info") as mock_info:
        manager.log_final_stats()
    (line,) = [call.args[1] for call in mock_info.call_args_list]
    assert f"admitted={NUM_CHUNKS} emitted={NUM_CHUNKS} " in line
    assert " dropped_evicted=0 " in line
    assert " pending=0" in line


def test_attention_only_model_stores_the_whole_front() -> None:
    """Without Mamba groups no block is a state: the due op stores every op
    in front of it."""
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, configs={}, receipts=_Receipts())
    attention, mamba = _anchored_request(pool, manager)

    (store,) = manager.on_scheduler_step(_step()).stores_to_submit

    assert _span(store) == (0, NUM_CHUNKS)
    assert store.op.block_ids == [attention, mamba]


def test_failed_later_segment_is_buffered_behind_the_earlier_ops() -> None:
    """A failed store of chunks 2..4 goes back behind chunks 0..1, so a
    release of the whole buffer stores the chunks in order."""
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, configs={}, receipts=_Receipts(), mamba_groups={1: 0})
    attention, mamba = _anchored_request(pool, manager)
    (failed,) = manager.on_scheduler_step(_step()).stores_to_submit
    assert _span(failed) == (2, NUM_CHUNKS)

    manager.on_store_results({"req"}, {"req": 1})
    (retried,) = _flush(manager, "req").stores_to_submit

    assert _span(retried) == (0, NUM_CHUNKS)
    assert retried.op.block_ids == [attention, mamba]


def test_segment_due_while_a_batch_is_in_flight_is_held() -> None:
    """Chunk 1's state comes due while chunks 2..4 are in flight: chunks 0..1
    are pinned, survive an allocation of the whole queue, and are stored
    once the batch settles."""
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, configs={}, receipts=_Receipts(), mamba_groups={1: 0})
    attention, mamba = _anchored_request(pool, manager)
    (later,) = manager.on_scheduler_step(_step()).stores_to_submit
    assert _span(later) == (2, NUM_CHUNKS)

    _bring_to_head(pool, mamba[1])
    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    pool.get_new_blocks(pool.get_num_free_blocks())
    manager.on_store_results(set(), {"req": 1})
    (earlier,) = manager.on_scheduler_step(_step()).stores_to_submit

    assert _span(earlier) == (0, 2)
    assert earlier.op.block_ids == [attention[:2], mamba[:2]]


@pytest.mark.parametrize(
    "removed_key", [*lazy_offload_policy.REMOVED_CONFIG_KEYS, None]
)
def test_removed_config_key_is_warned_about(
    monkeypatch: pytest.MonkeyPatch, removed_key: str | None
) -> None:
    """A removed key still in the config logs one warning naming it; a clean
    config logs none.

    The module logger does not propagate, so the test spies on it instead of
    using ``caplog``.
    """
    warnings: list[str] = []
    monkeypatch.setattr(
        lazy_offload_policy.logger,
        "warning",
        lambda msg, *args, **kwargs: warnings.append(str(msg) % args),
    )
    configs = {} if removed_key is None else {removed_key: 2.5}
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)

    _manager(pool, configs=configs)

    if removed_key is None:
        assert warnings == []
    else:
        assert len(warnings) == 1
        assert removed_key in warnings[0]


def _deferred_turn(
    num_chunks: int, finished: bool = True
) -> tuple[LazyOffloadManager, BlockPool, list[int]]:
    """A request "req" whose chunks stay buffered.

    Finished, its blocks are free: vLLM's hit over its own chain serves
    every chunk, so none is stored at its finish, and no allocation reaches
    its blocks. Running, its blocks are in use.

    Args:
        num_chunks: Chunks of the request, one op each.
        finished: Whether the request finished.

    Returns:
        The manager, its pool and the request's attention blocks, by chunk.
    """
    coordinator = _coordinator()
    pool = coordinator.block_pool
    null = pool.null_block.block_id
    manager, block_ids = _held_manager(coordinator, num_chunks)
    for chunk, block_id in enumerate(block_ids):
        manager.add_store_candidate(_op(chunk * TOKENS_PER_BLOCK, [[block_id], [null]]))
    if finished:
        manager.on_request_finished("req")
        pool.free_blocks([pool.blocks[block_id] for block_id in reversed(block_ids)])
    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    return manager, pool, block_ids


@pytest.mark.parametrize("finished", [True, False])
@pytest.mark.parametrize("local_hit_chunks", [0, 1])
def test_lookup_past_the_local_hit_waits_for_the_deferred_chunks(
    local_hit_chunks: int, finished: bool
) -> None:
    """A request whose prefix another request still buffers, past what
    vLLM's prefix cache serves it, has those chunks stored by the next
    drain together with the prefix before them, so L1 ages them as one, and
    its lookup waits until their receipt."""
    manager, pool, block_ids = _deferred_turn(3, finished)
    manager.on_request_arrived("next", _chain(4))
    local_hit = local_hit_chunks * TOKENS_PER_BLOCK

    assert manager.flush_for_lookup("next", local_hit)
    (store,) = manager.on_scheduler_step(_step()).stores_to_submit
    assert manager.flush_for_lookup("next", local_hit)
    manager.on_store_results(set(), {"req": 1})

    assert not manager.flush_for_lookup("next", local_hit)
    assert store.request_id == "req"
    assert _span(store) == (0, 3)
    assert store.op.block_ids[0] == block_ids
    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    assert not manager.has_inflight_store_work()


@pytest.mark.parametrize("case", ["other_prefix", "within_local_hit"])
def test_lookup_without_deferred_chunks_past_its_hit_does_not_wait(
    case: str,
) -> None:
    """A request sharing no buffered chunk, or only chunks vLLM's prefix
    cache serves it, looks up at once and stores nothing."""
    manager, pool, block_ids = _deferred_turn(3)
    if case == "other_prefix":
        chain = [BlockHash(bytes([i + 1]) * 31 + b"\x00") for i in range(4)]
        local_hit = 0
    else:
        chain = _chain(4)
        local_hit = 3 * TOKENS_PER_BLOCK
    manager.on_request_arrived("next", chain)

    assert not manager.flush_for_lookup("next", local_hit)
    assert manager.on_scheduler_step(_step()).stores_to_submit == []


def test_lookup_flushes_only_up_to_the_last_state() -> None:
    """In a hybrid model a hit ends at a chunk carrying a Mamba state, so a
    lookup flushes the deferred chunks up to the last one with a state."""
    pool = BlockPool(num_gpu_blocks=64, enable_caching=True, hash_block_size=4)
    manager = _manager(pool, receipts=_Receipts(), mamba_groups={1: 0})
    null = pool.null_block.block_id
    manager.on_request_arrived("req", _chain(4))
    attention = _hashed_blocks(pool, 4)
    (state,) = _hashed_blocks(pool, 1)
    for chunk in range(4):
        manager.add_store_candidate(
            _op(
                chunk * TOKENS_PER_BLOCK,
                [[attention[chunk]], [state if chunk == 1 else null]],
            )
        )
    manager.on_request_arrived("next", _chain(4))

    assert manager.flush_for_lookup("next", 0)
    (store,) = manager.on_scheduler_step(_step()).stores_to_submit

    assert _span(store) == (0, 2)
    assert store.op.block_ids == [attention[:2], [null, state]]


def test_state_block_freed_before_its_request_finished_survives_the_store() -> None:
    """A finished request's chunk vLLM's hit does not serve (speculative
    decoding drops the last block, and no earlier state is cached) is
    stored, its Mamba state included, which align mode freed before the
    finish. Once
    the store settles, the next allocation does not evict that state: the
    store leaves it where vLLM queued it, not at the free-queue head."""
    coordinator = _coordinator(mamba=True, num_blocks=64, use_eagle=True)
    pool = coordinator.block_pool
    null = pool.null_block.block_id
    chain = _chain(2)
    attention = [block.block_id for block in pool.get_new_blocks(2)]
    (state,) = [block.block_id for block in pool.get_new_blocks(1)]
    _cache(pool, chain, [attention, [null, state]])
    manager = _manager(pool, receipts=_Receipts())
    manager.bind_cache_coordinator(coordinator)
    manager.on_request_arrived("req", chain)
    manager.add_store_candidate(_op(0, [[attention[0]], [null]]))
    manager.add_store_candidate(_op(TOKENS_PER_BLOCK, [[attention[1]], [state]]))
    pool.free_blocks([pool.blocks[state]])
    assert manager.on_scheduler_step(_step()).stores_to_submit == []

    manager.on_request_finished("req")
    pool.free_blocks([pool.blocks[block_id] for block_id in reversed(attention)])
    (store,) = manager.on_scheduler_step(_step()).stores_to_submit
    assert state in store.op.block_ids[1]
    manager.on_store_results(set(), {"req": 1})

    assert pool.get_new_blocks(1)[0].block_id != state


def test_state_past_the_hit_is_stored_with_the_prefix_the_hit_serves() -> None:
    """A finished request whose hit ends at a state it does not store itself
    (cached by another request) stores its own state past the hit together
    with the prefix before it, in one segment, so L1 does not age the state
    ahead of the prefix it needs."""
    coordinator = _coordinator(mamba=True, num_blocks=64, use_eagle=True)
    pool = coordinator.block_pool
    null = pool.null_block.block_id
    chain = _chain(3)
    attention = [block.block_id for block in pool.get_new_blocks(3)]
    shared, own = [block.block_id for block in pool.get_new_blocks(2)]
    _cache(pool, chain, [attention, [shared, null, own]])
    manager = _manager(pool, receipts=_Receipts())
    manager.bind_cache_coordinator(coordinator)
    manager.on_request_arrived("req", chain)
    manager.add_store_candidate(_op(0, [[attention[0]], [null]]))
    manager.add_store_candidate(_op(TOKENS_PER_BLOCK, [[attention[1]], [null]]))
    manager.add_store_candidate(_op(2 * TOKENS_PER_BLOCK, [[attention[2]], [own]]))
    assert manager.on_scheduler_step(_step()).stores_to_submit == []

    manager.on_request_finished("req")
    pool.free_blocks([pool.blocks[block_id] for block_id in reversed(attention)])
    (store,) = manager.on_scheduler_step(_step()).stores_to_submit

    assert _span(store) == (0, 3)
    assert store.op.block_ids[1] == [null, null, own]


def test_block_a_lookup_flush_stored_is_not_evicted_next() -> None:
    """The free blocks a lookup flush stores keep their place in vLLM's
    eviction order: once the store settles, the next allocation takes none
    of them."""
    manager, pool, block_ids = _deferred_turn(3)
    manager.on_request_arrived("next", _chain(4))
    assert manager.flush_for_lookup("next", 0)
    (store,) = manager.on_scheduler_step(_step()).stores_to_submit
    manager.on_store_results(set(), {"req": 1})

    assert not manager.flush_for_lookup("next", 0)
    assert pool.get_new_blocks(1)[0].block_id not in block_ids
