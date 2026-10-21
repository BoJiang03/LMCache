# SPDX-License-Identifier: Apache-2.0
"""LazyOffloadManager admission of store operations on hybrid block tables."""

# Standard
from types import SimpleNamespace
from unittest.mock import patch

# Third Party
import pytest

pytest.importorskip("vllm", reason="the manager reads a vLLM BlockPool")

# Third Party
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)

# First Party
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

TOKENS_PER_BLOCK = 4
# Large enough that every admission here is scheduled in one chunk.
MAX_STEP_TOKENS = 1024
# Emit every buffered operation on the next drain, by its deferral deadline.
CONFIGS = {"lmcache.mp.lazy_offload_max_deferral_seconds": 1e-9}
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
    tables: dict[str, tuple[list[int], ...]] | None = None,
) -> SimpleNamespace:
    """A scheduler step running newly admitted requests from token zero.

    Args:
        scheduled: Request id -> tokens scheduled. None schedules one token
            of a request the manager never saw admitted, so no running
            request bounds the danger window.
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
        preempted_req_ids=None,
    )


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
    """Build a two-group manager bound to ``pool``."""
    manager = LazyOffloadManager(
        CONFIGS if configs is None else configs,
        [TOKENS_PER_BLOCK, TOKENS_PER_BLOCK],
        receipts or _NoReceipts(),
        max_step_tokens=max_step_tokens,
        num_lookahead_tokens=num_lookahead_tokens,
        max_model_len=max_model_len,
        mamba_groups=mamba_groups,
        prefill_alignment=prefill_alignment,
    )
    manager.bind_block_pool(pool)
    return manager


def _store(block_ids: list[list[int]]) -> LMCacheMPRequestMetadata:
    n_tokens = TOKENS_PER_BLOCK * len(block_ids[0])
    return LMCacheMPRequestMetadata(
        request_id="req",
        direction="STORE",
        op=LoadStoreOp(
            token_ids=list(range(n_tokens)),
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


def test_null_blocks_do_not_block_the_store() -> None:
    """A mamba group's null blocks (align mode) leave the store to its live blocks."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    attention = _hashed_blocks(pool, 2)
    state = _hashed_blocks(pool, 1)
    null = pool.null_block.block_id
    manager = _manager(pool)

    manager.add_store_candidate(_store([attention, [null, state[0]]]))
    actions = manager.on_scheduler_step(_step())

    assert len(actions.stores_to_submit) == 1
    stored = actions.stores_to_submit[0]
    assert stored.request_id == "req"
    assert stored.op.block_ids == [attention, [null, state[0]]]


def test_unhashed_live_block_still_rejects_the_store() -> None:
    """A live block without a hash may have been recycled: nothing is stored."""
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    attention = _hashed_blocks(pool, 1) + [b.block_id for b in pool.get_new_blocks(1)]
    manager = _manager(pool)

    manager.add_store_candidate(_store([attention, [pool.null_block.block_id] * 2]))
    actions = manager.on_scheduler_step(_step())

    assert actions.stores_to_submit == []


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
            token_ids=list(range(end)), block_ids=block_ids, start=start, end=end
        ),
    )


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
    the block survives the allocation, still buffered.
    """
    pool = BlockPool(num_gpu_blocks=32, enable_caching=True, hash_block_size=4)
    manager, block, _ = _buffered_block_in_queue(pool, position=ADMISSION_RESERVE + 5)
    num_free = pool.get_num_free_blocks()

    if admitted:
        manager.on_admission("new", ADMISSION_TOKENS, 0, 0, load_async=False)
        assert pool.get_num_free_blocks() == num_free
    allocated = pool.get_new_blocks(ADMISSION_BLOCKS)

    assert (block in {b.block_id for b in allocated}) is not admitted


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
    long as the model allows: it does not move the block, and its
    allocation leaves the block where it was, buffered.
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

    assert block not in {allocated_block.block_id for allocated_block in allocated}


def test_external_hit_of_mamba_groups_takes_a_state_block_of_its_own() -> None:
    """vLLM gives an external hit a Mamba state block before allocating.

    Each Mamba group takes one block for the external hit and one state
    block for the chunk, four in all. An earlier admission of the step has
    taken the reserve, which leaves the buffered block third in the queue,
    in reach. Saved by the admission notice, it is not recycled.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    manager, block, _ = _buffered_block_in_queue(
        pool, position=MAMBA_RESERVE + 2, mamba_groups={0: 0, 1: 0}
    )
    pool.get_new_blocks(MAMBA_RESERVE)

    manager.on_admission(
        "new", 20 * TOKENS_PER_BLOCK, 0, 10 * TOKENS_PER_BLOCK, load_async=False
    )
    allocated = pool.get_new_blocks(4)

    assert block not in {allocated_block.block_id for allocated_block in allocated}


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
    manager, block, _ = _buffered_block_in_queue(
        pool, position=ADMISSION_RESERVE + hit_blocks + 2
    )

    manager.on_admission(
        "new", TOKENS_PER_BLOCK + ADMISSION_TOKENS, 0, TOKENS_PER_BLOCK, load_async=True
    )
    pool.get_new_blocks(hit_blocks)
    if notified:
        manager.on_loads_finished({"new"})
    allocated = pool.get_new_blocks(ADMISSION_BLOCKS)

    assert (block in {b.block_id for b in allocated}) is not notified


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

    assert stores != []


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
        manager.add_store_candidate(store)
    pool.free_blocks([pool.blocks[block_id] for block_id in ids])

    first = manager.on_scheduler_step(_step({"running": 2 * TOKENS_PER_BLOCK}))
    pool.get_new_blocks(2)
    second = manager.on_scheduler_step(_step())

    stored = first.stores_to_submit + second.stores_to_submit
    assert {store.request_id for store in stored} == {"first", "second"}


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
    allocated = pool.get_new_blocks(new_blocks)

    assert block not in {allocated_block.block_id for allocated_block in allocated}


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
        mamba_groups={0: 0, 2: 0},
    )
    manager.bind_block_pool(pool)
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
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    null = pool.null_block.block_id
    (block,) = _hashed_blocks(pool, 1)
    manager = _manager(pool, receipts=_TwoWorkerReceipts())
    manager.add_store_candidate(_op(0, [[block], [null]]))
    assert len(manager.on_scheduler_step(_step()).stores_to_submit) == 1
    pool.free_blocks([pool.blocks[block]])

    manager.on_store_results({"req"}, {"req": 1})
    unsettled = manager.on_scheduler_step(_step()).stores_to_submit
    manager.on_store_results(set(), {"req": 1})
    retried = manager.on_scheduler_step(_step()).stores_to_submit
    manager.on_store_results(set(), {"req": 2})

    assert unsettled == []
    assert len(retried) == 1
    assert retried[0].op.block_ids[0] == [block]
    assert pool.blocks[block].ref_cnt == 0


def test_failed_store_of_a_preempted_request_is_not_retried() -> None:
    """A reset detaches the failed batch from the restarted request.

    The restart recomputes from token zero, so the old range would be stored
    from blocks the request no longer owns; its receipt only unpins them.
    """
    pool = BlockPool(num_gpu_blocks=16, enable_caching=True, hash_block_size=4)
    null = pool.null_block.block_id
    (block,) = _hashed_blocks(pool, 1)
    manager = _manager(pool, receipts=_Receipts())
    manager.add_store_candidate(_op(0, [[block], [null]]))
    assert len(manager.on_scheduler_step(_step()).stores_to_submit) == 1
    pool.free_blocks([pool.blocks[block]])

    manager.on_request_reset("req")
    manager.on_store_results({"req"}, {"req": 1})

    assert manager.on_scheduler_step(_step()).stores_to_submit == []
    assert pool.blocks[block].ref_cnt == 0
