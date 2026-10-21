# SPDX-License-Identifier: Apache-2.0
"""Scheduler-side contracts of ``LazyOffloadManager``: hybrid group spans,
null blocks in block tables and the retry of failed stores."""

# Standard
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest

pytest.importorskip("vllm", reason="lazy offload imports vLLM metadata types")

# Third Party
from vllm.v1.core.block_pool import BlockPool  # noqa: E402
from vllm.v1.core.kv_cache_utils import (  # noqa: E402
    BlockHash,
    make_block_hash_with_group_id,
)

# First Party
from lmcache.integration.vllm import lazy_offload_manager  # noqa: E402
from lmcache.integration.vllm.lazy_offload_manager import (  # noqa: E402
    LazyOffloadActions,
    LazyOffloadManager,
)
from lmcache.integration.vllm.lazy_offload_policy.base import (  # noqa: E402
    ConfigValue,
    DrainSignals,
    LazyOffloadDrain,
)
from lmcache.integration.vllm.lazy_offload_policy.fifo import (  # noqa: E402
    FIFOOffloadPolicy,
)
from lmcache.integration.vllm.lmcache_mp_metadata import (  # noqa: E402
    LMCacheMPRequestMetadata,
)
from lmcache.integration.vllm.vllm_multi_process_adapter import (  # noqa: E402
    LoadStoreOp,
)

pytestmark = pytest.mark.no_shared_allocator

TOKENS_PER_BLOCK = 4
# Emit every buffered operation on the next drain, by its deferral deadline.
CONFIGS = {"lmcache.mp.lazy_offload_max_deferral_seconds": 1e-9}


class _RecordingFIFOPolicy(FIFOOffloadPolicy):
    """FIFO policy that keeps every drain signal it receives."""

    def __init__(self, configs: dict[str, ConfigValue]) -> None:
        super().__init__(configs)
        self.signals: list[DrainSignals] = []

    def drain(self, signals: DrainSignals) -> LazyOffloadDrain:
        self.signals.append(signals)
        return super().drain(signals)


class _NoopCompletionTracker:
    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        return False


def _scheduler_output(total_num_scheduled_tokens: int) -> Any:
    return SimpleNamespace(
        total_num_scheduled_tokens=total_num_scheduled_tokens,
        scheduled_new_reqs=[],
        scheduled_cached_reqs=SimpleNamespace(new_block_ids=[]),
    )


@pytest.mark.parametrize(
    "group_tokens_per_block, scheduled_tokens, expected_blocks",
    [
        # Qwen3.8-Flash-Next / GLM-5.3-Flash style: a scratch ring group
        # (span 0) sits between cacheable groups.
        ([16, 0, 32], 48, 3 + 2),
        # Partial blocks round up per cacheable group.
        ([16, 0, 32], 17, 2 + 1),
        # Position does not matter to the manager: it takes any span list
        # the connector accepts.
        ([0, 64], 65, 2),
    ],
)
def test_scheduler_step_skips_scratch_groups_in_block_estimate(
    monkeypatch: pytest.MonkeyPatch,
    group_tokens_per_block: list[int],
    scheduled_tokens: int,
    expected_blocks: int,
) -> None:
    """A scratch group (span ``0``) adds no next-step blocks and never divides."""
    policies: list[_RecordingFIFOPolicy] = []

    def create_policy(
        configs: dict[str, ConfigValue], gpu_block_pool: object
    ) -> _RecordingFIFOPolicy:
        policies.append(_RecordingFIFOPolicy(configs))
        return policies[-1]

    monkeypatch.setattr(lazy_offload_manager, "create_offload_policy", create_policy)
    manager = LazyOffloadManager({}, group_tokens_per_block, _NoopCompletionTracker())
    manager.bind_block_pool(cast(Any, SimpleNamespace(blocks=[])))

    actions = manager.on_scheduler_step(_scheduler_output(scheduled_tokens))

    assert actions == LazyOffloadActions()
    assert [s.est_next_step_blocks for s in policies[0].signals] == [expected_blocks]


class _Receipts:
    """Every receipt settles its batch, as with a single worker."""

    def update_pending_store_count(self, request_id: str, count: int, /) -> bool:
        return True


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


def _step() -> Any:
    """A scheduler step of one token that allocates no block."""
    return _scheduler_output(1)


def _manager(pool: BlockPool, receipts: object | None = None) -> LazyOffloadManager:
    """Build a two-group manager bound to ``pool``."""
    manager = LazyOffloadManager(
        CONFIGS,
        [TOKENS_PER_BLOCK, TOKENS_PER_BLOCK],
        cast(Any, receipts or _NoopCompletionTracker()),
    )
    manager.bind_block_pool(pool)
    return manager


def _op(start: int, block_ids: list[list[int]]) -> LMCacheMPRequestMetadata:
    end = start + TOKENS_PER_BLOCK * len(block_ids[0])
    return LMCacheMPRequestMetadata(
        request_id="req",
        direction="STORE",
        op=LoadStoreOp(
            token_ids=list(range(end)), block_ids=block_ids, start=start, end=end
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

    manager.add_store_candidate(_op(0, [attention, [null, state[0]]]))
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

    manager.add_store_candidate(_op(0, [attention, [pool.null_block.block_id] * 2]))
    actions = manager.on_scheduler_step(_step())

    assert actions.stores_to_submit == []


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
