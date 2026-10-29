# SPDX-License-Identifier: Apache-2.0
"""Scheduler-step contracts of ``LazyOffloadManager``."""

# Standard
from types import SimpleNamespace
from typing import Any, cast

# Third Party
import pytest

pytest.importorskip("vllm", reason="lazy offload imports vLLM metadata types")

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
from lmcache.v1.multiprocess.token_codec import pack_token_ids  # noqa: E402

pytestmark = pytest.mark.no_shared_allocator


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


class _FakeBlockPool:
    """Block pool whose blocks keep stable hashes and ignore pinning."""

    def __init__(self, num_blocks: int) -> None:
        self.blocks = [
            SimpleNamespace(block_hash=f"hash-{i}".encode()) for i in range(num_blocks)
        ]

    def touch(self, blocks: list[Any]) -> None:
        pass

    def free_blocks(self, blocks: list[Any]) -> None:
        pass


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


def test_coalesced_store_keeps_request_configs() -> None:
    """Matching tags select the key namespace, so coalescing must keep them."""
    manager = LazyOffloadManager(
        {
            "lmcache.mp.lazy_offload_policy": "FIFO",
            "lmcache.mp.lazy_offload_threshold": 1,
        },
        [16],
        _NoopCompletionTracker(),
    )
    manager.bind_block_pool(cast(Any, _FakeBlockPool(2)))
    request_configs = {"lmcache.tag.tenant": "a"}
    token_bytes = pack_token_ids(range(32))
    manager.on_request_arrived("req")
    for block_id in range(2):
        manager.add_store_candidate(
            LMCacheMPRequestMetadata(
                request_id="req",
                direction="STORE",
                op=LoadStoreOp(
                    token_bytes=token_bytes,
                    block_ids=[[block_id]],
                    start=16 * block_id,
                    end=16 * (block_id + 1),
                ),
                cache_salt="salt",
                request_configs=request_configs,
            )
        )
    manager.on_request_finished("req")

    actions = manager.on_scheduler_step(_scheduler_output(1))

    [store] = actions.stores_to_submit
    assert (store.op.start, store.op.end) == (0, 32)
    assert store.cache_salt == "salt"
    assert store.request_configs == request_configs
