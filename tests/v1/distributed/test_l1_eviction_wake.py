# SPDX-License-Identifier: Apache-2.0

"""A write burst that does not fit in L1 gets space, whatever the usage."""

# Standard
import time

# Third Party
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
)
from lmcache.v1.distributed.error import L1Error
from lmcache.v1.distributed.l1_manager import L1Manager
from lmcache.v1.distributed.storage_controllers.eviction_controller import (
    L1EvictionController,
)
from tests.v1.distributed.utils import should_use_lazy_alloc

# 8 MiB per object, 8 objects fill the 64 MiB L1.
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([2048, 1024])], dtypes=[torch.float32])
CAPACITY = 8


def _key(i: int) -> ObjectKey:
    """Return the object key numbered ``i``."""
    return ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0)


def _write(manager: L1Manager, ids: list[int]) -> bool:
    """Write the objects numbered ``ids``; return whether L1 took them all."""
    keys = [_key(i) for i in ids]
    results = manager.reserve_write(keys, [False] * len(keys), LAYOUT)
    if any(err != L1Error.SUCCESS for err, _ in results.values()):
        return False
    manager.finish_write(keys)
    return True


def test_burst_that_does_not_fit_below_the_watermark_gets_space() -> None:
    manager = L1Manager(
        L1ManagerConfig(
            memory_config=L1MemoryManagerConfig(
                size_in_bytes=64 * 1024 * 1024,
                use_lazy=should_use_lazy_alloc(),
                init_size_in_bytes=64 * 1024 * 1024,
                align_bytes=0x1000,
            ),
            write_ttl_seconds=600,
            read_ttl_seconds=300,
        )
    )
    controller = L1EvictionController(
        l1_manager=manager,
        eviction_config=EvictionConfig(
            eviction_policy="LRU", trigger_watermark=0.95, eviction_ratio=0.25
        ),
    )
    controller.start()
    try:
        # 7 of 8 objects is below the watermark, so a periodic check alone
        # would never evict, yet a two-object burst does not fit.
        assert _write(manager, list(range(CAPACITY - 1)))
        assert not _write(manager, [CAPACITY, CAPACITY + 1])
        deadline = time.monotonic() + 5
        while not _write(manager, [CAPACITY, CAPACITY + 1]):
            assert time.monotonic() < deadline
            time.sleep(0.01)
    finally:
        controller.stop()
        manager.close()
