# SPDX-License-Identifier: Apache-2.0

"""A write that finds L1 full gets space before the next periodic check."""

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
    return ObjectKey(
        chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0
    )


def _write(manager: L1Manager, i: int) -> L1Error:
    err, _ = manager.reserve_write([_key(i)], [False], LAYOUT)[_key(i)]
    if err == L1Error.SUCCESS:
        manager.finish_write([_key(i)])
    return err


def test_write_into_full_l1_succeeds_before_the_next_periodic_check():
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
        # Let the first periodic check pass with L1 empty.
        time.sleep(1.1)
        for i in range(CAPACITY):
            assert _write(manager, i) == L1Error.SUCCESS
        start = time.monotonic()
        assert _write(manager, CAPACITY) == L1Error.OUT_OF_MEMORY
        while _write(manager, CAPACITY) != L1Error.SUCCESS:
            assert time.monotonic() - start < 0.5
            time.sleep(0.01)
    finally:
        controller.stop()
        manager.close()
