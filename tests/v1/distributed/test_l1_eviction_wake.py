# SPDX-License-Identifier: Apache-2.0

"""L1 makes room at once only for a write that asks for it.

A due store retries the keys L1 had no room for and asks for room
(``StorageManager.make_room``), so eviction runs at once, counting the burst
as used. Any other write keeps the periodic, watermark-only eviction check.
"""

# Standard
from collections.abc import Iterator
import time

# Third Party
import pytest
import torch

# First Party
from lmcache.v1.distributed.api import MemoryLayoutDesc, ObjectKey
from lmcache.v1.distributed.config import (
    EvictionConfig,
    L1ManagerConfig,
    L1MemoryManagerConfig,
    StorageManagerConfig,
)
from lmcache.v1.distributed.storage_manager import StorageManager
from tests.v1.distributed.utils import should_use_lazy_alloc

# 8 MiB per object, 8 objects fill the 64 MiB L1.
LAYOUT = MemoryLayoutDesc(shapes=[torch.Size([2048, 1024])], dtypes=[torch.float32])
CAPACITY = 8
BURST = [CAPACITY, CAPACITY + 1]


def _key(i: int) -> ObjectKey:
    """Return the object key numbered ``i``."""
    return ObjectKey(chunk_hash=ObjectKey.IntHash2Bytes(i), model_name="m", kv_rank=0)


def _write(manager: StorageManager, ids: list[int]) -> bool:
    """Write the objects numbered ``ids``; return whether L1 took them all."""
    keys = [_key(i) for i in ids]
    reserved = manager.reserve_write(keys, LAYOUT)
    manager.finish_write(list(reserved))
    return len(reserved) == len(keys)


@pytest.fixture
def manager() -> Iterator[StorageManager]:
    """A 64 MiB L1, 7 of 8 objects full: below a 0.95 watermark."""
    manager = StorageManager(
        StorageManagerConfig(
            l1_manager_config=L1ManagerConfig(
                memory_config=L1MemoryManagerConfig(
                    size_in_bytes=64 * 1024 * 1024,
                    use_lazy=should_use_lazy_alloc(),
                    init_size_in_bytes=64 * 1024 * 1024,
                    align_bytes=0x1000,
                ),
                write_ttl_seconds=600,
                read_ttl_seconds=300,
            ),
            eviction_config=EvictionConfig(
                eviction_policy="LRU", trigger_watermark=0.95, eviction_ratio=0.25
            ),
        )
    )
    assert _write(manager, list(range(CAPACITY - 1)))
    yield manager
    manager.close()


@pytest.mark.no_shared_allocator
def test_burst_that_asks_for_room_gets_it(manager: StorageManager) -> None:
    """A periodic check alone would never evict, yet the burst gets room."""
    assert not _write(manager, BURST)
    manager.make_room([_key(i) for i in BURST], LAYOUT)
    deadline = time.monotonic() + 5
    while not _write(manager, BURST):
        assert time.monotonic() < deadline
        time.sleep(0.01)


@pytest.mark.no_shared_allocator
def test_burst_that_does_not_ask_keeps_the_watermark(
    manager: StorageManager,
) -> None:
    """Without a request for room, L1 below the watermark evicts nothing."""
    assert not _write(manager, BURST)
    # Past the loop's one-second period.
    time.sleep(1.5)
    assert not _write(manager, BURST)
