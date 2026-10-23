# SPDX-License-Identifier: Apache-2.0
"""StoreLeadMeter times a stored chunk's overlap with vLLM's prefix cache."""

# Standard
from collections import Counter
from dataclasses import dataclass

# Third Party
import pytest

# First Party
from lmcache.integration.vllm.lazy_offload_lead import (
    LEAD_BUCKET_EDGES_S,
    NUM_LEAD_BUCKETS,
    StoreLeadMeter,
    lead_bucket,
)

TOKENS = 256


@dataclass
class _Block:
    """The fields of a vLLM KVCacheBlock the meter reads."""

    block_id: int
    ref_cnt: int
    block_hash: object


def _only(bucket: int, value: int = 1) -> list[int]:
    """A histogram row holding ``value`` in ``bucket`` alone."""
    row = [0] * NUM_LEAD_BUCKETS
    row[bucket] = value
    return row


@pytest.mark.parametrize(
    ("seconds", "bucket"),
    [
        (0.0, 0),
        (0.1, 0),
        (0.125, 1),
        (0.2, 1),
        (1.0, 4),
        (1.5, 4),
        (255.9, 11),
        (256.0, 12),
        (1e6, 12),
    ],
)
def test_bucket_upper_edge_is_exclusive(seconds: float, bucket: int) -> None:
    """A duration counts in the first bucket whose upper edge exceeds it."""
    assert lead_bucket(seconds) == bucket
    if bucket < len(LEAD_BUCKET_EDGES_S):
        assert seconds < LEAD_BUCKET_EDGES_S[bucket]


def test_store_then_free_then_recycle() -> None:
    """A store landing in use times the in-use and the free-queue overlap."""
    meter = StoreLeadMeter(Counter())
    block = _Block(5, ref_cnt=1, block_hash=b"h")
    meter.on_stored(10.0, block, b"h", TOKENS)
    meter.poll(10.5)
    block.ref_cnt = 0
    meter.poll(13.0)
    assert meter.snapshot().counts["alive_free"] == 1
    block.block_hash = None
    meter.poll(20.0)

    snapshot = meter.snapshot()
    assert snapshot.histograms["inuse"].chunks == _only(lead_bucket(3.0))
    assert snapshot.histograms["inuse"].tokens == _only(lead_bucket(3.0), TOKENS)
    assert snapshot.histograms["free_recycled"].chunks == _only(lead_bucket(7.0))
    assert snapshot.histograms["free_rehit"].chunks == [0] * NUM_LEAD_BUCKETS
    assert snapshot.counts["stored"] == snapshot.counts["stored_inuse"] == 1
    assert snapshot.counts["alive_inuse"] == snapshot.counts["alive_free"] == 0


def test_store_of_a_free_block_starts_the_free_overlap() -> None:
    """A store landing after vLLM freed the block times from the store."""
    meter = StoreLeadMeter(Counter())
    block = _Block(5, ref_cnt=0, block_hash=b"h")
    meter.on_stored(10.0, block, b"h", TOKENS)
    block.block_hash = b"other"
    meter.poll(10.05)

    snapshot = meter.snapshot()
    assert snapshot.histograms["free_recycled"].chunks == _only(0)
    assert snapshot.histograms["inuse"].chunks == [0] * NUM_LEAD_BUCKETS
    assert snapshot.counts["stored_inuse"] == 0


def test_rehit_ends_the_free_overlap() -> None:
    """A request referencing the free block again ends its overlap."""
    meter = StoreLeadMeter(Counter())
    block = _Block(5, ref_cnt=0, block_hash=b"h")
    meter.on_stored(10.0, block, b"h", TOKENS)
    meter.poll(11.0)
    block.ref_cnt = 1
    meter.poll(12.0)
    block.ref_cnt = 0
    meter.poll(30.0)

    snapshot = meter.snapshot()
    assert snapshot.histograms["free_rehit"].chunks == _only(lead_bucket(2.0))
    assert snapshot.histograms["free_recycled"].chunks == [0] * NUM_LEAD_BUCKETS
    assert snapshot.counts["alive_free"] == 0


def test_store_after_recycle_is_counted_and_not_tracked() -> None:
    """A store completing once the block lost its hash is a near-drop."""
    meter = StoreLeadMeter(Counter())
    meter.on_stored(10.0, _Block(5, ref_cnt=0, block_hash=None), b"h", TOKENS)
    meter.poll(11.0)

    snapshot = meter.snapshot()
    assert snapshot.counts["after_recycle"] == 1
    assert snapshot.counts["after_recycle_tokens"] == TOKENS
    assert snapshot.counts["alive_free"] == snapshot.counts["alive_inuse"] == 0
    assert all(
        hist.chunks == [0] * NUM_LEAD_BUCKETS for hist in snapshot.histograms.values()
    )


def test_own_pins_do_not_count_as_in_use() -> None:
    """References the caller pinned leave a block free in vLLM's terms."""
    pins: Counter[int] = Counter({5: 1})
    meter = StoreLeadMeter(pins)
    block = _Block(5, ref_cnt=1, block_hash=b"h")
    meter.on_stored(10.0, block, b"h", TOKENS)
    meter.poll(11.0)

    assert meter.snapshot().counts["alive_free"] == 1
    block.ref_cnt = 2
    meter.poll(12.0)
    assert meter.snapshot().histograms["free_rehit"].chunks == _only(lead_bucket(2.0))


def test_freed_and_recycled_between_polls_has_an_empty_free_overlap() -> None:
    """Both events seen in one poll count as happening at that poll."""
    meter = StoreLeadMeter(Counter())
    block = _Block(5, ref_cnt=1, block_hash=b"h")
    meter.on_stored(10.0, block, b"h", TOKENS)
    block.ref_cnt = 0
    block.block_hash = None
    meter.poll(14.0)

    snapshot = meter.snapshot()
    assert snapshot.histograms["inuse"].chunks == _only(lead_bucket(4.0))
    assert snapshot.histograms["free_recycled"].chunks == _only(0)


def test_render_lists_every_count_and_bucket() -> None:
    """The lead line carries each scalar and one value per bucket."""
    meter = StoreLeadMeter(Counter())
    meter.on_stored(10.0, _Block(5, ref_cnt=0, block_hash=b"h"), b"h", TOKENS)

    fields = dict(item.split("=") for item in meter.render().split())

    assert fields["stored"] == "1"
    assert fields["alive_free_tokens"] == str(TOKENS)
    for name in ("inuse", "free_recycled", "free_rehit"):
        assert len(fields[name].split(",")) == NUM_LEAD_BUCKETS
        assert len(fields[f"{name}_tokens"].split(",")) == NUM_LEAD_BUCKETS
