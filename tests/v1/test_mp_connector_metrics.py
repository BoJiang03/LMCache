# SPDX-License-Identifier: Apache-2.0
"""Tests for LMCache MP connector metrics."""

# Standard
from types import SimpleNamespace
from typing import Any

# Third Party
import pytest

pytest.importorskip("vllm", reason="MP connector metrics require vLLM")

# Third Party
from prometheus_client import Counter, Gauge, Histogram  # noqa: E402

# First Party
from lmcache.integration.vllm.l1_duplication import DuplicationSample  # noqa: E402
from lmcache.integration.vllm.lmcache_mp_metrics import (  # noqa: E402
    LMCacheMPConnectorStats,
    LMCacheMPPromMetrics,
)


class _BoundHistogram:
    def __init__(self) -> None:
        self.observations: list[float] = []

    def observe(self, value: float) -> None:
        self.observations.append(value)


class _Histogram:
    instance: "_Histogram"

    def __init__(self, **kwargs: Any) -> None:
        self.kwargs = kwargs
        self.children: dict[tuple[object, ...], _BoundHistogram] = {}
        _Histogram.instance = self

    def labels(self, *values: object) -> _BoundHistogram:
        return self.children.setdefault(values, _BoundHistogram())


def test_lookup_stats_snapshot_and_aggregation() -> None:
    stats = LMCacheMPConnectorStats()
    stats.record_lookup(0.025)

    snapshot = stats.clone_and_reset()
    aggregate = LMCacheMPConnectorStats()
    aggregate.aggregate(snapshot)

    assert stats.is_empty()
    assert aggregate.reduce() == {
        "LMCache MP lookup count": 1,
        "LMCache MP lookup avg latency (ms)": 25.0,
    }


def test_duplication_sample_is_a_latest_value() -> None:
    stats = LMCacheMPConnectorStats()
    stats.record_l1_duplication(DuplicationSample(1, 1, 1, 1, 0, 0))
    stats.record_l1_duplication(DuplicationSample(3, 300, 4, 1200, 2, 200))
    assert not stats.is_empty()

    accumulated = LMCacheMPConnectorStats(data=dict(stats.clone_and_reset().data))
    assert stats.is_empty()
    # A later interval without a sample keeps the last one.
    accumulated.aggregate(LMCacheMPConnectorStats())

    assert accumulated.reduce() == {
        "LMCache MP L1 dup byte_ratio": 0.25,
        "LMCache MP L1 dup object_ratio": 0.75,
        "LMCache MP L1 dup matched_objects": 3,
        "LMCache MP L1 dup matched_bytes": 300,
        "LMCache MP L1 dup model_objects": 4,
        "LMCache MP L1 dup model_bytes": 1200,
        "LMCache MP L1 dup matched_inuse_objects": 2,
        "LMCache MP L1 dup matched_inuse_bytes": 200,
        "LMCache MP L1 dup matched_demoted_objects": 0,
        "LMCache MP L1 dup matched_demoted_bytes": 0,
        **{
            f"LMCache MP L1 dup matched_mru_{cause}_{unit}": 0
            for cause in (
                "write",
                "lazy_touch",
                "lookup_touch",
                "end_session_touch",
                "other",
            )
            for unit in ("objects", "bytes")
        },
        "LMCache MP L1 dup matched_inuse_demoted_bytes": 0,
        **{
            f"LMCache MP L1 dup matched_inuse_mru_{cause}_bytes": 0
            for cause in (
                "write",
                "lazy_touch",
                "lookup_touch",
                "end_session_touch",
                "other",
            )
        },
        **{
            f"LMCache MP L1 dup {name}": 0
            for name in (
                "matched_inuse_demoted_objects",
                "matched_firm_objects",
                "matched_firm_bytes",
                "matched_firm_demoted_objects",
                "matched_firm_demoted_bytes",
                "matched_inuse_firm_objects",
                "matched_inuse_firm_bytes",
                "matched_inuse_firm_demoted_objects",
                "matched_inuse_firm_demoted_bytes",
            )
        },
    }


def test_duplication_sample_logs_demoted_and_mru_causes() -> None:
    stats = LMCacheMPConnectorStats()
    stats.record_l1_duplication(
        DuplicationSample(
            matched_objects=4,
            matched_bytes=400,
            model_objects=8,
            model_bytes=800,
            matched_inuse_objects=1,
            matched_inuse_bytes=100,
            matched_demoted_objects=1,
            matched_demoted_bytes=100,
            matched_mru_write_objects=2,
            matched_mru_write_bytes=200,
            matched_mru_lazy_touch_objects=1,
            matched_mru_lazy_touch_bytes=100,
        )
    )
    summary = stats.reduce()
    assert summary["LMCache MP L1 dup matched_demoted_bytes"] == 100
    assert summary["LMCache MP L1 dup matched_mru_write_objects"] == 2
    assert summary["LMCache MP L1 dup matched_mru_lazy_touch_bytes"] == 100
    assert summary["LMCache MP L1 dup matched_mru_other_objects"] == 0


def test_prometheus_observes_lookup_duration() -> None:
    metric_types: dict[Any, Any] = {
        Counter: _Histogram,
        Gauge: _Histogram,
        Histogram: _Histogram,
    }
    metrics = LMCacheMPPromMetrics(
        SimpleNamespace(kv_transfer_config=object()),
        metric_types,
        ["model_name", "engine"],
        {0: ["model", "0"]},
    )

    metrics.observe({"lookup_duration_seconds": [0.01, 0.02]})

    assert _Histogram.instance.kwargs["name"] == (
        "vllm:lmcache_mp_lookup_duration_seconds"
    )
    assert _Histogram.instance.children[("model", "0")].observations == [0.01, 0.02]
