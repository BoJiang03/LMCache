# SPDX-License-Identifier: Apache-2.0
"""vLLM metrics for the LMCache MP connector."""

# Future
from __future__ import annotations

# Standard
from dataclasses import asdict, dataclass
from typing import Any

# Third Party
from vllm.config import VllmConfig
from vllm.distributed.kv_transfer.kv_connector.v1.metrics import (
    KVConnectorPromMetrics,
    KVConnectorStats,
    PromMetric,
    PromMetricT,
)

# First Party
from lmcache.integration.vllm.l1_duplication import DuplicationSample

_LOOKUP_DURATION = "lookup_duration_seconds"
# Latest L1 duplication sample, {name: value} for byte_ratio, object_ratio and
# every DuplicationSample field; absent until one completes.
_L1_DUPLICATION = "l1_duplication"


@dataclass
class LMCacheMPConnectorStats(KVConnectorStats):
    """Lookup latencies and the latest L1 duplication sample.

    Aggregation keeps the newer sample: duplication is a level, not a count
    over the interval.
    """

    def __post_init__(self) -> None:
        self.data.setdefault(_LOOKUP_DURATION, [])

    def record_lookup(self, duration_seconds: float) -> None:
        self.data[_LOOKUP_DURATION].append(duration_seconds)

    def record_l1_duplication(self, sample: DuplicationSample) -> None:
        """Replace the latest L1 duplication sample.

        Args:
            sample: The completed sample.
        """
        self.data[_L1_DUPLICATION] = {
            "byte_ratio": sample.byte_ratio,
            "object_ratio": sample.object_ratio,
            **asdict(sample),
        }

    def clone_and_reset(self) -> "LMCacheMPConnectorStats":
        snapshot = LMCacheMPConnectorStats(data=dict(self.data))
        self.reset()
        return snapshot

    def reset(self) -> None:
        self.data: dict[str, Any] = {_LOOKUP_DURATION: []}

    def aggregate(self, other: KVConnectorStats) -> KVConnectorStats:
        if not isinstance(other, LMCacheMPConnectorStats):
            raise TypeError(f"Cannot aggregate {type(other)}")
        self.data[_LOOKUP_DURATION].extend(other.data[_LOOKUP_DURATION])
        if _L1_DUPLICATION in other.data:
            self.data[_L1_DUPLICATION] = other.data[_L1_DUPLICATION]
        return self

    def reduce(self) -> dict[str, int | float]:
        summary: dict[str, int | float] = {}
        durations = self.data[_LOOKUP_DURATION]
        if durations:
            summary["LMCache MP lookup count"] = len(durations)
            summary["LMCache MP lookup avg latency (ms)"] = round(
                sum(durations) / len(durations) * 1000, 3
            )
        for name, value in self.data.get(_L1_DUPLICATION, {}).items():
            summary[f"LMCache MP L1 dup {name}"] = (
                round(value, 3) if isinstance(value, float) else value
            )
        return summary

    def is_empty(self) -> bool:
        return not self.data[_LOOKUP_DURATION] and _L1_DUPLICATION not in self.data


class LMCacheMPPromMetrics(KVConnectorPromMetrics):
    """Expose LMCache MP lookup latency through vLLM Prometheus metrics."""

    def __init__(
        self,
        vllm_config: VllmConfig,
        metric_types: dict[type[PromMetric], type[PromMetricT]],
        labelnames: list[str],
        per_engine_labelvalues: dict[int, list[object]],
    ) -> None:
        super().__init__(vllm_config, metric_types, labelnames, per_engine_labelvalues)
        histogram = self._histogram_cls(
            name="vllm:lmcache_mp_lookup_duration_seconds",
            documentation="LMCache MP lookup latency observed by vLLM.",
            labelnames=labelnames,
        )
        self._lookup_duration = {
            engine_idx: histogram.labels(*labelvalues)
            for engine_idx, labelvalues in per_engine_labelvalues.items()
        }

    def observe(self, transfer_stats_data: dict[str, Any], engine_idx: int = 0) -> None:
        for duration in transfer_stats_data.get(_LOOKUP_DURATION, []):
            self._lookup_duration[engine_idx].observe(duration)
