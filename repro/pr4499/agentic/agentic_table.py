#!/usr/bin/env python
"""Render the agentic sweep's per-point results as comparison tables.

Reads every `AG_<mode>_s<sessions>_<rep>.json` in the results directory and
prints three tables: what each run served, what it cost, and the lazy
improvement per point. Percentages are the eviction-aware improvement over
eager: positive means eviction-aware is faster.

Decode time is reported separately from TTFT because the two answer
different questions in this workload -- TTFT is what the cache decides, and
decode is where per-scheduler-step policy work shows up.

Usage:
    python agentic_table.py [results_dir]
"""

import json
import math
from pathlib import Path
import statistics
import sys


def _percentile(values: list[float], fraction: float) -> float:
    """Percentile of a sample by nearest-rank, nan when empty."""
    ordered = sorted(value for value in values if not math.isnan(value))
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _continuation(document: dict) -> list[dict]:
    """Successful records of steps after a session's first."""
    return [
        record
        for record in document["records"]
        if record.get("ok") and record["step"] > 0
    ]


def _latencies(document: dict) -> dict[str, float]:
    """TTFT, decode and E2E percentiles over continuation steps."""
    records = _continuation(document)
    ttft = [record["ttft_ms"] for record in records]
    e2e = [record["e2e_ms"] for record in records]
    decode = [record["e2e_ms"] - record["ttft_ms"] for record in records]
    return {
        "ttft_p50": _percentile(ttft, 0.5),
        "ttft_p90": _percentile(ttft, 0.9),
        "decode_p50": _percentile(decode, 0.5),
        "e2e_p50": _percentile(e2e, 0.5),
        "e2e_p90": _percentile(e2e, 0.9),
        "ttft_mean": statistics.fmean(ttft) if ttft else math.nan,
    }


def _coverage(delta: dict) -> float:
    """Fraction of queried prompt tokens served by GPU APC or LMCache."""
    total = delta.get("apc_queries", 0.0)
    if not total:
        return math.nan
    return (delta["apc_hits"] + delta.get("ext_hits", 0.0)) / total


def _external(delta: dict) -> float:
    """Share of externally queried tokens LMCache served."""
    total = delta.get("ext_queries", 0.0)
    return delta["ext_hits"] / total if total else math.nan


def _gain(base: float, new: float) -> float:
    """Percent improvement of `new` over `base`, nan when undefined."""
    if not base or math.isnan(base) or math.isnan(new):
        return math.nan
    return (base - new) / base * 100.0


def _cell(value: float, digits: int = 1) -> str:
    """Format a possibly-nan number."""
    return "--" if math.isnan(value) else f"{value:.{digits}f}"


def main() -> int:
    """Print the sweep tables.

    Returns:
        Process exit code; 1 if no eager/lazy pairs were found.
    """
    root = Path(sys.argv[1] if len(sys.argv) > 1 else Path(__file__).parent / "results")
    points: dict[tuple[int, str], dict[str, dict]] = {}
    for path in sorted(root.glob("AG_*.json")):
        _, mode, sessions, rep = path.stem.split("_")
        points.setdefault((int(sessions[1:]), rep), {})[mode] = json.loads(path.read_text())
    pairs = {key: value for key, value in points.items() if {"eager", "lazy"} <= set(value)}
    if not pairs:
        print("no eager/lazy pairs found", file=sys.stderr)
        return 1

    print("### What each run served\n")
    print(
        "| sessions | working set | rep | policy | coverage | external hit | "
        "L1 written (GiB) | L1 peak | L1 cycles | admitted | emitted | pending |"
    )
    print("| ---: | ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (sessions, rep), modes in sorted(pairs.items()):
        for mode in ("eager", "lazy"):
            document = modes[mode]
            ledger = document["ledger"]
            l1 = document["l1"]
            print(
                f"| {sessions} | {document['working_set_gib']:.1f} GiB | {rep} | {mode} | "
                f"{_cell(_coverage(document['delta']), 3)} | "
                f"{_cell(_external(document['delta']), 3)} | "
                f"{_cell(l1.get('peak_used_gb', math.nan))} | "
                f"{_cell(l1.get('peak_ratio', math.nan), 2)} | {document['cycles']} | "
                f"{ledger.get('admitted', '--')} | {ledger.get('emitted', '--')} | "
                f"{ledger.get('pending', '--')} |"
            )

    print("\n### What each run cost (continuation steps)\n")
    print(
        "| sessions | rep | policy | TTFT p50 | TTFT p90 | decode p50 | E2E p50 | "
        "E2E p90 | wall (s) | lag p90 (ms) | preemptions | failed |"
    )
    print("| ---: | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (sessions, rep), modes in sorted(pairs.items()):
        for mode in ("eager", "lazy"):
            document = modes[mode]
            latency = _latencies(document)
            print(
                f"| {sessions} | {rep} | {mode} | "
                f"{_cell(latency['ttft_p50'])} | {_cell(latency['ttft_p90'])} | "
                f"{_cell(latency['decode_p50'])} | {_cell(latency['e2e_p50'])} | "
                f"{_cell(latency['e2e_p90'])} | {document['elapsed_s']:.0f} | "
                f"{_cell(document['lag']['p90_ms'])} | "
                f"{document['delta']['preemptions']:.0f} | {len(document['failed'])} |"
            )

    print("\n### Eviction-aware improvement over eager\n")
    print(
        "| sessions | working set | rep | coverage eager | coverage lazy | "
        "TTFT p50 | TTFT p90 | E2E p50 | E2E p90 |"
    )
    print("| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for (sessions, rep), modes in sorted(pairs.items()):
        eager, lazy = modes["eager"], modes["lazy"]
        left, right = _latencies(eager), _latencies(lazy)
        gains = [
            _gain(left["ttft_p50"], right["ttft_p50"]),
            _gain(left["ttft_p90"], right["ttft_p90"]),
            _gain(left["e2e_p50"], right["e2e_p50"]),
            _gain(left["e2e_p90"], right["e2e_p90"]),
        ]
        print(
            f"| {sessions} | {eager['working_set_gib']:.1f} GiB | {rep} | "
            f"{_cell(_coverage(eager['delta']), 3)} | {_cell(_coverage(lazy['delta']), 3)} | "
            + " | ".join(f"{_cell(value)}%" for value in gains)
            + " |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
