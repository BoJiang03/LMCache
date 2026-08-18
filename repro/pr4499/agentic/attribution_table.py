#!/usr/bin/env python
"""Render the attribution variants: what the decision loop costs per step.

Reads the run JSONs a `run_attribution.py` pass wrote under SMOKE_LOGDIR and
prints one row per variant, plus the decode time split into the four
quarters of the run -- the quarter split is the whole point, because the
pending queue grows monotonically when nothing is ever due, so a cost that
scales with queue depth shows up as a rising decode time inside one run.

Usage:
    python attribution_table.py [logdir] [rep]
"""

import json
import math
import os
from pathlib import Path
import statistics
import sys


def _percentile(values: list[float], fraction: float) -> float:
    """Percentile of a sample by nearest-rank, nan when empty."""
    ordered = sorted(value for value in values if not math.isnan(value))
    if not ordered:
        return math.nan
    return ordered[min(len(ordered) - 1, int(len(ordered) * fraction))]


def _cell(value: float, digits: int = 1) -> str:
    """Format a possibly-nan number."""
    return "--" if math.isnan(value) else f"{value:.{digits}f}"


def main() -> int:
    """Print the attribution table.

    Returns:
        Process exit code; 1 if no runs matched.
    """
    logdir = Path(sys.argv[1] if len(sys.argv) > 1 else os.environ["SMOKE_LOGDIR"])
    rep = sys.argv[2] if len(sys.argv) > 2 else os.environ.get("AGENTIC_ATTR_REP", "attr")
    documents = []
    for path in sorted(logdir.glob(f"ag_AG_*_{rep}*.json")):
        document = json.loads(path.read_text())
        label = document["config"]
        drain = document.get("extra_config", {}).get(
            "lmcache.mp.lazy_offload_max_drain_per_step"
        )
        documents.append((f"{label}-d{drain}" if drain else label, document))
    if not documents:
        print(f"no attribution runs for rep {rep} in {logdir}", file=sys.stderr)
        return 1
    print(
        "| variant | TTFT p50 | decode p50 | decode Q1 | decode Q2 | decode Q3 | "
        "decode Q4 | E2E p50 | wall (s) | pending at end | L1 written (GiB) |"
    )
    print("| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |")
    for label, document in documents:
        records = sorted(
            (record for record in document["records"] if record.get("ok")),
            key=lambda record: record["finished_s"],
        )
        decode = [record["e2e_ms"] - record["ttft_ms"] for record in records]
        quarter = max(1, len(records) // 4)
        quarters = [
            statistics.median(decode[index * quarter : (index + 1) * quarter])
            for index in range(4)
        ]
        continuation = [record for record in records if record["step"] > 0]
        print(
            f"| {label} | "
            f"{_cell(_percentile([r['ttft_ms'] for r in continuation], 0.5))} | "
            f"{_cell(_percentile([r['e2e_ms'] - r['ttft_ms'] for r in continuation], 0.5))} | "
            + " | ".join(_cell(value) for value in quarters)
            + f" | {_cell(_percentile([r['e2e_ms'] for r in continuation], 0.5))} | "
            f"{document['elapsed_s']:.0f} | "
            f"{document['ledger'].get('pending', '--')} | "
            f"{_cell(document['l1'].get('peak_used_gb', math.nan))} |"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
