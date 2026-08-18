#!/usr/bin/env python3
"""Fail fast when a benchmark result is vacuous or contains runtime errors."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("result", type=Path)
    parser.add_argument("--mode", choices=("eager", "lazy"), required=True)
    parser.add_argument("--kind", choices=("hot-cold", "gsm8k"), required=True)
    parser.add_argument("--requests", type=int, required=True)
    args = parser.parse_args()

    result = json.loads(args.result.read_text())
    errors: list[str] = []
    warnings = result.get("warnings") or []
    tp_size = result.get("tensor_parallel_size", 1)
    allowed_tp_warning_markers = (
        "allreduce_rms_fusion.py:930",
        "allreduce_rms_fusion.py:1054",
    )
    unexpected_warnings = [
        warning
        for warning in warnings
        if not (
            tp_size > 1
            and any(marker in warning for marker in allowed_tp_warning_markers)
        )
    ]
    if unexpected_warnings:
        errors.append(
            f"{len(unexpected_warnings)} unexpected warning(s): "
            f"{unexpected_warnings[:3]}"
        )
    if result.get("tracebacks"):
        errors.append(f"{len(result['tracebacks'])} traceback(s)")

    mode_lines = result.get("mode_lines", [])
    eviction_mode = any("EVICTION_AWARE policy" in line for line in mode_lines)
    if args.mode == "lazy" and not eviction_mode:
        errors.append("EVICTION_AWARE startup evidence is absent")
    if args.mode == "eager" and mode_lines:
        errors.append(f"eager run unexpectedly logged lazy mode: {mode_lines}")

    if args.kind == "hot-cold":
        actual = result.get("phases", {}).get("query", {}).get("requests")
    else:
        passes = result.get("passes", {})
        counts = [passes.get(name, {}).get("requests") for name in ("cold", "cached")]
        actual = counts if len(set(counts)) != 1 else counts[0]
    if actual != args.requests:
        errors.append(f"request count is {actual}, expected {args.requests}")

    if args.mode == "lazy":
        ledger = result.get("ledger") or {}
        admitted = ledger.get("admitted")
        terminal = ledger.get("pending", 0) + ledger.get("emitted", 0)
        terminal += sum(
            value
            for key, value in ledger.items()
            if key.startswith(("dropped_", "rejected_"))
        )
        if admitted is None or admitted != terminal:
            errors.append(
                f"counter ledger does not close: admitted={admitted}, rhs={terminal}"
            )

    if errors:
        raise SystemExit("invalid benchmark result: " + "; ".join(errors))
    print(f"[validate] {args.result.name}: all non-vacuity guards passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
