#!/usr/bin/env python3
"""Tabulate a QASPER working-set resweep (off/eager/lazy per cohort size).

Usage:
    qasper_panel.py <runtime_dir> <rep> [rep...]

Reads `QP_<mode>_u<users>r2q2g16_<rep>.{json,csv}` pairs produced by
`run_sweep2.py`. The request stream is deterministic (`apc_queries` is
byte-identical across modes), so round-2 latencies pair exactly by
`(user_id, question_id)` within a repetition.

Per (users, mode) it prints external-hit coverage and round-2 TTFT/E2E
p50, plus the per-user paired deltas against the same repetition's `off`
run. E2E here is `ttft + generation_time` for the request.
"""

import csv
import json
import statistics
import sys
from pathlib import Path

#: KV bytes per token for Qwen3-8B (36 layers x 8 KV heads x 128 dims x 2
#: tensors x 2 bytes), used only for the working-set estimate column.
KV_BYTES_PER_TOKEN = 147456

MODES = ("off", "eager", "lazy")
SIZES = (16, 24, 32, 40, 48)


def load_rows(path: Path) -> dict[tuple[int, int], dict[str, float]]:
    """Read one benchmark CSV keyed by (user_id, question_id).

    Args:
        path: The per-request CSV written by multi-round-qa.py.

    Returns:
        Mapping of (user_id, question_id) to that request's row with
        `ttft`, `e2e` (seconds), and `prompt_tokens`.

    Raises:
        ValueError: if a (user, question) key repeats.
    """
    rows: dict[tuple[int, int], dict[str, float]] = {}
    with path.open() as fh:
        for row in csv.DictReader(fh):
            key = (int(row["user_id"]), int(row["question_id"]))
            if key in rows:
                raise ValueError(f"{path}: duplicate request key {key}")
            rows[key] = {
                "ttft": float(row["ttft"]),
                "e2e": float(row["ttft"]) + float(row["generation_time"]),
                "prompt_tokens": float(row["prompt_tokens"]),
            }
    return rows


def one_point(runtime: Path, users: int, rep: str) -> dict[str, dict] | None:
    """Collect every mode's result for one cohort size and repetition.

    Args:
        runtime: Directory holding the QP_* files.
        users: Cohort size.
        rep: Repetition label.

    Returns:
        Mapping of mode to {"meta": result json, "rows": csv rows}, or
        None when any mode's files are missing (point not finished).
    """
    point: dict[str, dict] = {}
    for mode in MODES:
        tag = f"QP_{mode}_u{users}r2q2g16_{rep}"
        json_path = runtime / f"{tag}.json"
        csv_path = runtime / f"{tag}.csv"
        if not (json_path.exists() and csv_path.exists()):
            return None
        point[mode] = {
            "meta": json.loads(json_path.read_text()),
            "rows": load_rows(csv_path),
        }
    return point


def fmt_ms(seconds: float) -> str:
    return f"{seconds * 1000.0:7.1f}"


def main() -> int:
    if len(sys.argv) < 3:
        print(__doc__, file=sys.stderr)
        return 2
    runtime = Path(sys.argv[1])
    reps = sys.argv[2:]
    for rep in reps:
        print(f"\n=== repetition {rep} ===")
        header = (
            f"{'users':>5} {'KV est':>8} {'mode':>6} {'cover':>6} "
            f"{'r2 TTFT p50':>12} {'r2 E2E p50':>11} "
            f"{'dTTFT vs off':>13} {'dE2E vs off':>12} {'preempt':>7}"
        )
        print(header)
        for users in SIZES:
            point = one_point(runtime, users, rep)
            if point is None:
                print(f"{users:>5}  (incomplete)")
                continue
            keys = set(point["off"]["rows"])
            for mode in MODES:
                if set(point[mode]["rows"]) != keys:
                    print(f"{users:>5}  (row mismatch in {mode})")
            round2 = sorted(k for k in keys if k[1] == 2)
            kv_est = sum(
                point["off"]["rows"][k]["prompt_tokens"]
                for k in keys
                if k[1] == 1
            ) * KV_BYTES_PER_TOKEN / (1 << 30)
            for mode in MODES:
                meta = point[mode]["meta"]["delta"]
                rows = point[mode]["rows"]
                off_rows = point["off"]["rows"]
                cover = (
                    meta["ext_hits"] / meta["ext_queries"]
                    if meta["ext_queries"]
                    else 0.0
                )
                ttfts = [rows[k]["ttft"] for k in round2]
                e2es = [rows[k]["e2e"] for k in round2]
                d_ttft = statistics.median(
                    rows[k]["ttft"] - off_rows[k]["ttft"] for k in round2
                )
                d_e2e = statistics.median(
                    rows[k]["e2e"] - off_rows[k]["e2e"] for k in round2
                )
                print(
                    f"{users:>5} {kv_est:7.1f}G {mode:>6} {cover:6.3f} "
                    f"{fmt_ms(statistics.median(ttfts)):>12} "
                    f"{fmt_ms(statistics.median(e2es)):>11} "
                    f"{fmt_ms(d_ttft):>13} {fmt_ms(d_e2e):>12} "
                    f"{int(point[mode]['meta']['delta']['preemptions']):>7}"
                )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
