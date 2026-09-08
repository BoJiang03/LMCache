#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
"""Table the ladder: one CSV row per (arm, concurrency), plus the arm ratios.

Reads the artefacts ``run_point.sh`` leaves behind under the ladder's output
directory -- ``c<N>/<arm>/c<N>_warm.json`` for the measured round,
``external_counters.txt`` for the hit counters and ``disk.txt`` for the bytes
moved -- and prints a CSV to stdout.

Usage:
    ./collect.py <ladder-out-dir> [> ladder.csv]
"""

# Standard
from pathlib import Path
import json
import re
import sys

ARMS = ("ip_l2", "mp_l2_fs", "mp_ceiling")
COLUMNS = (
    "arm",
    "concurrency",
    "warm_tok_s",
    "mean_ttft_s",
    "bench_dur_s",
    "round2_window_s",
    "l2_read_GiB",
    "l2_read_GB_s",
    "ext_hits",
    "ext_queries",
    "hit_rate",
)


def read_bench(path: Path) -> dict[str, float]:
    """Return the measured-round fields from one ``vllm bench serve`` result.

    Args:
        path: Path to a ``c<N>_warm.json`` written by the bench.

    Returns:
        Mapping with ``warm_tok_s``, ``mean_ttft_s`` and ``bench_dur_s``, or an
        empty mapping when the file is missing or unreadable -- which is how a
        point that never produced a number is reported instead of crashing.
    """
    try:
        d = json.loads(path.read_text())
    except (OSError, ValueError):
        return {}
    return {
        "warm_tok_s": d.get("total_token_throughput", 0.0),
        "mean_ttft_s": d.get("mean_ttft_ms", 0.0) / 1000.0,
        "bench_dur_s": d.get("duration", 0.0),
    }


def read_counters(path: Path, conc: int) -> dict[str, int]:
    """Return the round-2 external-prefix-cache deltas for one point.

    Args:
        path: Path to the arm's ``external_counters.txt``.
        conc: Concurrency of the point, used to pick its line.

    Returns:
        Mapping with ``ext_hits`` and ``ext_queries``; zeros when absent.
    """
    out = {"ext_hits": 0, "ext_queries": 0}
    try:
        for line in path.read_text().splitlines():
            m = re.match(rf"c={conc} queries=(\d+) hits=(\d+)", line.strip())
            if m:
                out["ext_queries"] = int(m.group(1))
                out["ext_hits"] = int(m.group(2))
    except OSError:
        pass
    return out


def read_disk(path: Path, conc: int) -> dict[str, float]:
    """Return the bytes read during the measured round for one point.

    Args:
        path: Path to the arm's ``disk.txt``.
        conc: Concurrency of the point, used to pick its line.

    Returns:
        Mapping with ``round2_window_s``, ``l2_read_GiB`` and ``l2_read_GB_s``;
        zeros when the device counters were not available.
    """
    out = {"round2_window_s": 0.0, "l2_read_GiB": 0.0, "l2_read_GB_s": 0.0}
    pat = re.compile(
        rf"c={conc} round2 (\d+)s\s+read=([\d.]+)GiB \(([\d.]+) GB/s\)"
    )
    try:
        for line in path.read_text().splitlines():
            m = pat.search(line)
            if m:
                out["round2_window_s"] = float(m.group(1))
                out["l2_read_GiB"] = float(m.group(2))
                out["l2_read_GB_s"] = float(m.group(3))
    except OSError:
        pass
    return out


def collect(root: Path) -> list[dict[str, object]]:
    """Gather every point under a ladder output directory.

    Args:
        root: The ladder's ``OUT`` directory, holding ``c<N>/<arm>/`` trees.

    Returns:
        One row per point found, ordered by arm then concurrency.
    """
    rows: list[dict[str, object]] = []
    for arm in ARMS:
        for cdir in sorted(root.glob("c*"), key=lambda p: int(p.name[1:])):
            conc = int(cdir.name[1:])
            adir = cdir / arm
            bench = read_bench(adir / f"c{conc}_warm.json")
            if not bench:
                continue
            row: dict[str, object] = {"arm": arm, "concurrency": conc}
            row.update(bench)
            row.update(read_disk(adir / "disk.txt", conc))
            counters = read_counters(adir / "external_counters.txt", conc)
            row.update(counters)
            q = counters["ext_queries"]
            row["hit_rate"] = counters["ext_hits"] / q if q else 0.0
            rows.append(row)
    return rows


def emit_csv(rows: list[dict[str, object]]) -> None:
    """Print the rows as CSV on stdout, in ``COLUMNS`` order."""
    print(",".join(COLUMNS))
    for r in rows:
        print(",".join(_fmt(r.get(c, "")) for c in COLUMNS))


def _fmt(v: object) -> str:
    """Format one cell: integers bare, floats to a sensible precision."""
    if isinstance(v, float):
        return f"{v:.6f}" if v < 1 else f"{v:.1f}"
    return str(v)


def emit_summary(rows: list[dict[str, object]]) -> None:
    """Print the arm means and the two ratios the benchmark exists to show."""
    by_arm: dict[str, list[float]] = {}
    for r in rows:
        by_arm.setdefault(str(r["arm"]), []).append(float(r["warm_tok_s"]))
    print("\n# arm means (tok/s)", file=sys.stderr)
    for arm, vals in by_arm.items():
        mean = sum(vals) / len(vals)
        spread = (max(vals) / min(vals) - 1) * 100 if min(vals) else 0.0
        print(
            f"#   {arm:<11} n={len(vals)} mean={mean:,.0f} "
            f"min={min(vals):,.0f} max={max(vals):,.0f} spread={spread:.1f}%",
            file=sys.stderr,
        )

    ip = {int(r["concurrency"]): float(r["warm_tok_s"])
          for r in rows if r["arm"] == "ip_l2"}
    mp = {int(r["concurrency"]): float(r["warm_tok_s"])
          for r in rows if r["arm"] == "mp_l2_fs"}
    ceil = {int(r["concurrency"]): float(r["warm_tok_s"])
            for r in rows if r["arm"] == "mp_ceiling"}

    pairs = sorted(set(ip) & set(mp))
    if pairs:
        ratios = [ip[c] / mp[c] for c in pairs]
        print("\n# fair pair, IP / MP per point", file=sys.stderr)
        print("#   " + "  ".join(f"c{c}={r:.3f}" for c, r in zip(pairs, ratios)),
              file=sys.stderr)
        print(f"#   mean of ratios {sum(ratios)/len(ratios):.4f}", file=sys.stderr)

    gains = sorted(set(ceil) & set(mp))
    if gains:
        g = [(ceil[c] / mp[c] - 1) * 100 for c in gains]
        print("\n# MP ceiling over fair MP, per point (%)", file=sys.stderr)
        print("#   " + "  ".join(f"c{c}={v:+.1f}" for c, v in zip(gains, g)),
              file=sys.stderr)
        print(f"#   mean {sum(g)/len(g):+.2f}%", file=sys.stderr)


def main(argv: list[str]) -> int:
    """Entry point.

    Args:
        argv: Command-line arguments; ``argv[1]`` is the ladder output dir.

    Returns:
        0 on success, 2 on a usage error, 1 when no point produced a number.
    """
    if len(argv) != 2:
        print(__doc__, file=sys.stderr)
        return 2
    root = Path(argv[1])
    if not root.is_dir():
        print(f"not a directory: {root}", file=sys.stderr)
        return 2
    rows = collect(root)
    if not rows:
        print(f"no completed points found under {root}", file=sys.stderr)
        return 1
    emit_csv(rows)
    emit_summary(rows)
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
