#!/usr/bin/env python3
"""Diff two arms' merged pstats, excluding spin rows and reporting call deltas.

Written for the `mp` minus `nostore` question: of the cost the connector adds,
how much is work that can be pointed at, and how much is the busy-wait
absorbing the delay?

Spin rows are excluded by name, not by threshold, because their milliseconds
are dominated by cProfile's own per-call overhead at 3-5k calls/step -- their
CALL COUNTS are meaningful, their times are not. Everything they call
(``time.monotonic``, ``_thread.lock.__exit__``) inherits the same problem and
shows up as a large delta that is not connector work: read those as evidence
of more spinning, never as an optimisation target.

Rows whose call-count delta is 0.00 but whose time delta is large (GPU launch,
torch.mm) are the amplification landing on unchanged work -- also not a target.

Usage: nonspin_diff.py <a-prefix> <b-prefix> --steps N --workers N
       (prefixes name the dumps, e.g. `pns` and `pmp` for pns.<pid>.pstats)
"""

# Standard
import argparse
import glob
import os
import pstats

SPIN = (
    "sched_yield",
    "shm_broadcast",
    "memory_fence",
    "poll",
    "acquire",
    "wait",
    "check",
    "timeout_ms",
    "should_warn",
)


def load(prefix):
    merged = None
    for path in sorted(glob.glob(f"{prefix}.*.pstats")):
        stats = pstats.Stats(path)
        merged = stats if merged is None else merged.add(path)
    if merged is None:
        raise SystemExit(f"no pstats matching {prefix}.*.pstats")
    return merged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("a")
    ap.add_argument("b")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--top", type=int, default=24)
    ap.add_argument("--floor", type=float, default=0.004, help="ms/step cutoff")
    args = ap.parse_args()

    a, b = load(args.a), load(args.b)
    per = args.steps * args.workers

    rows = []
    for key in set(a.stats) | set(b.stats):
        an = a.stats.get(key, (0, 0, 0, 0, {}))
        bn = b.stats.get(key, (0, 0, 0, 0, {}))
        name = f"{os.path.basename(key[0])}:{key[1]}({key[2]})"
        if any(p in name for p in SPIN):
            continue
        d_ms = (bn[2] - an[2]) * 1000 / per
        if abs(d_ms) < args.floor:
            continue
        rows.append((d_ms, (bn[1] - an[1]) / per, name))

    rows.sort(key=lambda r: -abs(r[0]))
    print(f"{'dtot ms/step':>12} {'dcalls/step':>12}  frame   ({args.b} minus {args.a})")
    for d_ms, d_calls, name in rows[: args.top]:
        print(f"{d_ms:>12.4f} {d_calls:>12.2f}  {name}")
    print(f"\nsum of all non-spin rows: {sum(r[0] for r in rows):.3f} ms/step")
    print("NOTE: rows with dcalls 0.00 are unchanged work running slower --")
    print("      amplification, not an optimisation target.")


if __name__ == "__main__":
    main()
