#!/usr/bin/env python3
"""Diff two sets of cProfile dumps taken over a matched step window.

Each arm dumps one .pstats per worker (named <prefix>.<pid>.pstats), so the
totals here are summed over all ranks.  Everything is reported as
milliseconds per step per worker:

    ms/step/worker = 1000 * total_seconds / (steps * workers)

which is the same unit as the step probe's `cpu` column, so a line in this
table can be read directly against the 3.86 ms/step that nostore removes.

Sorted by |delta| in tottime -- tottime, not cumtime, because we are looking
for where the CPU is actually burned, and cumtime double-counts every caller
on the path down to it.  A cumtime column is printed alongside so a wrapper
that merely contains the cost can be told apart from the leaf that pays it.
"""
import argparse
import glob
import pstats
import sys


def load(prefix):
    files = sorted(glob.glob(f"{prefix}.*.pstats"))
    if not files:
        sys.exit(f"no pstats matching {prefix}.*.pstats")
    st = pstats.Stats(files[0])
    for f in files[1:]:
        st.add(f)
    return st, files


def table(st):
    """func -> (ncalls, tottime, cumtime), keyed by a printable name."""
    out = {}
    for func, (cc, nc, tt, ct, _callers) in st.stats.items():
        fn, ln, name = func
        if fn.startswith("<") or fn == "~":
            key = f"{name}"
        else:
            short = fn
            for marker in ("/site-packages/", "/lmcache/", "/vllm/"):
                i = short.find(marker)
                if i >= 0:
                    short = short[i + 1:]
                    break
            key = f"{short}:{ln}({name})"
        prev = out.get(key, (0, 0.0, 0.0))
        out[key] = (prev[0] + nc, prev[1] + tt, prev[2] + ct)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="baseline prefix")
    ap.add_argument("--b", required=True, help="regressed prefix")
    ap.add_argument("--a-name", default="a")
    ap.add_argument("--b-name", default="b")
    ap.add_argument("--steps", type=int, required=True)
    ap.add_argument("--workers", type=int, required=True)
    ap.add_argument("--top", type=int, default=40)
    args = ap.parse_args()

    sa, fa = load(args.a)
    sb, fb = load(args.b)
    ta, tb = table(sa), table(sb)
    denom = args.steps * args.workers / 1000.0  # seconds -> ms/step/worker

    print(f"{args.a_name}: {len(fa)} workers, total {sa.total_tt:.1f}s "
          f"= {sa.total_tt / denom:.2f} ms/step/worker")
    print(f"{args.b_name}: {len(fb)} workers, total {sb.total_tt:.1f}s "
          f"= {sb.total_tt / denom:.2f} ms/step/worker")
    print(f"delta (profiled tottime, all functions): "
          f"{(sb.total_tt - sa.total_tt) / denom:+.2f} ms/step/worker")
    print()

    keys = set(ta) | set(tb)
    rows = []
    for k in keys:
        na, xa, ca = ta.get(k, (0, 0.0, 0.0))
        nb, xb, cb = tb.get(k, (0, 0.0, 0.0))
        rows.append((xb - xa, cb - ca, na, nb, xa, xb, k))
    rows.sort(key=lambda r: -abs(r[0]))

    print(f"{'d_tot':>8} {'d_cum':>8} {'calls/step':>11} "
          f"{args.a_name[:9]:>9} {args.b_name[:9]:>9}  function")
    print(f"{'ms/step':>8} {'ms/step':>8} {'a -> b':>11} "
          f"{'ms/step':>9} {'ms/step':>9}")
    for dt, dc, na, nb, xa, xb, k in rows[:args.top]:
        cps = f"{na / args.steps / args.workers:.1f}->{nb / args.steps / args.workers:.1f}"
        print(f"{dt / denom:8.3f} {dc / denom:8.3f} {cps:>11} "
              f"{xa / denom:9.3f} {xb / denom:9.3f}  {k}")


if __name__ == "__main__":
    main()
