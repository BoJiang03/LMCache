#!/usr/bin/env python3
"""Aggregate the TIMER lines that timedconn/timed_mp_connector.py emits.

    scripts/timer_report.py results/phase1/1j_timed_mp/server.log [--lo=70] [--hi=130]

Each connector process prints one TIMER line per 200 engine steps.  This folds
them into per-hook ms/step, separately for the scheduler process and the eight
TP workers, and only over windows whose own wall_ms/step sits in the steady
prefill band -- startup, the ramp and the drain are not the regime the 5.7 ms
was measured in, and averaging them in would smear it.

The number to compare against is 5.7 ms/step (MP 91.0 minus no-connector 85.3).
`hooks` is the sum of every timed hook; if it does not add up to 5.7 the rest is
off-hook, and `cpu_busy` says whether it is at least CPU time.
"""
import re
import sys
import statistics as st
from collections import defaultdict

LINE = re.compile(
    r"TIMER pid=(\d+) role=(\S+) steps=(\d+) wall_ms/step=([\d.]+) "
    r"hooks_ms/step=([\d.]+) cpu_busy=([\d.]+) \| (.*)$")
HOOK = re.compile(r"(\w+)=([\d.]+)\(([\d.]+)x\)")

path = sys.argv[1]
lo, hi = 70.0, 130.0
for a in sys.argv[2:]:
    if a.startswith("--lo="):
        lo = float(a.split("=", 1)[1])
    if a.startswith("--hi="):
        hi = float(a.split("=", 1)[1])

windows = []
for raw in open(path, errors="ignore"):
    m = LINE.search(re.sub(r"\x1b\[[0-9;]*m", "", raw))
    if not m:
        continue
    pid, role, steps, wall, hooks, cpu, rest = m.groups()
    windows.append(dict(pid=int(pid), role=role, steps=int(steps),
                        wall=float(wall), hooks=float(hooks), cpu=float(cpu),
                        per={k: (float(v), float(c)) for k, v, c in HOOK.findall(rest)}))

if not windows:
    print(f"no TIMER lines in {path}")
    sys.exit(0)

kept = [w for w in windows if lo <= w["wall"] <= hi]
print(f"{len(windows)} TIMER windows, {len(kept)} inside the steady band "
      f"[{lo:g}, {hi:g}] ms/step  ({len({w['pid'] for w in kept})} processes)\n")

by_role = defaultdict(list)
for w in kept:
    by_role[w["role"]].append(w)

for role in sorted(by_role):
    ws = by_role[role]
    pids = sorted({w["pid"] for w in ws})
    print(f"=== role={role}  {len(ws)} windows over {len(pids)} process(es) ===")
    print(f"  wall     {st.median(w['wall'] for w in ws):8.2f} ms/step (median)")
    print(f"  hooks    {st.median(w['hooks'] for w in ws):8.3f} ms/step (median)"
          f"   <- compare with 5.7")
    print(f"  cpu_busy {st.median(w['cpu'] for w in ws):8.2f}")
    agg = defaultdict(list)
    for w in ws:
        for k, (ms, calls) in w["per"].items():
            agg[k].append((ms, calls))
    print(f"  {'hook':38} {'ms/step':>9} {'calls/step':>11}")
    for k in sorted(agg, key=lambda k: -st.median(m for m, _ in agg[k])):
        ms = st.median(m for m, _ in agg[k])
        ca = st.median(c for _, c in agg[k])
        if ms < 0.0005 and ca == 0:
            continue
        print(f"  {k:38} {ms:9.3f} {ca:11.1f}")
    print()
