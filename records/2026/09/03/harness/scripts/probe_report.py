#!/usr/bin/env python3
"""Steady-state host-side step accounting, from sitecustomize's probe.

    scripts/probe_report.py [results/phase1] [--skip 2]

WHY DELTAS AND NOT THE LAST LINE.  Each STEPPROBE line is a CUMULATIVE average
since the process started, which folds in CUDA graph capture, the profiling run
and the ramp.  The steady state is the difference between consecutive lines, so
totals are reconstructed as (per-step value x steps) and differenced -- the same
treatment timer_report.py gives the hook timers.

COLUMNS, per model-runner step, averaged over the TP workers

    loop      wall between successive entries of execute_model
    exec      wall inside execute_model
    cpu       thread CPU inside execute_model, same thread
    exec-cpu  the part of exec the main thread spent BLOCKED, not running
              bytecode.  execute_model launches asynchronously but ends by
              touching device results, so this is the GPU.

    exec-cpu grows with LMCache attached -> the device is doing more work
    cpu grows, exec-cpu flat             -> the cost is host-side, the GIL
"""
import glob
import os
import re
import subprocess
import sys
from collections import defaultdict

LINE = re.compile(
    r"STEPPROBE pid=(\d+) steps=(\d+) loop_ms/step=([\d.]+) "
    r"exec_wall_ms/step=([\d.]+) exec_cpu_ms/step=([\d.]+)")

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
root = argv[0] if argv else "results/phase1"
SKIP = 2
for a in sys.argv[1:]:
    if a.startswith("--skip="):
        SKIP = int(a.split("=", 1)[1])

rows = []
for d in sorted(glob.glob(os.path.join(root, "*"))):
    log = os.path.join(d, "server.log")
    if not os.path.exists(log):
        continue
    txt = subprocess.run(f"sed 's/\\x1b\\[[0-9;]*m//g' {log}", shell=True,
                         capture_output=True, text=True).stdout
    per_pid = defaultdict(list)
    for line in txt.splitlines():
        m = LINE.search(line)
        if m:
            steps = int(m.group(2))
            loop, ew, ec = (float(m.group(i)) for i in (3, 4, 5))
            per_pid[int(m.group(1))].append((steps, loop * steps, ew * steps,
                                             ec * steps))
    if not per_pid:
        continue
    tot = [0.0, 0.0, 0.0, 0.0]
    wins = 0
    for seq in per_pid.values():
        for a_, b_ in zip(seq[SKIP:], seq[SKIP + 1:]):
            ds = b_[0] - a_[0]
            if ds <= 0:
                continue
            wins += 1
            tot[0] += ds
            for i in (1, 2, 3):
                tot[i] += b_[i] - a_[i]
    if tot[0] == 0:
        continue
    n = tot[0]
    rows.append((os.path.basename(d), wins, len(per_pid),
                 tot[1] / n, tot[2] / n, tot[3] / n))

if not rows:
    print(f"no STEPPROBE lines under {root}")
    sys.exit(0)

print(f"{'arm':<22}{'wins':>6}{'procs':>6}{'loop':>9}{'exec':>9}{'cpu':>8}"
      f"{'exec-cpu':>10}   (ms/step, steady state)")
for name, wins, procs, loop, ew, ec in sorted(rows, key=lambda r: r[3]):
    print(f"{name:<22}{wins:>6}{procs:>6}{loop:>9.2f}{ew:>9.2f}{ec:>8.2f}"
          f"{ew - ec:>10.2f}")

if len(rows) >= 2:
    base = min(rows, key=lambda r: r[3])
    print(f"\nagainst the fastest arm ({base[0]}):")
    for name, _w, _p, loop, ew, ec in sorted(rows, key=lambda r: r[3]):
        print(f"  {name:<22} loop{loop - base[3]:>+8.2f}"
              f"  exec{ew - base[4]:>+8.2f}  cpu{ec - base[5]:>+8.2f}"
              f"  blocked{(ew - ec) - (base[4] - base[5]):>+8.2f}")
    print("\nread: `blocked` tracks the loop delta -> the GPU is doing more work;"
          "\n      `cpu` tracks it -> the cost is host-side.")
