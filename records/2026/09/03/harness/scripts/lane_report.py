#!/usr/bin/env python3
"""Rank the TP=4 lane's arms -- but only after the lane proves it is valid.

The lane exists because another tenant took GPUs 4-7 and TP=8 became
unlaunchable.  TP=4 numbers are NOT comparable with phase1's 85.3/91.0/97.5;
every comparison must be internal to the lane.  That is only sound if the lane
still shows the thing under investigation, so this refuses to rank arms until

    mp (or timed) is at least 2% slower per step than none

which is the phenomenon phase1 measured at +6.7%.  If the lane does not
reproduce it, the arms are measuring a different regime and their agreement or
disagreement means nothing.

Underlying numbers come from scripts/engine_rate.py, which reads vLLM's own
in-engine `Avg prompt throughput` lines rather than the bench client's
end-to-end aggregate -- the two differ by ramp and drain, and mixing them is
mistake 2 of 2026-09-02 record 9.
"""
import re
import subprocess
import sys
import os

HERE = os.path.dirname(os.path.abspath(__file__))
root = sys.argv[1] if len(sys.argv) > 1 else os.path.join(HERE, "..", "results", "lane")

out = subprocess.run([sys.executable, os.path.join(HERE, "engine_rate.py"),
                      root, "--min-outstanding=0"],
                     capture_output=True, text=True).stdout
print(out)

ROW = re.compile(r"^(\S+)\s+\d\d:\d\d:\d\d\.\.\d\d:\d\d:\d\d\s+(\d+)s\s+(\d+)"
                 r"\s+([\d,]+)\s+([\d.]+)\s+(\d+)")
ms = {}
for line in out.splitlines():
    m = ROW.match(line)
    if m:
        # An arm can produce several blocks; keep the longest, which is the
        # measured pass rather than a warmup or a drain fragment.
        arm, dur, val = m.group(1), int(m.group(2)), float(m.group(5))
        if arm not in ms or dur > ms[arm][1]:
            ms[arm] = (val, dur)

if "none" not in ms:
    print("VERDICT: the baseline arm `none` has not run yet; nothing to rank.")
    sys.exit(0)

base = ms["none"][0]
ref = ms.get("mp") or ms.get("timed")
print(f"lane baseline `none` = {base:.1f} ms/step")
if ref is None:
    print("VERDICT: neither `mp` nor `timed` has run; the lane is unvalidated.")
    sys.exit(0)

gap = ref[0] - base
print(f"lane stock LMCache      = {ref[0]:.1f} ms/step   gap = {gap:+.1f} ms/step "
      f"({100 * gap / base:+.1f}%)")
if gap < 0.02 * base:
    print("VERDICT: LANE VOID -- TP=4 does not reproduce the connector tax, so the")
    print("         store/lookup arms below cannot be interpreted.  Wait for GPUs")
    print("         4-7 and redo this at TP=8.")
    sys.exit(0)
print("VERDICT: lane reproduces the tax; the arms below are interpretable.\n")
print(f"{'arm':<12}{'ms/step':>9}{'vs none':>10}{'tax removed':>14}")
for arm, (v, _d) in sorted(ms.items(), key=lambda kv: kv[1][0]):
    removed = "" if gap <= 0 else f"{100 * (ref[0] - v) / gap:>12.0f}%"
    print(f"{arm:<12}{v:>9.1f}{v - base:>+10.1f}{removed:>14}")
