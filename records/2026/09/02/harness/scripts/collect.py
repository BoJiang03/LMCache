#!/usr/bin/env python3
"""Aggregate vllm bench serve json results into a comparison table."""
import json, sys, glob, os, re
from collections import defaultdict

root = sys.argv[1] if len(sys.argv) > 1 else "results/phase1"
pass_kind = sys.argv[2] if len(sys.argv) > 2 else "warm"

rows = defaultdict(dict)
for f in sorted(glob.glob(os.path.join(root, "*", f"*_{pass_kind}.json"))):
    cfg = os.path.basename(os.path.dirname(f))
    m = re.search(r"c(\d+)_", os.path.basename(f))
    if not m:
        continue
    c = int(m.group(1))
    try:
        d = json.load(open(f))
    except Exception:
        continue
    rows[c][cfg] = d

cfgs = sorted({k for v in rows.values() for k in v})
if not cfgs:
    print(f"no {pass_kind} results under {root}"); sys.exit(0)

def cell(d, key):
    return d.get(key) if d else None

for metric, label, scale, fmt in [
    ("p99_ttft_ms", "P99 TTFT (s)", 1000.0, "9.1f"),
    ("mean_ttft_ms", "mean TTFT (s)", 1000.0, "9.1f"),
    ("total_token_throughput", "total tok/s", 1.0, "9.0f"),
]:
    print(f"\n=== {label}  [{pass_kind} pass] ===")
    hdr = f"{'conc':>6} " + " ".join(f"{c:>20}" for c in cfgs)
    if len(cfgs) == 2:
        hdr += f"{'delta':>12}"
    print(hdr)
    for c in sorted(rows):
        vals = []
        for cfg in cfgs:
            v = cell(rows[c].get(cfg), metric)
            vals.append(v / scale if v is not None else None)
        line = f"{c:>6} " + " ".join(
            (f"{v:>20.1f}" if v is not None else f"{'-':>20}") for v in vals)
        if len(cfgs) == 2 and all(v is not None for v in vals) and vals[0]:
            line += f"{(vals[1]/vals[0]-1)*100:>11.1f}%"
        print(line)
