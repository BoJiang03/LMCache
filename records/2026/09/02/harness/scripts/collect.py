#!/usr/bin/env python3
"""Aggregate `vllm bench serve` result JSONs into a comparison table.

    scripts/collect.py [results/phase1] [warm] [--base CFG]

Runs that did not complete (engine crash -> completed == 0) are shown as
"crash", never as 0.0 -- a zero here reads as a real, very slow measurement and
that is exactly how a lost c=1500 point nearly got quoted as data.
"""
import json, sys, glob, os, re
from collections import defaultdict

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
base_cfg = None
for a in sys.argv[1:]:
    if a.startswith("--base="):
        base_cfg = a.split("=", 1)[1]

root = argv[0] if len(argv) > 0 else "results/phase1"
pass_kind = argv[1] if len(argv) > 1 else "warm"

rows = defaultdict(dict)
crashed = set()
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
    if not d.get("completed"):          # 0/N -> the run died; not a datapoint
        crashed.add((c, cfg))
        continue
    rows[c][cfg] = d

cfgs = sorted({k for v in rows.values() for k in v} | {c for _, c in crashed})
if not cfgs:
    print(f"no {pass_kind} results under {root}")
    sys.exit(0)
if base_cfg and base_cfg not in cfgs:
    print(f"--base={base_cfg} not among {cfgs}")
    sys.exit(1)

W = max(14, max(len(c) for c in cfgs) + 2)

for metric, label, scale, higher_better in [
    ("p99_ttft_ms", "P99 TTFT (s)", 1000.0, False),
    ("mean_ttft_ms", "mean TTFT (s)", 1000.0, False),
    ("total_token_throughput", "total tok/s", 1.0, True),
]:
    print(f"\n=== {label}  [{pass_kind} pass] ===")
    head = f"{'conc':>6} " + " ".join(f"{c:>{W}}" for c in cfgs)
    if base_cfg:
        head += "   " + " ".join(f"{c+'/'+base_cfg:>{W}}" for c in cfgs if c != base_cfg)
    print(head)
    for c in sorted(rows | {k: None for k, _ in crashed}):
        cells, vals = [], {}
        for cfg in cfgs:
            if (c, cfg) in crashed:
                cells.append(f"{'crash':>{W}}"); continue
            d = rows.get(c, {}).get(cfg)
            v = d.get(metric) if d else None
            if v is None:
                cells.append(f"{'-':>{W}}")
            else:
                v /= scale
                vals[cfg] = v
                cells.append(f"{v:>{W},.1f}" if scale == 1.0 else f"{v:>{W}.1f}")
        line = f"{c:>6} " + " ".join(cells)
        if base_cfg:
            b = vals.get(base_cfg)
            ratios = []
            for cfg in cfgs:
                if cfg == base_cfg:
                    continue
                v = vals.get(cfg)
                # always report as a cost multiplier: >1 means worse than base
                r = (b / v if higher_better else v / b) if (b and v) else None
                ratios.append(f"{r:>{W}.2f}x" if r else f"{'-':>{W}}")
            line += "   " + " ".join(ratios)
        print(line)

if crashed:
    print("\ncrashed / incomplete:", ", ".join(f"c={c} {cfg}" for c, cfg in sorted(crashed)))
