#!/usr/bin/env python3
"""Per-step cost decomposition from vLLM's own in-engine counters.

    scripts/engine_rate.py [results/phase1] [--min-outstanding 700]

Why this exists.  End-to-end `vllm bench serve` numbers pair runs from different
sittings on a shared box, and one arm drifted 28% between sessions once.  This
reads vLLM's `Avg prompt throughput` stat lines out of each arm's server.log
instead, which is a measurement taken INSIDE the engine during the run.  It
reproduced the end-to-end ratios independently, and it converts them into a
number that points at code: milliseconds per forward step.

The conversion is only valid because every arm runs the same
`max_num_batched_tokens` (8192 here, asserted below), so a step is a fixed
number of tokens and

    ms/step = 1000 * max_num_batched_tokens / (tokens per second)

The reported rate is quantised in units of ~6000 tok/s (one 60,000-token prompt
per 10 s logging interval), so the histogram is the honest view: the mode is the
steady state and the tail is startup and drain.

Blocks are split on gaps > 60 s, and only blocks whose peak outstanding request
count clears --min-outstanding are reported, which selects the c>=1000 phases
and drops the small-concurrency warmups from a multi-point run.
"""
import re, sys, glob, os, subprocess
from collections import Counter

argv = [a for a in sys.argv[1:] if not a.startswith("--")]
root = argv[0] if argv else "results/phase1"
MIN_OUT = 700
for a in sys.argv[1:]:
    if a.startswith("--min-outstanding="):
        MIN_OUT = int(a.split("=", 1)[1])

STAT = re.compile(
    r"INFO \d\d-\d\d (\d\d:\d\d:\d\d).*Avg prompt throughput: ([\d.]+) tokens/s"
    r".*Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: (\d+) reqs)?"
)
BATCH = re.compile(r"max_num_batched_tokens=(\d+)")

def secs(t):
    h, m, s = map(int, t.split(":"))
    return h * 3600 + m * 60 + s

def read(path):
    # server.log carries ANSI colour; py-spy runs put vLLM's stdout in pyspy.log
    txt = subprocess.run(
        f"sed 's/\\x1b\\[[0-9;]*m//g' {path}", shell=True,
        capture_output=True, text=True).stdout
    rows, batch = [], None
    for l in txt.splitlines():
        if batch is None:
            m = BATCH.search(l)
            if m:
                batch = int(m.group(1))
        m = STAT.search(l)
        if m:
            rows.append((secs(m.group(1)), m.group(1), float(m.group(2)),
                         int(m.group(3)), int(m.group(4)), int(m.group(5) or 0)))
    return rows, batch

results = []
for d in sorted(glob.glob(os.path.join(root, "*"))):
    log = next((p for p in (os.path.join(d, "server.log"), os.path.join(d, "pyspy.log"))
                if os.path.exists(p)), None)
    if not log:
        continue
    rows, batch = read(log)
    if not rows:
        continue
    blocks, cur = [], []
    for r in rows:
        if cur and r[0] - cur[-1][0] > 60:
            blocks.append(cur); cur = []
        cur.append(r)
    if cur:
        blocks.append(cur)
    for b in blocks:
        peak = max(r[3] + r[4] + r[5] for r in b)
        if peak < MIN_OUT:
            continue
        rates = sorted(r[2] for r in b)
        n = len(rates)
        results.append(dict(
            arm=os.path.basename(d), batch=batch, start=b[0][1], end=b[-1][1],
            dur=b[-1][0] - b[0][0], peak=peak, n=n,
            mean=sum(rates) / n, p50=rates[n // 2],
            hist=dict(sorted(Counter(round(r[2] / 6000) for r in b).items())),
            defmax=max(r[5] for r in b)))

if not results:
    print(f"no blocks with peak outstanding >= {MIN_OUT} under {root}")
    sys.exit(0)

batches = {r["batch"] for r in results if r["batch"]}
if len(batches) > 1:
    print(f"!! max_num_batched_tokens differs across arms {sorted(batches)}; "
          "ms/step is NOT comparable.  Showing rates only.")
B = batches.pop() if len(batches) == 1 else None
print(f"max_num_batched_tokens = {B}  (a step is this many tokens in every arm)\n")

print(f"{'arm':<22}{'block':<19}{'dur':>6}{'n':>5}{'p50 tok/s':>12}"
      f"{'ms/step':>10}{'Def max':>9}  histogram (units of 6000 tok/s)")
for r in sorted(results, key=lambda x: -x["p50"]):
    ms = 1000 * B / r["p50"] if B and r["p50"] else float("nan")
    print(f"{r['arm']:<22}{r['start']+'..'+r['end']:<19}{r['dur']:>5}s{r['n']:>5}"
          f"{r['p50']:>12,.0f}{ms:>10.1f}{r['defmax']:>9}  {r['hist']}")

best = max(r["p50"] for r in results)
print(f"\ndeltas against the fastest block ({best:,.0f} tok/s = "
      f"{1000*B/best:.1f} ms/step):" if B else "")
for r in sorted(results, key=lambda x: -x["p50"]):
    if B and r["p50"]:
        print(f"  {r['arm']:<22} {1000*B/r['p50'] - 1000*B/best:>+7.1f} ms/step"
              f"   ({best/r['p50']:.3f}x slower)")
