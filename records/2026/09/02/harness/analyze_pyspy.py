#!/usr/bin/env python3
"""Aggregate a py-spy speedscope profile and attribute time to the KV connector.

    scripts/analyze_pyspy.py <profile.speedscope.json> [--top N] [--min-share 0.005]

The question this answers: of the wall clock a vLLM process spends, how much sits
underneath LMCache or vLLM's kv_transfer plumbing?  1h's target is 5.6 ms per
8192-token forward step, which is ~6% of wall clock -- so a share of a few
percent under connector frames is a hit, and ~0% means the cost is native and
the follow-up is a --native profile or a no-connector baseline to diff against.

Every number is a SHARE OF SAMPLES, i.e. of wall clock, not of CPU: the profile
is taken with --idle so a thread blocked in a CUDA sync or a socket read still
counts.  That is deliberate -- a block is as good an explanation as a burn --
but it means a share is only comparable within the same thread.
"""
import json, sys, re
from collections import defaultdict, Counter

args = [a for a in sys.argv[1:] if not a.startswith("--")]
TOP = 15
MIN_SHARE = 0.005
for a in sys.argv[1:]:
    if a.startswith("--top="): TOP = int(a.split("=", 1)[1])
    if a.startswith("--min-share="): MIN_SHARE = float(a.split("=", 1)[1])
if not args:
    print(__doc__); sys.exit(1)

# A frame is "connector" if it belongs to LMCache or to vLLM's KV-transfer
# plumbing.  Kept as substrings of the frame's file/name so a rename upstream
# shows up as a miss rather than as a silent zero.
CONNECTOR = re.compile(
    r"lmcache|kv_transfer|kv_connector|maybe_transfer_kv_layer|"
    r"kv_connector_model_runner_mixin",
    re.I,
)
PROC_RE = re.compile(r'^process (\d+):"(.*)"$', re.S)

d = json.load(open(args[0]))
frames = d["shared"]["frames"]
fname = [f.get("name", "?") for f in frames]
ffile = [f.get("file", "") or "" for f in frames]
label = [f"{n}  ({p})" if p else n for n, p in zip(fname, ffile)]
is_conn = [bool(CONNECTOR.search(n + " " + p)) for n, p in zip(fname, ffile)]

# process frame -> cmdline, so profiles can be named EngineCore / Worker / API
proc_cmd = {}
for i, n in enumerate(fname):
    m = PROC_RE.match(n)
    if m:
        proc_cmd[i] = (m.group(1), m.group(2))

def role(cmd: str) -> str:
    c = cmd or ""
    if "EngineCore" in c or "engine_core" in c: return "EngineCore"
    if "VLLM::Worker" in c or "worker" in c.lower(): return "Worker"
    if "lmcache" in c.lower(): return "LMCacheServer"
    if "api_server" in c or "vllm" in c.lower(): return "APIServer/other"
    return "other"

print(f"profile: {args[0]}")
print(f"threads: {len(d['profiles'])}  frames: {len(frames)}  "
      f"connector frames: {sum(is_conn)}")

rows = []
grand_tot = 0.0
grand_conn = 0.0
for prof in d["profiles"]:
    samples = prof.get("samples", [])
    weights = prof.get("weights") or [1.0] * len(samples)
    tot = sum(weights)
    if tot <= 0:
        continue
    conn = 0.0
    self_t = Counter()
    conn_entry = Counter()      # the outermost connector frame -> time under it
    for st, w in zip(samples, weights):
        if not st:
            continue
        self_t[st[-1]] += w
        hit = next((f for f in st if is_conn[f]), None)
        if hit is not None:
            conn += w
            conn_entry[hit] += w
    # Name the thread by its OWN process, not the tree root: with
    # --subprocesses a stack reads "process parent;process child;frames...",
    # so the innermost process frame is the one that owns these samples.
    pid, cmd = "?", ""
    for st in samples:
        if not st:
            continue
        procs = [f for f in st if f in proc_cmd]
        if procs:
            pid, cmd = proc_cmd[procs[-1]]
            break
    rows.append((prof.get("name", "?"), role(cmd), pid, tot, conn, self_t, conn_entry))
    grand_tot += tot
    grand_conn += conn

rows.sort(key=lambda r: -r[3])
print(f"\n=== per-thread wall clock, connector share ===")
print(f"{'role':<16}{'pid':>8}{'thread':<34}{'samples':>10}{'conn%':>8}")
for name, rl, pid, tot, conn, _, _ in rows:
    if tot / grand_tot < 1e-4:
        continue
    th = name.split("Thread", 1)[-1].strip() if "Thread" in name else name
    print(f"{rl:<16}{pid:>8}{th[:33]:<34}{tot:>10,.0f}{100*conn/tot:>7.2f}%")
print(f"{'ALL':<16}{'':>8}{'':<34}{grand_tot:>10,.0f}{100*grand_conn/grand_tot:>7.2f}%")

print(f"\n=== where the connector time actually enters (per thread, share >= {MIN_SHARE:.1%}) ===")
any_hit = False
for name, rl, pid, tot, conn, self_t, conn_entry in rows:
    if conn / max(tot, 1) < MIN_SHARE:
        continue
    any_hit = True
    th = name.split("Thread", 1)[-1].strip() if "Thread" in name else name
    print(f"\n  {rl} pid={pid} {th}   connector = {100*conn/tot:.2f}% of {tot:,.0f} samples")
    for fi, w in conn_entry.most_common(TOP):
        if w / tot < MIN_SHARE / 4:
            break
        print(f"     {100*w/tot:>6.2f}%  {label[fi]}")
if not any_hit:
    print(f"  (no thread spends >= {MIN_SHARE:.1%} of its wall clock under a connector frame)")
    print("  -> the tax is not Python-level.  Next: same run with --native, or a")
    print("     no-connector baseline profile to diff frame-by-frame.")

print(f"\n=== top self-time frames in the busiest threads (for the baseline diff) ===")
for name, rl, pid, tot, conn, self_t, _ in rows[:4]:
    th = name.split("Thread", 1)[-1].strip() if "Thread" in name else name
    print(f"\n  {rl} pid={pid} {th}  ({tot:,.0f} samples)")
    for fi, w in self_t.most_common(TOP):
        if w / tot < MIN_SHARE:
            break
        mark = " <-- connector" if is_conn[fi] else ""
        print(f"     {100*w/tot:>6.2f}%  {label[fi]}{mark}")
