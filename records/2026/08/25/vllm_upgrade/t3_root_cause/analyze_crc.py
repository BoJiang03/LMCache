"""Cross-run CRC analysis for the T3 store-corruption root cause.

Reads mp_server_run{A,B,C}.log (T3DBG instrumentation from sitecustomize.py):
  FW lines carry per-key CRC32 of the chunk bytes at finish_write time;
  RR fail lines carry ordered key lists, from which within-chain position
  (prev-key) maps are built.

Outputs:
  - pairwise mismatch counts and shift attribution (X[k] == Y[prev(k)]
    means X stored chunk k-1's bytes under key k, i.e. X is shifted);
  - within-run adjacent-duplicate CRC pairs (obj_i == obj_{i-1}), the
    in-run fingerprint of the mis-copy onset.

Result on 2026-08-25 (qwen2-vl-2b, vllm 0.23.0 venv, GPU 6):
  A vs C: 169 mismatches, A-shifted 133, C-shifted 0
  B vs C: 154 mismatches, B-shifted 128, C-shifted 0
  within-run duplicates: A=31 B=36 C=0
Run C differs from A/B only by a stream-synchronize inserted before the
torch-fallback lmcache_memcpy_async's synchronous cudaMemcpy.
"""
import glob
import re
import sys

BASE = sys.argv[1] if len(sys.argv) > 1 else "."

def fw(path):
    d = {}
    for line in open(path, errors="replace"):
        if "] FW " in line:
            for k, c in re.findall(r"\('([0-9a-f]{10})', '([0-9a-f]{8})'\)", line):
                d.setdefault(k, c)
    return d

prev = {}
for path in glob.glob(f"{BASE}/mp_server_run*.log"):
    for line in open(path, errors="replace"):
        if "] RR " not in line or "keys=" not in line:
            continue
        keys = re.findall(r"'([0-9a-f]{10})'", line.split("keys=")[-1])
        for i in range(1, len(keys)):
            prev.setdefault(keys[i], keys[i - 1])

runs = {r: fw(f"{BASE}/mp_server_run{r}.log") for r in ("A", "B", "C")}

def direction(x, y):
    X, Y = runs[x], runs[y]
    mism = [k for k in set(X) & set(Y) if X[k] != Y[k]]
    xs = sum(1 for k in mism if k in prev and prev[k] in Y and X[k] == Y[prev[k]])
    ys = sum(1 for k in mism if k in prev and prev[k] in X and Y[k] == X[prev[k]])
    print(f"{x} vs {y}: mismatch={len(mism)} {x}-shifted={xs} {y}-shifted={ys}")

direction("A", "B"); direction("A", "C"); direction("B", "C")
for r, X in runs.items():
    n = sum(1 for k, p in prev.items() if k in X and p in X and X[k] == X[p])
    print(f"{r}: within-run adjacent-duplicate CRC pairs: {n}")
