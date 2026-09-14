#!/usr/bin/env bash
# The headline cell, measured properly.  chunk_size=256 -> 6 MiB objects is
# LMCache's default and the number the PR leads with, and it is the one cell
# where the pool has only ever been run at depth=64.
#
# N3 just measured num_workers=64 at 53.0 GB/s here against the pool's 50.0 at
# depth=64, i.e. the pool LOSING 6% at the default setting.  If that holds at
# depth=256 the PR's headline is wrong.
#
# At 6 MiB the budget allows 1536/6 = 256 objects, so depth and budget bind at
# the same point: depth=256 should match w=256, depth=64 should not.
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/dflt
OUT=${WORK:?directory on the storage under test}/expN/default_cell.log
TREE=${TREE:?LMCache checkout under test, built in place}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "default cell: 6 MiB objects (chunk_size 256)  commit=$(git -C "$TREE" rev-parse --short HEAD)  $(date -Is)" >> "$OUT"
run() {
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib 6144 --corpus-gib 48 --workers "$3" --batch 256 \
    --outstanding 3 --io-depth "$4" --budget-mib "$5" \
    --duration-s 25 --probe-gap-ms 100 --arm "$1" --keep-corpus $2 2>&1 \
  | grep -E "^RESULT|Error|Traceback" >> "$OUT"
}
run w4        ""               4   0   0
run w64       "--reuse-corpus" 64  0   0
run w256      "--reuse-corpus" 256 0   0
run POOL_d64  "--reuse-corpus" 4   64  1536
run POOL_d256 "--reuse-corpus" 4   256 1536
run POOL_d512 "--reuse-corpus" 4   512 1536
run w64_drift "--reuse-corpus" 64  0   0
echo "finished $(date -Is)" >> "$OUT"
