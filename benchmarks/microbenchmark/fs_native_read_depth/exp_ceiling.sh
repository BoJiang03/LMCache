#!/usr/bin/env bash
# THE CORRECTED COMPARISON.  The earlier probe run tested the wrong cell.
#
# It compared num_workers=64 against POOL at depth=64.  At small objects those
# are the SAME concurrency (64 objects in flight) and the budget never binds,
# so a tie there says nothing.  It never tested the pool's actual claim.
#
# The claim is "depth is a ceiling, not a target": set depth HIGH so small
# objects reach bandwidth, and the byte budget stops that same setting from
# over-queueing large objects, because bytes/object_size falls as objects grow.
#
#   1.5 MiB, budget 1536 MiB -> 1024 objects allowed, so DEPTH binds
#   192 MiB, budget 1536 MiB ->    8 objects allowed, so BUDGET binds
#
# A static num_workers cannot do that: it is one count at every object size.
#
# Already measured at 192 MiB: w=4 -> 53.66 GB/s p95 142.4; 48 objects in
# flight (any large num_workers) -> 49.99 GB/s p95 218.7; pool at 8 objects ->
# 52.65 GB/s p95 146.8.  So high concurrency already costs 5.3% bandwidth and
# 49% tail there.  The missing number is whether POOL at depth=256 matches
# w=256's 49.64 GB/s at 1.5 MiB.
#
# PASS: within 5% of best bandwidth AND within 2x best probe p50, AT BOTH ends.
# PREDICTION: only POOL at depth>=256 passes.
# FALSIFIER: a single num_workers passes both -> the pool adds nothing here.
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/ceil
OUT=${WORK:?directory on the storage under test}/expN/ceiling.log
TREE=${TREE:?LMCache checkout under test, built in place}
DUR=20
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "depth-ceiling test  commit=$(git -C "$TREE" rev-parse --short HEAD) started $(date -Is)" >> "$OUT"

run() {  # arm obj batch workers depth budget reuse
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib 48 --workers "$4" --batch "$3" \
    --outstanding 3 --io-depth "$5" --budget-mib "$6" \
    --duration-s "$DUR" --probe-gap-ms 100 --arm "$1" --keep-corpus $7 2>&1 \
  | grep -E "^RESULT|Error|Traceback|error" >> "$OUT"
}

echo "======== 1.5 MiB (chunk_size 64): budget allows 1024 objects, DEPTH binds ========" >> "$OUT"
run w4            1536 512 4   0   0    ""
run w256          1536 512 256 0   0    "--reuse-corpus"
run POOL_d64      1536 512 4   64  1536 "--reuse-corpus"
run POOL_d256     1536 512 4   256 1536 "--reuse-corpus"
run POOL_d512     1536 512 4   512 1536 "--reuse-corpus"
run w256_drift    1536 512 256 0   0    "--reuse-corpus"
rm -rf "$CORPUS"

echo "======== 192 MiB (chunk_size 8192): budget allows 8 objects, BUDGET binds ========" >> "$OUT"
run w4            196608 16 4   0   0    ""
run w256          196608 16 256 0   0    "--reuse-corpus"
run POOL_d256     196608 16 4   256 1536 "--reuse-corpus"
run POOL_d512     196608 16 4   512 1536 "--reuse-corpus"
run w4_drift      196608 16 4   0   0    "--reuse-corpus"
rm -rf "$CORPUS"
echo "finished $(date -Is)" >> "$OUT"
