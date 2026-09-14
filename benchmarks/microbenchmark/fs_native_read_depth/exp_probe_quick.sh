#!/usr/bin/env bash
# N4-lite: the fastest thing that can answer "does num_workers fail".
#
# Only the two ends of the object-size range, because that is where a single
# num_workers is forced to choose:
#   1.5 MiB  small objects need MANY in flight to reach bandwidth
#   192 MiB  large objects need FEW, or the queue is pure latency
# Little's law: inflight = BW x latency.  Past saturation BW is pinned, so
# every extra byte in flight is paid entirely in latency.
#
# legacy inflight = num_workers x object_size  (scales with object size)
# pooled inflight = budget                      (constant, in bytes)
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/quick
OUT=${WORK:?directory on the storage under test}/expN/probe_quick.log
TREE=${TREE:?LMCache checkout under test, built in place}
DUR=20
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "N4-lite  commit=$(git -C "$TREE" rev-parse --short HEAD)  started $(date -Is)" >> "$OUT"

run() {  # arm obj batch workers depth budget reuse
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib 48 --workers "$4" --batch "$3" \
    --outstanding 3 --io-depth "$5" --budget-mib "$6" \
    --duration-s "$DUR" --probe-gap-ms 100 --arm "$1" --keep-corpus $7 2>&1 \
  | grep -E "^RESULT|Error|Traceback|error" >> "$OUT"
}

echo "======== 1.5 MiB (chunk_size 64) ========" >> "$OUT"
run w4    1536 512 4   0  0    ""
run w64   1536 512 64  0  0    "--reuse-corpus"
run w256  1536 512 256 0  0    "--reuse-corpus"
run POOL  1536 512 4   64 1536 "--reuse-corpus"
rm -rf "$CORPUS"

echo "======== 192 MiB (chunk_size 8192) ========" >> "$OUT"
run w4    196608 16 4  0  0    ""
run w64   196608 16 64 0  0    "--reuse-corpus"
run POOL  196608 16 4  64 1536 "--reuse-corpus"
rm -rf "$CORPUS"
echo "finished $(date -Is)" >> "$OUT"
