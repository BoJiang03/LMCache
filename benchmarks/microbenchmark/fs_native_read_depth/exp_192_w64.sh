#!/usr/bin/env bash
# The one cell where a tuned num_workers=64 might lose: 192 MiB objects,
# where 64 in flight is 12 GiB.  The earlier single pass (probe_quick) read
# w64 49.99 vs POOL 52.65 vs w4 53.66 with p95 219 vs 147 ms, uncontrolled.
# Same arms, now with a discarded warmup and two interleaved rounds.
#
# CLAIM ONLY IF: POOL - w64 gap exceeds twice the larger within-arm drift.
# Otherwise the PR says "ties a tuned num_workers at every object size".
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/w64_192
OUT=${WORK:?directory on the storage under test}/expN/w64_192.log
TREE=${TREE:?LMCache checkout under test, built in place}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "192 MiB, w64 vs pool, controlled  tree=$(git -C "$TREE" rev-parse --short HEAD)+move  $(date -Is)" >> "$OUT"
run() {  # arm workers depth budget reuse dur
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib 196608 --corpus-gib 48 --workers "$2" --batch 16 \
    --outstanding 3 --io-depth "$3" --budget-mib "$4" \
    --duration-s "$6" --probe-gap-ms 100 --arm "$1" --keep-corpus $5 2>&1 \
  | grep -E "^RESULT|Error|Traceback" >> "$OUT"
}
run warmup_discard 4 0 0 "" 15
for rep in 1 2; do
  run "w64_r${rep}"  64 0  0    "--reuse-corpus" 20
  run "POOL_r${rep}" 4  64 1536 "--reuse-corpus" 20
  run "w4_r${rep}"   4  0  0    "--reuse-corpus" 20
done
echo "finished $(date -Is)" >> "$OUT"
