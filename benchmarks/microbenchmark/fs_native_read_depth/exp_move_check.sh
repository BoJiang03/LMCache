#!/usr/bin/env bash
# Does the pool moved into FSConnector (pr1move) read what the base-class
# version (pr1, tiles=1 build) read, and does the ceil(n/depth) tile rule
# help a single large batch?
#
# Arms, 6 MiB objects, 48 GiB corpus, one discarded warmup then two
# interleaved rounds:
#   NEW_POOL_b16     pr1move  w4 d64 b1536  batch 16  outstanding 16
#   OLD_POOL_b16     pr1      same            (identical code path: 1 tile)
#   NEW_POOL_b480_o1 pr1move  w4 d64 b1536  batch 480 outstanding 1
#   OLD_POOL_b480_o1 pr1      same  (tiles=1: one worker, 384 MiB in flight)
#   NEW_LEG_w64_b16  pr1move  w64 d0          legacy reference
#
# PREDICTIONS: NEW_b16 == OLD_b16 within drift.  NEW_b480_o1 > OLD_b480_o1,
# because the new rule splits 480 objects into 4 tiles so all four workers'
# 384 MiB shares are in flight (1536 MiB) instead of one (384 MiB).
# FALSIFIER: NEW_b480_o1 <= OLD_b480_o1 within drift -> the tile rule buys
# nothing for a lone batch; keep it only for the small-batch case.
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/move_check
OUT=${WORK:?directory on the storage under test}/expN/move_check.log
NEW=${NEW_TREE:?checkout with the pool in FSConnector}
OLD=${OLD_TREE:?checkout with the pool in ConnectorBase}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "move check  new=$(git -C "$NEW" rev-parse --short HEAD)+$(git -C "$NEW" diff --stat | tail -1 | tr -s ' ')  old=$(git -C "$OLD" rev-parse --short HEAD)+tilefix  $(date -Is)" >> "$OUT"
run() {  # arm tree workers depth budget batch outstanding reuse dur
  PYTHONPATH="$2" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib 6144 --corpus-gib 48 --workers "$3" --batch "$6" \
    --outstanding "$7" --io-depth "$4" --budget-mib "$5" \
    --duration-s "$9" --probe-gap-ms 100 --arm "$1" --keep-corpus $8 2>&1 \
  | grep -E "^RESULT|Error|Traceback" >> "$OUT"
}
run warmup_discard "$NEW" 4 64 1536 16 16 "" 15
for rep in 1 2; do
  run "NEW_POOL_b16_r${rep}"     "$NEW" 4  64 1536 16  16 "--reuse-corpus" 20
  run "OLD_POOL_b16_r${rep}"     "$OLD" 4  64 1536 16  16 "--reuse-corpus" 20
  run "NEW_POOL_b480_o1_r${rep}" "$NEW" 4  64 1536 480 1  "--reuse-corpus" 20
  run "OLD_POOL_b480_o1_r${rep}" "$OLD" 4  64 1536 480 1  "--reuse-corpus" 20
  run "NEW_LEG_w64_b16_r${rep}"  "$NEW" 64 0  0    16  16 "--reuse-corpus" 20
done
echo "finished $(date -Is)" >> "$OUT"
