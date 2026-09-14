#!/usr/bin/env bash
# Sliding-window dispatch (`NEW_TREE`) against the grouped dispatch
# it replaces (`OLD_TREE`, bcb4c109), 48 GiB corpus, one discarded warmup
# then two interleaved rounds of 20 s each.
#
# PREDICTIONS: b16 arms tie within drift (a worker still blocks on its own
# 16-object tile in both designs).  NEW_b480_o1 >= OLD_b480_o1: the lone
# tile now gets the whole 1536 MiB budget with no per-group barrier, where
# the grouped version split it into four 120-object tiles drained 64 at a
# time.  At 1.5 MiB, NEW ties OLD: 36k lock round trips per second is not
# a contended mutex.
# FALSIFIERS: NEW_b480_o1 < OLD_b480_o1 by more than drift -> the window
# costs something the groups did not.  NEW_1.5MiB < OLD_1.5MiB by more than
# drift -> lock traffic shows.
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/window_check
OUT=${WORK:?directory on the storage under test}/expN/window_check.log
NEW=${NEW_TREE:?checkout with the sliding-window dispatch}
OLD=${OLD_TREE:?checkout with the grouped dispatch}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "window check  new=$(git -C "$NEW" rev-parse --short HEAD)+window(uncommitted)  old=$(git -C "$OLD" rev-parse --short HEAD) grouped  $(date -Is)  load=$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT"
run() {  # arm tree obj_kib workers depth budget batch outstanding reuse dur
  PYTHONPATH="$2" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$3" --corpus-gib 48 --workers "$4" --batch "$7" \
    --outstanding "$8" --io-depth "$5" --budget-mib "$6" \
    --duration-s "${10}" --probe-gap-ms 100 --arm "$1" --keep-corpus $9 2>&1 \
  | grep -E "^RESULT|Error|Traceback" >> "$OUT"
}
echo "======== 6 MiB" >> "$OUT"
run warmup_discard "$NEW" 6144 4 64 1536 16 16 "" 15
for rep in 1 2; do
  run "NEW_POOL_b16_r${rep}"     "$NEW" 6144 4  64 1536 16  16 "--reuse-corpus" 20
  run "OLD_POOL_b16_r${rep}"     "$OLD" 6144 4  64 1536 16  16 "--reuse-corpus" 20
  run "NEW_POOL_b480_o1_r${rep}" "$NEW" 6144 4  64 1536 480 1  "--reuse-corpus" 20
  run "OLD_POOL_b480_o1_r${rep}" "$OLD" 6144 4  64 1536 480 1  "--reuse-corpus" 20
  run "LEG_w64_b16_r${rep}"      "$NEW" 6144 64 0  0    16  16 "--reuse-corpus" 20
done
rm -rf "$CORPUS"
echo "======== 1.5 MiB" >> "$OUT"
run warmup_discard "$NEW" 1536 4 64 1536 16 16 "" 15
for rep in 1 2; do
  run "NEW_POOL_b16_r${rep}"  "$NEW" 1536 4  64 1536 16 16 "--reuse-corpus" 20
  run "OLD_POOL_b16_r${rep}"  "$OLD" 1536 4  64 1536 16 16 "--reuse-corpus" 20
  run "LEG_w64_b16_r${rep}"   "$NEW" 1536 64 0  0    16 16 "--reuse-corpus" 20
done
echo "finished $(date -Is)  load=$(cut -d' ' -f1 /proc/loadavg)" >> "$OUT"
