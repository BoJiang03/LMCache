#!/usr/bin/env bash
# Is 1536 MiB simply the wrong constant for local storage?
#
# The default-cell run found the best in-flight bytes to be ~384 MiB at 6 MiB
# objects and ~768 MiB at 192 MiB objects, while the shipped default budget is
# 1536 MiB.  That is 2-4x too large, and it is why the pool loses to
# num_workers=64 at the default chunk_size: not the mechanism, the constant.
#
# 1536 MiB was chosen as the worst-case-best across four storage conditions
# INCLUDING emulated 8 ms and 32 ms latency, where the design doc measured
# 768 MiB losing 42%.  On a local array that trade does not apply.
#
# depth is held at 256 everywhere so it is never the binding constraint except
# where it should be; the budget is the only thing that moves.
#
# PASS: one budget within 2% of the best arm at ALL THREE object sizes.
# FALSIFIER: no budget does it -> the byte budget cannot be one constant on
# this hardware either, and the PR must either ship a per-deployment value or
# narrow its claim to the object sizes where it wins.
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/bsweep
OUT=${WORK:?directory on the storage under test}/expN/budget_sweep.log
TREE=${TREE:?LMCache checkout under test, built in place}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "budget sweep at depth=256  commit=$(git -C "$TREE" rev-parse --short HEAD) $(date -Is)" >> "$OUT"
run() {  # arm obj batch depth budget reuse
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib 48 --workers 4 --batch "$3" \
    --outstanding 3 --io-depth "$4" --budget-mib "$5" \
    --duration-s 20 --probe-gap-ms 100 --arm "$1" --keep-corpus $6 2>&1 \
  | grep -E "^RESULT|Error|Traceback" >> "$OUT"
}
for cell in 1536:512 6144:256 196608:16; do
  obj=${cell%%:*}; batch=${cell##*:}
  echo "======== object $obj KiB ========" >> "$OUT"
  first=""
  for b in 192 384 768 1536; do
    run "b${b}" "$obj" "$batch" 256 "$b" "$first"
    first="--reuse-corpus"
  done
  # best legacy reference for this size, and a drift control
  case "$obj" in
    1536)   ref=256 ;;
    6144)   ref=64  ;;
    196608) ref=4   ;;
  esac
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$obj" --corpus-gib 48 --workers "$ref" --batch "$batch" \
    --outstanding 3 --io-depth 0 --budget-mib 0 --duration-s 20 \
    --probe-gap-ms 100 --arm "best_legacy_w${ref}" --keep-corpus \
    --reuse-corpus 2>&1 | grep -E "^RESULT|Error" >> "$OUT"
  run "b384_drift" "$obj" "$batch" 256 384 "--reuse-corpus"
  rm -rf "$CORPUS"
done
echo "finished $(date -Is)" >> "$OUT"
