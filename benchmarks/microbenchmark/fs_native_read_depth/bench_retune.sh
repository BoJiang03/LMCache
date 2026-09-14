#!/usr/bin/env bash
# BENCHMARK 4: the cost of not retuning when chunk_size moves.
#
# THE CLAIM UNDER TEST
#   One fixed PR setting holds across the chunk_size range.  No fixed
#   num_workers does.
#
# WHY NOT A SIMPLE A/B
#   Against the DEFAULT the pool wins 1.70x, but against num_workers=64 it
#   wins nothing: 09-10 measured 49.2 vs 49.0 at 6 MiB and 50.85 vs 50.5 at
#   24 MiB, inside 0.4% noise.  So bandwidth at one object size cannot
#   justify this PR and must not be the headline.  What num_workers cannot
#   do is stay correct when the object size moves, because it counts
#   OBJECTS while the hardware cares about BYTES, and object size is
#   chunk_size x bytes-per-token-per-rank.
#
# SHAPE: tune once at the default, then freeze and let the world move.
#   Every arm is a FIXED configuration swept across the whole size axis.
#   The operator picks their arm by looking only at the chunk_size=256
#   column, which is what they would actually have measured, and then lives
#   with the rest of their row.
#
# PRE-COMMITTED FALSIFIER
#   If any single fixed num_workers stays within 5% of the best arm at
#   EVERY rung, the operator-facing argument for this PR fails and the pure
#   config change is sufficient.  Existing evidence says it will not
#   (w=64 is -11% at 192 MiB, w=4 is about -78% at 1.5 MiB) but this run
#   decides, not that evidence.
#
# ORDERING
#   Each rung runs the 7 arms forward, then the same 7 in reverse.  Every
#   arm therefore gets one early slot and one late slot, so the settling
#   drift that broke bench1's 24 MiB rung falls on all arms equally.  An
#   arm's two blocks disagreeing IS its error bar.
#
# usage: bench_retune.sh <label> <corpus_dir> <corpus_gib> [extra flags]
set -u

LABEL="${1:?label}"; CORPUS="${2:?corpus dir}"; CORPUS_GIB="${3:?corpus GiB}"
EXTRA="${4:-}"

PY=${PY:-python3}
BENCH=$(dirname "$0")/benchB.py
TREE=${TREE:?LMCache checkout under test, built in place}
OUT=${WORK:?directory on the storage under test}/bench4/${LABEL}.log

OUTSTANDING=3; PASSES=3
DEPTH=64; BUDGET=1536
# arm tag : workers : io_depth : budget_mib
ARMS="W4:4:0:0 W8:8:0:0 W16:16:0:0 W32:32:0:0 W64:64:0:0 W128:128:0:0 P:4:${DEPTH}:${BUDGET}"
# obj_kib:batch  -- chunk_size 64,128,256,512,1024,2048,4096,8192 at 24 KiB/token/rank
SIZES="1536:2048 3072:1024 6144:512 12288:256 24576:128 49152:64 98304:32 196608:16"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"
{
  echo "BENCH4 retune-cost sweep  label=$LABEL corpus=$CORPUS (${CORPUS_GIB} GiB) extra='$EXTRA'"
  echo "tree=$TREE commit=$(git -C "$TREE" rev-parse --short HEAD)"
  echo "arms: $ARMS"
  echo "each rung: arms forward, then the same arms reversed"
  echo "started $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
} >> "$OUT"

trap 'rm -rf "$CORPUS"' EXIT INT TERM HUP

run() {  # tag obj batch workers depth budget reuse passes
  echo "### $1 obj=$2 workers=$4 depth=$5 budget=$6" >> "$OUT"
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib "$CORPUS_GIB" --workers "$4" \
    --batch "$3" --outstanding "$OUTSTANDING" --passes "$8" \
    --io-depth "$5" --budget-mib "$6" --keep-corpus $7 $EXTRA 2>&1 \
    | grep -E "^pass|busy=|budget=|Error|Traceback" >> "$OUT"
}

for pair in $SIZES; do
  obj=${pair%%:*}; batch=${pair##*:}
  echo "======== obj_kib=$obj batch=$batch loadavg=$(cut -d' ' -f1 /proc/loadavg) ========" >> "$OUT"
  run warmup_discard "$obj" "$batch" 4 0 0 "" 1
  fwd="$ARMS"
  rev=$(echo "$ARMS" | tr ' ' '\n' | tac | tr '\n' ' ')
  for pass_dir in "$fwd" "$rev"; do
    for a in $pass_dir; do
      IFS=: read -r tag w d b <<< "$a"
      run "$tag" "$obj" "$batch" "$w" "$d" "$b" "--reuse-corpus" "$PASSES"
    done
  done
  rm -rf "$CORPUS"
done

echo "finished $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"
