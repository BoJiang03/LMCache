#!/usr/bin/env bash
# BENCHMARK 2: legacy vs pooled across the full object-size ladder.
#
# object size = chunk_size x bytes-per-token-per-rank.  At 24 KiB/token/rank
# (gpt-oss-20b, TP=2) the ladder below is chunk_size 64 .. 8192, so every
# point is a setting a deployment could actually run.  chunk_size 256 (6 MiB)
# is LMCache's default.
#
#   1.5 MiB = cs 64     12 MiB = cs 512     96 MiB = cs 4096
#     3 MiB = cs 128    24 MiB = cs 1024   192 MiB = cs 8192
#     6 MiB = cs 256    48 MiB = cs 2048
#
# WHY THIS SUPERSEDES BENCH1's SHAPE
# Bench1 ran one legacy arm, one pooled arm, one legacy drift control per
# size.  At 24 MiB the drift control came back 4.1 GB/s ABOVE the first
# legacy run, i.e. the ordering bias exceeded the effect, because the first
# arm at each size starts while the drives are still settling from the 96 GiB
# corpus write.  Here each size gets a discarded warmup run, then the arms
# alternate L P L P, and each arm is scored on the median of its 6 passes.
# Ordering bias then falls on both arms roughly equally, and the L-to-L and
# P-to-P spreads measure what is left.
#
# usage: bench_objsize.sh <label> <corpus_dir> <corpus_gib> [--no-odirect]
set -u

LABEL="${1:?label}"; CORPUS="${2:?corpus dir}"; CORPUS_GIB="${3:?corpus GiB}"
ODIRECT="${4:-}"

PY=${PY:-python3}
BENCH=$(dirname "$0")/benchB.py
TREE=${TREE:?LMCache checkout under test, built in place}
OUT=${WORK:?directory on the storage under test}/bench2/${LABEL}.log

WORKERS=4; DEPTH=64; BUDGET=1536; OUTSTANDING=3; PASSES=3
# obj_kib:batch   batch chosen so one batch is ~3 GiB at every size
SIZES="1536:2048 3072:1024 6144:512 12288:256 24576:128 49152:64 98304:32 196608:16"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"
{
  echo "BENCH2 objsize sweep  label=$LABEL  corpus=$CORPUS (${CORPUS_GIB} GiB) odirect_flag='$ODIRECT'"
  echo "tree=$TREE commit=$(git -C "$TREE" rev-parse --short HEAD)"
  echo "workers=$WORKERS depth=$DEPTH budget=${BUDGET}MiB passes=$PASSES arms=L,P,L,P"
  echo "started $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
} >> "$OUT"

trap 'rm -rf "$CORPUS"' EXIT

run() {  # $1=tag $2=obj_kib $3=batch $4=depth $5=budget $6=reuse $7=passes
  echo "### $1 obj=$2 depth=$4" >> "$OUT"
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib "$CORPUS_GIB" --workers "$WORKERS" \
    --batch "$3" --outstanding "$OUTSTANDING" --passes "$7" \
    --io-depth "$4" --budget-mib "$5" --keep-corpus $6 $ODIRECT 2>&1 \
    | grep -E "^pass|budget=|Error|Traceback|error" >> "$OUT"
}

for pair in $SIZES; do
  obj=${pair%%:*}; batch=${pair##*:}
  echo "======== object $((obj / 1024)) MiB  batch=$batch  loadavg=$(cut -d' ' -f1 /proc/loadavg) ========" >> "$OUT"
  run warmup_discard "$obj" "$batch" 0 0 "" 1
  run L1 "$obj" "$batch" 0        0        "--reuse-corpus" "$PASSES"
  run P1 "$obj" "$batch" "$DEPTH" "$BUDGET" "--reuse-corpus" "$PASSES"
  run L2 "$obj" "$batch" 0        0        "--reuse-corpus" "$PASSES"
  run P2 "$obj" "$batch" "$DEPTH" "$BUDGET" "--reuse-corpus" "$PASSES"
  rm -rf "$CORPUS"
done

echo "finished $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"
