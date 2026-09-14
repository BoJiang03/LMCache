#!/usr/bin/env bash
# BENCHMARK 3: the arm that decides whether the PR is needed at all.
#
# Bench2 compared w=4 d=0 against w=4 d=64.  That shows the pool beats the
# DEFAULT, which nobody disputes.  The question a reviewer actually asks is
# whether raising the knob that already exists does the same job, and on
# 09-10 at a fixed small object size it did: 6 MiB gave w=64 49.2 against the
# PR's 49.0, and 24 MiB gave 50.85 against 50.5, both inside 0.4% noise.
#
# So the comparison has to be made the way a deployment experiences it: ONE
# configuration held fixed across the whole object-size axis, scored by its
# WORST rung, because chunk_size is chosen by the user and not by whoever
# tuned num_workers.
#
#   W4  w=4  d=0            upstream default
#   W64 w=64 d=0            pure config, no code change, the PR's real rival
#   P   w=4  d=64 b=1536    this PR
#
# Order is W4 W64 P W4 W64 P so ordering bias falls on all three equally;
# each arm is scored on the median of its 6 passes and its two blocks'
# disagreement is its drift.  rd_ticks_sum is kept in the log because it is
# the cost axis: w=64 buys its bandwidth with 64 threads and 64 connections
# on all four lanes, and previously showed 2.7x worse per-I/O latency at
# equal throughput.
#
# usage: bench_objsize_3arm.sh <label> <corpus_dir> <corpus_gib> [extra flags]
set -u

LABEL="${1:?label}"; CORPUS="${2:?corpus dir}"; CORPUS_GIB="${3:?corpus GiB}"
EXTRA="${4:-}"

PY=${PY:-python3}
BENCH=$(dirname "$0")/benchB.py
TREE=${TREE:?LMCache checkout under test, built in place}
OUT=${WORK:?directory on the storage under test}/bench3/${LABEL}.log

DEPTH=64; BUDGET=1536; OUTSTANDING=3; PASSES=3
SIZES="1536:2048 3072:1024 6144:512 12288:256 24576:128 49152:64 98304:32 196608:16"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"
{
  echo "BENCH3 three-arm objsize sweep  label=$LABEL corpus=$CORPUS (${CORPUS_GIB} GiB) extra='$EXTRA'"
  echo "tree=$TREE commit=$(git -C "$TREE" rev-parse --short HEAD)"
  echo "arms: W4(w=4,d=0)  W64(w=64,d=0)  P(w=4,d=$DEPTH,b=${BUDGET}MiB), each twice"
  echo "started $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
} >> "$OUT"

trap 'rm -rf "$CORPUS"' EXIT INT TERM HUP

run() {  # tag obj batch workers depth budget reuse passes
  echo "### $1 obj=$2 workers=$4 depth=$5" >> "$OUT"
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
  for rep in 1 2; do
    run "W4_$rep"  "$obj" "$batch"  4 0        0        "--reuse-corpus" "$PASSES"
    run "W64_$rep" "$obj" "$batch" 64 0        0        "--reuse-corpus" "$PASSES"
    run "P_$rep"   "$obj" "$batch"  4 "$DEPTH" "$BUDGET" "--reuse-corpus" "$PASSES"
  done
  rm -rf "$CORPUS"
done

echo "finished $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"
