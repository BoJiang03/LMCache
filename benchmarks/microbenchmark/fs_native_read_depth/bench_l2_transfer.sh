#!/usr/bin/env bash
# BENCHMARK 1: fs_native L2 -> host read bandwidth, legacy vs pooled.
#
# No vLLM, no GPU.  Talks to LMCacheFSClient directly, the same C++ connector
# the mp_l2 arm uses, O_DIRECT, and brackets each pass with /proc/diskstats so
# achieved bandwidth is stated against device busy time, not wall clock alone.
#
# Tree under test: the checkout under test @ a6c4b337 (ConnectorBase pool,
# shape B: reader threads hold their own ConnectionType and call the
# backend's unmodified do_single_get).
#
# Array ceiling, measured independently with fio O_DIRECT: ~53.5 GB/s.
#
# PRE-COMMITTED PREDICTIONS  (median of 3 passes, "GB/s vs wall", app bytes)
#   6 MiB objects (chunk_size=256, the default)
#     legacy d=0   28-34 GB/s   anchor.  Outside means the array or a
#                               neighbour moved and nothing below compares
#                               to the numbers in the PR body.
#     pooled d=64  44-51 GB/s   PRIMARY.  Falsifier: < 38 GB/s means the
#                               base-class pool does not reproduce the
#                               fs-only pool's win.
#   192 MiB objects (chunk_size=8192)
#     legacy d=0   47-54 GB/s   already saturated without the pool.
#     pooled d=64  within 3 GB/s of legacy.  Falsifier: pooled < legacy - 3
#                               means the pool costs bandwidth where it is
#                               not needed, which is a regression.
#   24 MiB objects (chunk_size=1024)  no prediction, shape-filling only.
#   drift control: legacy repeated last at each size, within 3 GB/s of the
#                  first legacy run at that size.
set -u

PY=${PY:-python3}
BENCH=$(dirname "$0")/benchB.py
CORPUS=${WORK:?directory on the storage under test}/bench1/corpus
TREE=${TREE:?LMCache checkout under test, built in place}
OUT=${WORK:?directory on the storage under test}/bench1/l2_transfer.log

WORKERS=4          # LMCache's default num_workers
DEPTH=64           # read_io_depth for the pooled arm
BUDGET=1536        # MiB; kDefaultReadMaxBytesInFlight
OUTSTANDING=3
PASSES=3
CORPUS_GIB=96      # >> the drives' combined DRAM, so no pass is served from it
SIZES="6144:512 24576:128 196608:16"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"
{
  echo "BENCH1 l2 transfer  tree=$TREE commit=$(git -C "$TREE" rev-parse --short HEAD)"
  echo "workers=$WORKERS depth=$DEPTH budget=${BUDGET}MiB corpus=${CORPUS_GIB}GiB passes=$PASSES"
  echo "started $(date -Is)"
} >> "$OUT"

run() {  # $1=label $2=obj_kib $3=batch $4=io_depth $5=budget_mib $6=extra
  echo "### $1  obj=$2KiB depth=$4 budget=$5MiB" >> "$OUT"
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib "$CORPUS_GIB" --workers "$WORKERS" \
    --batch "$3" --outstanding "$OUTSTANDING" --passes "$PASSES" \
    --io-depth "$4" --budget-mib "$5" $6 2>&1 \
    | grep -E "^corpus|^write|^pass|busy=|member|budget=|Error|Traceback" >> "$OUT"
}

for pair in $SIZES; do
  obj=${pair%%:*}; batch=${pair##*:}
  echo "======== object $((obj / 1024)) MiB  batch=$batch ========" >> "$OUT"
  run "legacy"        "$obj" "$batch" 0  0        "--keep-corpus"
  run "pooled"        "$obj" "$batch" "$DEPTH" "$BUDGET" "--keep-corpus --reuse-corpus"
  run "legacy_drift"  "$obj" "$batch" 0  0        "--keep-corpus --reuse-corpus"
  rm -rf "$CORPUS"
done

echo "finished $(date -Is)" >> "$OUT"
