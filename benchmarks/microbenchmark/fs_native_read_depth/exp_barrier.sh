#!/usr/bin/env bash
# Is the per-group barrier the limit at small batches?
#
# After the choose_num_tiles fix a GET batch is ONE tile, so one worker drives
# the whole batch through the pool and then blocks in read_done_cv_.wait until
# every object in it lands.  With batch=16 that is a full drain per batch, so
# concurrency is num_workers x batch, NOT depth.
#
# Direct test: hold depth at 64 and add WORKER threads.  If throughput scales
# with num_workers while depth is fixed, the limit is the number of
# independent stop-start pipelines, i.e. the barrier.
#
# DESIGN NOTE, learned the hard way twice today: the first arm after a 48 GiB
# corpus write reads low while the drives settle, and the previous version of
# this experiment drifted 26% from first arm to last, which made every
# cross-arm comparison in it worthless.  So: one discarded warmup, then two
# interleaved rounds, and each arm is scored on the median of both rounds.
#
# FALSIFIER: throughput flat as workers rise -> the barrier is not the limit
# and removing it would be wasted work.
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/barrier
OUT=${WORK:?directory on the storage under test}/expN/barrier.log
TREE=${TREE:?LMCache checkout under test, built in place}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "barrier test: depth=64 fixed, workers varied, batch=16  commit=$(git -C "$TREE" rev-parse --short HEAD)+tilefix  $(date -Is)" >> "$OUT"
run() {  # arm workers depth reuse dur
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib 6144 --corpus-gib 48 --workers "$2" --batch 16 \
    --outstanding 16 --io-depth "$3" --budget-mib 1536 \
    --duration-s "$5" --probe-gap-ms 100 --arm "$1" --keep-corpus $4 2>&1 \
  | grep -E "^RESULT|Error|Traceback" >> "$OUT"
}
run warmup_discard 4 64 "" 15
for rep in 1 2; do
  for w in 4 8 16 32; do
    run "POOL_w${w}_r${rep}" "$w" 64 "--reuse-corpus" 20
  done
  # legacy reference in the same round, so it drifts with everything else
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib 6144 --corpus-gib 48 --workers 64 --batch 16 \
    --outstanding 16 --io-depth 0 --budget-mib 0 --duration-s 20 \
    --probe-gap-ms 100 --arm "legacy_w64_r${rep}" --keep-corpus \
    --reuse-corpus 2>&1 | grep -E "^RESULT|Error" >> "$OUT"
done
echo "finished $(date -Is)" >> "$OUT"
