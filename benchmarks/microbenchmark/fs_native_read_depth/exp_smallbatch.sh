#!/usr/bin/env bash
# SMALL BATCH: the case every benchmark so far has hidden.
#
# submit_load_task is called once per request per adapter, so the batch size
# is the number of chunks that request needs.  Our E2E uses ISL=122880 at
# chunk_size=256, i.e. 480 chunks per request, and every microbenchmark used
# batch 256-512.  A 4096-token request is 16 chunks.
#
# BEFORE the choose_num_tiles change, a 16-key batch was split into
# min(num_workers,16) tiles, each worker handed its OWN tile (4 objects) to
# the pool and blocked on it: 4 workers x 4 = 16 reads in flight, and
# read_io_depth did nothing.  Legacy at num_workers=64 got 64 in flight by
# taking tiles from several requests at once.  So the pool LOST at small
# batches, which is where real requests live.
#
# AFTER: a GET batch is one tile, so one worker drives the whole batch through
# the pool.  4 workers x 16 = 64 in flight from 4 THREADS, where legacy needs
# 64 threads (and on redis, 64 connections) for the same number.
#
# outstanding=16 stands in for ~16 concurrent requests.
#
# PREDICTION: POOL at workers=4 now matches w64 within 5%.
# FALSIFIER: POOL still far below w64 -> the tile change did not lift the cap
# and the diagnosis is wrong.
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/small
OUT=${WORK:?directory on the storage under test}/expN/smallbatch.log
TREE=${TREE:?LMCache checkout under test, built in place}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "small batch (16 keys = a 4096-token request at chunk_size 256)  commit=$(git -C "$TREE" rev-parse --short HEAD)+tilefix  $(date -Is)" >> "$OUT"
run() {  # arm workers depth budget reuse
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib 6144 --corpus-gib 48 --workers "$2" --batch 16 \
    --outstanding 16 --io-depth "$3" --budget-mib "$4" \
    --duration-s 20 --probe-gap-ms 100 --arm "$1" --keep-corpus $5 2>&1 \
  | grep -E "^RESULT|Error|Traceback" >> "$OUT"
}
run w4          4   0   0    ""
run w64         64  0   0    "--reuse-corpus"
run w256        256 0   0    "--reuse-corpus"
run POOL_d64    4   64  1536 "--reuse-corpus"
run POOL_d256   4   256 1536 "--reuse-corpus"
run w64_drift   64  0   0    "--reuse-corpus"
echo "finished $(date -Is)" >> "$OUT"
