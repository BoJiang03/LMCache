#!/usr/bin/env bash
# N2-lite: does a num_workers picked for READS disturb the STORE path?
#
# This is the last of the three surviving differentiators that is untested,
# and it is the only one that is structural rather than a matter of degree:
# on fs there is exactly one worker pool for every op.  ConnectorBase::
# start_workers creates dedicated lanes only from per_op_workers, and the fs
# pybind does not accept per_op_workers (only mooncake's does).  So a value
# chosen to get reads in flight IS the store concurrency, with no way to
# separate them.  The read pool leaves num_workers alone.
#
# FALSIFIER: write bandwidth flat from workers=4 to workers=256.  Then raising
# num_workers costs nothing on the store path either, and the last structural
# argument for the knob is gone on this backend.
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/benchN.py
CORPUS=${WORK:?directory on the storage under test}/expN/wq
OUT=${WORK:?directory on the storage under test}/expN/write_quick.log
TREE=${TREE:?LMCache checkout under test, built in place}
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT
echo "N2-lite write sweep  commit=$(git -C "$TREE" rev-parse --short HEAD) started $(date -Is)" >> "$OUT"
# Two rounds so ordering bias does not land on one value.
for rep in 1 2; do
  for w in 4 16 64 256; do
    echo "### rep=$rep workers=$w" >> "$OUT"
    PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
      --obj-kib 6144 --corpus-gib 48 --workers "$w" --batch 512 \
      --outstanding 3 --passes 1 --io-depth 0 --budget-mib 0 2>&1 \
      | grep -E "^write:|Error|Traceback" >> "$OUT"
    rm -rf "$CORPUS"
  done
done
echo "finished $(date -Is)" >> "$OUT"
