#!/usr/bin/env bash
# EXPERIMENT N3: store latency while loads saturate.
#
# N1 is expected to show that num_workers CAN buy the read bandwidth, so
# bandwidth alone cannot answer "why not just raise num_workers".  This can.
#
# On fs every op shares one queue served by num_workers threads, because
# ConnectorBase::start_workers creates dedicated lanes only from
# per_op_workers and the fs pybind does not accept it.  So:
#
#   A  workers=4  depth=0    default: loads slow, and they occupy all 4
#   B  workers=64 depth=0    "just raise it": loads fast, but a store now
#                            queues behind loads on all 64 threads
#   C  workers=4  depth=64   loads run on reader threads; do_batch_get
#                            returns immediately and the 4 workers stay free
#
# PREDICTION: C's store latency is far below B's at equal read bandwidth.
# FALSIFIER: B's store latency comparable to C's -> the isolation argument
# fails too, and num_workers is an adequate substitute on this backend.
set -u

PY=${PY:-python3}
TREE=${TREE:?LMCache checkout under test, built in place}
BENCH=$(dirname "$0")/bench_mixed.py
CORPUS=${WORK:?directory on the storage under test}/expN/mixed
OUT=${WORK:?directory on the storage under test}/expN/mixed.log
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT

# Self-check first: a 3 GiB, 5 s run that must print a RESULT line.  If the
# harness is broken, fail here having cost seconds, not after the long arms.
echo "### smoke" >> "$OUT"
PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
  --obj-kib 6144 --corpus-gib 3 --workers 4 --batch 64 --outstanding 2 \
  --duration-s 5 --arm smoke --keep-corpus >> "$OUT" 2>&1
if ! grep -q "RESULT arm=smoke" "$OUT"; then
  echo "SMOKE FAILED, not running the arms" >> "$OUT"; exit 1
fi
rm -rf "$CORPUS"
echo "smoke OK $(date -Is)" >> "$OUT"

run() {  # arm workers depth budget reuse
  echo "### arm=$1" >> "$OUT"
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib 6144 --corpus-gib 64 --workers "$2" --batch 512 \
    --outstanding 3 --io-depth "$3" --budget-mib "$4" \
    --duration-s 45 --store-batch 4 --store-gap-ms 50 \
    --arm "$1" --keep-corpus $5 2>&1 \
  | grep -E "^arm=|^RESULT|^corpus|Error|Traceback" >> "$OUT"
}

# Two rounds, interleaved, so ordering bias does not land on one arm.
for rep in 1 2; do
  reuse=""; [ "$rep" = 2 ] && reuse="--reuse-corpus"
  run "A_w4_d0_r$rep"   4  0  0    "$reuse"
  run "B_w64_d0_r$rep"  64 0  0    "--reuse-corpus"
  run "C_w4_d64_r$rep"  4  64 1536 "--reuse-corpus"
done
echo "finished $(date -Is)" >> "$OUT"
