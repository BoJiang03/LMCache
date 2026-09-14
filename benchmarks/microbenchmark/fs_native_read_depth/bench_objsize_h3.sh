#!/usr/bin/env bash
# BENCHMARK 2, H2 arm: single NVMe (/home, nvme2n1p1, Micron 7450 960GB).
#
# /home is a SHARED filesystem at 96% and the standing rule is that nothing
# which grows lives there.  So: abort unless there is real headroom, cap the
# corpus at 8 GiB, and remove it on every exit path including a signal.
#
# Batch is sized so the driver keeps 2.25 GiB of objects in flight at EVERY
# object size.  That matters: it is above the connector's 1536 MiB budget, so
# the connector is the limiter rather than the driver's own queue, and it
# leaves the 8 GiB corpus about 10.6 batch-groups deep at every rung.
set -u

CORPUS=/dev/shm/l2bench_h3_corpus
OUT=${WORK:?directory on the storage under test}/bench2/H3_tmpfs.log
MIN_FREE_GB=100
CORPUS_GIB=32

PY=${PY:-python3}
BENCH=$(dirname "$0")/benchB.py
TREE=${TREE:?LMCache checkout under test, built in place}
WORKERS=4; DEPTH=64; BUDGET=1536; OUTSTANDING=3; PASSES=3
SIZES="1536:512 3072:256 6144:128 12288:64 24576:32 49152:16 98304:8 196608:4"

cleanup() { rm -rf "$CORPUS"; }
trap cleanup EXIT INT TERM HUP

free_gb() { df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9'; }

mkdir -p "$(dirname "$OUT")"
: > "$OUT"
f=$(free_gb)
if [ "$f" -lt "$MIN_FREE_GB" ]; then
  echo "ABORT: /dev/shm has ${f} GB free, need >= ${MIN_FREE_GB} GB. Nothing written." >> "$OUT"
  echo "ABORT: /dev/shm has ${f} GB free"; exit 1
fi
{
  echo "BENCH2 H2 single NVMe  corpus=$CORPUS (${CORPUS_GIB} GiB, /dev/shm had ${f} GB free)"
  echo "device: tmpfs on /dev/shm, reads are memcpy, no block device"
  echo "tree=$TREE commit=$(git -C "$TREE" rev-parse --short HEAD)"
  echo "workers=$WORKERS depth=$DEPTH budget=${BUDGET}MiB passes=$PASSES arms=L,P,L,P"
  echo "started $(date -Is)  loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
} >> "$OUT"

run() {  # tag obj batch depth budget reuse passes
  echo "### $1 obj=$2 depth=$4" >> "$OUT"
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib "$CORPUS_GIB" --workers "$WORKERS" \
    --batch "$3" --outstanding "$OUTSTANDING" --passes "$7" \
    --io-depth "$4" --budget-mib "$5" --dev md1 --no-odirect --keep-corpus $6 2>&1 \
    | grep -E "^pass|budget=|Error|Traceback|error" >> "$OUT"
}

for pair in $SIZES; do
  obj=${pair%%:*}; batch=${pair##*:}
  # Re-check headroom before every corpus write, not only once at the top.
  f=$(free_gb)
  if [ "$f" -lt "$MIN_FREE_GB" ]; then
    echo "STOP at $((obj / 1024)) MiB: /dev/shm fell to ${f} GB free" >> "$OUT"; break
  fi
  echo "======== object $((obj / 1024)) MiB  batch=$batch  shm_free=${f}GB ========" >> "$OUT"
  run warmup_discard "$obj" "$batch" 0 0 "" 1
  run L1 "$obj" "$batch" 0        0        "--reuse-corpus" "$PASSES"
  run P1 "$obj" "$batch" "$DEPTH" "$BUDGET" "--reuse-corpus" "$PASSES"
  run L2 "$obj" "$batch" 0        0        "--reuse-corpus" "$PASSES"
  run P2 "$obj" "$batch" "$DEPTH" "$BUDGET" "--reuse-corpus" "$PASSES"
  rm -rf "$CORPUS"
done

echo "finished $(date -Is)  shm_free=$(free_gb)GB" >> "$OUT"
