#!/usr/bin/env bash
# tmpfs: does the pool beat a TUNED num_workers there, or only the default?
# bench2/H3 compared against w=4 only (4.84x at 6 MiB).  tmpfs reads are
# memcpy, so bandwidth should scale with copying threads, and w=64 legacy
# and POOL d=64 are both 64 copying threads.
#
# PREDICTION: w64 ties or beats POOL at both sizes.  If POOL wins by more than
# twice the larger drift, the budget/grouping matters even for memcpy, which
# would be worth understanding before claiming it.
set -u
PY=${PY:-python3}
BENCH=$(dirname "$0")/benchB.py
CORPUS=/dev/shm/l2bench_h3_w64
OUT=${WORK:?directory on the storage under test}/expN/h3_w64.log
TREE=${TREE:?LMCache checkout under test, built in place}
CORPUS_GIB=16; MIN_FREE_GB=200
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
cleanup() { rm -rf "$CORPUS"; }
trap cleanup EXIT INT TERM HUP
free_gb() { df -BG --output=avail /dev/shm | tail -1 | tr -dc '0-9'; }
f=$(free_gb)
if [ "$f" -lt "$MIN_FREE_GB" ]; then echo "ABORT: /dev/shm has ${f} GB free" | tee -a "$OUT"; exit 1; fi
echo "tmpfs, w64 vs pool, controlled  tree=$(git -C "$TREE" rev-parse --short HEAD)+move  corpus=${CORPUS_GIB}GiB  $(date -Is)" >> "$OUT"
run() {  # arm obj_kib batch workers depth budget reuse passes
  [ "$(free_gb)" -lt "$MIN_FREE_GB" ] && { echo "ABORT mid-run: /dev/shm low" >> "$OUT"; exit 1; }
  echo "ARM $1 obj=$2 workers=$4 depth=$5 budget=$6" >> "$OUT"
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib "$CORPUS_GIB" --workers "$4" --batch "$3" \
    --io-depth "$5" --budget-mib "$6" --outstanding 3 --dev md1 --no-odirect --keep-corpus \
    --passes "$8" $7 2>&1 | grep -E "GB/s|pass|Error|Traceback" >> "$OUT"
}
for pair in 6144:128 1536:512; do
  obj=${pair%%:*}; batch=${pair##*:}
  rm -rf "$CORPUS"
  run warmup_discard "$obj" "$batch" 4 0 0 "" 1
  for rep in 1 2; do
    run "w64_r${rep}"  "$obj" "$batch" 64 0  0    "--reuse-corpus" 3
    run "POOL_r${rep}" "$obj" "$batch" 4  64 1536 "--reuse-corpus" 3
    run "w4_r${rep}"   "$obj" "$batch" 4  0  0    "--reuse-corpus" 3
  done
done
echo "finished $(date -Is)" >> "$OUT"
