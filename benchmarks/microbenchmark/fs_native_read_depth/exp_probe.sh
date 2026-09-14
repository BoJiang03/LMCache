#!/usr/bin/env bash
# EXPERIMENT N4: the one that can actually retire "just raise num_workers".
#
# Bandwidth alone cannot, because raising num_workers DOES buy bandwidth.
# The cost is latency, and Little's law ties them:  inflight = BW x latency.
# Past saturation the BW term is pinned, so extra inflight is pure latency.
#
# On the legacy path inflight = num_workers x object_size, so it SCALES with
# object size.  On the pooled path it is a constant in bytes.  Therefore one
# num_workers must choose between reaching bandwidth at small objects and not
# over-queueing at large ones.  One byte budget need not choose.
#
# Arithmetic this is meant to confirm or refute, at 192 MiB on a 54 GB/s array:
#   workers=4    768 MiB inflight ->  ~14 ms
#   workers=64    12 GiB inflight -> ~220 ms
#   pool         1536 MiB inflight ->  ~28 ms
# and at 1.5 MiB, workers=4 is only 6 MiB inflight, which is why it reads
# 10.88 GB/s there against 48 for the pool.
#
# PASS CONDITION for an arm: within 5% of the best stream bandwidth AND within
# 2x of the best probe p50, AT EVERY object size.
# PREDICTION: only POOL passes.
# FALSIFIER: if a single num_workers passes too, the knob does not earn its
# place on this backend and the PR must say so.
set -u

PY=${PY:-python3}
BENCH=$(dirname "$0")/bench_probe.py
CORPUS=${WORK:?directory on the storage under test}/expN/probe
OUT=${WORK:?directory on the storage under test}/expN/probe.log
TREE=${TREE:?LMCache checkout under test, built in place}
DUR=30
mkdir -p "$(dirname "$OUT")"
: > "$OUT"
trap 'rm -rf "$CORPUS"' EXIT

echo "EXPERIMENT N4 probe latency under load  commit=$(git -C "$TREE" rev-parse --short HEAD)" >> "$OUT"
echo "started $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"

# Self-check: a 6 GiB, 8 s run that must print a RESULT line with probes>0.
# bench_probe.py is new code; find its bugs in seconds, not after 20 minutes.
PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" --obj-kib 6144 \
  --corpus-gib 6 --workers 4 --batch 32 --outstanding 2 --duration-s 8 \
  --probe-gap-ms 50 --arm smoke >> "$OUT" 2>&1
if ! grep -qE "RESULT arm=smoke .*probes=[1-9]" "$OUT"; then
  echo "SMOKE FAILED (no RESULT line, or zero probes) -- not running the arms" >> "$OUT"
  exit 1
fi
echo "smoke OK $(date -Is)" >> "$OUT"

run() {  # arm obj batch workers depth budget reuse
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib 64 --workers "$4" --batch "$3" \
    --outstanding 3 --io-depth "$5" --budget-mib "$6" \
    --duration-s "$DUR" --probe-gap-ms 100 --arm "$1" --keep-corpus $7 2>&1 \
  | grep -E "^RESULT|Error|Traceback" >> "$OUT"
}

# obj_kib:batch:ladder   ladder stops where batch*outstanding can still feed it
for cell in 1536:512:4,16,64,256 6144:256:4,16,64,256 196608:32:4,16,64; do
  obj=${cell%%:*}; r=${cell#*:}; batch=${r%%:*}; ladder=${r##*:}
  echo "======== object $obj KiB batch=$batch load=$(cut -d' ' -f1 /proc/loadavg) ========" >> "$OUT"
  first=1
  for w in ${ladder//,/ }; do
    reuse="--reuse-corpus"; [ $first -eq 1 ] && reuse=""
    first=0
    run "w${w}" "$obj" "$batch" "$w" 0 0 "$reuse"
  done
  run "POOL" "$obj" "$batch" 4 64 1536 "--reuse-corpus"
  run "w4_drift" "$obj" "$batch" 4 0 0 "--reuse-corpus"
  rm -rf "$CORPUS"
done
echo "finished $(date -Is)" >> "$OUT"
