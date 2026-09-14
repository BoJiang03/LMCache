#!/usr/bin/env bash
# THE DECIDING EXPERIMENT for whether an adaptive budget is worth building.
#
# Today's H2 sweep found the pool 7-14% SLOWER than legacy on a single NVMe,
# at the static default budget of 1536 MiB.  Two possible causes, and they
# point opposite ways:
#
#   (a) the budget is simply wrong for this device.  Record 2026-09-11/9 put
#       the single-NVMe optimum at 192 MiB and had a controller settle at
#       106 MiB scoring 99.8%.  If a small budget recovers the loss here, the
#       device spread is real, no constant covers both devices, and that is
#       the case for adaptive.
#   (b) it is the pool's own dispatch overhead, which a budget cannot fix.
#       Today's probe run saw the pool at 44.1 ms against 23.2 ms for plain
#       workers at IDENTICAL concurrency, so this is not hypothetical.
#       If small budgets do not recover it, adaptive is pointless here.
#
# /home is shared and 96% full: 8 GiB cap, abort under 25 GB free, remove on
# every exit path.
set -u
CORPUS=${WORK:?directory on the storage under test}/l2bench_h2b
OUT=${WORK:?directory on the storage under test}/expN/h2_budget.log
PY=${PY:-python3}
BENCH=$(dirname "$0")/benchN.py
TREE=${TREE:?LMCache checkout under test, built in place}
MIN_FREE_GB=25
cleanup() { rm -rf "$CORPUS"; }
trap cleanup EXIT INT TERM HUP
free_gb() { df -BG --output=avail /home | tail -1 | tr -dc '0-9'; }
mkdir -p "$(dirname "$OUT")"; : > "$OUT"
f=$(free_gb)
if [ "$f" -lt "$MIN_FREE_GB" ]; then echo "ABORT: /home ${f} GB free" >> "$OUT"; exit 1; fi
echo "H2 budget sweep  6 MiB objects  /home had ${f} GB free  started $(date -Is)" >> "$OUT"

run() {  # tag depth budget reuse
  echo "### $1" >> "$OUT"
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib 6144 --corpus-gib 8 --workers 4 --batch 128 --outstanding 3 \
    --passes 3 --io-depth "$2" --budget-mib "$3" --dev nvme2n1 \
    --keep-corpus $4 2>&1 | grep -E "^pass|Error|Traceback" >> "$OUT"
}
run warmup_discard 0 0 ""
for rep in 1 2; do
  run "legacy_d0_r$rep"   0  0    "--reuse-corpus"
  for b in 48 96 192 384 1536; do
    run "d64_b${b}_r$rep" 64 "$b" "--reuse-corpus"
  done
done
echo "finished $(date -Is)  /home free=$(free_gb)GB" >> "$OUT"
