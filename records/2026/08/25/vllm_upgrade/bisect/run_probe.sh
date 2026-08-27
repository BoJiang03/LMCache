#!/bin/bash
# run_probe.sh <venv> <gpu> <out_json> <log>
# Runs the text-accuracy probe and kills it once the result json is complete
# (probes hang at engine teardown holding GPU memory -- lesson from 08-25).
set -uo pipefail
VENV="$1"; GPU="$2"; OUT="$3"; LOG="$4"
PROBE=/home/bo/LMCache-worktrees/multi_modal/records/2026/08/25/vllm_upgrade/text_accuracy_probe_chunkparam.py
rm -f "$OUT"
CUDA_VISIBLE_DEVICES=$GPU PYTHONPATH=/home/bo/LMCache-worktrees/multi_modal PROBE_CHUNK=16 \
  setsid "$VENV/bin/python" "$PROBE" "$OUT" >"$LOG" 2>&1 &
PID=$!
PGID=$(ps -o pgid= -p $PID | tr -d ' ')
for i in $(seq 1 240); do   # up to 20 min
  sleep 5
  if [ -f "$OUT" ] && grep -qE '"stage": "(done|error)"' "$OUT"; then
    sleep 10   # let the final write settle
    kill -TERM -"$PGID" 2>/dev/null; sleep 5; kill -KILL -"$PGID" 2>/dev/null
    echo "[runner] probe finished, stage=$(grep -o '"stage": "[a-z_]*"' "$OUT"), killed pgid $PGID"
    exit 0
  fi
  if ! kill -0 $PID 2>/dev/null; then
    echo "[runner] probe exited on its own"
    exit 0
  fi
done
echo "[runner] TIMEOUT after 20min, killing pgid $PGID"
kill -KILL -"$PGID" 2>/dev/null
exit 1
