#!/usr/bin/env bash
# Run the four groups once the reference vLLM (baseline.sh) is healthy.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
until curl -sf "localhost:$BASELINE_PORT/health" > /dev/null; do
  if ! pgrep -f "port $BASELINE_PORT" > /dev/null; then
    echo "baseline died"
    grep -E "Error" "$OUT_ROOT/baseline.log" | tail -5
    exit 1
  fi
  sleep 2
done
for spec in "g1-prefix:$CONTROLS/prefix:0" "g2-fix:$REPO:0" \
  "g3-utilsonly-lazy:$CONTROLS/utilsonly:1" "g4-fix-lazy:$REPO:1"; do
  IFS=: read -r label tree lazy <<< "$spec"
  bash "$E2E/run.sh" "$tree" "$label" "$lazy"
  cat "$OUT_ROOT/$label/tree.txt" "$OUT_ROOT/$label/client.log"
done
