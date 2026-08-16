#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SMOKE_PYTHON:-python3}"
REPETITIONS="${REPETITIONS:-3}"
QUESTIONS="${QUESTIONS:-120}"
CONCURRENCY="${CONCURRENCY:-4}"
L1_GB="${L1_GB:-68}"
export SMOKE_REPO="${SMOKE_REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}"
export SMOKE_HORIZON="${SMOKE_HORIZON:-2.5}"

for mode in eager lazy; do
  for ((rep=0; rep<REPETITIONS; rep++)); do
    "$PYTHON" "$HERE/accuracy.py" run \
      "$mode" "$rep" "$QUESTIONS" "$CONCURRENCY" "$L1_GB"
    "$PYTHON" "$HERE/validate_result.py" \
      "$HERE/logs/ac_A_${mode}_n${QUESTIONS}_l${L1_GB}_${rep}.json" \
      --mode "$mode" --kind gsm8k --requests "$QUESTIONS"
  done
done
"$PYTHON" "$HERE/accuracy.py" table
