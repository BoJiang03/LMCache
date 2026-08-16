#!/usr/bin/env bash
set -euo pipefail
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SMOKE_PYTHON:-python3}"
REPETITIONS="${REPETITIONS:-3}"
L1_GB="${L1_GB:-40}"
export SMOKE_REPO="${SMOKE_REPO:-$(git -C "$HERE" rev-parse --show-toplevel)}"
export SMOKE_HORIZON="${SMOKE_HORIZON:-2.5}"

for mode in eager lazy; do
  for ((rep=0; rep<REPETITIONS; rep++)); do
    "$PYTHON" "$HERE/longdoc.py" run "$mode" "$rep" "$L1_GB"
    "$PYTHON" "$HERE/validate_result.py" \
      "$HERE/logs/ld_L_${mode}_h3c11_${rep}.json" \
      --mode "$mode" --kind hot-cold --requests 120
  done
done
"$PYTHON" "$HERE/longdoc.py" table
