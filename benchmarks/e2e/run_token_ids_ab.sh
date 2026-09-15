#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the three-point end-to-end A/B for the connector's token_ids transport
# and print the comparison table.
#
#   baseline      a checkout from before this work (list of ints, whole sequence)
#   packed        this branch with lmcache.mp.delta_token_ids=false
#   packed+delta  this branch as it ships
#
# The baseline checkout must be a worktree at the commit this branch forked
# from, with the compiled extensions copied in (they are built from the same
# csrc, so copying is valid):
#
#   git worktree add --detach "$BASELINE" <base-commit>
#   cp lmcache/*.so lmcache/_version.py "$BASELINE/lmcache/"
#
# Usage: benchmarks/e2e/run_token_ids_ab.sh [input_len] [num_prompts] [concurrency]

set -euo pipefail

BRANCH="${BRANCH:-/raid/bo/delta_token_ids}"
BASELINE="${BASELINE:-/raid/bo/delta_baseline}"
PY="${PY:-/home/bo/LMCache-worktrees/mtp/.venv/bin/python}"
OUT_DIR="${OUT_DIR:-$BRANCH/benchmarks/e2e/results}"

INPUT_LEN="${1:-120000}"
NUM_PROMPTS="${2:-24}"
CONCURRENCY="${3:-8}"

RUNNER="$BRANCH/benchmarks/e2e/token_ids_transport_e2e.py"
COMMON=(--input-len "$INPUT_LEN" --num-prompts "$NUM_PROMPTS"
        --concurrency "$CONCURRENCY" --out-dir "$OUT_DIR")

run_one() {
  local label="$1" checkout="$2" delta="$3"
  echo "=== $label ==="
  # Serially, on distinct ports, so a stale server from a crashed run cannot
  # be mistaken for this one's.
  "$PY" "$RUNNER" run \
    --label "$label" --checkout "$checkout" --delta "$delta" \
    --lmcache-port 5599 --vllm-port 8199 "${COMMON[@]}"
}

run_one baseline "$BASELINE" unset
run_one packed "$BRANCH" false
run_one packed+delta "$BRANCH" true

"$PY" "$RUNNER" report \
  "$OUT_DIR/baseline.json" "$OUT_DIR/packed.json" "$OUT_DIR/packed+delta.json"
