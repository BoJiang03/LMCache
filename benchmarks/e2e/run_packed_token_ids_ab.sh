#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# Run the two-point end-to-end A/B for the connector's token_ids encoding and
# print the comparison table.
#
#   baseline   a checkout from before this change (token ids as list[int])
#   packed     this branch (token ids as a big-endian uint32 buffer)
#
# The baseline checkout must be a worktree at the commit this branch forked
# from, with the compiled extensions copied in (they are built from the same
# csrc, so copying is valid):
#
#   git worktree add --detach "$BASELINE" <base-commit>
#   cp lmcache/*.so lmcache/_version.py "$BASELINE/lmcache/"
#
# Prompts must stay long: the decode this change removes scales with context
# length, and below ~16k tokens packing is a small pessimization. The saving
# also multiplies by TP rank, so TP=1 understates it. The default 30k sits
# well past that crossover and inside Qwen2.5-7B's 32k position limit.
#
# Usage: benchmarks/e2e/run_packed_token_ids_ab.sh [input_len] [num_prompts] [concurrency] [tp]

set -euo pipefail

# FlashInfer JIT-compiles its sampling kernels at engine start. Without this
# the build picks up whatever `nvcc` is first on PATH -- on this box that is
# CUDA 11.5, which does not know `compute_90a` and fails the whole startup.
export CUDA_HOME="${CUDA_HOME:-/usr/local/cuda-13.0}"
export PATH="$CUDA_HOME/bin:$PATH"

BRANCH="${BRANCH:-/raid/bo/delta_token_ids_pr}"
BASELINE="${BASELINE:-/raid/bo/delta_baseline}"
PY="${PY:-/home/bo/LMCache-worktrees/mtp/.venv/bin/python}"
OUT_DIR="${OUT_DIR:-$BRANCH/benchmarks/e2e/results}"

L1_SIZE_GB="${L1_SIZE_GB:-120}"
# Must not exceed the model's max_position_embeddings. vLLM only warns when
# it does, then a prompt past the RoPE table trips a device-side assert in
# the compiled kernel and takes the engine core down mid-run. Qwen2.5-7B is
# a 32k model; 128k needs YaRN rope scaling turned on explicitly.
MAX_MODEL_LEN="${MAX_MODEL_LEN:-32768}"

INPUT_LEN="${1:-30000}"
NUM_PROMPTS="${2:-16}"
CONCURRENCY="${3:-4}"
TP="${4:-4}"

# The runner lives next to this script, which is not the checkout under
# test: BRANCH is what gets measured, this file is just the driver.
RUNNER="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)/packed_token_ids_e2e.py"
COMMON=(--input-len "$INPUT_LEN" --num-prompts "$NUM_PROMPTS"
        --concurrency "$CONCURRENCY" --tp "$TP" --out-dir "$OUT_DIR"
        --l1-size-gb "$L1_SIZE_GB" --max-model-len "$MAX_MODEL_LEN")

run_one() {
  local label="$1" checkout="$2"
  echo "=== $label ==="
  # Serially, on distinct ports, so a stale server from a crashed run cannot
  # be mistaken for this one's.
  "$PY" "$RUNNER" run \
    --label "$label" --checkout "$checkout" \
    --lmcache-port 5599 --vllm-port 8199 "${COMMON[@]}"
}

run_one baseline "$BASELINE"
run_one packed "$BRANCH"

"$PY" "$RUNNER" report "$OUT_DIR/baseline.json" "$OUT_DIR/packed.json"
