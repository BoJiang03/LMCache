#!/bin/bash
# Full-benchmark parity for one model on vLLM 0.27.1, MP path.
# $1=gpu  $2=log/tag  $3.. = benchmark_parity.py args
set -u
SP=/tmp/claude-1016/-home-bo-LMCache-worktrees-multi-modal/911c8e4e-a468-4726-ba44-ae873957a060/scratchpad
TREE=/home/bo/LMCache-worktrees/multi_modal_verify
GPU=$1; TAG=$2; shift 2
T=/tmp/mm27/p_$TAG
rm -rf "$T"; mkdir -p "$T"
cd "$TREE/tests/e2e_mm" || exit 2
exec env \
  CUDA_VISIBLE_DEVICES="$GPU" \
  TMPDIR="$T" \
  HF_HUB_CACHE=/raid/data/hub \
  PYTHONUNBUFFERED=1 \
  LMCACHE_MM_E2E=1 \
  PYTHONPATH="$SP/pyguard:$TREE" \
  /home/bo/venvs/vllm-mm/bin/python benchmark_parity.py "$@" \
  > "$SP/parity0271/$TAG.log" 2>&1
