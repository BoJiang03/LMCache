#!/bin/bash
# One 0.27.1 MP-only FULL suite run (no -k: preemption included).
# $1=model key  $2=gpu  $3=log name  $4=tmp tag
set -u
SP=/tmp/claude-1016/-home-bo-LMCache-worktrees-multi-modal/911c8e4e-a468-4726-ba44-ae873957a060/scratchpad
TREE=/home/bo/LMCache-worktrees/multi_modal_verify
T=/tmp/mm27/$4
rm -rf "$T"; mkdir -p "$T"
cd "$TREE/tests/e2e_mm" || exit 2
exec env \
  CUDA_VISIBLE_DEVICES="$2" \
  TMPDIR="$T" \
  HF_HUB_CACHE=/raid/data/hub \
  LMCACHE_MM_E2E=1 \
  LMCACHE_MM_E2E_MODELS="$1" \
  PYTHONPATH="$SP/pyguard:$TREE" \
  /home/bo/venvs/vllm-mm/bin/python -m pytest . -q -p no:randomly \
  > "$SP/run0271/$3" 2>&1
