#!/bin/bash
# One 0.27.1 MP-only suite run.  $1=model key  $2=gpu  $3=pytest -k expr  $4=log
set -u
GUARD=/tmp/claude-1016/-home-bo-LMCache-worktrees-multi-modal/2fd28306-24c3-492e-810e-96e902af066e/scratchpad/pyguard
TREE=/home/bo/LMCache-worktrees/multi_modal_verify
T=/tmp/mm27/${5:-$1}
rm -rf "$T"; mkdir -p "$T"
cd "$TREE/tests/e2e_mm" || exit 2
exec env \
  CUDA_VISIBLE_DEVICES="$2" \
  TMPDIR="$T" \
  HF_HUB_CACHE=/raid/data/hub \
  LMCACHE_MM_E2E=1 \
  LMCACHE_MM_E2E_MODELS="$1" \
  PYTHONPATH="$GUARD:$TREE" \
  /home/bo/venvs/vllm-mm/bin/python -m pytest . -q -p no:randomly -k "$3" > "$4" 2>&1
