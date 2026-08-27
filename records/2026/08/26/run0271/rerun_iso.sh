#!/bin/bash
# Re-run one isolated scenario directly, full stderr kept.  $1=scenario $2=model $3=gpu $4=tag
set -u
GUARD=/tmp/claude-1016/-home-bo-LMCache-worktrees-multi-modal/2fd28306-24c3-492e-810e-96e902af066e/scratchpad/pyguard
TREE=/home/bo/LMCache-worktrees/multi_modal_verify
OUT=/tmp/claude-1016/-home-bo-LMCache-worktrees-multi-modal/2fd28306-24c3-492e-810e-96e902af066e/scratchpad/run0271
T=/tmp/mm27/$4
rm -rf "$T"; mkdir -p "$T"
cd "$TREE/tests/e2e_mm" || exit 2
exec env \
  CUDA_VISIBLE_DEVICES="$3" \
  TMPDIR="$T" \
  HF_HUB_CACHE=/raid/data/hub \
  LMCACHE_MM_E2E=1 \
  VLLM_ENABLE_V1_MULTIPROCESSING=0 \
  PYTHONHASHSEED=0 \
  PYTHONPATH="$GUARD:$TREE" \
  /home/bo/venvs/vllm-mm/bin/python isolated_cases.py "$1" "$2" "$T/$1_$2.json" \
  > "$OUT/iso_$4.log" 2>&1
