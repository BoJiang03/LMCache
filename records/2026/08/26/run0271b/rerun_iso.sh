#!/bin/bash
# Re-run one isolated scenario directly, full stderr kept.  $1=scenario $2=model $3=gpu $4=tag
set -u
SP=/tmp/claude-1016/-home-bo-LMCache-worktrees-multi-modal/911c8e4e-a468-4726-ba44-ae873957a060/scratchpad
TREE=/home/bo/LMCache-worktrees/multi_modal_verify
OUT=$SP/run0271
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
  PYTHONPATH="$SP/pyguard:$TREE" \
  /home/bo/venvs/vllm-mm/bin/python isolated_cases.py "$1" "$2" "$T/$1_$2.json" \
  > "$OUT/iso_$4.log" 2>&1
