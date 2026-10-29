#!/usr/bin/env bash
# Reference vLLM without LMCache, shared by every group. Runs in foreground.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
mkdir -p "$OUT_ROOT"
cd "$E2E"
CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=0 VLLM_BATCH_INVARIANT=1 \
  exec "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$BASELINE_PORT" \
  --enable-lora --max-lora-rank 8 --max-loras 2 \
  --lora-modules lora-a="$ADAPTERS/lora-a" lora-b="$ADAPTERS/lora-b" \
  --gpu-memory-utilization 0.3 --max-model-len 8192 --max-num-batched-tokens 1024 \
  --no-async-scheduling --attention-backend FLASH_ATTN \
  > "$OUT_ROOT/baseline.log" 2>&1
