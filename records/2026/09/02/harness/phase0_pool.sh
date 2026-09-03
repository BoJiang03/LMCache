#!/usr/bin/env bash
# Phase 0: quantify the --disable-hybrid-kv-cache-manager confound.
# Launch vLLM twice with identical flags except the hybrid allocator, and read
# the GPU KV cache pool size out of the startup log. No benchmark, no LMCache.
set -uo pipefail
source "$(dirname "$0")/env.sh"
OUT=$REPRO_ROOT/results/phase0; mkdir -p "$OUT"

run() {
  local tag="$1"; shift
  local log="$OUT/$tag.log"
  echo "=== [$tag] launching: extra flags: $* ==="
  $VLLM serve "$MODEL" \
    --host 0.0.0.0 --port "$PORT" \
    --served-model-name "$SERVED_NAME" \
    --tensor-parallel-size 8 \
    --max-model-len 131072 \
    --enable-prefix-caching \
    --block-size=64 \
    --max-num-seqs 256 \
    "$@" > "$log" 2>&1 &
  local pid=$!
  # wait for health or death
  for i in $(seq 1 180); do
    if curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1; then
      echo "[$tag] up after ${i}0s"; break
    fi
    kill -0 $pid 2>/dev/null || { echo "[$tag] DIED early"; break; }
    sleep 10
  done
  echo "--- [$tag] pool lines ---"
  grep -iE "GPU KV cache size|Maximum concurrency|kv_cache_manager|KV cache group|available_kv_cache_memory|num_gpu_blocks" "$log" | tail -20
  kill $pid 2>/dev/null; wait $pid 2>/dev/null
  pkill -f "VLLM::EngineCore" 2>/dev/null
  sleep 20
}

run hybrid_ON
run hybrid_OFF --disable-hybrid-kv-cache-manager
echo "=== done, logs in $OUT ==="
