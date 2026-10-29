#!/usr/bin/env bash
# One group: LMCache MP server + vLLM with LMCacheMPConnector, then client.py.
# Usage: run.sh <lmcache_tree> <label> <lazy 0|1>
set -u
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
TREE=$1 LABEL=$2 LAZY=$3
OUT=$OUT_ROOT/$LABEL
mkdir -p "$OUT"
export PYTHONPATH=$TREE CUDA_VISIBLE_DEVICES=$GPU PYTHONHASHSEED=0 VLLM_BATCH_INVARIANT=1

# Record which tree the processes import; cwd wins over PYTHONPATH.
cd "$TREE"
"$PY" -c "import lmcache, lmcache.integration.vllm.utils as u; print('lmcache from', lmcache.__file__, 'lora_tag', hasattr(u, 'LORA_TAG_KEY'))" > "$OUT/tree.txt" 2>&1

"$PY" -m lmcache.v1.multiprocess.server --l1-size-gb 20 --chunk-size 256 \
  --eviction-policy LRU --port "$LMCACHE_PORT" > "$OUT/lmcache.log" 2>&1 &
LMC=$!
sleep 15
kill -0 $LMC 2> /dev/null || { echo "lmcache server died"; tail -3 "$OUT/lmcache.log"; exit 1; }

EXTRA="\"lmcache.mp.host\": \"tcp://localhost\", \"lmcache.mp.port\": $LMCACHE_PORT"
if [[ $LAZY == 1 ]]; then
  # FIFO with threshold 1 drains a finished request on the next step.
  EXTRA="$EXTRA"', "lmcache.mp.lazy_offload": true, "lmcache.mp.lazy_offload_policy": "FIFO", "lmcache.mp.lazy_offload_threshold": 1'
fi
VLLM_SERVER_DEV_MODE=1 "$PY" -m vllm.entrypoints.openai.api_server \
  --model "$MODEL" --port "$VLLM_PORT" \
  --enable-lora --max-lora-rank 8 --max-loras 2 \
  --lora-modules lora-a="$ADAPTERS/lora-a" lora-b="$ADAPTERS/lora-b" \
  --gpu-memory-utilization 0.3 --max-model-len 8192 --max-num-batched-tokens 1024 \
  --no-async-scheduling --attention-backend FLASH_ATTN \
  --kv-transfer-config "{\"kv_connector\": \"LMCacheMPConnector\", \"kv_role\": \"kv_both\", \"kv_connector_extra_config\": {$EXTRA}}" \
  > "$OUT/vllm.log" 2>&1 &
VLLM=$!

for _ in $(seq 300); do
  curl -sf "localhost:$VLLM_PORT/health" > /dev/null && break
  kill -0 $VLLM 2> /dev/null || break
  sleep 2
done

"$PY" "$E2E/client.py" "$VLLM_PORT" "$BASELINE_PORT" "$MODEL" "$OUT/results.json" > "$OUT/client.log" 2>&1
echo "client exit=$?" >> "$OUT/client.log"

kill $VLLM; wait $VLLM 2> /dev/null
kill $LMC; wait $LMC 2> /dev/null
echo "done $LABEL"
