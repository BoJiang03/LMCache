#!/usr/bin/env bash
# vLLM DP4/EP4 across two nodes.  Serving flags follow perf-regression-testing
# (config.toml [vllm], tuned on the same agentic workload; see its CONFIG.md),
# plus the multi-node DP flags and the DeepSeek-V4 model flags.
# --data-parallel-hybrid-lb: each node's API server (port 8100) serves only its
# local DP ranks; vllm-router decides which node a request goes to.
#
#   ./launch_vllm.sh            # same command on node 26 (head) and node 16
#   SPEC=0 ./launch_vllm.sh     # without DSpark speculative decoding
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
SPEC=${SPEC:-1}

# NCCL (EP all2all) uses the rails of the GPUs in use.
HCAS=$(for r in $(gpu_rails); do rail_dev "$r"; done | paste -sd,)

export CUDA_VISIBLE_DEVICES=$GPUS
export PYTHONPATH=$LM_EXTRA_PATH
export VLLM_USE_RUST_FRONTEND=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
export VLLM_SERVER_DEV_MODE=1
export VLLM_USE_V2_MODEL_RUNNER=1
export VLLM_HTTP_TIMEOUT_KEEP_ALIVE=600
export PYTHONHASHSEED=0
export NCCL_IB_HCA=$HCAS
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0

ARGS=(
  --trust-remote-code
  --kv-cache-dtype fp8
  --block-size 256
  --enable-expert-parallel
  --host 0.0.0.0 --port $VLLM_PORT
  --served-model-name $SERVED_NAME
  --data-parallel-size $DP_SIZE
  --data-parallel-size-local $DP_LOCAL
  --data-parallel-address $HEAD_IP
  --data-parallel-rpc-port $DP_RPC_PORT
  --data-parallel-hybrid-lb
  --max-num-batched-tokens 8192
  --gpu-memory-utilization 0.95
  --no-enable-flashinfer-autotune
  --max-num-seqs 256
  --max-model-len auto
  --enable-ep-weight-filter
  --reasoning-parser deepseek_v4
  --reasoning-config '{"reasoning_parser":"deepseek_v4","reasoning_start_str":"","reasoning_end_str":""}'
  --enable-prompt-tokens-details
  --generation-config vllm
  --enable-prefix-caching
  --prefix-cache-retention-interval 1024
  --disable-cascade-attn
  --kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LM_MQ_PORT,\"lmcache.mp.mq_timeout\":1500}}"
)
[ "$SPEC" = 1 ] && ARGS+=(--speculative-config '{"method":"dspark","num_speculative_tokens":7,"draft_sample_method":"probabilistic"}')
[ "$NODE" = 16 ] && ARGS+=(--data-parallel-start-rank $DP_LOCAL)

LOG=$LOGDIR/vllm.log
nohup "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1 &
echo "vllm pid $! node $NODE gpus $GPUS nccl_hca $HCAS spec=$SPEC log $LOG"
