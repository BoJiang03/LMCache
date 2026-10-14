#!/usr/bin/env bash
# DP-attention + EP DeepSeek-V4-Flash across node 26 (head, DP rank 0) and
# node 16 (headless, DP rank 1), one GPU per node (GPU 6 -> rail7 on both).
#
#   ./launch_dpep.sh head   [base|lmcache]     # on node 26
#   ./launch_dpep.sh worker [base|lmcache]     # on node 16
#
# Overridable: GPU, DP_LOCAL, ALL2ALL, EP_HCA, MAX_MODEL_LEN, GPU_MEM_UTIL.
set -euo pipefail
ROLE=${1:?head|worker}
MODE=${2:-base}

VENV=/raid/bo/p2pbench/venv
MODEL=/raid/bo/p2pbench/models/DeepSeek-V4-Flash
LM_EXTRA_PATH=/raid/bo/p2pbench/lmcache_dev:/raid/bo/p2pbench/extra
LOGDIR=/raid/bo/dpep/logs
HEAD_IP=10.2.201.26
DP_RPC_PORT=13400
API_PORT=8300
LM_MQ_PORT=6755

GPU=${GPU:-6}
DP_LOCAL=${DP_LOCAL:-1}
DP_SIZE=$(( DP_LOCAL * 2 ))
ALL2ALL=${ALL2ALL:-allgather_reducescatter}
MAX_MODEL_LEN=${MAX_MODEL_LEN:-32768}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}

# NCCL (EP all2all) is pinned to the rail of the GPU in use.
if [ "$(hostname)" = "yotta-jap-h200-26" ]; then EP_HCA_DEFAULT=roce_rail7; else EP_HCA_DEFAULT=mlx5_8; fi
EP_HCA=${EP_HCA:-$EP_HCA_DEFAULT}

export CUDA_VISIBLE_DEVICES=$GPU
export CUDA_HOME=/usr/local/cuda-13.0
export TMPDIR=/raid/bo/p2pbench/tmp
export TORCHINDUCTOR_CACHE_DIR=/raid/bo/p2pbench/cache/inductor
export NCCL_IB_HCA=$EP_HCA
export NCCL_IB_GID_INDEX=3
export NCCL_SOCKET_IFNAME=bond0
export GLOO_SOCKET_IFNAME=bond0
export VLLM_SERVER_DEV_MODE=1
export VLLM_ENGINE_READY_TIMEOUT_S=3600
mkdir -p "$LOGDIR" "$TMPDIR" "$TORCHINDUCTOR_CACHE_DIR"

ARGS=(
  --data-parallel-size "$DP_SIZE"
  --data-parallel-size-local "$DP_LOCAL"
  --data-parallel-address "$HEAD_IP"
  --data-parallel-rpc-port "$DP_RPC_PORT"
  --enable-expert-parallel
  --all2all-backend "$ALL2ALL"
  --tensor-parallel-size 1
  --max-model-len "$MAX_MODEL_LEN"
  --gpu-memory-utilization "$GPU_MEM_UTIL"
  --block-size 256
  --enable-prompt-tokens-details
  --no-enable-flashinfer-autotune
  --no-enable-prefix-caching
)
if [ "$ROLE" = worker ]; then
  ARGS+=(--headless --data-parallel-start-rank "$DP_LOCAL")
else
  ARGS+=(--host 0.0.0.0 --port "$API_PORT")
fi
if [ "$MODE" = lmcache ]; then
  export PYTHONPATH=$LM_EXTRA_PATH
  ARGS+=(--kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_load_failure_policy\":\"recompute\",\"kv_connector_extra_config\":{\"lmcache.mp.port\":$LM_MQ_PORT}}")
fi

LOG=$LOGDIR/vllm_${ROLE}_${MODE}.log
nohup "$VENV/bin/vllm" serve "$MODEL" "${ARGS[@]}" > "$LOG" 2>&1 &
echo "pid $! log $LOG"
