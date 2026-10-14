#!/usr/bin/env bash
# AgentX (weka cc-traces 256k) replay with the prt-aiperf image, against the
# vllm-router on node 26.  Same aiperf args/env as perf-regression-testing's
# `agentic` workload; model/tokenizer/url switched to this deployment.
#
#   ./run_aiperf.sh <label> <concurrency> [duration_s]      # run on node 26
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
LABEL=${1:?label}
CONC=${2:?concurrency}
DURATION=${3:-960}
IMAGE=${AIPERF_IMAGE:-prt-aiperf:0047a047b}
TRACE=/raid/bo/prt-cache/work/weka-062126-256k-dir
# With HF_HUB_OFFLINE=1 AIPerf rejects a local tokenizer path and only resolves
# repo ids from the HF cache; this repo id is aliased in $HF to symlinks into $MODEL.
HF=/raid/bo/dpep/hf
TOKENIZER=deepseek-ai/DeepSeek-V4-Flash-0731
OUT=/raid/bo/dpep/aiperf/$LABEL
[ -e "$OUT" ] && { echo "$OUT exists" >&2; exit 1; }
mkdir -p "$OUT"/aiperf_mmap/base "$OUT"/aiperf_mmap/cache

docker run --rm --name "dpep-aiperf-$LABEL" \
  --network=host --ipc=host --user "$(id -u):$(id -g)" -e HOME=/tmp \
  -v "$MODEL:$MODEL:ro" -v "$TRACE:$TRACE:ro" -v "$OUT:$OUT" -v "$HF:$HF:ro" \
  -e HF_HUB_OFFLINE=1 -e HF_HOME="$HF" \
  -e AIPERF_DATASET_CONFIGURATION_TIMEOUT=3600 \
  -e AIPERF_SERVICE_PROFILE_CONFIGURE_TIMEOUT=3600 \
  -e AIPERF_DATASET_WEKA_PARALLEL_WORKERS=1 \
  -e AIPERF_DATASET_MMAP_BASE_PATH="$OUT/aiperf_mmap/base" \
  -e AIPERF_DATASET_MMAP_CACHE_DIR="$OUT/aiperf_mmap/cache" \
  -e PRT_LIVE_HISTORY_TOKENIZER="$MODEL" \
  -e PRT_LIVE_HISTORY_LOG_DIR="$OUT/aiperf/prt_live_history" \
  "$IMAGE" \
  aiperf profile \
    --url "http://127.0.0.1:$ROUTER_PORT" \
    --model "$SERVED_NAME" \
    --tokenizer "$TOKENIZER" \
    --endpoint-type chat --streaming \
    --extra-inputs ignore_eos:true \
    --use-server-token-count \
    --random-seed 0 --export-level records --slice-duration 30 --no-gpu-telemetry \
    --scenario inferencex-agentx-mvp --unsafe-override \
    --custom-dataset-type weka_trace --input-file "$TRACE" \
    --max-context-length 262144 --prompt-corpus coding --cache-bust first_turn_prefix \
    --concurrency "$CONC" \
    --benchmark-duration "$DURATION" --benchmark-grace-period 60 \
    --server-metrics "http://10.2.201.26:$VLLM_PORT/metrics" "http://10.2.201.16:$VLLM_PORT/metrics" \
    --ui simple \
    --artifact-dir "$OUT/aiperf" \
  2>&1 | tee "$OUT/aiperf.log"
