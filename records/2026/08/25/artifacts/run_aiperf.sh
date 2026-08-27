#!/bin/bash
# Replay a short AgentX slice against the live vLLM+LMCache server.
# usage: run_aiperf.sh <tag> [duration_s] [concurrency] [entries]
set -u
. "$(dirname "$0")/env.sh"
TAG=${1:-lazy}
DUR=${2:-180}
CONC=${3:-4}
ENTRIES=${4:-12}
ART=$SD/artifacts/$TAG
rm -rf "$ART"; mkdir -p "$ART"

set -x
$AIPERF profile \
  --model agentx \
  --url "http://127.0.0.1:$VLLM_PORT" \
  --endpoint-type chat \
  --streaming \
  --tokenizer "$MODEL" \
  --scenario inferencex-agentx-mvp \
  --unsafe-override \
  --public-dataset semianalysis-cc-traces-weka-062126 \
  --max-context-length "${MAXCTX:-$MAX_MODEL_LEN}" \
  --num-dataset-entries "$ENTRIES" \
  --concurrency "$CONC" \
  --benchmark-duration "$DUR" \
  --benchmark-grace-period 60 \
  --use-think-time-only \
  --random-seed 1234 \
  --output-artifact-dir "$ART" \
  2>&1 | tee "$LOGDIR/${TAG}_aiperf.log"
