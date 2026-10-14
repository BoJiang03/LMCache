#!/usr/bin/env bash
# vllm-router in front of both nodes' API servers (node 26 only).
#   ./launch_router.sh [policy]     # default consistent_hash
# No --intra-node-data-parallel-size: the router would send X-data-parallel-rank
# 0/1 to both nodes, but node 16 owns ranks 2/3 ("engine-core error").  The router
# picks the node; each node's API server balances across its local ranks, which
# share that node's LMCache server.
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
POLICY=${1:-consistent_hash}

nohup /raid/bo/tools/router-venv/bin/vllm-router \
  --host 0.0.0.0 --port $ROUTER_PORT \
  --policy "$POLICY" \
  --worker-urls "http://10.2.201.26:$VLLM_PORT" "http://10.2.201.16:$VLLM_PORT" \
  --worker-startup-timeout-secs 1800 \
  > "$LOGDIR/router_$POLICY.log" 2>&1 &
echo "router pid $! policy $POLICY port $ROUTER_PORT"
