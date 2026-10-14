#!/usr/bin/env bash
# LMCache MP server (one per node) + coordinator (node 26 only).
# Flags follow perf-regression-testing's vllm_lmcache deployment, except
# observability stays on (the P2P counters are what we measure) and the only L2
# is P2P; P2P/coordinator flags added.  P2P traffic is pinned to
# one rail via UCX_NET_DEVICES and the advertised IP.
#
#   ./launch_lmcache.sh <p2p_rail>      # node 26 first (starts the coordinator), then node 16
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
P2P_RAIL=${1:?p2p rail number 1..8}
L1_GB=${L1_GB:-256}

IP=$(rail_ip "$P2P_RAIL")
DEV=$(rail_dev "$P2P_RAIL")

export PYTHONPATH=$LM_EXTRA_PATH
export UCX_TLS=rc_mlx5,ud_mlx5
export UCX_NET_DEVICES=${DEV}:1
export UCX_IB_GID_INDEX=3
export UCX_RCACHE_MAX_UNRELEASED=1024

if [ "$NODE" = 26 ]; then
  nohup "$VENV/bin/lmcache" coordinator --host 0.0.0.0 --port $COORD_PORT \
    --chunk-size 1024 --hash-algorithm blake3 > "$LOGDIR/coordinator.log" 2>&1 &
  echo "coordinator pid $!"
  sleep 5
fi

CUDA_VISIBLE_DEVICES=$GPUS nohup "$VENV/bin/lmcache" server \
  --host 0.0.0.0 --port $LM_MQ_PORT --http-port $LM_HTTP_PORT --http-host 0.0.0.0 \
  --l1-size-gb "$L1_GB" --l1-init-size-gb 20 \
  --max-gpu-workers 1 --max-cpu-workers 8 \
  --chunk-size 1024 --separate-object-groups --l2-prefetch-policy retain \
  --eviction-trigger-watermark 0.95 --eviction-ratio 0.05 --eviction-policy LRU \
  --supported-transfer-mode lmcache_driven \
  --hash-algorithm blake3 \
  --instance-id "node-$NODE" \
  --coordinator-url "http://$HEAD_IP:$COORD_PORT" \
  --coordinator-advertise-ip "$IP" \
  --p2p-advertise-url "$IP:$LM_P2P_PORT" \
  > "$LOGDIR/lmcache_server.log" 2>&1 &
echo "lmcache server pid $! p2p on rail$P2P_RAIL ($DEV $IP)"
