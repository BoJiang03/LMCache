#!/usr/bin/env bash
# LMCache coordinator (node 26 only) + one LMCache MP server per node.
# P2P traffic is pinned to one rail via UCX_NET_DEVICES and the advertised IP.
#
#   ./launch_lmcache.sh <rail>          # rail in 1..8; run on node 26, then node 16
#
# Same rail number on both nodes -> same physical rail.  rail7 = GPU 6's NIC,
# the rail the EP all2all uses in launch_dpep.sh.
set -euo pipefail
RAIL=${1:?rail number 1..8}

VENV=/raid/bo/p2pbench/venv
LM_EXTRA_PATH=/raid/bo/p2pbench/lmcache_dev:/raid/bo/p2pbench/extra
LOGDIR=/raid/bo/dpep/logs
GPU=${GPU:-6}
L1_GB=${L1_GB:-100}
LM_MQ_PORT=6755
LM_HTTP_PORT=7755
LM_P2P_PORT=8755
COORD_PORT=9455
COORD_HOST=10.2.201.26

# railN IP: 172.(14+2N).0.40 on node 26, .30 on node 16; the verbs device is
# looked up from the netdev (names differ: roce_railN on 26, mlx5_* on 16).
if [ "$(hostname)" = "yotta-jap-h200-26" ]; then NODE=26; HOSTOCT=40; else NODE=16; HOSTOCT=30; fi
IP=172.$((14 + 2 * RAIL)).0.$HOSTOCT
NETDEV=$(ip -4 -o addr | awk -v ip="$IP" '$4 ~ "^"ip"/" {print $2}')
[ -n "$NETDEV" ] || { echo "rail IP $IP not on this host" >&2; exit 1; }
DEV=$(ls -d /sys/class/infiniband/*/device/net/"$NETDEV" | cut -d/ -f5)

export PYTHONPATH=$LM_EXTRA_PATH
export TMPDIR=/raid/bo/p2pbench/tmp
export UCX_TLS=rc_mlx5,ud_mlx5
export UCX_NET_DEVICES=${DEV}:1
export UCX_IB_GID_INDEX=3
export UCX_RCACHE_MAX_UNRELEASED=1024
mkdir -p "$LOGDIR"

if [ "$NODE" = 26 ]; then
  nohup "$VENV/bin/lmcache" coordinator --host 0.0.0.0 --port $COORD_PORT \
    --chunk-size 256 --hash-algorithm blake3 > "$LOGDIR/coordinator.log" 2>&1 &
  echo "coordinator pid $!"
  sleep 5
fi

CUDA_VISIBLE_DEVICES=$GPU nohup "$VENV/bin/lmcache" server \
  --host 0.0.0.0 --port $LM_MQ_PORT --http-port $LM_HTTP_PORT --http-host 0.0.0.0 \
  --chunk-size 256 --l1-size-gb "$L1_GB" --eviction-policy LRU \
  --l1-align-bytes 65536 --hash-algorithm blake3 \
  --instance-id "node-$NODE" \
  --coordinator-url "http://$COORD_HOST:$COORD_PORT" \
  --coordinator-advertise-ip "$IP" \
  --p2p-advertise-url "$IP:$LM_P2P_PORT" \
  --separate-object-groups \
  > "$LOGDIR/lmcache_server_rail${RAIL}.log" 2>&1 &
echo "lmcache server pid $! rail$RAIL $DEV $IP"
