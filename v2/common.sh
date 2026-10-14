# Shared settings for the v2 deployment: DeepSeek-V4-Flash-0731, DP4/EP4 split
# 2+2 across node 26 (head, DP ranks 0-1) and node 16 (DP ranks 2-3).
# Sourced by the launch scripts; based on the team's single-node reference config.

VENV=/raid/bo/p2pbench/venv
LM_EXTRA_PATH=/raid/bo/p2pbench/lmcache_dev:/raid/bo/p2pbench/extra
MODEL=/raid/bo/models/DeepSeek-V4-Flash-0731   # identical on both nodes (LMCache keys include it)
LOGDIR=/raid/bo/dpep/logs/v2

GPUS=${GPUS:-4,6}                # GPU 4 -> rail5, GPU 6 -> rail7 on both nodes
DP_LOCAL=2
DP_SIZE=4
HEAD_IP=10.2.201.26
HEAD_NODE=yotta-jap-h200-26

VLLM_PORT=8100
DP_RPC_PORT=13400
LM_MQ_PORT=5655
LM_HTTP_PORT=8180
LM_P2P_PORT=8755
COORD_PORT=9455
ROUTER_PORT=8200
SERVED_NAME=dsv4flash

export CUDA_HOME=/usr/local/cuda-13.0
export TMPDIR=/raid/bo/p2pbench/tmp
export TORCHINDUCTOR_CACHE_DIR=/raid/bo/p2pbench/cache/inductor
mkdir -p "$LOGDIR" "$TMPDIR"

if [ "$(hostname)" = "$HEAD_NODE" ]; then NODE=26; HOSTOCT=40; else NODE=16; HOSTOCT=30; fi

# railN IP is 172.(14+2N).0.{40 on 26, 30 on 16}; verbs device names differ per
# node (roce_railN on 26, mlx5_* on 16), so look them up from the netdev.
rail_ip() { echo "172.$((14 + 2 * $1)).0.$HOSTOCT"; }
rail_dev() {
  local ip netdev
  ip=$(rail_ip "$1")
  netdev=$(ip -4 -o addr | awk -v ip="$ip" '$4 ~ "^"ip"/" {print $2}')
  [ -n "$netdev" ] || { echo "rail$1 IP $ip not on $(hostname)" >&2; return 1; }
  ls -d /sys/class/infiniband/*/device/net/"$netdev" | cut -d/ -f5
}
# Rail attached to each GPU in use (GPU g -> rail g+1, see nvidia-smi topo -m).
gpu_rails() { local g; for g in ${GPUS//,/ }; do echo $((g + 1)); done; }
