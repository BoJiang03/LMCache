#!/bin/bash
# Hard teardown: kill every lmcache MP server and every vllm engine/worker this
# smoke owns, then verify the GPUs and ports are actually clear. up.sh refuses
# to start until this reports clean, so a stale server can never silently keep
# the port and serve a run whose L1 budget it does not have.
. "$(dirname "$0")/env.sh"
for pid in $(pgrep -f 'lmcache.v1.multiprocess.http_server'); do kill -9 "$pid" 2>/dev/null; done
for pid in $(pgrep -f "vllm serve|VLLM::EngineCore"); do kill -9 "$pid" 2>/dev/null; done
sleep 6
owned=$(echo "$GPUS" | tr ',' ' ')
for g in $owned; do
  bus=$(nvidia-smi --query-gpu=index,pci.bus_id --format=csv,noheader | awk -F', ' -v i="$g" '$1==i{print $2}')
  for pid in $(nvidia-smi --query-compute-apps=gpu_bus_id,pid --format=csv,noheader | awk -F', ' -v b="$bus" '$1==b{print $2}'); do
    kill -9 "$pid" 2>/dev/null && echo "killed leftover $pid on gpu $g"
  done
done
sleep 5
rm -f "$LOGDIR"/*.pid
bad=0
for g in $owned; do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits | awk -F', ' -v i="$g" '$1==i{print $2}')
  [ "$used" -gt 1000 ] && { echo "DIRTY: gpu $g still holds ${used} MiB"; bad=1; }
done
ss -ltn 2>/dev/null | grep -qE ":($MP_PORT|$HTTP_PORT|$VLLM_PORT)\b" && { echo "DIRTY: a port is still bound"; bad=1; }
[ $bad -eq 0 ] && echo "clean" || exit 1
