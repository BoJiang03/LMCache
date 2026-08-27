#!/bin/bash
# Bring up LMCache MP server + vLLM for the AgentX smoke.
# usage: up.sh <off|eager|lazy>
set -u
CONFIG=${1:-lazy}
. "$(dirname "$0")/env.sh"
mkdir -p "$LOGDIR"

# Refuse to start on a dirty node: an MP server that lost the port race keeps
# running with the wrong L1 budget, and /healthcheck answers from whichever one
# owns the port -- so "mp-server up" is not evidence the new one is serving.
if ss -ltn 2>/dev/null | grep -qE ":($MP_PORT|$HTTP_PORT|$VLLM_PORT)\b"; then
  echo "refusing to start: one of $MP_PORT/$HTTP_PORT/$VLLM_PORT is already bound; run down.sh" >&2
  exit 1
fi
if pgrep -f 'lmcache.v1.multiprocess.http_server' >/dev/null; then
  echo "refusing to start: an MP server is already running; run down.sh" >&2
  exit 1
fi

# The cwd shadows the editable-install finder ($REPO/lmcache is a real
# package dir), so cwd -- not the finder -- decides which tree is served.
# Make that explicit and assert it, so a baseline run cannot silently import
# the branch.
cd "$REPO" || exit 1
served=$($PY -c 'import lmcache; print(lmcache.__file__)' 2>/dev/null | tail -1)
case "$served" in
  "$REPO"/*) echo "serving lmcache from $REPO" ;;
  *) echo "refusing to start: lmcache resolves to $served, not $REPO" >&2; exit 1 ;;
esac

export CUDA_VISIBLE_DEVICES=$GPUS
export PYTHONPATH=$REPO
export VLLM_SERVER_DEV_MODE=1
export PYTHONFAULTHANDLER=1
export CPATH

if [ "$CONFIG" != "off" ]; then
  echo "--- starting MP server (L1=${L1_GB}GB) on $MP_PORT / http $HTTP_PORT"
  nohup $PY -m lmcache.v1.multiprocess.http_server \
    --host 127.0.0.1 --port $MP_PORT \
    --http-host 127.0.0.1 --http-port $HTTP_PORT \
    --l1-size-gb $L1_GB --eviction-policy LRU \
    --script-allowed-imports hashlib --max-workers 4 \
    > "$LOGDIR/${CONFIG}_server.log" 2>&1 &
  spid=$!
  echo $spid > "$LOGDIR/server.pid"
  up=0
  for i in $(seq 90); do
    curl -sf "http://127.0.0.1:$HTTP_PORT/healthcheck" >/dev/null && { up=1; break; }
    kill -0 $spid 2>/dev/null || { echo "mp-server died, see $LOGDIR/${CONFIG}_server.log" >&2; exit 1; }
    sleep 1
  done
  [ $up -eq 1 ] || { echo "mp-server never healthy" >&2; exit 1; }
  # Assert the *listening* server is the one we just started, by reading the L1
  # target out of its own log. /status reports the LazyMemoryAllocator's
  # currently expanded size, which grows ~10 GB at a time in the background --
  # so it is a progress gauge, not the configured cap, and cannot be used to
  # identify the server.
  want=$(( L1_GB * 1024 ))
  got=""
  for i in $(seq 40); do
    got=$(grep -oE 'pinned memory, now total is [0-9]+ MB / [0-9]+ MB' "$LOGDIR/${CONFIG}_server.log" \
          | tail -1 | awk '{print $(NF-1)}')
    [ -n "$got" ] && break
    sleep 2
  done
  if [ "${got:-0}" != "$want" ]; then
    echo "refusing to run: listening MP server reports L1 target ${got:-unknown} MB, asked ${want} MB" >&2
    exit 1
  fi
  echo "mp-server up (pid $spid, L1 target ${want} MB confirmed)"
fi

KV_BASE="\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT"
KV_ARGS=()
if [ "$CONFIG" = "eager" ]; then
  KV_ARGS=(--kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$KV_BASE}}")
elif [ "$CONFIG" = "lazy" ]; then
  KV_ARGS=(--kv-transfer-config "{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$KV_BASE,\"lmcache.mp.lazy_offload\":true,\"lmcache.mp.lazy_offload_policy\":\"EVICTION_AWARE\",\"lmcache.mp.lazy_offload_horizon_steps\":$HORIZON}}")
fi

echo "--- starting vllm serve $MODEL tp=$TP pool=${POOL_BLOCKS}blk (${POOL_GIB}GiB) config=$CONFIG"
nohup $VLLM serve "$MODEL" \
  --port $VLLM_PORT \
  --tensor-parallel-size $TP \
  --max-model-len $MAX_MODEL_LEN \
  --block-size 16 \
  --gpu-memory-utilization 0.60 \
  --num-gpu-blocks-override $POOL_BLOCKS \
  --served-model-name agentx \
  "${KV_ARGS[@]}" \
  > "$LOGDIR/${CONFIG}_vllm.log" 2>&1 &
vpid=$!
echo $vpid > "$LOGDIR/vllm.pid"
for i in $(seq 90); do
  [ "$(curl -s -o /dev/null -w '%{http_code}' http://127.0.0.1:$VLLM_PORT/health)" = "200" ] && { echo "vllm up (pid $vpid)"; exit 0; }
  kill -0 $vpid 2>/dev/null || { echo "vllm died, see $LOGDIR/${CONFIG}_vllm.log" >&2; exit 1; }
  sleep 8
done
echo "vllm never healthy" >&2; exit 1
