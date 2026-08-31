#!/bin/bash
# Lazy-vs-eager A/B chain on Trinity. Each arm: fresh MP server (empty L1),
# fresh engine (empty GPU pool), same corpus/seed/concurrency/duration.
# Usage: ab_chain.sh <arm>...   where arm is like e48, l48, e72, l72
#   e = eager (plain MP connector), l = lazy (record 08/29/5 recipe), number = aiperf concurrency.
set -u
W=/tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/aa12d55f-c087-43f5-b7de-9e19b1dcd21f/scratchpad/sweep
# The harness may re-execute a Bash call carrying the nohup; a duplicate
# instance killed the first one's MP server mid-arm on 08-30 (bad_run1).
exec 9>"$W/ab_chain.lock"
flock -n 9 || { echo "[chain] another instance holds the lock, exiting"; exit 0; }
M=/raid/data/hub/models--arcee-ai--Trinity-Large-Thinking-FP8-Block/snapshots/6412c1ad1588c664977b0f830b80a20273173318
PY=/home/bo/venvs/vllm-lazy/bin/python
VLLM=/home/bo/venvs/vllm-lazy/bin/vllm
AIPERF=/home/bo/venvs/aiperf/bin/aiperf
export PATH=/usr/local/cuda/bin:$PATH
export CPATH=/raid/data/hub/pr4499_agentic/pydev/usr/include/python3.12:/raid/data/hub/pr4499_agentic/pydev/usr/include
export CUDA_VISIBLE_DEVICES=1,2,3,4

LAZY_KEYS='"lmcache.mp.lazy_offload":true,
  "lmcache.mp.lazy_offload_policy":"EVICTION_AWARE",
  "lmcache.mp.lazy_offload_horizon_steps":2.5,
  "lmcache.mp.lazy_offload_store_release":"lru_tail",
  "lmcache.mp.lazy_offload_max_drain_blocks_per_step":0,
  "lmcache.mp.lazy_offload_danger_floor_max_blocks":8192,
  "lmcache.mp.lazy_offload_announce_hits":false,
  "lmcache.mp.lazy_offload_max_deferral_seconds":30,'

kill_pidfile() {
  local pf=$1 pat=$2
  [ -f "$pf" ] || return 0
  local pid; pid=$(cat "$pf")
  if ps -p "$pid" -o args= 2>/dev/null | grep -q "$pat"; then
    kill "$pid" 2>/dev/null
    for i in $(seq 60); do ps -p "$pid" >/dev/null 2>&1 || break; sleep 2; done
    ps -p "$pid" >/dev/null 2>&1 && kill -9 "$pid" 2>/dev/null
  fi
  rm -f "$pf"
}

wait_gpu_free() {
  for i in $(seq 60); do
    local used; used=$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits -i 1,2,3,4 | sort -n | tail -1)
    [ "$used" -lt 3000 ] && return 0
    sleep 5
  done
  echo "[chain] WARN: GPU memory not fully released"; return 1
}

start_mp() {
  kill_pidfile $W/mp.pid "multiprocess.http_server"
  # the pre-chain MP server was started outside this script; clear it by port
  local old; old=$(pgrep -u 1016 -f "multiprocess.http_server.*--port 8971" | head -1)
  if [ -n "$old" ]; then kill "$old"; for i in $(seq 30); do ps -p "$old" >/dev/null 2>&1 || break; sleep 2; done; fi
  nohup $PY -m lmcache.v1.multiprocess.http_server \
    --host 127.0.0.1 --port 8971 --http-host 127.0.0.1 --http-port 8972 \
    --l1-size-gb 250 --chunk-size 256 --eviction-policy LRU \
    --separate-object-groups \
    --script-allowed-imports hashlib --max-workers 4 > $W/${1}_mp.log 2>&1 9>&- &
  echo $! > $W/mp.pid
  for i in $(seq 90); do curl -sf http://127.0.0.1:8972/healthcheck >/dev/null 2>&1 && return 0; sleep 1; done
  echo "[chain] FATAL: MP server not healthy"; return 1
}

FIFO_KEYS='"lmcache.mp.lazy_offload":true,
  "lmcache.mp.lazy_offload_policy":"FIFO",'

start_engine() {
  local name=$1 mode=$2 extra=""
  [ "$mode" = l ] && extra=$LAZY_KEYS
  [ "$mode" = f ] && extra=$FIFO_KEYS
  local kv="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{$extra\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":8971}}"
  nohup $VLLM serve "$M" --served-model-name agentx \
    --host 127.0.0.1 --port 8973 --tensor-parallel-size 4 \
    --max-model-len 262144 --kv-cache-dtype fp8 --gpu-memory-utilization 0.90 \
    --kv-transfer-config "$kv" > $W/${name}_server.log 2>&1 9>&- &
  echo $! > $W/engine.pid
  for i in $(seq 250); do
    curl -sf http://127.0.0.1:8973/v1/models >/dev/null 2>&1 && return 0
    ps -p "$(cat $W/engine.pid)" >/dev/null 2>&1 || { echo "[chain] FATAL: engine died during start"; return 1; }
    sleep 6
  done
  echo "[chain] FATAL: engine not ready in 25 min"; return 1
}

sampler() {
  local out=$1
  while true; do
    T=$(date +%s)
    {
      echo "=== t=$T"
      curl -sf --max-time 5 http://127.0.0.1:8973/metrics 2>/dev/null | grep -E '^vllm:(num_requests_running|num_requests_waiting|kv_cache_usage_perc|gpu_cache_usage_perc|prompt_tokens_by_source|prompt_tokens_total|generation_tokens_total)'
      curl -sf --max-time 5 http://127.0.0.1:8972/metrics 2>/dev/null | grep -E '^lmcache_mp_(l1_read_chunks_total|l1_write_chunks_total|l1_memory_usage_bytes|l1_usage_ratio|lookup_hit_tokens_total|lookup_requested_tokens_total|num_chunks_loaded_total)'
    } >> "$out" 2>/dev/null
    sleep 15
  done
}

run_arm() {
  local name=$1
  local mode=${name:0:1}
  local conc=${name:1}
  echo "[chain] arm $name starting $(date +%T)"
  start_mp "$name" || return 1
  start_engine "$name" "$mode" || return 1
  echo "[chain] arm $name engine ready $(date +%T)"
  sampler $W/${name}_samples.log 9>&- & local spid=$!
  (cd $W && $AIPERF profile --model agentx \
    --endpoint-type chat --streaming \
    --url http://127.0.0.1:8973 \
    --tokenizer "$M" \
    --scenario inferencex-agentx-mvp \
    --public-dataset semianalysis-cc-traces-weka-062126-256k \
    --concurrency "$conc" --benchmark-duration 1800 \
    --benchmark-grace-period 600 --random-seed 1234 \
    --artifact-dir $W/${name}_artifacts) > $W/${name}_client.log 2>&1
  local rc=$?
  kill $spid 2>/dev/null
  curl -sf http://127.0.0.1:8972/metrics > $W/${name}_mp_final.prom 2>/dev/null
  curl -sf http://127.0.0.1:8973/metrics > $W/${name}_vllm_final.prom 2>/dev/null
  kill_pidfile $W/engine.pid "vllm serve"
  wait_gpu_free
  echo "[chain] arm $name done rc=$rc $(date +%T)"
  return 0
}

for arm in "$@"; do
  run_arm "$arm" || { echo "[chain] ABORT at $arm"; exit 1; }
done
echo "[chain] ALL_DONE"
