#!/usr/bin/env bash
# Phase 1 — reproduce claim (1): "GPU Only + LMCache" slower than "GPU Only w/o LMCache".
#   1a = plain vLLM
#   1b = vLLM + LMCacheConnectorV1, LMCache with NO cpu tier and NO remote backend
# All other flags identical to the VAST config (PDF p.2).
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-100 300 600 1000 1500}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )

run_cfg() {
  local tag="$1"; shift
  local dir="$OUT/$tag"; mkdir -p "$dir"
  echo "=== [$tag] launching $(date +%H:%M:%S) ==="
  spawn "$dir/server.log" "$VLLM" "$@"; local pid=$SPAWNED_PID
  if ! wait_health "$dir/server.log" "$pid" 1500; then
    echo "[$tag] FAILED TO START"; teardown; return 1
  fi
  grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
  local c; for c in $CONC; do bench_point "$tag" "$dir" "$c"; done
  teardown
}

run_cfg 1a_vllm_only "${BASE[@]}"

export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/lmcache_gpu_only.yaml
run_cfg 1b_vllm_lmcache "${BASE[@]}" \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'

echo "=== phase1 done $(date +%H:%M:%S) -> $OUT ==="
