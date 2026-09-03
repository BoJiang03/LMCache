#!/usr/bin/env bash
# Control re-run: 1c (plain vLLM, allocator forced off) at the concurrency points
# where 1b beat it, executed back-to-back with 1b on the SAME machine state.
#
# Why: 1c's original run started 17:53 on Sep 1 -- 15 minutes after the OOM
# incident, while root's two k8s lmcache pods were restarting and re-claiming
# 200 GB each.  1b (Sep 2 11:00) was 24% faster at c=300 with an identical
# 13,724,416-token pool and identical prompts.  Either the connector really
# changes scheduling, or 1c was measured on a perturbed box.  This tells us which.
#
# Results land in 1c_rerun_* so the original 1c JSONs stay untouched.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-300 600}"
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

unset LMCACHE_CONFIG_FILE
run_cfg 1c_rerun "${BASE[@]}" --disable-hybrid-kv-cache-manager
echo "=== 1c control done $(date +%H:%M:%S) ==="
