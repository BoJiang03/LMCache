#!/usr/bin/env bash
# Control re-run: 1a (plain vLLM, hybrid allocator ON) at the same concurrency
# points as the 1c control, on the SAME machine state.
#
# Why: the 1c control settled the c=300 anomaly -- the ORIGINAL 1c was the bad
# measurement (155.6s, taken 15 min after the OOM incident); the re-run gives
# 122.0s, within 3% of 1b.  That invalidates the 1.86x pool-halving figure, but
# it does not license quoting 1.47x either: 1a was measured Sep 1 15:30, and the
# box has now been shown to swing a single point by 28% between sessions.  The
# pool-halving cost needs both arms measured in the same session.
#
# Results land in 1a_rerun_* so the original 1c JSONs stay untouched.
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
run_cfg 1a_rerun "${BASE[@]}"
echo "=== 1c control done $(date +%H:%M:%S) ==="
