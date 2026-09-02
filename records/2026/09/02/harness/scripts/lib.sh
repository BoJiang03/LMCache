# shared launch/bench helpers
# Shared box: only ever touch processes THIS script started.
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"

MY_PIDS=()   # process-group leaders we spawned

# spawn <logfile> <cmd...>  -- setsid so we own a killable process group.
# Sets $SPAWNED_PID in the CALLER's shell. Never use $(spawn ...): the subshell
# would drop the pid from MY_PIDS and teardown would kill nothing.
SPAWNED_PID=""
spawn() {
  local log="$1"; shift
  setsid "$@" > "$log" 2>&1 &
  SPAWNED_PID=$!
  MY_PIDS+=("$SPAWNED_PID")
}

wait_health() {  # wait_health <logfile> <pid> <timeout_s>
  local log="$1" pid="$2" t="${3:-1200}" i=0
  while [ $i -lt "$t" ]; do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  up after ${i}s"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "  process exited early"; tail -25 "$log"; return 1; }
    sleep 5; i=$((i+5))
  done
  echo "  TIMEOUT after ${t}s"; return 1
}

teardown() {
  local pid
  for pid in "${MY_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  done
  # wait for OUR pids only (never poll global nvidia-smi: other tenants live here)
  local i=0
  while [ $i -lt 120 ]; do
    local alive=0
    for pid in "${MY_PIDS[@]:-}"; do
      [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
    done
    [ $alive -eq 0 ] && break
    sleep 5; i=$((i+5))
  done
  for pid in "${MY_PIDS[@]:-}"; do
    [ -n "$pid" ] && kill -KILL -"$pid" 2>/dev/null
  done
  MY_PIDS=()
  sleep 20   # let the driver release HBM
}
# EXIT runs teardown on normal end; INT/TERM must ALSO exit -- a bare trap
# handler returns and the script would carry on running the next config.
on_signal() { echo "  [signal] tearing down"; teardown; exit 130; }
trap teardown EXIT
trap on_signal INT TERM

# bench_point <tag> <outdir> <concurrency>  -- cold pass then warm pass
bench_point() {
  local tag="$1" out="$2" c="$3"
  local common=( --backend vllm --base-url "http://127.0.0.1:$PORT"
    --model "$MODEL" --served-model-name "$SERVED_NAME" --tokenizer "$MODEL"
    --dataset-name random --random-input-len 60000 --random-output-len 1
    --random-range-ratio 0.0 --ignore-eos --seed 42
    --num-prompts "$c" --max-concurrency "$c"
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99
    --save-result --result-dir "$out" )
  local p
  for p in cold warm; do
    echo "  [$tag c=$c] $p pass  ($(date +%H:%M:%S))"
    $VLLM bench serve "${common[@]}" --result-filename "c${c}_${p}.json" \
        > "$out/c${c}_${p}.log" 2>&1
  done
  $PY - "$out/c${c}_warm.json" <<'PYEOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"    -> warm p99_ttft={d.get('p99_ttft_ms',0)/1000:.1f}s "
          f"mean_ttft={d.get('mean_ttft_ms',0)/1000:.1f}s "
          f"total_tok/s={d.get('total_token_throughput',0):.0f}")
except Exception as e:
    print("    -> WARM RESULT MISSING:", e)
PYEOF
}
