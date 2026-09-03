#!/usr/bin/env bash
# 1h -- py-spy profile of the MP arm, to localise the ~9% tax common to IP and MP.
#
# WHY MP AND NOT IP.  MP is the clean arm for the common tax:
#   - Deferred is 0 for every stat line in 1d and 1e, so the async-lookup client
#     is not involved at all and cannot contaminate the profile.
#   - MP shows the same ~9% at BOTH pool sizes (1d/1a = 1.090x at 25.8M,
#     1e/1c = 1.088x at 13.7M), so it is a property of attaching a connector,
#     not of the pool.
#
# WHAT WE ARE LOOKING FOR.  All arms run max_num_batched_tokens=8192, so a step
# is a fixed 8192 tokens and the in-engine "Avg prompt throughput" converts
# directly to per-step time:
#     no connector 1c : 96,000 tok/s -> 85.4 ms/step
#     MP 1e           : 90,000 tok/s -> 91.0 ms/step   -> +5.6 ms/step
# The target is 5.6 ms per forward step.  Over a 686 s cold pass that is ~6% of
# every process's wall clock, which 30 Hz sampling resolves comfortably.
#
# CONFIGURATION IS IDENTICAL TO 1e in every respect (same L1_GB=8, same
# --disable-hybrid-kv-cache-manager, same configs/mp.yaml, same ISL, same
# c=1000) except that the vLLM process tree is launched under py-spy.
#
# WHY py-spy CAN ATTACH.  kernel.yama.ptrace_scope is 1 on this box, so a
# process may only trace its own descendants.  "py-spy record -- vllm serve"
# makes py-spy the parent, and --subprocesses follows the EngineCore and the 8
# TP workers because they are its descendants.  Verified before writing this:
# spawned children's frames are captured, 0 errors, and py-spy flushes its
# output file when the tree is torn down.  No sudo, no sysctl change.
#
# --idle is ON deliberately.  The 5.6 ms could be CPU burn OR a block (a CUDA
# sync, a ZMQ wait, a lock).  Without --idle a block is invisible, and a
# one-run budget cannot afford to miss half the hypothesis space.  Frame names
# separate the two afterwards.
#
# PRE-REGISTERED READING OF THE RESULT:
#   - If frames under lmcache.* / vllm.distributed.kv_transfer.* /
#     kv_connector_model_runner_mixin / maybe_transfer_kv_layer account for
#     roughly 5-7% of EngineCore or worker wall clock, the tax is Python-level
#     and named, and the next step is to read that specific call.
#   - If those frames are ~0%, the cost is native (GPU, memcpy, allocator) and
#     the follow-up is the same run with --native, or a no-connector baseline
#     profile to diff against.
# Either outcome is informative; there is no way for this run to say nothing.
#
# SANITY GATE: the profile is only meaningful if this run actually reproduces
# the tax.  The script prints the in-engine rate histogram at the end; the mode
# must be ~90,000 tok/s (15 units of 6000).  If py-spy's sampling has depressed
# it below that, the profile describes a distorted run and must be discarded.
set -uo pipefail
source "$(dirname "$0")/lib.sh"
OUT=$REPRO_ROOT/results/phase1; mkdir -p "$OUT"
CONC="${CONC:-1000}"
PASSES="${PASSES:-cold}"
L1_GB="${L1_GB:-8}"
RATE="${RATE:-30}"
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

PYSPY=$VENV/bin/py-spy
[ -x "$PYSPY" ] || { echo "[1h] ABORT: no py-spy at $PYSPY"; exit 1; }

avail=$(free -g | awk '/^Mem:/{print $7}')
need=$((L1_GB + 250))
if (( avail < need )); then echo "[1h] ABORT: ${avail}GB available, need ${need}GB."; exit 1; fi
echo "memory check ok: ${avail}GB available, need ${need}GB"

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size 8 --max-model-len 131072 --enable-prefix-caching
       --block-size=64 --max-num-seqs 256 )
MP_KVCFG="{\"kv_connector\":\"LMCacheMPConnector\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT,\"lmcache.mp.heartbeat_interval\":60.0}}"

dir=$OUT/1h_mp_profile; mkdir -p "$dir"
PROF=$dir/mp_c1000.speedscope.json

echo "=== lmcache server (l1=${L1_GB}GB) $(date +%H:%M:%S) ==="
spawn "$dir/lmcache_server.log" "$VENV/bin/lmcache" server \
  --host 127.0.0.1 --port "$MP_PORT" --l1-size-gb "$L1_GB" \
  --eviction-policy noop --eviction-trigger-watermark 0.8 --eviction-ratio 0.2 \
  --chunk-size 8192 --l2-prefetch-max-in-flight 4 \
  --max-gpu-workers 8 --max-cpu-workers 8 \
  --worker-reap-timeout-seconds 180 --l1-align-bytes 1048576
mp_pid=$SPAWNED_PID
t=0; up=0
while (( t < 300 )); do
  if ss -ltn 2>/dev/null | grep -q "127.0.0.1:$MP_PORT"; then up=1; break; fi
  kill -0 "$mp_pid" 2>/dev/null || { echo "  lmcache server died after ${t}s:"; tail -30 "$dir/lmcache_server.log"; teardown; exit 1; }
  sleep 5; t=$((t+5))
done
(( up == 1 )) || { echo "[1h] ABORT: lmcache server never listened on $MP_PORT"; tail -30 "$dir/lmcache_server.log"; teardown; exit 1; }
echo "  lmcache server listening on $MP_PORT after ${t}s"

export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/mp.yaml
echo "=== [1h] launching vllm UNDER py-spy (rate=${RATE}Hz) $(date +%H:%M:%S) ==="
# -d is a ceiling only; the real stop is the SIGINT after the bench.  It is set
# well past the expected run so py-spy never stops sampling early and silently
# profiles half the pass.
spawn "$dir/pyspy.log" "$PYSPY" record --subprocesses --idle -r "$RATE" -d 5400 \
      -f speedscope -o "$PROF" -- "$VLLM" "${BASE[@]}" \
      --disable-hybrid-kv-cache-manager --kv-transfer-config "$MP_KVCFG"
pyspy_pid=$SPAWNED_PID

# py-spy forwards nothing to our log, so health has to be read from the port.
i=0; ok=0
while [ $i -lt 1500 ]; do
  curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  up after ${i}s"; ok=1; break; }
  kill -0 "$pyspy_pid" 2>/dev/null || { echo "  py-spy/vllm exited early"; tail -30 "$dir/pyspy.log"; break; }
  sleep 5; i=$((i+5))
done
[ $ok -eq 1 ] || { echo "[1h] FAILED TO START"; teardown; exit 1; }

# vLLM's own log went to py-spy's stdout capture; split out what we need.
grep -iE "GPU KV cache size|Maximum concurrency" "$dir/pyspy.log" | tail -4 | tee "$dir/pool.txt"
pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/pyspy.log" | tail -1 | tr -cd '0-9')
echo "  pool=${pool:-unknown} (expect 13724416, same as 1e)"
if [ "${pool:-0}" -ne 13724416 ]; then
  echo "[1h] ABORT: pool is ${pool}, not 1e's 13,724,416; the profile would not match any baseline."
  teardown; exit 1
fi

for c in $CONC; do
  for p in $PASSES; do
    echo "  [1h c=$c] $p pass  ($(date +%H:%M:%S))"
    $VLLM bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" \
      --model "$MODEL" --served-model-name "$SERVED_NAME" --tokenizer "$MODEL" \
      --dataset-name random --random-input-len 60000 --random-output-len 1 \
      --random-range-ratio 0.0 --ignore-eos --seed 42 \
      --num-prompts "$c" --max-concurrency "$c" \
      --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99 \
      --save-result --result-dir "$dir" --result-filename "c${c}_${p}.json" \
      > "$dir/c${c}_${p}.log" 2>&1
    $PY - "$dir/c${c}_${p}.json" "$p" <<'PYEOF'
import json,sys
try:
    d=json.load(open(sys.argv[1]))
    print(f"    -> {sys.argv[2]} dur={d.get('duration',0):.1f}s "
          f"tok/s={d.get('total_token_throughput',0):.0f} "
          f"p99_ttft={d.get('p99_ttft_ms',0)/1000:.1f}s   (1e cold was 686.0s / 87,462)")
except Exception as e:
    print(f"    -> {sys.argv[2]} RESULT MISSING:", e)
PYEOF
  done
done

# Stop py-spy FIRST and give it time to serialise, then tear the tree down.
echo "=== stopping py-spy $(date +%H:%M:%S) ==="
kill -INT "$pyspy_pid" 2>/dev/null
for i in $(seq 1 60); do
  [ -s "$PROF" ] && { echo "  profile written after ${i}s: $(du -h "$PROF" | cut -f1)"; break; }
  sleep 5
done
[ -s "$PROF" ] || echo "  WARNING: $PROF still empty; teardown may still flush it"
teardown
[ -s "$PROF" ] && echo "  profile: $PROF ($(du -h "$PROF" | cut -f1))" || echo "  PROFILE LOST"

# Sanity gate: did this run actually reproduce the tax we set out to profile?
$PY - "$dir/pyspy.log" <<'PYEOF'
import re,sys
from collections import Counter
pat=re.compile(r'Avg prompt throughput: ([\d.]+) tokens/s.*Running: (\d+) reqs, Waiting: (\d+) reqs(?:, Deferred: (\d+) reqs)?')
rows=[]
for l in open(sys.argv[1],errors='ignore'):
    m=pat.search(l)
    if m: rows.append((float(m.group(1)),int(m.group(2)),int(m.group(3)),int(m.group(4) or 0)))
busy=[r for r in rows if r[2]+r[3]>200]
if not busy:
    print("  SANITY GATE: no busy stat lines found -- cannot verify the tax reproduced.")
else:
    rates=sorted(r[0] for r in busy); n=len(rates)
    h=dict(sorted(Counter(round(r[0]/6000) for r in busy).items()))
    print(f"  SANITY GATE: busy={n} mean={sum(rates)/n:,.0f} p50={rates[n//2]:,.0f} hist(units of 6000)={h}")
    print(f"     expect p50 ~90,000 (mode 15).  1c no-connector was 96,000 (16); IP 1b was 84,000 (14).")
    print(f"     Deferred max={max(r[3] for r in busy)} (expect 0 for MP)")
PYEOF
echo "=== 1h done $(date +%H:%M:%S) -> $dir ==="
