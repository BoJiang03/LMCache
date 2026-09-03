#!/usr/bin/env bash
# The TP=4 lane -- one arm per invocation:  ARM=<name> [TAG=<dir>] [PROBE=1] scripts/lane.sh
#
# WHY A NEW LANE.  Every phase1 arm ran TP=8 across all eight GPUs.  Another
# tenant took GPUs 4-7 (~120 GB each) after arm 1l finished, so TP=8 cannot be
# launched and will not be until they leave.  GPUs 0-3 are free.
#
# TP=4 changes the absolute numbers -- half the GPUs for the same FLOPs, a
# different pool, different collectives -- so NOTHING here may be compared with
# phase1's 85.3 / 91.0 / 97.5.  The lane carries its own baseline and every
# comparison is internal to it.  The lane is only trustworthy if it first
# reproduces the phenomenon: arm `mp` must be clearly slower per step than arm
# `none`.  scripts/lane_report.py checks that before it will rank anything.
#
# WHAT IS HELD FIXED ACROSS ARMS.  Same model, same TP, same ISL/OSL, same
# concurrency, same seed, same --max-num-batched-tokens 8192 (pinned, not
# defaulted, because ms/step = 1000*8192/tok-per-s is only meaningful if a step
# is a fixed number of tokens), same --disable-hybrid-kv-cache-manager, same
# --gpu-memory-utilization (pinned so a neighbour's allocation on GPU 0 cannot
# move the pool between arms), and the pool size is asserted equal to the
# lane's first arm.
#
# WHICH LMCacheMPConnector.  vLLM's factory maps the bare name
# "LMCacheMPConnector" to ITS OWN in-tree copy under
# vllm/distributed/kv_transfer/kv_connector/v1/, which is a DIFFERENT class
# from the one in lmcache/integration/vllm/ that the timed subclasses inherit
# from.  Comparing an arm built from one against an arm built from the other
# would silently compare two implementations, so every MP arm here passes
# kv_connector_module_path explicitly -- which is also exactly what VAST's
# reported config does, and what phase1's 1d/1e did.
#
# TWO LANES, ONE SCRIPT.  LANE_OUT and the geometry variables decide which:
#   TP=4 lane   LANE_OUT=results/lane, GPUS=0,1,2,3, GPUMEM=0.80, 300 prompts.
#               Self-contained; needs its own `none` baseline; ~9 min per arm.
#   TP=8 lane   LANE_OUT=results/phase1, GPUS=0..7, GPUMEM= (unset), 1000
#               prompts.  This is phase1's exact protocol, so new arms land in
#               the same table as 1a-1l and inherit their baselines: no
#               connector 85.3 ms/step, MP 91.0, IP 97.5.  Prefer it whenever
#               all eight GPUs are actually free.
#
# ARMS
#   none        no connector at all                         -- the baseline
#   null        NullConnector: every hook implemented, empty -- vLLM plumbing
#   mp          stock LMCacheMPConnector                     -- the phenomenon
#   timed       + 20 hook timers                             -- instrument check
#   nostore     timed, worker never submits the KV copy-out
#   nolookup    timed, scheduler never sends the LOOKUP
#   storeprobe  timed, plus sub-timers splitting the worker store submission
#   nowait      timed, LOOKUP sent but its discarded ack not waited on
#               (pair with EAGER=1: submit at on_new_request time)
#   ip          stock LMCacheConnectorV1 (in-process)        -- the second loss
#   timedip     + 19 hook timers on the IP connector
#   ipstoreprobe + timers on the engine calls IP's wait_for_save makes
#
# PROBE=1 additionally arms sitecustomize.py's CUDA-event probe on
# GPUModelRunner.execute_model, which reports GPU time per step next to wall
# and thread CPU.  It works in every arm including `none`, which is the point:
# a connector subclass cannot instrument a run that has no connector.  Use
# TAG to keep a probed arm in its own directory next to the unprobed one.
set -uo pipefail
source "$(dirname "$0")/lib.sh"

ARM="${ARM:?set ARM=none|null|mp|timed|nostore|nolookup|ip|timedip}"
TAG="${TAG:-$ARM}"
GPUS="${GPUS:-0,1,2,3}"
TP="${TP:-4}"
NPROMPTS="${NPROMPTS:-300}"
CONC="${CONC:-300}"
ISL="${ISL:-60000}"
# ${GPUMEM-0.80} not ${GPUMEM:-0.80}: an explicitly EMPTY value must stay
# empty so the TP=8 lane can omit the flag and land on phase1's pool.
GPUMEM="${GPUMEM-0.80}"
L1_GB="${L1_GB:-8}"
L1_EVICT="${L1_EVICT:-noop}"
# The lmcache server also opens an HTTP status port, default 8080.  Another
# tenant took 8080 overnight -- it was free during phase1 -- and the bind
# failure is FATAL: the server shuts itself down 15 s in and every MP arm
# aborts on "lmcache server died".  Pin it next to $PORT so this box's other
# LMCache servers cannot take the lane out again.
HTTP_PORT="${HTTP_PORT:-8766}"
# NUM_BLOCKS pins the KV pool with --num-gpu-blocks-override so it is
# IDENTICAL in every arm regardless of what else is on the device.  Without
# it the pool is derived from free memory at profiling time, and on a shared
# box that is not reproducible: the `mp` arm profiled 52.87 GiB of available
# KV against the baseline's 92.48 GiB on GPU 0 alone, giving 3,080,064
# tokens against 5,387,584, and the comparability assert -- correctly --
# threw the arm away.  A block is 36 layers x 2 x 64 x 2 x 64 x 2 B =
# 1,179,648 B per rank, so 30,000 blocks is 33 GiB, well inside the worst
# availability seen, and 1.92M tokens.  The workload never holds more than a
# couple of requests of KV at once (GPU KV cache usage ran 0.3-0.5% of a
# 13.7M-token pool in phase1, and OSL=1 means a request frees as soon as it
# has prefilled), so this is not a binding constraint on the measurement.
NUM_BLOCKS="${NUM_BLOCKS:-}"
PASSES="${PASSES:-cold}"
# APC=0 drops --enable-prefix-caching.  Needed only for the warm/finding-#2
# runs: vLLM's own paged pool here is 8 x 117.8 GiB = 942 GiB of KV, larger
# than any LMCache tier this box can host, so with APC on a warm pass is
# served entirely by vLLM and never reaches LMCache -- the retrieve path
# under test would not execute at all.
APC="${APC:-1}"
# IP arms take their yaml from here so a warm run can point at a config with
# a CPU tier big enough to hold the working set.
IP_YAML="${IP_YAML:-lmcache_gpu_only.yaml}"
OUT="${LANE_OUT:-$REPRO_ROOT/results/lane}"
dir=$OUT/$TAG
rm -rf "$dir"; mkdir -p "$dir" "$dir/timers"
export CUDA_VISIBLE_DEVICES="$GPUS"
export PYTHONPATH="$REPRO_ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LMC_TIMER_DIR="$dir/timers"
export LMC_TIMER_REPORT_EVERY="${LMC_TIMER_REPORT_EVERY:-200}"
if [ "${PROBE:-0}" = "1" ]; then
  export STEPPROBE=1
  export STEPPROBE_EVERY="${STEPPROBE_EVERY:-200}"
  echo "[lane] step probe armed (sitecustomize.py on GPUModelRunner.execute_model)"
fi
ulimit -n 1048576 2>/dev/null || ulimit -n 65535

# arm -> (connector class, module, needs lmcache mp server, lmcache yaml)
CLS=""; MOD=""; NEEDS_SERVER=0; YAML=""
case "$ARM" in
  none)     ;;
  null)     CLS=NullConnector;        MOD=nullconn.null_connector ;;
  mp)       CLS=LMCacheMPConnector;   MOD=lmcache.integration.vllm.lmcache_mp_connector; NEEDS_SERVER=1; YAML=mp.yaml ;;
  timed)    CLS=TimedMPConnector;     MOD=timedconn.timed_mp_connector;    NEEDS_SERVER=1; YAML=mp.yaml ;;
  nostore)  CLS=NoStoreMPConnector;   MOD=timedconn.nostore_mp_connector;  NEEDS_SERVER=1; YAML=mp.yaml ;;
  tinykey)  CLS=TinyKeyMPConnector;   MOD=timedconn.tinykey_mp_connector;  NEEDS_SERVER=1; YAML=mp.yaml ;;
  nolookup) CLS=NoLookupMPConnector;  MOD=timedconn.nolookup_mp_connector; NEEDS_SERVER=1; YAML=mp.yaml ;;
  nowait)   CLS=NoWaitMPConnector;    MOD=timedconn.nowait_mp_connector;   NEEDS_SERVER=1; YAML=mp.yaml ;;
  storeprobe) CLS=StoreProbeMPConnector; MOD=timedconn.storeprobe_mp_connector; NEEDS_SERVER=1; YAML=mp.yaml ;;
  ip)       CLS=LMCacheConnectorV1;   MOD="";                              YAML=$IP_YAML ;;
  timedip)  CLS=TimedIPConnector;     MOD=timedconn.timed_ip_connector;    YAML=$IP_YAML ;;
  ipstoreprobe) CLS=IPStoreProbeConnector; MOD=timedconn.ipstoreprobe_connector; YAML=$IP_YAML ;;
  *) echo "unknown ARM=$ARM"; exit 1 ;;
esac

avail=$(free -g | awk '/^Mem:/{print $7}')
need=$((L1_GB + 200))
if (( avail < need )); then echo "[$TAG] ABORT: ${avail}GB available, need ${need}GB."; exit 1; fi

# Free-VRAM gate.  The whole reason this lane exists is that a neighbour took
# four GPUs; launching into a GPU that is not actually free wastes 10 minutes
# and produces a differently-sized pool that then fails the pool assert anyway.
$PY - "$GPUS" <<'PY' || exit 1
import subprocess, sys
want = [int(x) for x in sys.argv[1].split(",")]
q = subprocess.run(["nvidia-smi", "--query-gpu=index,memory.free",
                    "--format=csv,noheader,nounits"], capture_output=True, text=True)
free = {int(a): int(b) for a, b in
        (l.split(", ") for l in q.stdout.strip().splitlines())}
bad = [(g, free.get(g, -1)) for g in want if free.get(g, 0) < 100000]
if bad:
    print(f"[lane] ABORT: these GPUs are not free (MiB): {bad}")
    print("       another tenant is using them; wait or pick different GPUs.")
    sys.exit(1)
print(f"[lane] GPUs {want} free: " + ", ".join(f"{g}={free[g]}MiB" for g in want))
PY

# Pre-flight the connector class in seconds rather than after a 3-minute load.
if [ -n "$CLS" ]; then
$PY - "$CLS" "$MOD" <<'PY' || exit 1
import sys, inspect
from vllm.config.kv_transfer import KVTransferConfig
from vllm.distributed.kv_transfer.kv_connector.factory import KVConnectorFactory
cls_name, mod = sys.argv[1], sys.argv[2]
cfg = KVTransferConfig(kv_connector=cls_name, kv_role="kv_both",
                       **({"kv_connector_module_path": mod} if mod else {}))
try:
    cls = KVConnectorFactory.get_connector_class(cfg)
except Exception as e:
    print(f"[lane] ABORT: factory cannot load {cls_name}: {e}"); sys.exit(1)
if inspect.isabstract(cls):
    print(f"[lane] ABORT: {cls_name} is abstract."); sys.exit(1)
print(f"[lane] {cls_name} -> {cls.__module__}.{cls.__qualname__}")
if mod.startswith("timedconn.timed_ip") or mod.startswith("timedconn.") and "ip" in mod:
    pass
if mod in ("timedconn.timed_mp_connector", "timedconn.nostore_mp_connector",
           "timedconn.nolookup_mp_connector", "timedconn.nowait_mp_connector",
           "timedconn.storeprobe_mp_connector",
           "timedconn.tinykey_mp_connector"):
    from lmcache.integration.vllm.lmcache_mp_connector import LMCacheMPConnector
    from timedconn.timed_mp_connector import TimedMPConnector, _wrapped
    from vllm.distributed.kv_transfer.kv_connector.v1 import supports_hma
    if LMCacheMPConnector not in cls.__mro__ or TimedMPConnector not in cls.__mro__:
        print("[lane] ABORT: mro is", cls.__mro__[:4]); sys.exit(1)
    # Same pool, same cudagraph mode, same attention layout as the real MP
    # connector, or the arm is not comparable to `mp`.
    for name, got, want in [
        ("supports_hma", supports_hma(cls), supports_hma(LMCacheMPConnector)),
        ("piecewise", cls.requires_piecewise_for_cudagraph({}),
                      LMCacheMPConnector.requires_piecewise_for_cudagraph({})),
        ("layout", cls.get_required_kvcache_layout({}),
                   LMCacheMPConnector.get_required_kvcache_layout({})),
    ]:
        if got != want:
            print(f"[lane] ABORT: {name} differs: {got} vs {want}"); sys.exit(1)
    # Descriptor-level diff: the subclass must differ from LMCacheMPConnector in
    # exactly the timed hooks and nothing else.  The first version of the timing
    # wrapper turned the @property transfer_intermediate_tensors into a method;
    # a bound method is always truthy, so every worker died at init asking the
    # server for a feature it does not advertise.  Catch that class of mistake
    # here, not 90 seconds into a model load.
    allowed = set(_wrapped) | {"__init__", "__doc__", "__dict__", "__module__",
                               "__abstractmethods__", "_abc_impl", "wait_for_save"}
    diff = []
    for name in set(dir(cls)) | set(dir(LMCacheMPConnector)):
        if name in allowed: continue
        a = inspect.getattr_static(cls, name, "<missing>")
        b = inspect.getattr_static(LMCacheMPConnector, name, "<missing>")
        if a is not b: diff.append(f"{name}: {type(a).__name__} vs {type(b).__name__}")
    if diff:
        print("[lane] ABORT: differs outside the timed hooks: " + "; ".join(diff))
        sys.exit(1)
    if not isinstance(inspect.getattr_static(cls, "transfer_intermediate_tensors"), property):
        print("[lane] ABORT: transfer_intermediate_tensors is no longer a property.")
        sys.exit(1)
    print("[lane] comparable to LMCacheMPConnector: same hma/piecewise/layout,"
          " no descriptor differs outside the timed hooks.")
if mod in ("timedconn.timed_ip_connector", "timedconn.ipstoreprobe_connector"):
    from vllm.distributed.kv_transfer.kv_connector.v1.lmcache_connector import (
        LMCacheConnectorV1)
    if LMCacheConnectorV1 not in cls.__mro__:
        print("[lane] ABORT: TimedIPConnector does not subclass the connector vLLM"
              " actually builds for the name LMCacheConnectorV1."); sys.exit(1)
    print("[lane] TimedIPConnector subclasses vLLM's LMCacheConnectorV1 shim.")
PY
fi

if [ "$NEEDS_SERVER" = "1" ]; then
  echo "=== [$TAG] lmcache server (l1=${L1_GB}GB) $(date +%H:%M:%S) ==="
  spawn "$dir/lmcache_server.log" "$VENV/bin/lmcache" server \
    --host 127.0.0.1 --port "$MP_PORT" --http-port "$HTTP_PORT" --l1-size-gb "$L1_GB" \
    --eviction-policy "$L1_EVICT" --eviction-trigger-watermark 0.8 --eviction-ratio 0.2 \
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
  (( up == 1 )) || { echo "  ABORT: lmcache server never listened on $MP_PORT"; tail -30 "$dir/lmcache_server.log"; teardown; exit 1; }
  echo "  lmcache server listening after ${t}s"
  # A fatal bind error can arrive AFTER the MQ port opens, so listening is
  # not proof the server lives.  This exact failure -- the HTTP port taken
  # by another tenant -- killed an arm 15 s in with the MQ socket already up.
  if grep -q "address already in use" "$dir/lmcache_server.log" 2>/dev/null; then
    echo "[$TAG] ABORT: the lmcache server hit a port conflict:"
    grep -m2 "address already in use" "$dir/lmcache_server.log"
    teardown; exit 1
  fi
fi
[ -n "$YAML" ] && export LMCACHE_CONFIG_FILE=$REPRO_ROOT/configs/$YAML

BASE=( serve "$MODEL" --host 127.0.0.1 --port "$PORT" --served-model-name "$SERVED_NAME"
       --tensor-parallel-size "$TP" --max-model-len 131072
       --block-size=64 --max-num-seqs 256
       --disable-hybrid-kv-cache-manager )
if [ "$APC" = "1" ]; then BASE+=( --enable-prefix-caching ); else
  BASE+=( --no-enable-prefix-caching )
  echo "[lane] vLLM prefix caching DISABLED: a warm hit must come from LMCache."
fi
# MNBT is deliberately EMPTY by default.  Passing --max-num-batched-tokens 8192
# explicitly -- even though 8192 is exactly what vLLM derives anyway and is what
# the log then reports -- changes how the KV cache specs are sized for this
# hybrid model: the sliding-window group's per-request budget is computed from
# scheduler_config.max_num_batched_tokens, and pinning it collapsed the pool
# from 13,724,416 tokens to 1,223,040 (max concurrency 104.71x -> 9.33x) with
# identical available memory of 117.8 GiB per GPU.  No phase1 arm passed it.
# The step size is asserted from the log below instead, which is what makes
# ms/step = 1000*8192/tok-per-s well defined.
[ -n "${MNBT:-}" ] && BASE+=( --max-num-batched-tokens "$MNBT" )
[ -n "$NUM_BLOCKS" ] && BASE+=( --num-gpu-blocks-override "$NUM_BLOCKS" )
# Empty GPUMEM means "do not pass the flag", which is what every phase1 arm did
# and is the only way to land on their pool of 13,724,416 tokens.  A pinned
# value is for the TP=4 lane, where a neighbour's allocation on GPU 0 would
# otherwise move the pool between arms.
[ -n "$GPUMEM" ] && BASE+=( --gpu-memory-utilization "$GPUMEM" )

if [ -n "$CLS" ]; then
  MODFIELD=""
  [ -n "$MOD" ] && MODFIELD="\"kv_connector_module_path\":\"$MOD\","
  if [ "$NEEDS_SERVER" = "1" ]; then
    EAGERFIELD=""
    [ "${EAGER:-0}" = "1" ] && EAGERFIELD=",\"lmcache.mp.eager_prefetch\":true"
    KVCFG="{\"kv_connector\":\"$CLS\",${MODFIELD}\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT,\"lmcache.mp.heartbeat_interval\":60.0${EAGERFIELD}}}"
  else
    KVCFG="{\"kv_connector\":\"$CLS\",${MODFIELD}\"kv_role\":\"kv_both\"}"
  fi
  BASE+=( --kv-transfer-config "$KVCFG" )
fi

echo "=== [$TAG] launching vllm TP=$TP on GPUs $GPUS $(date +%H:%M:%S) ==="
spawn "$dir/server.log" "$VLLM" "${BASE[@]}"
pid=$SPAWNED_PID
if ! wait_health "$dir/server.log" "$pid" 1500; then
  echo "[$TAG] FAILED TO START"; sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | tail -40; teardown; exit 1
fi
if grep -q "marked as init failed" "$dir/server.log" 2>/dev/null; then
  echo "[$TAG] ABORT: LMCache init failed -- would measure degraded mode"; teardown; exit 1
fi

clean() { sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log"; }

# Per-arm proof that the arm is the arm.  Without these an intervention that
# silently did not apply reads as "no effect", which is exactly the conclusion
# these runs are trying to draw -- the most expensive possible false negative.
if [ -n "$CLS" ]; then
  n=$(grep -c "Creating v1 connector with name: $CLS" "$dir/server.log" 2>/dev/null); n=${n:-0}
  echo "  connector instantiations: $n (expect $((TP+1)))"
  if [ "$n" -lt 2 ]; then echo "[$TAG] ABORT: factory did not build $CLS on both sides."; teardown; exit 1; fi
  if ! clean | grep -q "EngineCore.*Creating v1 connector with name: $CLS"; then
    echo "[$TAG] ABORT: no scheduler-side connector."; teardown; exit 1; fi
fi
case "$ARM" in
  nostore)
    clean | grep -m1 "NoStoreMPConnector attached .*role=WORKER" | tee "$dir/patch.txt"
    grep -q "no_store=True" "$dir/patch.txt" 2>/dev/null || {
      echo "[$TAG] ABORT: the workers did not get the no-store patch; this would"
      echo "     silently re-run 'timed' and answer nothing."; teardown; exit 1; } ;;
  nolookup)
    clean | grep -m1 "NoLookupMPConnector attached .*role=SCHEDULER" | tee "$dir/patch.txt"
    grep -q "no_lookup=True" "$dir/patch.txt" 2>/dev/null || {
      echo "[$TAG] ABORT: the scheduler did not get the no-lookup patch."; teardown; exit 1; } ;;
  nowait)
    clean | grep -m1 "NoWaitMPConnector attached .*role=SCHEDULER" | tee "$dir/patch.txt"
    grep -q "nowait=True" "$dir/patch.txt" 2>/dev/null || {
      echo "[$TAG] ABORT: the scheduler did not get the nowait patch."; teardown; exit 1; }
    if [ "${EAGER:-0}" = "1" ] && ! grep -q "eager_prefetch=True" "$dir/patch.txt"; then
      echo "[$TAG] ABORT: EAGER=1 was asked for but the connector read"
      echo "     eager_prefetch as False, so the LOOKUP is still submitted late"
      echo "     and this arm is just a slower nolookup."; teardown; exit 1
    fi ;;
esac
if [ "${PROBE:-0}" = "1" ]; then
  nprobe=$(clean | grep -c "STEPPROBE installed"); nprobe=${nprobe:-0}
  echo "  step probes installed: $nprobe (expect $TP, one per worker)"
  if [ "$nprobe" -lt "$TP" ]; then
    echo "[$TAG] ABORT: the probe did not reach every worker, so gpu_ms/step would"
    echo "     be an average over an unknown subset."; teardown; exit 1
  fi
fi

grep -iE "GPU KV cache size|Maximum concurrency" "$dir/server.log" | tail -4 | tee "$dir/pool.txt"
pool=$(grep -o "GPU KV cache size: [0-9,]*" "$dir/server.log" | tail -1 | tr -cd '0-9')
echo "  pool=${pool:-unknown}"
ref_file=$OUT/POOL_REF
if [ -s "$ref_file" ]; then
  ref=$(cat "$ref_file")
  if [ "${pool:-0}" -ne "$ref" ]; then
    echo "[$TAG] ABORT: pool is $pool, lane reference is $ref -- not comparable."
    teardown; exit 1
  fi
  echo "  pool matches the lane reference ($ref)."
else
  echo "${pool:-0}" > "$ref_file"; echo "  pool reference for this lane set to ${pool}."
fi
mnbt=$(clean | grep -o "max_num_batched_tokens=[0-9]*" | head -1)
echo "  $mnbt"
if [ "$mnbt" != "max_num_batched_tokens=8192" ]; then
  echo "[$TAG] ABORT: a step is not 8192 tokens, so ms/step is not defined."; teardown; exit 1
fi

# PASSES defaults to "cold": every phase1 arm measured the miss path only, and
# ms/step is quoted from it.  "cold warm" adds a second pass over the SAME
# prompts (same seed), which is the regime VAST's second finding lives in --
# there MP warm runs at roughly half IP warm.  The warm pass is only meaningful
# if LMCache can actually hold the working set, so pair it with a large L1_GB
# and L1_EVICT=LRU; with the default 8 GB the cache is full after ~113 chunks
# and the warm pass is a second cold pass wearing a different name.
for p in $PASSES; do
  echo "  [$TAG] $p pass, $NPROMPTS prompts x $ISL tokens, c=$CONC  ($(date +%H:%M:%S))"
  $VLLM bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --served-model-name "$SERVED_NAME" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len 1 \
    --random-range-ratio 0.0 --ignore-eos --seed 42 \
    --num-prompts "$NPROMPTS" --max-concurrency "$CONC" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99 \
    --save-result --result-dir "$dir" --result-filename "$p.json" \
    > "$dir/$p.log" 2>&1
  $PY - "$dir/$p.json" "$TAG" "$p" <<'PYEOF'
import json, sys
try:
    d = json.load(open(sys.argv[1]))
    print(f"    -> {sys.argv[2]} {sys.argv[3]}: dur={d.get('duration',0):.1f}s "
          f"end-to-end tok/s={d.get('total_token_throughput',0):.0f} "
          f"mean_ttft={d.get('mean_ttft_ms',0)/1000:.1f}s")
    print("       AUTHORITATIVE: scripts/lane_report.py")
except Exception as e:
    print("    -> RESULT MISSING:", e)
PYEOF
done

case "$ARM" in
  nostore)  clean | grep -o "NOSTORE .*skipped_requests=[0-9]*" | tail -3 | tee "$dir/skipped.txt" ;;
  nolookup) clean | grep -o "NOLOOKUP .*lookups_skipped=[0-9]*" | tail -3 | tee "$dir/skipped.txt" ;;
esac
[ "${PROBE:-0}" = "1" ] && clean | grep -o "STEPPROBE pid=.*" | tail -8 | tee "$dir/stepprobe.txt"
clean | grep -o "Deferred: [0-9]*" | sort | uniq -c | sort -rn | head -3 | tee "$dir/deferred.txt"
teardown
echo "=== [$TAG] done $(date +%H:%M:%S) -> $dir ==="
