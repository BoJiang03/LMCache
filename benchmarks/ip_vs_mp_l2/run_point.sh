#!/usr/bin/env bash
# SPDX-License-Identifier: Apache-2.0
#
# One arm of the IP-vs-MP L2 benchmark, over one or more concurrency points.
# Normally invoked by run_ladder.sh; runnable on its own:
#
#   ARM=ip_l2 CONC=100 OUT=./out ./run_point.sh
#
# WHY TWO ROUNDS.  The thing being measured is the L2 -> GPU load path, so the
# measured round must not be servable from vLLM's own GPU prefix cache.  With a
# single round it is: every prompt of a 120k x c corpus fits inside the GPU KV
# pool at the lower concurrencies, so after the first pass the KV is still
# resident and the connector is never asked for anything.  That failure is
# silent -- it looks like a fast warm pass.  So, per point:
#
#   1. POST /reset_prefix_cache            -- round 1 is a true full prefill
#   2. round 1, result DISCARDED           -- this is what populates L2
#   3. drain: wait for the L2 directory to stop growing, so the async store
#      path has finished.  Skipping this lets part of the corpus miss L2, and
#      because the two arms have different store pipelines the contamination
#      would be arm-dependent, i.e. it would bias the ratio.
#   4. POST /reset_prefix_cache again      -- clears vLLM's block-pool hash map
#      and every block hash, and leaves the connector untouched.  Both query
#      params default to false; reset_external MUST stay off or LMCache is
#      wiped too.
#   5. round 2, MEASURED                   -- same corpus, served from L2.
#
# TWO ACCEPTANCE GATES, both mandatory, both symmetric across arms:
#
#   G1  the reset must actually have happened.  The endpoint returns HTTP 200
#       unconditionally, and block_pool.reset_prefix_cache() refuses (with only
#       a logger.warning) while any block is still referenced.  So the real
#       signal is the server log: "Successfully reset prefix cache" vs "Failed
#       to reset prefix cache because some blocks (N) are not freed yet".  A
#       silently-failed reset is indistinguishable from a good run in the
#       numbers, which is why this is checked and retried rather than assumed.
#   G2  round 2 must have been served by the connector.  Measured with vLLM's
#       own Prometheus counters vllm:external_prefix_cache_queries / _hits,
#       sampled either side of round 2 and differenced.  These are emitted for
#       both connectors.  A point with zero hits is labelled and kept, not
#       dropped -- a zero-hit arm is a finding, not a run error.
set -uo pipefail
HERE=$(cd "$(dirname "$0")" && pwd)
source "$HERE/env.sh"
source "$HERE/lib.sh"

ARM="${ARM:-ip_l2}"
CONC="${CONC:-100}"
OUT="${OUT:-$PWD/out}"
DRAIN_MAX_S="${DRAIN_MAX_S:-1200}"
DRAIN_STABLE="${DRAIN_STABLE:-3}"       # consecutive equal du samples
DRAIN_INTERVAL="${DRAIN_INTERVAL:-20}"
GPU_WAIT_MIN="${GPU_WAIT_MIN:-0}"       # >0 to wait for a busy shared box
GPU_MIN_FREE_MIB="${GPU_MIN_FREE_MIB:-100000}"
mkdir -p "$OUT"

[ -x "${VLLM:-}" ] || { echo "ABORT: vllm not found; set VLLM=/path/to/vllm"; exit 1; }
[ -x "${PY:-}" ]   || { echo "ABORT: python not found; set PY=/path/to/python"; exit 1; }

echo "[point] arm=$ARM conc='$CONC' tp=$TP l1=${L1_GB}GB out=$OUT"
"$PY" -c 'import lmcache,sys; print("[point] lmcache:", lmcache.__file__)' || exit 1

# ---- preflight --------------------------------------------------------------

avail=$(free -g | awk '/^Mem:/{print $7}')
need_mem=$((L1_GB + 250))
if (( avail < need_mem )); then
  echo "ABORT: ${avail}GB RAM available, need ${need_mem}GB for L1_GB=${L1_GB}."
  echo "       Both arms are given ${L1_GB}GB of DRAM; lower L1_GB to fit, but"
  echo "       lower it for BOTH arms or the comparison is void (see README)."
  exit 1
fi
echo "[point] memory ok: ${avail}GB available, need ${need_mem}GB"

if (( GPU_WAIT_MIN > 0 )); then
  deadline=$(( $(date +%s) + GPU_WAIT_MIN * 60 ))
  while :; do
    busy=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | awk -F', ' -v f="$GPU_MIN_FREE_MIB" '$2 < f {printf "%s(%sMiB) ", $1, $2}')
    [ -z "$busy" ] && { echo "[point] GPUs free at $(date +%H:%M:%S)"; break; }
    [ "$(date +%s)" -ge "$deadline" ] && { echo "[point] gave up waiting; busy: $busy"; exit 1; }
    echo "[point] $(date +%H:%M:%S) waiting, busy: $busy"; sleep 60
  done
fi

# ---- protocol helpers -------------------------------------------------------

disk_rw_bytes() {  # -> "read_bytes written_bytes" for the L2 device
  [ -n "$L2_DEV" ] && [ -r "/sys/block/$L2_DEV/stat" ] || { echo "0 0"; return; }
  awk '{printf "%.0f %.0f", $3*512, $7*512}' "/sys/block/$L2_DEV/stat"
}

ext_counters() {  # -> "queries hits"; vLLM's connector-side counters
  curl -sS --max-time 20 "http://127.0.0.1:$PORT/metrics" 2>/dev/null \
    | awk '/_created/{next}
           /^vllm:external_prefix_cache_queries/{q+=$NF}
           /^vllm:external_prefix_cache_hits/{h+=$NF}
           END{printf "%.0f %.0f", q+0, h+0}'
}

# G1.  Re-POST with backoff and only count log lines that appear after this
# call, since earlier points leave their own lines behind.
reset_prefix_cache() {  # reset_prefix_cache <logfile> <label>; 0 = verified
  local log="$1" label="$2" ok0 bad0 ok1 bad1 attempt t
  ok0=$(grep -c "Successfully reset prefix cache" "$log" 2>/dev/null); ok0=${ok0:-0}
  bad0=$(grep -c "Failed to reset prefix cache because some blocks" "$log" 2>/dev/null); bad0=${bad0:-0}
  for attempt in 1 2 3 4 5 6; do
    curl -sS --max-time 60 -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" >/dev/null 2>&1
    t=0
    while (( t < 20 )); do
      ok1=$(grep -c "Successfully reset prefix cache" "$log" 2>/dev/null); ok1=${ok1:-0}
      (( ok1 > ok0 )) && { echo "    [$label] prefix cache reset verified (attempt $attempt)"; return 0; }
      bad1=$(grep -c "Failed to reset prefix cache because some blocks" "$log" 2>/dev/null); bad1=${bad1:-0}
      (( bad1 > bad0 )) && break        # refused this time; back off and retry
      sleep 2; t=$((t+2))
    done
    bad0=${bad1:-$bad0}
    echo "    [$label] reset refused or silent (attempt $attempt); waiting 15s"
    sleep 15
  done
  echo "    [$label] RESET NOT VERIFIED after 6 attempts --"
  sed 's/\x1b\[[0-9;]*m//g' "$log" | grep -E "reset prefix cache" | tail -3
  return 1
}

drain_l2() {  # drain_l2 <label>
  local label="$1" prev=-1 stable=0 t=0 now
  while (( t < DRAIN_MAX_S )); do
    now=$(du -sb "$L2_DIR" 2>/dev/null | awk '{print $1}'); now=${now:-0}
    if [ "$now" = "$prev" ]; then
      stable=$((stable+1))
      (( stable >= DRAIN_STABLE )) && {
        echo "    [$label] L2 drained after ${t}s ($(numfmt --to=iec "$now"))"; return 0; }
    else
      stable=0
    fi
    prev=$now; sleep "$DRAIN_INTERVAL"; t=$((t+DRAIN_INTERVAL))
  done
  echo "    [$label] WARNING: L2 still growing after ${t}s; proceeding"
  return 0
}

one_round() {  # one_round <dir> <c> <name>
  local dir="$1" c="$2" name="$3"
  "$VLLM" bench serve --backend vllm --base-url "http://127.0.0.1:$PORT" \
    --model "$MODEL" --served-model-name "$SERVED_NAME" --tokenizer "$MODEL" \
    --dataset-name random --random-input-len "$ISL" --random-output-len 1 \
    --random-range-ratio 0.0 --ignore-eos --seed 42 \
    --num-prompts "$c" --max-concurrency "$c" \
    --percentile-metrics ttft,tpot,itl,e2el --metric-percentiles 99 \
    --save-result --result-dir "$dir" --result-filename "c${c}_${name}.json" \
    > "$dir/c${c}_${name}.log" 2>&1
}

point() {  # point <dir> <c>
  local dir="$1" c="$2" log="$dir/server.log"
  echo "  [$ARM c=$c] === point start $(date +%H:%M:%S) ==="

  local need=$(( c * 9 + 300 )) free_gb
  free_gb=$(df -BG --output=avail "$L2_DIR" | tail -1 | tr -dc '0-9')
  if (( free_gb < need )); then
    echo "  [$ARM c=$c] SKIPPED: ${free_gb}GB free, this point needs ~${need}GB" \
      | tee -a "$dir/FAILED"
    return 1
  fi
  echo "    disk: ${free_gb}GB free, this point needs ~${need}GB"

  reset_prefix_cache "$log" "$ARM c=$c pre" || {
    echo "  [$ARM c=$c] FAILED (pre-reset)" | tee -a "$dir/FAILED"; return 1; }

  echo "  [$ARM c=$c] round 1 / prefill (discarded) $(date +%H:%M:%S)"
  one_round "$dir" "$c" prefill

  drain_l2 "$ARM c=$c"

  reset_prefix_cache "$log" "$ARM c=$c mid" || {
    echo "  [$ARM c=$c] FAILED (mid-reset) -- round 2 would be prefix-cache served" \
      | tee -a "$dir/FAILED"; return 1; }

  local q0 h0 dr0 dw0 q1 h1 dr1 dw1 t0 t1
  read -r q0 h0 <<<"$(ext_counters)"
  read -r dr0 dw0 <<<"$(disk_rw_bytes)"
  t0=$(date +%s)
  echo "  [$ARM c=$c] round 2 / warm (MEASURED) $(date +%H:%M:%S)"
  one_round "$dir" "$c" warm
  t1=$(date +%s)
  read -r q1 h1 <<<"$(ext_counters)"
  read -r dr1 dw1 <<<"$(disk_rw_bytes)"

  # Bytes and rate over the measured round.  If both arms sit at the device
  # ceiling the ratio is masked by the disk and the point says nothing about
  # the connectors, so this number is not optional.
  local secs=$(( t1 - t0 )); (( secs > 0 )) || secs=1
  local rd=$(( ${dr1:-0} - ${dr0:-0} )) wr=$(( ${dw1:-0} - ${dw0:-0} ))
  "$PY" - "$rd" "$wr" "$secs" "$ARM" "$c" "$dir/disk.txt" <<'PY'
import sys
rd, wr, secs, arm, c, out = (int(sys.argv[1]), int(sys.argv[2]), int(sys.argv[3]),
                             sys.argv[4], sys.argv[5], sys.argv[6])
line = (f"c={c} round2 {secs}s  read={rd/2**30:.1f}GiB ({rd/secs/1e9:.2f} GB/s)"
        f"  write={wr/2**30:.1f}GiB ({wr/secs/1e9:.2f} GB/s)")
print("    " + line)
open(out, "a").write(f"{arm} {line}\n")
PY

  local dq=$(( ${q1:-0} - ${q0:-0} )) dh=$(( ${h1:-0} - ${h0:-0} ))
  echo "    external prefix cache during round 2: queries=+$dq hits=+$dh"
  echo "c=$c queries=$dq hits=$dh" >> "$dir/external_counters.txt"

  "$PY" - "$dir/c${c}_warm.json" "$ARM" "$c" "$dh" "$dq" <<'PY'
import json, sys
path, arm, c, dh, dq = sys.argv[1:6]
try:
    d = json.load(open(path))
    print(f"    -> {arm} c={c} WARM total tok/s={d.get('total_token_throughput',0):.0f} "
          f"mean_ttft={d.get('mean_ttft_ms',0)/1000:.1f}s "
          f"dur={d.get('duration',0):.1f}s  ext_hits=+{dh}/{dq}")
except Exception as e:
    print(f"    -> {arm} c={c} WARM MISSING ({e})")
PY

  # G2 is reported, not enforced: a zero-hit arm is the interesting case, so it
  # must still produce a number.  Points are labelled instead of dropped.
  if (( dh <= 0 )); then
    echo "  [$ARM c=$c] NOTE: G2 zero-hit -- round 2 was a full re-prefill." \
         "If this is the IP arm at TP>1, check that your LMCache has the" \
         "cross-process hash-seed fix (see README)." | tee -a "$dir/ZERO_HIT"
  fi
  du -sh "$L2_DIR" 2>/dev/null | sed 's/^/    L2 on disk: /'
  return 0
}

# ---- the arms ---------------------------------------------------------------
#
# ip_l2       LMCacheConnectorV1, in-process, L2 = the `fs` remote connector.
# mp_l2_fs    LMCacheMPConnector, separate server, L2 = the `fs` L2 adapter.
#             This is the fair counterpart to ip_l2: same L2 mechanism, same
#             DRAM, hybrid KV cache manager disabled on both.
# mp_ceiling  LMCacheMPConnector with nothing held back for comparability:
#             the C++ `fs_native` L2 adapter, the hybrid KV cache manager left
#             ON, and --separate-object-groups.  Never read this arm as an
#             MP-vs-IP number; it answers "how fast is MP unconstrained".

dir="$OUT/$ARM"
rm -rf "$dir"; mkdir -p "$dir"

echo "=== [$ARM] wiping $L2_DIR $(date +%H:%M:%S) ==="
rm -rf "${L2_DIR:?}/ip" "${L2_DIR:?}/mp"
mkdir -p "$L2_DIR/ip" "$L2_DIR/mp"
df -h "$L2_DIR" | tail -1

base=( serve "$MODEL" --host 127.0.0.1 --port "$PORT"
  --served-model-name "$SERVED_NAME" --tensor-parallel-size "$TP"
  --max-model-len 131072 --enable-prefix-caching --block-size=64
  --max-num-seqs 256 )
cls=""; needs_server=0; yaml=""; l2_adapter=""; server_extra=()

case "$ARM" in
  ip_l2)
    cls=LMCacheConnectorV1; yaml=ip_l2.yaml
    # vLLM disables the hybrid KV cache manager by itself for a connector that
    # does not subclass SupportsHMA, which LMCacheConnectorV1 does not.  Passing
    # the flag explicitly makes the two command lines differ only in
    # --kv-transfer-config, and makes the asymmetry visible rather than implicit.
    base+=( --disable-hybrid-kv-cache-manager ) ;;
  mp_l2_fs)
    cls=LMCacheMPConnector; yaml=mp.yaml; needs_server=1
    l2_adapter="{\"type\":\"fs\",\"base_path\":\"$L2_DIR/mp\",\"use_odirect\":true,\"read_ahead_size\":8192}"
    base+=( --disable-hybrid-kv-cache-manager ) ;;
  mp_ceiling)
    cls=LMCacheMPConnector; yaml=mp.yaml; needs_server=1
    l2_adapter="{\"type\":\"fs_native\",\"base_path\":\"$L2_DIR/mp\",\"num_workers\":4,\"use_odirect\":true,\"max_capacity_gb\":0,\"read_ahead_size\":8192}"
    server_extra=( --separate-object-groups ) ;;
  *) echo "unknown arm '$ARM' (ip_l2 | mp_l2_fs | mp_ceiling)"; exit 1 ;;
esac

if [ "$needs_server" = "1" ]; then
  echo "=== [$ARM] lmcache server (l1=${L1_GB}GB, L2=$l2_adapter, skip_l1 ${server_extra[*]-}) $(date +%H:%M:%S) ==="
  spawn "$dir/lmcache_server.log" lmcache server \
    --host 127.0.0.1 --port "$MP_PORT" --http-port "$HTTP_PORT" \
    --l1-size-gb "$L1_GB" --eviction-policy noop \
    --eviction-trigger-watermark 0.8 --eviction-ratio 0.2 \
    --chunk-size 8192 --l2-prefetch-max-in-flight 4 \
    --max-gpu-workers "$TP" --max-cpu-workers "$TP" \
    --worker-reap-timeout-seconds 180 \
    --l2-adapter "$l2_adapter" \
    --l1-align-bytes 1048576 \
    --l2-store-policy skip_l1 "${server_extra[@]}"
  mp_pid=$SPAWNED_PID; t=0; up=0
  while (( t < 600 )); do
    [[ "$(ss -ltn 2>/dev/null || true)" == *"127.0.0.1:$MP_PORT"* ]] && { up=1; break; }
    kill -0 "$mp_pid" 2>/dev/null || { echo "  server died after ${t}s"; tail -30 "$dir/lmcache_server.log"; exit 1; }
    sleep 5; t=$((t+5))
  done
  (( up == 1 )) || { echo "  ABORT: server never listened"; exit 1; }
  echo "  lmcache server listening after ${t}s"
  grep -q "address already in use" "$dir/lmcache_server.log" 2>/dev/null && {
    echo "[$ARM] ABORT: port conflict on $MP_PORT"; exit 1; }
  # Without this the run would silently have no L2 at all.
  grep -qiE "fs_native|L2 ?adapter" "$dir/lmcache_server.log" || {
    echo "[$ARM] ABORT: no L2 adapter line in the server log"; exit 1; }
fi

# Render the engine yaml: LMCache's config file has no env-var interpolation,
# so the paths and ports are substituted here to keep one source of truth.
cpu_gb=$("$PY" -c "print(f'{$L1_GB / $TP:.1f}')")
sed -e "s|@L2_DIR@|$L2_DIR|g" -e "s|@MP_PORT@|$MP_PORT|g" -e "s|@CPU_GB@|$cpu_gb|g" \
  "$HERE/configs/$yaml.in" > "$dir/$yaml"
export LMCACHE_CONFIG_FILE="$dir/$yaml"
echo "  engine yaml: $dir/$yaml (max_local_cpu_size ${cpu_gb} x TP$TP = ${L1_GB}GB)"
if [ "$needs_server" = "1" ]; then
  kvcfg="{\"kv_connector\":\"$cls\",\"kv_connector_module_path\":\"lmcache.integration.vllm.lmcache_mp_connector\",\"kv_role\":\"kv_both\",\"kv_connector_extra_config\":{\"lmcache.mp.host\":\"tcp://127.0.0.1\",\"lmcache.mp.port\":$MP_PORT,\"lmcache.mp.heartbeat_interval\":60.0}}"
else
  kvcfg="{\"kv_connector\":\"$cls\",\"kv_role\":\"kv_both\"}"
fi
base+=( --kv-transfer-config "$kvcfg" )

echo "=== [$ARM] launching vllm TP=$TP $(date +%H:%M:%S) ==="
printf '%q ' "$VLLM" "${base[@]}" > "$dir/cmdline.txt"; echo >> "$dir/cmdline.txt"
spawn "$dir/server.log" "$VLLM" "${base[@]}"
pid=$SPAWNED_PID
wait_health "$dir/server.log" "$pid" 1800 || {
  echo "[$ARM] FAILED TO START"; sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" | tail -40; exit 1; }
grep -q "marked as init failed" "$dir/server.log" 2>/dev/null && {
  echo "[$ARM] ABORT: LMCache init failed -- would measure degraded mode"; exit 1; }

n=$(grep -c "Creating v1 connector with name: $cls" "$dir/server.log" 2>/dev/null); n=${n:-0}
echo "  connector instantiations: $n (expect $((TP+1)): scheduler + $TP workers)"
(( n >= 2 )) || { echo "[$ARM] ABORT: factory did not build $cls on both sides"; exit 1; }

if ! curl -sS --max-time 30 -o /dev/null -w '%{http_code}' \
      -X POST "http://127.0.0.1:$PORT/reset_prefix_cache" | grep -q '^200$'; then
  echo "[$ARM] ABORT: /reset_prefix_cache not mounted (VLLM_SERVER_DEV_MODE?)"; exit 1
fi
echo "  /reset_prefix_cache reachable"

sed 's/\x1b\[[0-9;]*m//g' "$dir/server.log" \
  | grep -iE "GPU KV cache size|Maximum concurrency|hybrid kv cache manager" \
  | tail -6 | tee "$dir/pool.txt"

for c in $CONC; do
  point "$dir" "$c" || echo "  [$ARM c=$c] point failed; continuing"
done
echo "=== [$ARM] done $(date +%H:%M:%S) ==="
