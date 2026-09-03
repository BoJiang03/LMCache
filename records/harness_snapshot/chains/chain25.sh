#!/usr/bin/env bash
# chain25 -- verify the delta-key fix.
#
# The fix: STORE stops shipping the request's growing prompt prefix.  The op
# and the IPC key now carry only [start, end) with token_offset=start; the
# server's Session splices the delta in and continues the rolling chunk hash
# it already cached.  Touched: lmcache_mp_metadata.GetStoreMetadata,
# LoadStoreOp, IPCCacheServerKey, MPCacheServerContext.resolve_obj_keys,
# Session.extend_tokens, LMCacheDrivenTransfer.store, blend.store.
#
# Second attempt.  The first run VOIDED itself: extend_tokens replaced the
# token list whenever token_offset==0, so a straggler rank's first delta
# truncated the session shared by all 8 ranks and every rank further along
# gapped (6599 skipped stores -> the arm degenerated to `nostore`).  The
# splice no longer shortens the sequence; a new 8-rank interleave test
# gives 242 gaps on the old code and 0 on the new.
#
# ARM 1 tp8_mpfix -- the number.  Identical knobs to the frozen tp8_mp.
#   Compare against the frozen step-probe figures in the same directory:
#       tp8_none 83.94   tp8_nostore 85.71   tp8_tinykey 88.53   tp8_mp 91.90
#   Pre-registered reading:
#     ~88.5  the fix recovers what the tinykey diagnostic showed was there
#            (3.38 of the 7.97 ms/step, 42%); the rest is elsewhere.
#     <88.5  it also took back part of the scheduler-side +1.77, because the
#            connector metadata vLLM broadcasts every step got smaller too.
#     ~91.9  no effect -- the key size was not the cost after all, and record 6
#            is wrong.
#
# ARM 2 tp8_mpfix_warm -- does the cache still HIT.  The unit test proves the
#   server derives identical chunk hashes from a delta; this proves the whole
#   plumbing agrees end to end.  APC=0 so vLLM's own prefix cache cannot serve
#   the warm pass, L1 big enough to hold 16 x 60000 tokens (~10 GB at the
#   ~8.6 KB/token this config gets), LRU so it does not wedge when full.
#     warm much faster than cold -> retrieves are hitting, keys still match.
#     warm ~= cold             -> the delta broke key derivation.  VOID the fix.
set -uo pipefail
cd /home/bo/vast_profiling_problem
exec 2>&1

OUT=/home/bo/vast_profiling_problem/results/phase1_v2
WOUT=/home/bo/vast_profiling_problem/results/deltafix_warm
mkdir -p "$WOUT"

while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 20

# ---- neighbour watch (chain24's third-generation version) -----------------
compute_pids() { nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' '; }
is_ours() {
  local u a
  u=$(ps -o user= -p "$1" 2>/dev/null | tr -d ' ')
  [ "$u" = "bo" ] || return 1
  a=$(ps -o args=,comm= -p "$1" 2>/dev/null)
  case "$a" in
    *vast_profiling_problem*|*VLLM::*|*"lmcache server"*) return 0 ;;
    *) return 1 ;;
  esac
}
BASELINE=" $(compute_pids | tr '\n' ' ') "
echo "######## CHAIN25: pre-existing compute pids (the real neighbours):"
for p in $BASELINE; do
  echo "    $p  $(ps -o user=,comm=,etime= -p "$p" 2>/dev/null | tr -s ' ')"
done
( while :; do
    new=""
    for p in $(compute_pids); do
      case "$BASELINE" in *" $p "*) continue ;; esac
      is_ours "$p" && continue
      new="$new $p($(ps -o user=,comm= -p "$p" 2>/dev/null | tr -s ' ' | tr ' ' '/'))"
    done
    [ -n "$new" ] && echo "$(date +%H:%M:%S) NEW NEIGHBOUR:$new"
    sleep 10
  done ) > "$OUT/neighbour_watch_chain25.txt" 2>&1 &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

# ---------------------------------------------------------------- ARM 1
echo "######## CHAIN25: starting tp8_mpfix at $(date +%H:%M:%S)"
env GPUS=0,1,2,3,4,5,6,7 TP=8 GPUMEM= MNBT= NUM_BLOCKS= PROBE=1 \
    NPROMPTS=1000 CONC=1000 LANE_OUT="$OUT" \
    ARM=mp TAG=tp8_mpfix bash scripts/lane.sh 2>&1 | tee logs/lane_tp8_mpfix.out
echo "######## CHAIN25: tp8_mpfix exited at $(date +%H:%M:%S)"

# ---------------------------------------------------------------- ARM 2
echo "######## CHAIN25: starting tp8_mpfix_warm at $(date +%H:%M:%S)"
env GPUS=0,1,2,3,4,5,6,7 TP=8 GPUMEM= MNBT= NUM_BLOCKS= PROBE=1 \
    NPROMPTS=16 CONC=16 APC=0 PASSES="cold warm" L1_GB=32 L1_EVICT=LRU \
    LANE_OUT="$WOUT" ARM=mp TAG=tp8_mpfix_warm bash scripts/lane.sh 2>&1 \
    | tee logs/lane_tp8_mpfix_warm.out
echo "######## CHAIN25: tp8_mpfix_warm exited at $(date +%H:%M:%S)"

kill $WATCH 2>/dev/null

# ---------------------------------------------------------------- readout
echo "######## CHAIN25: step probe, differenced (the authoritative number)"
$PWD/.venv/bin/python scripts/probe_report.py "$OUT" 2>&1 | tail -30

echo "######## CHAIN25: step-count comparability (ms/step is meaningless if these differ)"
for d in tp8_none tp8_nostore tp8_tinykey tp8_mp tp8_mpfix; do
  f="$OUT/$d/stepprobe.txt"; [ -f "$f" ] || continue
  printf "  %-14s %s\n" "$d" "$(head -1 "$f" | grep -o 'steps=[0-9]*')"
done

echo "######## CHAIN25: end-to-end client duration"
for d in tp8_none tp8_nostore tp8_tinykey tp8_mp tp8_mpfix; do
  f="$OUT/$d/cold.json"; [ -f "$f" ] || continue
  printf "  %-14s %s\n" "$d" "$($PWD/.venv/bin/python -c "
import json,sys; d=json.load(open('$f'))
print(f\"dur={d.get('duration',0):.1f}s  tok/s={d.get('total_token_throughput',0):.0f}  ttft={d.get('mean_ttft_ms',0)/1000:.1f}s\")")"
done

echo "######## CHAIN25: correctness -- warm pass must beat cold"
for p in cold warm; do
  f="$WOUT/tp8_mpfix_warm/$p.json"; [ -f "$f" ] || { echo "  $p MISSING"; continue; }
  printf "  %-5s %s\n" "$p" "$($PWD/.venv/bin/python -c "
import json; d=json.load(open('$f'))
print(f\"dur={d.get('duration',0):.1f}s  tok/s={d.get('total_token_throughput',0):.0f}  mean_ttft={d.get('mean_ttft_ms',0)/1000:.2f}s\")")"
done

echo "######## CHAIN25: gap warnings (must be none)"
for d in "$OUT/tp8_mpfix" "$WOUT/tp8_mpfix_warm"; do
  n=$(sed 's/\x1b\[[0-9;]*m//g' "$d/lmcache_server.log" 2>/dev/null \
      | grep -c "Skipping STORE\|SessionTokenGapError")
  echo "  $(basename "$d"): $n"
done

echo "######## CHAIN25: new neighbours during the run:"
sort -u "$OUT/neighbour_watch_chain25.txt" 2>/dev/null | head -10
[ -s "$OUT/neighbour_watch_chain25.txt" ] || echo "  none"
echo "######## CHAIN25: batch done $(date +%H:%M:%S)"
