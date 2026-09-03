#!/usr/bin/env bash
# chain24 -- size the prize before writing the fix.
#
#   `tinykey` is `mp` with the STORE key cut down to the chunk it actually
#   stores (token_ids[start:end], start=0) instead of the whole grown prompt
#   prefix.  Compared against the frozen FULL-protocol pair in the same
#   directory:  tp8_none 85.34, tp8_mp 91.04 ms/step.
#
#   Reading, pre-registered:
#     -> ~86  the key IS the loss; the spin amplification collapses with it and
#             the real fix (raw-buffer encoding / delta shipping) is worth the
#             full 6.7%.
#     -> ~90  the key is ~1 ms of a 5.7 ms problem; the fix is worth ~1% and
#             the cost is mostly elsewhere in the store path.  Keep digging.
#     TRUNCATED == 0 or FELLBACK >> 0 in the log  ->  VOID, the patch missed.
#
#   Same FULL knobs as tp8_none / tp8_mp: no --gpu-memory-utilization, no
#   --num-gpu-blocks-override, no --max-num-batched-tokens, 1000 prompts,
#   c=1000, GPUs 0-7, TP=8.
set -uo pipefail
cd /home/bo/vast_profiling_problem
exec 2>&1

OUT=/home/bo/vast_profiling_problem/results/phase1_v2

while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 20

# ---- neighbour watch, third attempt --------------------------------------
# chain21/22 flagged on free memory and utilisation, so our own workers tripped
# it.  chain23 flagged on session id, but lane.sh spawns the server detached, so
# our own workers tripped that too.  This one takes a BASELINE of the compute
# pids already on the GPUs before our arm starts -- those are the real
# neighbours, named once -- and then only reports pids that are neither in the
# baseline nor attributable to this lane.  A pid is ours if it runs as bo and
# its args or comm name it: the repo path, a VLLM worker, or an lmcache server.
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
BASELINE=" $(compute_pids | tr '\n' ' ')"
echo "######## CHAIN24: pre-existing compute pids (the real neighbours):"
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
  done ) > "$OUT/neighbour_watch_chain24.txt" 2>&1 &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

echo "######## CHAIN24: starting tp8_tinykey at $(date +%H:%M:%S)"
env GPUS=0,1,2,3,4,5,6,7 TP=8 GPUMEM= MNBT= NUM_BLOCKS= PROBE=1 \
    NPROMPTS=1000 CONC=1000 LANE_OUT="$OUT" \
    ARM=tinykey TAG=tp8_tinykey bash scripts/lane.sh 2>&1 | tee logs/lane_tp8_tinykey.out
echo "######## CHAIN24: tp8_tinykey exited at $(date +%H:%M:%S)"

kill $WATCH 2>/dev/null
echo "######## CHAIN24: validity -- last TINYKEY counters per worker"
sed 's/\x1b\[[0-9;]*m//g' "$OUT/tp8_tinykey/server.log" | grep -o "TINYKEY pid=.*" | tail -8
echo "######## CHAIN24: in-engine ms/step (median Avg prompt throughput)"
for d in tp8_none tp8_mp tp8_nostore tp8_tinykey; do
  f="$OUT/$d/server.log"; [ -f "$f" ] || continue
  m=$(sed 's/\x1b\[[0-9;]*m//g' "$f" | grep -o "Avg prompt throughput: [0-9.]*" | awk '{print $4}' \
      | sort -n | awk '{a[NR]=$1} END{if(NR)printf "%.1f",(NR%2? a[(NR+1)/2] : (a[NR/2]+a[NR/2+1])/2)}')
  printf "  %-14s median=%-10s ms/step=%s\n" "$d" "$m" \
    "$(awk -v m=$m 'BEGIN{if(m>0)printf "%.2f",8192000/m; else print "NA"}')"
done
echo "######## CHAIN24: new neighbours during the run:"
cat "$OUT/neighbour_watch_chain24.txt" 2>/dev/null | sort -u | head -10
[ -s "$OUT/neighbour_watch_chain24.txt" ] || echo "  none"
echo "######## CHAIN24: batch done $(date +%H:%M:%S)"
