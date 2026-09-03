#!/usr/bin/env bash
# chain21
#   1. c1000_mp  -- redo the arm killed with chain17 at 06:22:18, so the
#      c1000 pair is complete.  TP=4, GPUs 0-3, pool pinned 1,920,000.
#   2. the TP=8 triple -- none / mp / nostore.  WHY nostore: at TP=4 the whole
#      loss is accounted for by 0.89 ms/step of worker hook wall time, but at
#      TP=8 the loss is 5.70 ms/step against the same 0.81 ms/step of hooks.
#      4.9 ms/step at TP=8 is in no hook and does not exist at TP=4.  nostore
#      says whether that residue is downstream of the store submission.
#
#   The TP=8 mode is chosen at run time from what GPUs 4-7 actually have free:
#     FULL  (>=100 GiB free on every one of 4-7) -- phase1's exact protocol:
#           no --gpu-memory-utilization, no --num-gpu-blocks-override, no
#           --max-num-batched-tokens, 1000 prompts, c=1000.  Lands in the same
#           table as 85.3 / 91.0 / 97.5.
#     LEAN  (otherwise) -- fits beside root's SGLang: GPUMEM=0.145,
#           NUM_BLOCKS=30000, 300 prompts, c=300.  A fourth regime, comparable
#           only within itself.
#
#   A watchdog samples GPUs 4-7 every 10 s for the whole TP=8 phase.  Any
#   sample with util>0 or free<18 GiB means a neighbour moved and the arms it
#   overlaps are NOT trustworthy; the report must say so rather than average
#   it away.
set -uo pipefail
cd /home/bo/vast_profiling_problem
exec 2>&1

while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 20

run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN21: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN21: $tag exited at $(date +%H:%M:%S)"
}

# ---- 1. finish the c1000 pair -------------------------------------------
run mp c1000_mp \
    GPUS=0,1,2,3 TP=4 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
    LANE_OUT=/home/bo/vast_profiling_problem/results/lane_c1000 \
    NPROMPTS=1000 CONC=1000

# ---- 2. pick the TP=8 mode from the live GPU state ----------------------
min_free=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits \
           | awk -F', ' '$1>=4 {if (m=="" || $2<m) m=$2} END{print m+0}')
echo "######## CHAIN21: min free on GPUs 4-7 = ${min_free} MiB"

if [ "$min_free" -ge 102400 ]; then
  MODE=FULL
  OUT=/home/bo/vast_profiling_problem/results/phase1_v2
  COMMON=(GPUS=0,1,2,3,4,5,6,7 TP=8 GPUMEM= MNBT= NUM_BLOCKS= PROBE=1
          NPROMPTS=1000 CONC=1000 LANE_OUT=$OUT)
else
  MODE=LEAN
  OUT=/home/bo/vast_profiling_problem/results/tp8_lean
  COMMON=(GPUS=0,1,2,3,4,5,6,7 TP=8 GPUMEM=0.145 MNBT=8192 NUM_BLOCKS=30000 PROBE=1
          NPROMPTS=300 CONC=300 LANE_OUT=$OUT)
fi
echo "######## CHAIN21: TP=8 mode = $MODE  -> $OUT"
mkdir -p "$OUT"

# ---- watchdog on the neighbour -----------------------------------------
( while :; do
    echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
          --format=csv,noheader,nounits | awk -F', ' '$1>=4{printf "gpu%s:free=%s,util=%s ", $1,$2,$3}')"
    sleep 10
  done ) > "$OUT/neighbour_watch.txt" 2>&1 &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

for arm in none mp nostore; do
  run "$arm" "tp8_$arm" "${COMMON[@]}"
done

kill $WATCH 2>/dev/null
echo "######## CHAIN21: neighbour watch summary"
awk '{for(i=2;i<=NF;i++){split($i,a,":");split(a[2],b,",");
      split(b[1],f,"=");split(b[2],u,"=");
      if (u[2]+0>0 || f[2]+0<18432) print "  DIRTY " $1 " " $i}}' \
    "$OUT/neighbour_watch.txt" | head -20
echo "######## CHAIN21: batch done $(date +%H:%M:%S)"
