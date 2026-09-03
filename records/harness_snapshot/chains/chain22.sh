#!/usr/bin/env bash
# chain22 -- close the last gap on loss #1: is the 8x amplification TP degree
# or pool depth?
#
# WHAT IS ALREADY EXCLUDED
#   client concurrency -- TP=4, pool 1.92M: c=300 gave +0.84%, c=1000 +0.51%.
#                         The pool caps in-flight at ~32 either way.
#   pool depth, UPPER RANGE -- phase1 ran both 13,724,416 and 25,798,626 and
#                         got 85.3 / 91.0 in both (records 2026/09/02/7, and
#                         "Every block inside an arm agrees to 0.1 ms").
#
# THE GAP
#   The TP=4 lane ran at 1,920,000 -- 7x below the smallest pool phase1 ever
#   tested.  So pool is excluded from 13.7M upward, not below it.
#
# THIS PAIR
#   TP=8, everything identical to the phase1_v2 FULL arms EXCEPT
#   --num-gpu-blocks-override 30000 (pool 1,920,000).  One variable.
#     mp lands near 91.0  -> pool irrelevant across the whole range; TP degree
#                            is the amplifier (allreduce straggler hypothesis).
#     mp lands near 85-86 -> it is the pool after all; the TP hypothesis dies.
set -uo pipefail
cd /home/bo/vast_profiling_problem
exec 2>&1

while pgrep -u bo -f "chain21[.]sh" >/dev/null; do sleep 30; done
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 20

OUT=/home/bo/vast_profiling_problem/results/tp8_smallpool
mkdir -p "$OUT"

( while :; do
    echo "$(date +%H:%M:%S) $(nvidia-smi --query-gpu=index,memory.free,utilization.gpu \
          --format=csv,noheader,nounits | awk -F', ' '$1>=4{printf "gpu%s:free=%s,util=%s ", $1,$2,$3}')"
    sleep 10
  done ) > "$OUT/neighbour_watch.txt" 2>&1 &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

for arm in none mp; do
  tag="tp8sp_$arm"
  echo "######## CHAIN22: starting $tag at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3,4,5,6,7 TP=8 GPUMEM= MNBT= NUM_BLOCKS=30000 PROBE=1 \
      NPROMPTS=1000 CONC=1000 LANE_OUT="$OUT" ARM="$arm" TAG="$tag" \
      bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN22: $tag exited at $(date +%H:%M:%S)"
done

kill $WATCH 2>/dev/null
echo "######## CHAIN22: neighbour watch summary"
awk '{for(i=2;i<=NF;i++){split($i,a,":");split(a[2],b,",");
      split(b[1],f,"=");split(b[2],u,"=");
      if (u[2]+0>0 || f[2]+0<18432) print "  DIRTY " $1 " " $i}}' \
    "$OUT/neighbour_watch.txt" | head -20
echo "######## CHAIN22: batch done $(date +%H:%M:%S)"
