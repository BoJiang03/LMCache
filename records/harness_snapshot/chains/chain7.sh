#!/usr/bin/env bash
# TP=4 lane on GPUs 0-3.  Another tenant took GPUs 4-7 again mid-launch, so
# TP=8 is out; 0-3 have been free and stable all night and arms take 9 minutes
# instead of 21.  MNBT=8192 and GPUMEM=0.80 reproduce exactly what the existing
# `none` baseline (136.5 ms/step) was run with, so it can be reused.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane.sh" >/dev/null; do sleep 10; done
sleep 25
export LANE_OUT=/home/bo/vast_profiling_problem/results/lane
export GPUS=0,1,2,3 TP=4 NPROMPTS=300 CONC=300 GPUMEM=0.80 MNBT=8192
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN7: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env "$@" ARM="$arm" TAG="$tag" PROBE=1 bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN7: $tag exited at $(date +%H:%M:%S)"
}
run mp       mp          # lane validity: does TP=4 reproduce the connector tax
run nolookup nolookup    # the clean test 1l could not be
run nostore  nostore
run timedip  timedip     # the second loss
echo "######## CHAIN7: batch done $(date +%H:%M:%S)"
