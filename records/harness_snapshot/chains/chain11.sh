#!/usr/bin/env bash
# TP=4 lane, GPUs 0-3, KV pool pinned at 30,000 blocks so every arm is
# comparable regardless of what else lands on the devices.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 10; done
sleep 25
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN11: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3 TP=4 LANE_OUT=/home/bo/vast_profiling_problem/results/lane \
      NPROMPTS=300 CONC=300 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
      "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN11: $tag exited at $(date +%H:%M:%S)"
}
run none     none      # baseline first: it sets the pinned-pool reference
run mp       mp        # lane validity: does TP=4 reproduce the connector tax
run nolookup nolookup  # the clean test 1l could not be
run nostore  nostore
run timedip  timedip   # the second loss
echo "######## CHAIN11: batch done $(date +%H:%M:%S)"
