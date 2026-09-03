#!/usr/bin/env bash
# TP=4 lane on GPUs 0-3, with the lmcache HTTP port pinned to 8766 (8080 is
# held by another tenant's server and the bind failure is fatal).
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 10; done
sleep 25
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN9: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3 TP=4 LANE_OUT=/home/bo/vast_profiling_problem/results/lane \
      NPROMPTS=300 CONC=300 GPUMEM=0.80 MNBT=8192 PROBE=1 \
      "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN9: $tag exited at $(date +%H:%M:%S)"
}
run mp       mp
run nolookup nolookup
run nostore  nostore
run timedip  timedip
run none     none_probe
echo "######## CHAIN9: batch done $(date +%H:%M:%S)"
