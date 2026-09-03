#!/usr/bin/env bash
# TP=8 probe batch, phase1 protocol so the arms land in the same table as 1a-1l.
cd /home/bo/vast_profiling_problem
# The TP=4 arm that was interrupted may still be tearing down; its teardown
# holds ~60 GB per GPU until the driver releases it.
while pgrep -u bo -f "bash scripts/lane.sh" >/dev/null; do sleep 10; done
sleep 20
export LANE_OUT=/home/bo/vast_profiling_problem/results/phase1
export GPUS=0,1,2,3,4,5,6,7 TP=8 NPROMPTS=1000 CONC=1000 GPUMEM=""
run() {
  local arm="$1" tag="$2"
  echo "######## CHAIN3: starting $tag ($arm) at $(date +%H:%M:%S)"
  ARM="$arm" TAG="$tag" PROBE=1 bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN3: $tag exited at $(date +%H:%M:%S)"
}
run none     1m_probe_none
run mp       1n_probe_mp
run nostore  1o_probe_nostore
run nolookup 1p_probe_nolookup
echo "######## CHAIN3: batch done $(date +%H:%M:%S)"
