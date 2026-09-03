#!/usr/bin/env bash
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane.sh" >/dev/null; do sleep 10; done
sleep 20
export LANE_OUT=/home/bo/vast_profiling_problem/results/phase1
export GPUS=0,1,2,3,4,5,6,7 TP=8 NPROMPTS=1000 CONC=1000 GPUMEM="" MNBT=""
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN4: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env "$@" ARM="$arm" TAG="$tag" PROBE=1 bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN4: $tag exited at $(date +%H:%M:%S)"
}
run none     1m_probe_none
run mp       1n_probe_mp
run nostore  1o_probe_nostore
run nolookup 1p_probe_nolookup
echo "######## CHAIN4: batch done $(date +%H:%M:%S)"
