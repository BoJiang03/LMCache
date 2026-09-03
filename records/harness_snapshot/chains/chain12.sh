#!/usr/bin/env bash
# VAST finding #2: MP vs IP on the WARM path.  Runs after chain11.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "chain11[.]sh" >/dev/null; do sleep 20; done
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 10; done
sleep 25
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN12: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3 TP=4 LANE_OUT=/home/bo/vast_profiling_problem/results/warm \
      NPROMPTS=100 CONC=100 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 APC=0 \
      PROBE=1 PASSES="cold warm" "$@" ARM="$arm" TAG="$tag" \
      bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN12: $tag exited at $(date +%H:%M:%S)"
}
run mp warm_mp L1_GB=500 L1_EVICT=noop
run ip warm_ip IP_YAML=ip_l1_big.yaml
echo "######## CHAIN12: batch done $(date +%H:%M:%S)"
