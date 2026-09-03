#!/usr/bin/env bash
# Runs after chain11 (which ends with timedip).  Order is by value:
#   1. storeprobe  -- localise the 1.60 ms/step of worker blocking that nostore
#                     proved is the entire connector tax.  Decides the fix.
#   2. warm pair   -- VAST finding #2, MP vs IP on the hit path.
#   3. c1000 pair  -- separate TP degree from concurrency, since phase1 was
#                     TP=8 AND c=1000 while tonight's lane is TP=4 AND c=300.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "chain11[.]sh" >/dev/null; do sleep 20; done
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 10; done
sleep 25
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN14: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3 TP=4 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
      "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN14: $tag exited at $(date +%H:%M:%S)"
}
L=/home/bo/vast_profiling_problem/results/lane
run storeprobe storeprobe LANE_OUT=$L NPROMPTS=300 CONC=300

W=/home/bo/vast_profiling_problem/results/warm
run mp warm_mp LANE_OUT=$W NPROMPTS=100 CONC=100 APC=0 PASSES="cold warm" \
  L1_GB=500 L1_EVICT=noop
run ip warm_ip LANE_OUT=$W NPROMPTS=100 CONC=100 APC=0 PASSES="cold warm" \
  IP_YAML=ip_l1_big.yaml

C=/home/bo/vast_profiling_problem/results/lane_c1000
run none c1000_none LANE_OUT=$C NPROMPTS=1000 CONC=1000
run mp   c1000_mp   LANE_OUT=$C NPROMPTS=1000 CONC=1000
echo "######## CHAIN14: batch done $(date +%H:%M:%S)"
