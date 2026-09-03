#!/usr/bin/env bash
# Runs after chain7.  Two jobs:
#   1. none_probe -- the TP=4 `none` baseline was taken before the step probe
#      existed, so the probe's own columns have no zero without this.
#   2. VAST finding #2 -- MP vs IP on the WARM path, the regime their chart is
#      in.  Own LANE_OUT because APC is off there and the pool reference differs.
# Every knob is passed as an ARGUMENT to run(), not as a prefix assignment:
# bash does not export prefix assignments to a child process unless the name is
# already exported, so `LANE_OUT=x run ...` would have silently written to the
# default directory.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "chain7.sh" >/dev/null; do sleep 20; done
while pgrep -u bo -f "bash scripts/lane.sh" >/dev/null; do sleep 10; done
sleep 25
export GPUS=0,1,2,3 TP=4
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN8: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN8: $tag exited at $(date +%H:%M:%S)"
}
run none none_probe \
  LANE_OUT=/home/bo/vast_profiling_problem/results/lane \
  NPROMPTS=300 CONC=300 GPUMEM=0.80 MNBT=8192 PROBE=1

# Finding #2.  100 prompts x 60,000 tokens = 6M tokens = ~442 GB of KV in
# either connector, so both tiers get ~500 GB and the comparison is between
# code paths rather than cache sizes.  APC=0 because vLLM's own paged pool is
# larger than any LMCache tier this box can host: with prefix caching on the
# warm pass never reaches LMCache at all.
run mp warm_mp \
  LANE_OUT=/home/bo/vast_profiling_problem/results/warm \
  NPROMPTS=100 CONC=100 GPUMEM=0.80 MNBT=8192 APC=0 PASSES="cold warm" PROBE=1 \
  L1_GB=500 L1_EVICT=noop
run ip warm_ip \
  LANE_OUT=/home/bo/vast_profiling_problem/results/warm \
  NPROMPTS=100 CONC=100 GPUMEM=0.80 MNBT=8192 APC=0 PASSES="cold warm" PROBE=1 \
  IP_YAML=ip_l1_big.yaml
echo "######## CHAIN8: batch done $(date +%H:%M:%S)"
