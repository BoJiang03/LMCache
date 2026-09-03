#!/usr/bin/env bash
# VAST finding #2: MP vs IP on the WARM path.  Runs after chain9.
#
# 100 prompts x 60,000 tokens = 6M tokens = ~442 GB of KV in either connector,
# so both tiers get ~500 GB and the comparison is between code paths rather
# than cache sizes.  APC=0 because vLLM's own paged pool is larger than any
# LMCache tier this box can host: with prefix caching on, the warm pass is
# served by vLLM and never reaches LMCache at all.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "chain9[.]sh" >/dev/null; do sleep 20; done
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 10; done
sleep 25
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN10: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3 TP=4 LANE_OUT=/home/bo/vast_profiling_problem/results/warm \
      NPROMPTS=100 CONC=100 GPUMEM=0.80 MNBT=8192 APC=0 PROBE=1 \
      PASSES="cold warm" "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 \
      | tee "logs/lane_$tag.out"
  echo "######## CHAIN10: $tag exited at $(date +%H:%M:%S)"
}
run mp warm_mp L1_GB=500 L1_EVICT=noop
run ip warm_ip IP_YAML=ip_l1_big.yaml
echo "######## CHAIN10: batch done $(date +%H:%M:%S)"
