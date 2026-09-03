#!/usr/bin/env bash
# 1. warm_ip_apc -- IP's warm pass logged "LMCache hit tokens: 0" for all 200
#    requests while paying 140 ms/step of exec to store.  The one alternative
#    explanation is that APC=0 (needed so a hit must come from LMCache) broke
#    the retrieve path rather than exposing it.  This arm turns prefix caching
#    back ON, which is VAST's actual config.  The working set is 6M tokens
#    against a 1.92M-token pool, so vLLM's own cache cannot serve most of it
#    and LMCache still has to.  Hits > 0 here means APC=0 was the artifact;
#    still 0 means IP's cache really is write-only.
# 2. the c1000 pair -- separate TP degree from concurrency for loss #1.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 25
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN17: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3 TP=4 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
      "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN17: $tag exited at $(date +%H:%M:%S)"
}
W=/home/bo/vast_profiling_problem/results/warm_apc
C=/home/bo/vast_profiling_problem/results/lane_c1000
run ip warm_ip_apc LANE_OUT=$W NPROMPTS=100 CONC=100 APC=1 PASSES="cold warm" IP_YAML=ip_l1_big.yaml
run mp warm_mp_apc LANE_OUT=$W NPROMPTS=100 CONC=100 APC=1 PASSES="cold warm" L1_GB=500 L1_EVICT=noop
run none c1000_none LANE_OUT=$C NPROMPTS=1000 CONC=1000
run mp   c1000_mp   LANE_OUT=$C NPROMPTS=1000 CONC=1000
echo "######## CHAIN17: batch done $(date +%H:%M:%S)"
