#!/usr/bin/env bash
# Disentangle TP degree from concurrency.
#
# phase1 measured the +6.7% tax at TP=8 AND c=1000.  Tonight's lane measures
# +0.8% at TP=4 AND c=300.  Two variables moved, so "the tax scales with TP" is
# not yet established -- it could just as well scale with the depth of the
# waiting queue.  This pair holds TP=4 and moves concurrency to 1000, which is
# the phase1 value:
#     tax appears at TP=4/c=1000  -> concurrency is the driver, not TP
#     tax still ~0                -> TP is the driver
# Same pinned pool as every other arm so all four cells are comparable.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "chain1[12][.]sh" >/dev/null; do sleep 20; done
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 10; done
sleep 25
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN13: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3 TP=4 LANE_OUT=/home/bo/vast_profiling_problem/results/lane_c1000 \
      NPROMPTS=1000 CONC=1000 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
      "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN13: $tag exited at $(date +%H:%M:%S)"
}
run none c1000_none
run mp   c1000_mp
echo "######## CHAIN13: batch done $(date +%H:%M:%S)"
