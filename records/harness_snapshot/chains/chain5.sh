#!/usr/bin/env bash
cd /home/bo/vast_profiling_problem
# wait for chain4 to finish
while pgrep -u bo -f "chain4.sh" >/dev/null; do sleep 20; done
while pgrep -u bo -f "bash scripts/lane.sh" >/dev/null; do sleep 10; done
sleep 20
export LANE_OUT=/home/bo/vast_profiling_problem/results/phase1
export GPUS=0,1,2,3,4,5,6,7 TP=8 NPROMPTS=1000 CONC=1000 GPUMEM="" MNBT=""
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN5: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env "$@" ARM="$arm" TAG="$tag" PROBE=1 bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN5: $tag exited at $(date +%H:%M:%S)"
}
# The second loss: IP's extra +6.5 ms/step.  IP's lookup_client returns None
# for an in-flight async lookup, so vLLM re-asks EVERY deferred request EVERY
# step; IP's Deferred counter sits at 415-926 where MP's is 0.  This arm counts
# the hook calls, which is the whole hypothesis.
run timedip 1q_timedip
# Whether phase1's MP series measured a degenerate store path: with l1=8GB and
# eviction noop only 113 chunk stores ever succeeded.  600 GB with LRU makes
# every store succeed.
run mp 1r_mp_bigl1 L1_GB=600 L1_EVICT=LRU
echo "######## CHAIN5: batch done $(date +%H:%M:%S)"
