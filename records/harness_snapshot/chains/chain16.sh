#!/usr/bin/env bash
# Order by what decides the fix.
#
#   bigl1        storeprobe showed the store SUBMIT is cheap (0.454 ms/step,
#                0.03 s blocked over the run) yet no-op'ing it removes all 1.70
#                ms/step of worker blocking.  So the cost is the server ACTING
#                on the submission, concurrently, on the same GPUs and cores.
#                In the 8 GB / noop-eviction config 99.3% of those stores are
#                rejected for lack of room.  If the tax disappears when stores
#                actually succeed, the fix is backpressure; if it persists, the
#                fix has to be in how the transfer contends.  Decisive either way.
#   warm pair    VAST finding #2, MP vs IP on the hit path.
#   ipstoreprobe split IP's 73.9 ms/step wait_for_save.
#   c1000 pair   separate TP degree from concurrency.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 10; done
sleep 25
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN16: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3 TP=4 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
      "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN16: $tag exited at $(date +%H:%M:%S)"
}
L=/home/bo/vast_profiling_problem/results/lane
W=/home/bo/vast_profiling_problem/results/warm
C=/home/bo/vast_profiling_problem/results/lane_c1000
run mp bigl1 LANE_OUT=$L NPROMPTS=300 CONC=300 L1_GB=500 L1_EVICT=LRU
run mp warm_mp LANE_OUT=$W NPROMPTS=100 CONC=100 APC=0 PASSES="cold warm" L1_GB=500 L1_EVICT=noop
run ip warm_ip LANE_OUT=$W NPROMPTS=100 CONC=100 APC=0 PASSES="cold warm" IP_YAML=ip_l1_big.yaml
run ipstoreprobe ipstoreprobe LANE_OUT=$L NPROMPTS=300 CONC=300
run none c1000_none LANE_OUT=$C NPROMPTS=1000 CONC=1000
run mp   c1000_mp   LANE_OUT=$C NPROMPTS=1000 CONC=1000
echo "######## CHAIN16: batch done $(date +%H:%M:%S)"
