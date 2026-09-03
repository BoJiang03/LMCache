#!/usr/bin/env bash
# THE FIX ARM.  lmcache/integration/vllm/vllm_v1_adapter.py wait_for_save now
# does the skip_leading_tokens check BEFORE slot_mapping.to(self.device).
# Baseline to beat, same lane, same pinned pool:
#     none        296.9 s   loop 131.78  exec  81.82
#     timedip     326.1 s   loop 143.28  exec 140.39   wait_for_save 73.938 ms/step
#     ipstoreprobe 326.0 s               exec ~140     wait_for_save 72.921, of
#                                        which lmcache_engine.store is only 4.953
# If the hoist is the fix, ipfixed's wait_for_save collapses toward ~5 ms/step
# and its end-to-end moves from 326 s toward the ~300 s the store cost allows.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 25
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN18: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3 TP=4 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
      "$@" ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN18: $tag exited at $(date +%H:%M:%S)"
}
L=/home/bo/vast_profiling_problem/results/lane
run ipstoreprobe ipfixed LANE_OUT=$L NPROMPTS=300 CONC=300
echo "######## CHAIN18: fix arm done $(date +%H:%M:%S)"
