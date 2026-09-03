#!/usr/bin/env bash
# The four arms that answer the two questions the user asked, no step probe.
# The probe killed a TP worker on its first inference step (silent death, no
# traceback) so it is out until it can be hardened; phase1 already supplies the
# baselines these arms are measured against, at the identical protocol and the
# identical pool of 13,724,416 tokens:
#     no connector 85.3 ms/step (1a x6, 1c x3, 1i)
#     LMCache MP   91.0        (1d x2, 1e x2, 1j, 1k, 1l)
#     LMCache IP   97.5        (1b x6, 1f x2, 1g)
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane.sh" >/dev/null; do sleep 10; done
sleep 25
export LANE_OUT=/home/bo/vast_profiling_problem/results/phase1
export GPUS=0,1,2,3,4,5,6,7 TP=8 NPROMPTS=1000 CONC=1000 GPUMEM="" MNBT=""
run() {
  local arm="$1" tag="$2"; shift 2
  echo "######## CHAIN6: starting $tag ($arm) $* at $(date +%H:%M:%S)"
  env "$@" ARM="$arm" TAG="$tag" PROBE=0 bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN6: $tag exited at $(date +%H:%M:%S)"
}
run nolookup 1m_nolookup
run nostore  1n_nostore
run timedip  1o_timedip
run mp       1p_mp_bigl1 L1_GB=600 L1_EVICT=LRU
echo "######## CHAIN6: batch done $(date +%H:%M:%S)"
