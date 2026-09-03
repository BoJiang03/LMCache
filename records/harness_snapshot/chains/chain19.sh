#!/usr/bin/env bash
# Profile IP's worker instead of guessing at its 67 ms/step.
# Window is steps 900..1100 -- well inside steady state (the run does ~2,200)
# and 200 steps is enough samples while keeping cProfile's own overhead from
# distorting the whole run.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 25
rm -f /home/bo/.claude/jobs/ba4f4ca8/tmp/prof.*.pstats
echo "######## CHAIN19: starting ipprof at $(date +%H:%M:%S)"
env GPUS=0,1,2,3 TP=4 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
    STEPPROBE_CPROFILE=900:1100 \
    STEPPROBE_CPROFILE_OUT=/home/bo/.claude/jobs/ba4f4ca8/tmp/prof \
    LANE_OUT=/home/bo/vast_profiling_problem/results/lane NPROMPTS=300 CONC=300 \
    ARM=ipstoreprobe TAG=ipprof bash scripts/lane.sh 2>&1 | tee logs/lane_ipprof.out
echo "######## CHAIN19: ipprof exited at $(date +%H:%M:%S)"
