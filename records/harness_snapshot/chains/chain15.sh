#!/usr/bin/env bash
# Localize IP's 73.9 ms/step wait_for_save.  Runs after chain14.
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "chain14[.]sh" >/dev/null; do sleep 20; done
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 10; done
sleep 25
echo "######## CHAIN15: starting ipstoreprobe at $(date +%H:%M:%S)"
env GPUS=0,1,2,3 TP=4 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
    LANE_OUT=/home/bo/vast_profiling_problem/results/lane NPROMPTS=300 CONC=300 \
    ARM=ipstoreprobe TAG=ipstoreprobe bash scripts/lane.sh 2>&1 \
    | tee logs/lane_ipstoreprobe.out
echo "######## CHAIN15: ipstoreprobe exited at $(date +%H:%M:%S)"
