#!/usr/bin/env bash
# FIX ARM 2: stage slot_mapping through a reusable pinned buffer so the H2D
# copy is asynchronous instead of a synchronous stall inside the forward pass.
# Profiled cause: vllm_v1_adapter.py wait_for_save -> Tensor.to(), 217 calls /
# 7.321 s = 33.7 ms per call over a 200-step window.
# Numbers to beat, same lane, same pinned pool, same instrument:
#     none          296.9 s  loop 131.78  exec  81.82
#     ipstoreprobe  326.0 s  loop 143.64  exec 140.47  wait_for_save 72.921
cd /home/bo/vast_profiling_problem
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 25
echo "######## CHAIN20: starting ipfix2 at $(date +%H:%M:%S)"
env GPUS=0,1,2,3 TP=4 GPUMEM=0.80 MNBT=8192 NUM_BLOCKS=30000 PROBE=1 \
    LANE_OUT=/home/bo/vast_profiling_problem/results/lane NPROMPTS=300 CONC=300 \
    ARM=ipstoreprobe TAG=ipfix2 bash scripts/lane.sh 2>&1 | tee logs/lane_ipfix2.out
echo "######## CHAIN20: ipfix2 exited at $(date +%H:%M:%S)"
