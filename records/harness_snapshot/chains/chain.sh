#!/usr/bin/env bash
# Run the lane arms back to back. Each arm tears down its own processes.
cd /home/bo/vast_profiling_problem
while kill -0 2170096 2>/dev/null; do sleep 10; done
for arm in mp nostore nolookup; do
  echo "######## CHAIN: starting $arm at $(date +%H:%M:%S)"
  ARM=$arm bash scripts/lane.sh > logs/lane_$arm.out 2>&1
  echo "######## CHAIN: $arm exited rc=$? at $(date +%H:%M:%S)"
done
echo "######## CHAIN: all done $(date +%H:%M:%S)"
