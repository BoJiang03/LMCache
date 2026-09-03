#!/usr/bin/env bash
cd /home/bo/vast_profiling_problem
run() {  # run <ARM> [TAG] [PROBE]
  local arm="$1" tag="${2:-$1}" probe="${3:-0}"
  echo "######## CHAIN: starting $tag at $(date +%H:%M:%S)"
  ARM="$arm" TAG="$tag" PROBE="$probe" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN: $tag exited at $(date +%H:%M:%S)"
}
run mp
run nostore
run nolookup
run timed
echo "######## CHAIN: batch 1 done $(date +%H:%M:%S)"
