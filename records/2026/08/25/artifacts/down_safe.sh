#!/bin/bash
# Scoped teardown. Unlike the old down.sh (which pattern-killed every
# `vllm serve`/`VLLM::EngineCore` on the box -- other users' containers
# included), this only kills:
#   1. the pid trees recorded in $LOGDIR/{server,vllm}.pid, and
#   2. compute apps on *our* GPUs that are owned by *our* uid.
set -u
. /tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/7445f449-aa3b-46be-b351-5a22de3af76a/scratchpad/smoke/env.sh 2>/dev/null \
  || { echo "cannot source env.sh" >&2; exit 1; }
ME=$(id -u)

descendants() {  # recursive child walk
  local p=$1 c
  for c in $(ps -o pid= --ppid "$p" 2>/dev/null); do descendants "$c"; done
  echo "$p"
}

for f in server vllm; do
  pf="$LOGDIR/$f.pid"
  [ -f "$pf" ] || continue
  root=$(cat "$pf")
  [ -n "$root" ] || continue
  for pid in $(descendants "$root"); do
    owner=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$owner" = "$ME" ] || { echo "skip $pid (uid $owner, not mine)"; continue; }
    kill -9 "$pid" 2>/dev/null && echo "killed $f pid $pid"
  done
  rm -f "$pf"
done

# GPU sweep, restricted to our GPU indices AND our uid.
for g in $(echo "$GPUS" | tr ',' ' '); do
  uuid=$(nvidia-smi --query-gpu=index,uuid --format=csv,noheader \
         | awk -F', ' -v g="$g" '$1==g{print $2}')
  [ -n "$uuid" ] || continue
  for pid in $(nvidia-smi --query-compute-apps=pid,gpu_uuid --format=csv,noheader \
               | awk -F', ' -v u="$uuid" '$2==u{print $1}'); do
    owner=$(ps -o uid= -p "$pid" 2>/dev/null | tr -d ' ')
    [ "$owner" = "$ME" ] || { echo "skip gpu$g pid $pid (uid $owner, not mine)"; continue; }
    kill -9 "$pid" 2>/dev/null && echo "killed leftover $pid on gpu $g"
  done
done

sleep 3
for g in $(echo "$GPUS" | tr ',' ' '); do
  used=$(nvidia-smi --query-gpu=index,memory.used --format=csv,noheader,nounits \
         | awk -F', ' -v g="$g" '$1==g{print $2}')
  echo "gpu$g used=${used}MiB"
done
ss -ltn 2>/dev/null | grep -E ":($MP_PORT|$HTTP_PORT|$VLLM_PORT)\b" && { echo "ports still bound" >&2; exit 1; }
echo clean
