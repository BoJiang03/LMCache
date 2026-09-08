# SPDX-License-Identifier: Apache-2.0
# Process helpers.  These only ever touch processes THIS script started, so the
# benchmark is safe to run on a shared machine.

MY_PIDS=()   # process-group leaders we spawned

# spawn <logfile> <cmd...> -- setsid so we own a killable process group.
# Sets $SPAWNED_PID in the CALLER's shell.  Never use $(spawn ...): the subshell
# would drop the pid from MY_PIDS and teardown would kill nothing.
SPAWNED_PID=""
spawn() {
  local log="$1"; shift
  setsid "$@" > "$log" 2>&1 &
  SPAWNED_PID=$!
  MY_PIDS+=("$SPAWNED_PID")
}

# wait_health <logfile> <pid> <timeout_s>
wait_health() {
  local log="$1" pid="$2" t="${3:-1800}" i=0
  while [ "$i" -lt "$t" ]; do
    curl -sf "http://127.0.0.1:$PORT/health" >/dev/null 2>&1 && { echo "  up after ${i}s"; return 0; }
    kill -0 "$pid" 2>/dev/null || { echo "  process exited early"; tail -25 "$log"; return 1; }
    sleep 5; i=$((i+5))
  done
  echo "  TIMEOUT after ${t}s"; return 1
}

teardown() {
  local pid i alive
  for pid in "${MY_PIDS[@]:-}"; do
    [ -n "$pid" ] || continue
    kill -TERM -"$pid" 2>/dev/null || kill -TERM "$pid" 2>/dev/null
  done
  i=0
  while [ "$i" -lt 120 ]; do
    alive=0
    for pid in "${MY_PIDS[@]:-}"; do
      [ -n "$pid" ] && kill -0 "$pid" 2>/dev/null && alive=1
    done
    [ "$alive" -eq 0 ] && break
    sleep 5; i=$((i+5))
  done
  for pid in "${MY_PIDS[@]:-}"; do
    [ -n "$pid" ] && kill -KILL -"$pid" 2>/dev/null
  done
  MY_PIDS=()
  sleep 20   # let the driver release HBM
}

# EXIT runs teardown on a normal end; INT/TERM must ALSO exit -- a bare trap
# handler returns and the script would carry on with the next point.
on_signal() { echo "  [signal] tearing down"; teardown; exit 130; }
trap teardown EXIT
trap on_signal INT TERM
