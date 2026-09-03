#!/usr/bin/env bash
# chain23 -- the cProfile diff that loss #1's fix is waiting on.
#
#   nostore returns exec/cpu/blocked all three to baseline, but it only skips
#   ~0.48 ms/step of instrumented in-hook work (_create_key 0.246, event export
#   0.040, MQ send ~0.19) while the worker main thread's CPU falls 3.86 ms/step
#   (+3.29 -> -0.57).  ~3.4 ms/step of host CPU is burned inside execute_model
#   and outside every hook we timed.  It is thread_time, so it is really
#   consumed -- not GPU contention (would land in `blocked`) and not core
#   starvation (160 cores, load 11).  cProfile around a matched step window in
#   both arms names the function.
#
#   Window 2400:3000 (600 steps) out of 7200: steady state, well past ramp,
#   well before drain.  ~55 s of profiled wall per worker per arm.  All 8
#   workers dump their own pstats keyed by pid; the diff aggregates all 8 so a
#   per-rank asymmetry (the allreduce-straggler shape) cannot hide in rank 0.
#
#   Everything else is phase1's exact protocol -- the same FULL knobs that
#   produced 85.34 / 91.04 -- so the profiled runs sit in the same table.
#   cProfile inflates both arms symmetrically; only the DIFFERENCE is read.
set -uo pipefail
cd /home/bo/vast_profiling_problem
exec 2>&1

TMP=/home/bo/.claude/jobs/ba4f4ca8/tmp
OUT=/home/bo/vast_profiling_problem/results/prof_tp8
mkdir -p "$OUT"

# wait for chain22's arms to clear
while pgrep -u bo -f "bash scripts/lane[.]sh" >/dev/null; do sleep 15; done
sleep 20

rm -f "$TMP"/pmp.*.pstats "$TMP"/pns.*.pstats

# ---- ownership-aware neighbour watch ------------------------------------
# The chain21/22 watchdog flagged on free-memory and utilisation alone, so it
# marked OUR OWN arms DIRTY the moment the workers allocated.  This one asks
# who the compute processes actually are: a pid is ours iff it shares this
# script's session id.  Anything else on GPUs 4-7 is a real neighbour.
MYSID=$(ps -o sid= -p $$ | tr -d ' ')
( while :; do
    foreign=""
    while read -r pid; do
      [ -z "$pid" ] && continue
      sid=$(ps -o sid= -p "$pid" 2>/dev/null | tr -d ' ')
      [ "$sid" = "$MYSID" ] || foreign="$foreign $pid($(ps -o user=,comm= -p "$pid" 2>/dev/null | tr -s ' ' | tr ' ' '/'))"
    done < <(nvidia-smi --query-compute-apps=pid --format=csv,noheader 2>/dev/null | tr -d ' ')
    [ -n "$foreign" ] && echo "$(date +%H:%M:%S) FOREIGN:$foreign"
    sleep 10
  done ) > "$OUT/neighbour_watch.txt" 2>&1 &
WATCH=$!
trap 'kill $WATCH 2>/dev/null' EXIT

run() {
  local arm="$1" tag="$2" pfx="$3"
  echo "######## CHAIN23: starting $tag ($arm) at $(date +%H:%M:%S)"
  env GPUS=0,1,2,3,4,5,6,7 TP=8 GPUMEM= MNBT= NUM_BLOCKS= PROBE=1 \
      NPROMPTS=1000 CONC=1000 LANE_OUT="$OUT" \
      STEPPROBE_CPROFILE=2400:3000 \
      STEPPROBE_CPROFILE_OUT="$TMP/$pfx" \
      ARM="$arm" TAG="$tag" bash scripts/lane.sh 2>&1 | tee "logs/lane_$tag.out"
  echo "######## CHAIN23: $tag exited at $(date +%H:%M:%S)"
  echo "  pstats written: $(ls -1 "$TMP/$pfx".*.pstats 2>/dev/null | wc -l) (expect 8)"
}

run mp      prof_mp      pmp
run nostore prof_nostore pns

kill $WATCH 2>/dev/null
echo "######## CHAIN23: foreign compute processes seen during the run:"
sort -u "$OUT/neighbour_watch.txt" | head -20
[ -s "$OUT/neighbour_watch.txt" ] || echo "  none -- GPUs were ours for the whole run"

echo "######## CHAIN23: profile diff"
/home/bo/vast_profiling_problem/.venv/bin/python scripts/prof_diff.py \
    --a "$TMP/pns" --b "$TMP/pmp" --a-name nostore --b-name mp --steps 600 --workers 8 \
    2>&1 | tee "$OUT/prof_diff.txt"
echo "######## CHAIN23: batch done $(date +%H:%M:%S)"
