#!/bin/bash
# Event stream for the in-flight full-MME parity runs. v4.
#
# Phase machine, read out of benchmark_parity.py main() (0.27.1 tree):
#   A   configure_environment() + benchmark.load_items()   log 0 B, no GPU, no child
#   B   :958 "[parity] N MME questions loaded"             first bytes in the log
#   C1  :964 spawn --role baseline; the CHILD repeats      child present, holds NO GPU
#       load_items() at :803 before building its LLM
#   C2  child's engine up, baseline generation             child holds GPU
#   D   child exits; parent starts MP server + engine      parent holds GPU
#   E   report written
#
# Liveness is a CPU-TICK DELTA from /proc/<pid>/stat, never `ps -o %cpu`:
# that column is CPU-time/elapsed averaged over the whole lifetime, so a
# parent blocked in subprocess.run still reads 55-100% (measured 100 -> 71
# -> 55 purely by sitting in wait()), and a wedged process keeps reading
# high for a long time. In phase C the work is in the CHILD -- sample that.
#
# Progress comes from vLLM's tqdm bars, which the log carries because the
# child inherits our fds. One event per NEW bar label (tells us which pass
# we are in), plus a 30-minute heartbeat with the current count so an ETA
# is computable. Bar labels seen are persisted, so a restart of this
# watcher does not replay them.
set -u
SP=/tmp/claude-1016/-home-bo-LMCache-worktrees-multi-modal/911c8e4e-a468-4726-ba44-ae873957a060/scratchpad
D=$SP/parity0271
STATE=$D/.pwatch_state
# vLLM prints "Free memory on device (139.29/139.8 GiB) on startup" on EVERY
# healthy start, so only the "... is less than desired ..." form is a failure.
# A false alarm here is worse than a miss: it teaches you to ignore the stream.
FAIL='Traceback \(most recent call last\)|CUDA out of memory|OutOfMemoryError|Cannot reach the LMCache MP server|baseline subprocess failed|AssertionError|Killed|on startup is less than desired'
BAR='[A-Za-z][A-Za-z ]*: *[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+'
RUNS="qwen2vl2b_c|Qwen2-VL-2B|parity_qwen2-vl-2b_r2.json gemma4e4b_full|gemma-4-E4B|parity_gemma-4-e4b.json"

gpu_of() { nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null \
             | awk -F', ' -v p="$1" '$1==p{print $2; exit}'; }
ticks()  { awk '{print $14+$15}' /proc/"$1"/stat 2>/dev/null || echo 0; }
lastbar() { tail -c 4000 "$1" 2>/dev/null | tr '\r' '\n' | grep -aoE "$BAR" | tail -1; }

declare -A PH OFF LASTSZ LASTCH STALL FIN BEAT SEEN
for spec in $RUNS; do t=${spec%%|*}
  PH[$t]=A; OFF[$t]=0; LASTSZ[$t]=-1; LASTCH[$t]=$(date +%s); STALL[$t]=0; FIN[$t]=""
  BEAT[$t]=0; SEEN[$t]=""
done
if [ -f "$STATE" ]; then
  while IFS='=' read -r k v; do
    case "$k" in *.bars) t=${k%.bars}; [ -n "${SEEN[$t]:-}" ] || [ -n "${PH[$t]:-}" ] && SEEN[$t]=$v ;;
                 *) [ -n "${PH[$k]:-}" ] && PH[$k]=$v ;;
    esac
  done < "$STATE"
fi
save() { : > "$STATE"; for spec in $RUNS; do t=${spec%%|*}
  printf '%s=%s\n%s.bars=%s\n' "$t" "${PH[$t]}" "$t" "${SEEN[$t]}" >> "$STATE"; done; }
say() { printf '%s %s\n' "$(date +%H:%M:%S)" "$*"; save; }

while :; do
  for spec in $RUNS; do
    IFS='|' read -r tag pat outj <<<"$spec"
    [ -n "${FIN[$tag]}" ] && continue
    log=$D/$tag.log; out=$D/$outj
    pid=$(pgrep -f "benchmark_parit[y].py .*$pat" 2>/dev/null | head -1)
    kid=""
    [ -n "$pid" ] && kid=$(ps --ppid "$pid" -o pid=,args= --no-headers 2>/dev/null \
                            | grep -- '--role baseline' | awk '{print $1}' | head -1)

    if [ -f "$out" ]; then
      say "[$tag] DONE report written"
      python3 - "$out" <<'PY' 2>&1 | sed "s/^/           /"
import json,sys
r=json.load(open(sys.argv[1])); g=r.get("gate",r)
for k in ["pass","deployment_path","num_questions","flips_pass1_vs_baseline",
          "flips_pass2_vs_pass1","score_baseline","score_pass1","score_pass2",
          "score_delta_pass1","score_delta_pass2","pass2_hit_coverage",
          "pass2_lookup_hit_ratio","baseline_answer_parse_ratio",
          "cache_granularity_tokens","failures"]:
    for src in (g,r):
        if k in src: print(f"{k} = {src[k]}"); break
PY
      FIN[$tag]=done; continue
    fi
    if [ -z "$pid" ]; then
      say "[$tag] GONE process vanished with no report (phase ${PH[$tag]}). log tail:"
      tail -c 1500 "$log" 2>/dev/null | tr '\r' '\n' | tail -14 | sed "s/^/           /"
      FIN[$tag]=gone; continue
    fi

    sz=$(stat -c %s "$log" 2>/dev/null || echo 0)
    if [ "${PH[$tag]}" = A ] && [ "$sz" -gt 0 ]; then
      PH[$tag]=B; say "[$tag] A->B dataset loaded, elapsed $(ps -o etime= -p "$pid" | tr -d ' ')"
    fi
    if [ -n "$kid" ]; then
      case "${PH[$tag]}" in
        A|B) PH[$tag]=C1; say "[$tag] ->C1 baseline child $kid up; it reloads the dataset before building its engine" ;;
        C1)  g=$(gpu_of "$kid"); [ -n "$g" ] && { PH[$tag]=C2; say "[$tag] C1->C2 child $kid engine up, gpu=$g"; } ;;
      esac
    else
      case "${PH[$tag]}" in
        C1|C2) PH[$tag]=D; say "[$tag] ->D baseline child exited; parent gpu=$(gpu_of "$pid")" ;;
      esac
    fi

    if [ "$sz" -gt "${OFF[$tag]}" ]; then
      chunk=$(tail -c +$(( ${OFF[$tag]} + 1 )) "$log" 2>/dev/null | tr '\r' '\n')
      printf '%s' "$chunk" | grep -aE "$FAIL" | head -4 \
        | while read -r l; do say "[$tag] !! ${l:0:170}"; done
      for lbl in $(printf '%s' "$chunk" | grep -aoE "$BAR" | sed 's/:.*//' | tr ' ' '_' | sort -u); do
        case ",${SEEN[$tag]}," in
          *",$lbl,"*) ;;
          *) SEEN[$tag]="${SEEN[$tag]:+${SEEN[$tag]},}$lbl"
             say "[$tag] bar '${lbl//_/ }' started -- $(lastbar "$log")" ;;
        esac
      done
      OFF[$tag]=$sz
    fi

    if [ "$sz" -ne "${LASTSZ[$tag]}" ]; then LASTSZ[$tag]=$sz; LASTCH[$tag]=$(date +%s); STALL[$tag]=0; fi
    now=$(date +%s)
    if [ $(( now - ${BEAT[$tag]} )) -ge 1800 ]; then
      BEAT[$tag]=$now
      [ "${PH[$tag]}" != A ] && say "[$tag] .. ${PH[$tag]}, elapsed $(ps -o etime= -p "$pid" | tr -d ' '), $(lastbar "$log")"
    fi
    quiet=$(( now - ${LASTCH[$tag]} ))
    if [ "$quiet" -gt 900 ] && [ "${STALL[$tag]}" -lt 1 ]; then
      work=${kid:-$pid}
      t0=$(ticks "$work"); sleep 4; t1=$(ticks "$work")
      if [ "$(( t1 - t0 ))" -lt 20 ]; then
        STALL[$tag]=1
        say "[$tag] ?? quiet ${quiet}s in ${PH[$tag]}, pid $work burned $(( t1 - t0 )) ticks in 4s (idle), gpu=$(gpu_of "$work") -- suspect, not proof"
      fi
    fi
  done
  alldone=1; for spec in $RUNS; do t=${spec%%|*}; [ -n "${FIN[$t]}" ] || alldone=0; done
  [ "$alldone" = 1 ] && { say "both runs terminal"; exit 0; }
  sleep 60
done
