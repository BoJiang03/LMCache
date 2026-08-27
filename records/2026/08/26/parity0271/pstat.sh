#!/bin/bash
# One-line-per-run status for the in-flight parity runs.
# Signals, in the order that actually discriminates:
#   ALIVE   pid still there
#   CPU     %cpu of the parent (0.0 + no child = suspect)
#   KIDS    child processes -- the `--role baseline` subprocess. Its
#           presence is what tells "waiting on a child" apart from "wedged";
#           this is the check that was skipped on 2026-08-26 (records 7_ §四.4).
#   GPU     MiB this pid holds, i.e. the engine really came up
#   LOG     bytes + age; 0 bytes is NORMAL for the first ~15 min
#   OUT     result json present = finished
SP=/tmp/claude-1016/-home-bo-LMCache-worktrees-multi-modal/911c8e4e-a468-4726-ba44-ae873957a060/scratchpad
D=$SP/parity0271
now=$(date +%s)
printf '== %s ==\n' "$(date +%H:%M:%S)"
for spec in "qwen2vl2b_c:parity_qwen2-vl-2b_r2.json" "gemma4e4b_full:parity_gemma-4-e4b.json"; do
  tag=${spec%%:*}; outj=${spec#*:}
  pid=$(pgrep -f "benchmark_parity.py .*$( [ "$tag" = qwen2vl2b_c ] && echo Qwen2-VL-2B || echo gemma-4-E4B )" | head -1)
  log=$D/$tag.log
  if [ -n "$pid" ]; then
    read -r et st cpu rss <<<"$(ps -o etime=,stat=,%cpu=,rss= -p "$pid" | tr -s ' ')"
    kids=$(ps --ppid "$pid" -o args= --no-headers 2>/dev/null | grep -c .)
    role=$(ps --ppid "$pid" -o args= --no-headers 2>/dev/null | grep -o '\-\-role [a-z]*' | tr '\n' ',')
    gpu=$(nvidia-smi --query-compute-apps=pid,used_memory --format=csv,noheader 2>/dev/null | awk -F', ' -v p="$pid" '$1==p{print $2}')
    printf '%-16s pid %-8s %-8s %s cpu=%-6s rss=%-7s kids=%s %s gpu=%s\n' \
      "$tag" "$pid" "$et" "$st" "$cpu" "$((rss/1024))M" "$kids" "${role:-—}" "${gpu:-none}"
  else
    printf '%-16s pid GONE\n' "$tag"
  fi
  if [ -f "$log" ]; then
    age=$(( now - $(stat -c %Y "$log") ))
    printf '%-16s log %s bytes, %ss old  |  bar: %s\n' "" "$(stat -c %s "$log")" "$age" \
      "$(tail -c 4000 "$log" 2>/dev/null | tr '\r' '\n' | grep -aoE '[A-Za-z][A-Za-z ]*: *[0-9]+%\|[^|]*\| *[0-9]+/[0-9]+' | tail -1)"
    printf '%-16s     last non-bar: %s\n' "" "$(tail -c 6000 "$log" 2>/dev/null | tr '\r' '\n' | grep -av '%|' | grep -av '^$' | tail -1 | cut -c1-110)"
  fi
  exit0=1; for f in "$outj" "${outj%.json}.baseline.json" "${outj%.json}.answers.json"; do
    [ -f "$D/$f" ] && printf '%-16s OUT %s (%s bytes)\n' "" "$f" "$(stat -c %s "$D/$f")"
  done
done
exit 0
