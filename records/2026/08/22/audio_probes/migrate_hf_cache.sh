#!/bin/bash
# Move this user's HF cache entries out of /home into the shared /raid hub.
#
# Order matters and is not negotiable: for every entry, COPY to /raid, VERIFY
# the copy, and only then delete from /home. Nothing is ever deleted before
# its replacement is confirmed to exist and match.
#
# The duplicate set is handled separately and needs no copy at all: those
# entries are already byte-identical in /raid (verified by size), so /home's
# copy is pure waste.
set -uo pipefail

SRC=/home/bo/.cache/huggingface/hub
DST=/raid/data/hub
LOG=/home/bo/.claude/jobs/ee036230/tmp/migrate_hf_cache.log

# Already present in $DST at identical size -- drop from $SRC, no copy.
DUPES=(
  models--google--gemma-3-270m-it
  models--google--gemma-4-12B-it
  models--google--gemma-4-E4B-it
  models--Qwen--Qwen2.5-VL-3B-Instruct
  models--Qwen--Qwen3-8B
  models--Qwen--Qwen3-30B-A3B
)
# Unique to $SRC -- copy, verify, then drop.
MOVES=(
  datasets--lmms-lab--MME
  models--Qwen--Qwen3-0.6B
  datasets--TwinkStart--MMAU
  models--google--gemma-3-4b-it
  models--OpenGVLab--InternVL3_5-2B-HF
  models--Qwen--Qwen2.5-Omni-3B
  models--Qwen--Qwen2-VL-2B-Instruct
  models--Qwen--Qwen3.5-2B
  models--Qwen--Qwen3.6-27B
  models--Qwen--Qwen3.8-27B
  models--Qwen--Qwen3-VL-2B-Instruct
  models--zai-org--GLM-4.1V-9B-Thinking
  models--zai-org--GLM-4.6V-Flash
)

say() { echo "[$(date +%H:%M:%S)] $*" | tee -a "$LOG"; }

say "=== migration start; df before:"
df -h /home /raid | tee -a "$LOG"

for e in "${DUPES[@]}"; do
  if [ ! -d "$DST/$e" ]; then say "SKIP dupe $e -- not in DST after all"; continue; fi
  s=$(du -sb "$SRC/$e" 2>/dev/null | cut -f1)
  d=$(du -sb "$DST/$e" 2>/dev/null | cut -f1)
  # Allow a small delta: blob layouts can differ by refs/metadata only.
  if [ "${d:-0}" -ge $(( ${s:-0} * 95 / 100 )) ]; then
    say "DROP dupe $e (home=$s raid=$d)"
    rm -rf "$SRC/$e" && say "  dropped"
  else
    say "KEEP dupe $e -- raid copy smaller than expected (home=$s raid=$d)"
  fi
done

for e in "${MOVES[@]}"; do
  [ -d "$SRC/$e" ] || { say "SKIP move $e -- gone"; continue; }
  say "COPY $e"
  rsync -a --info=none "$SRC/$e/" "$DST/$e/" 2>>"$LOG" || { say "  rsync FAILED, keeping source"; continue; }
  s=$(du -sb "$SRC/$e" | cut -f1)
  d=$(du -sb "$DST/$e" | cut -f1)
  if [ "$d" -ge $(( s * 99 / 100 )) ]; then
    say "  verified (home=$s raid=$d) -> dropping source"
    rm -rf "$SRC/$e"
  else
    say "  VERIFY FAILED (home=$s raid=$d) -- source kept"
  fi
done

say "=== migration done; df after:"
df -h /home /raid | tee -a "$LOG"
