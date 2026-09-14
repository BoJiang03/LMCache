#!/usr/bin/env bash
# EXPERIMENT N: is raising num_workers an adequate substitute for the read pool?
#
# THE OBJECTION THIS TESTS
# The PR says reads in flight are pinned at num_workers because
# ConnectorBase::choose_num_tiles returns min(worker_count_for_op(op), n) and
# the default do_batch_get runs a tile serially.  The obvious reply is: then
# raise num_workers.  My earlier ablation pass never tested that -- it compared
# the byte budget against NO bound at all, which is the wrong reference point,
# so its conclusion was worthless.  This is the right reference point.
#
# WHAT num_workers ACTUALLY COSTS, and why it is not a free dial
# For the fs connector num_workers is the ONLY lever: per_op_workers exists in
# WorkerPoolConfig but is wired into mooncake's pybind only, so fs cannot size
# reads separately from writes.  One number governs every op.  Hence part N2.
#
# PRE-COMMITTED PREDICTIONS
#  N1a  The best num_workers MOVES with object size: 1.5 MiB needs >= 256,
#       192 MiB is already at the array ceiling with 4.
#  N1b  ONE pooled config (workers=4, depth=64, budget=1536 MiB) lands within
#       5% of the best legacy config at EVERY size.
#       Falsifier: pooled more than 5% below best-legacy anywhere.
#  N1c  THE DECISIVE ONE.  No single num_workers is within 5% of the best at
#       every size.
#       FALSIFIER: if some single value (64 is the likely one) IS within 5%
#       everywhere, then for reads on this backend num_workers tuning is an
#       adequate substitute, the bandwidth argument for the new knob fails,
#       and the PR has to rest on N2 plus the networked-backend case instead.
#       I do not know which way this goes.  At 1.5 MiB the pool is capped by
#       depth=64, not by the budget, so legacy at workers=64 should tie it
#       there; the question is whether anything separates them at 192 MiB.
#  N2   Write bandwidth changes materially with num_workers, so a value picked
#       for reads is not free -- it is also the write path's setting.
#
# WHAT THE BANDWIDTH NUMBER CANNOT SHOW, stated up front so it is not smuggled
# in later: bytes in flight.  Legacy at workers=64 with 192 MiB objects has
# 64 x 192 MiB = 12 GiB of reads outstanding; the pooled arm has 1536 MiB by
# construction.  That is an 8x difference in committed host memory and device
# queue pressure at identical bandwidth.  qdepth and svc ms/req below are the
# observable shadow of it.
set -u

PY=${PY:-python3}
BENCH=$(dirname "$0")/benchN.py      # instrumented copy; benchB.py is in use
CORPUS=${WORK:?directory on the storage under test}/expN/corpus
TREE=${TREE:?LMCache checkout under test, built in place}
OUT=${WORK:?directory on the storage under test}/expN/numworkers.log

OUTSTANDING=3; PASSES=3; CORPUS_GIB=96
DEPTH=64; BUDGET=1536

# obj_kib:batch:workers_ladder   batch*OUTSTANDING caps reachable concurrency,
# so the ladder stops where the batch can no longer feed it.
CELLS="1536:2048:4,16,64,256 6144:512:4,16,64,256 24576:128:4,16,64,256 196608:32:4,16,64"

mkdir -p "$(dirname "$OUT")"
: > "$OUT"
{
  echo "EXPERIMENT N  num_workers vs read pool"
  echo "tree=$TREE commit=$(git -C "$TREE" rev-parse --short HEAD)"
  echo "corpus=${CORPUS_GIB}GiB outstanding=$OUTSTANDING passes=$PASSES"
  echo "pooled arm is ONE config at every size: workers=4 depth=$DEPTH budget=${BUDGET}MiB"
  echo "started $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)"
} >> "$OUT"

trap 'rm -rf "$CORPUS"' EXIT

run() {  # tag obj batch workers depth budget reuse passes
  echo "### $1 obj=$2 workers=$4 depth=$5" >> "$OUT"
  PYTHONPATH="$TREE" "$PY" "$BENCH" --base-path "$CORPUS" \
    --obj-kib "$2" --corpus-gib "$CORPUS_GIB" --workers "$4" \
    --batch "$3" --outstanding "$OUTSTANDING" --passes "$8" \
    --io-depth "$5" --budget-mib "$6" --keep-corpus $7 2>&1 \
    | grep -E "^pass|^write|qdepth=|budget=|Error|Traceback" >> "$OUT"
}

echo "" >> "$OUT"
echo "############ N1: read bandwidth vs num_workers ############" >> "$OUT"
for cell in $CELLS; do
  obj=${cell%%:*}; rest=${cell#*:}; batch=${rest%%:*}; ladder=${rest##*:}
  echo "======== object $obj KiB  batch=$batch  ladder=$ladder  load=$(cut -d' ' -f1 /proc/loadavg) ========" >> "$OUT"
  run warmup_discard "$obj" "$batch" 4 0 0 "" 1
  # Two full rounds so ordering bias falls on every arm, not just the first.
  for rep in 1 2; do
    for w in ${ladder//,/ }; do
      run "W${w}_r${rep}" "$obj" "$batch" "$w" 0 0 "--reuse-corpus" "$PASSES"
    done
    run "POOL_r${rep}" "$obj" "$batch" 4 "$DEPTH" "$BUDGET" "--reuse-corpus" "$PASSES"
  done
  rm -rf "$CORPUS"
done

echo "" >> "$OUT"
echo "############ N2: write bandwidth vs num_workers ############" >> "$OUT"
echo "# num_workers is one number for every op on fs.  A value chosen to get" >> "$OUT"
echo "# reads in flight is also the store path's concurrency." >> "$OUT"
for w in 4 16 64 256; do
  echo "======== write sweep workers=$w obj=6144KiB ========" >> "$OUT"
  run "WRITE_w${w}" 6144 512 "$w" 0 0 "" 1
  rm -rf "$CORPUS"
done

echo "finished $(date -Is) loadavg=$(cut -d' ' -f1-3 /proc/loadavg)" >> "$OUT"
