#!/usr/bin/env bash
# Per-second tx/rx Gbps on the rails of the GPUs in use, plus this node's
# cumulative LMCache P2P load chunks.  Runs until killed.
#   ./rail_sampler.sh <out.csv>      # start on each node
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
OUT=${1:?out csv}
RAILS=($(gpu_rails))
DEVS=($(for r in "${RAILS[@]}"; do rail_dev "$r"; done))

hdr="ts"; for r in "${RAILS[@]}"; do hdr+=",rail${r}_tx_gbps,rail${r}_rx_gbps"; done
echo "$hdr,p2p_load_chunks" > "$OUT"
read_ctr() { cat /sys/class/infiniband/$1/ports/1/counters/port_$2_data; }
declare -A prev
for d in "${DEVS[@]}"; do prev[$d.tx]=$(read_ctr $d xmit); prev[$d.rx]=$(read_ctr $d rcv); done
t0=$(date +%s.%N)
while sleep 1; do
  t1=$(date +%s.%N); dt=$(awk -v a="$t0" -v b="$t1" 'BEGIN{print b-a}'); t0=$t1
  line=$(date +%s)
  for d in "${DEVS[@]}"; do
    tx=$(read_ctr $d xmit); rx=$(read_ctr $d rcv)
    # counters are in 4-byte words
    line+=$(awk -v t="$tx" -v pt="${prev[$d.tx]}" -v r="$rx" -v pr="${prev[$d.rx]}" -v dt="$dt" \
      'BEGIN{printf ",%.2f,%.2f", (t-pt)*32/dt/1e9, (r-pr)*32/dt/1e9}')
    prev[$d.tx]=$tx; prev[$d.rx]=$rx
  done
  c=$(curl -s -m 1 localhost:$LM_HTTP_PORT/metrics | awk '/^lmcache_mp_l2_prefetch_load_completed_chunks_total/ {print $2}' || true)
  echo "$line,${c:-}" >> "$OUT"
done
