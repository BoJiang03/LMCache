#!/usr/bin/env bash
# Start coordinator + LMCache (both nodes, P2P on <rail>) + vLLM (both nodes) and
# wait until both API servers are healthy.  Everything must be stopped first;
# the router (stateless) is left running.  Previous logs move to logs/v2/<ts>/.
#   ./restart_all.sh <p2p_rail>          # on node 26
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
RAIL=${1:?p2p rail}
R16=/raid/bo/dpep/v2

if pgrep -u bo -f "^$VENV/bin/python $VENV/bin/(vllm serve|lmcache)" >/dev/null ||
   ssh yotta-h200-16 "pgrep -u bo -f '^$VENV/bin/python $VENV/bin/(vllm serve|lmcache)'" >/dev/null; then
  echo "vLLM/LMCache still running; stop them first" >&2; exit 1
fi

TS=$(date +%Y%m%d_%H%M%S)
for h in local yotta-h200-16; do
  cmd="mkdir -p $LOGDIR/$TS && find $LOGDIR -maxdepth 1 -name '*.log' ! -name 'router_*' -exec mv {} $LOGDIR/$TS/ \;"
  if [ $h = local ]; then bash -c "$cmd"; else ssh $h "$cmd"; fi
done

./launch_lmcache.sh "$RAIL"
ssh yotta-h200-16 "cd $R16 && ./launch_lmcache.sh $RAIL"
until grep -q "Registered instance node-16" "$LOGDIR/coordinator.log"; do sleep 3; done
./launch_vllm.sh
ssh yotta-h200-16 "cd $R16 && ./launch_vllm.sh"

for i in $(seq 1 120); do
  a=$(curl -s -o /dev/null -w '%{http_code}' localhost:$VLLM_PORT/health || true)
  b=$(ssh yotta-h200-16 "curl -s -o /dev/null -w '%{http_code}' localhost:$VLLM_PORT/health" || true)
  [ "$a" = 200 ] && [ "$b" = 200 ] && { echo "ready after $((i * 15))s (p2p rail$RAIL)"; exit 0; }
  sleep 15
done
echo "not ready after 30 min" >&2; exit 1
