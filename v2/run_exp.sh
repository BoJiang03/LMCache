#!/usr/bin/env bash
# One contention experiment: agentx at the fixed concurrency (200, 600 s) with
# per-second network sampling (rails 1, 5, 7) on both nodes.
#   ./run_exp.sh <label> [concurrency] [duration_s]      # on node 26
# Output: /raid/bo/dpep/aiperf/<label>/ and /raid/bo/dpep/rails/<label>_{26,16}.csv
set -euo pipefail
cd "$(dirname "$0")"
source ./common.sh
LABEL=${1:?label}
CONC=${2:-200}
DURATION=${3:-600}
S=/raid/bo/dpep/rails
RAILS="1 5 7"
mkdir -p $S; ssh yotta-h200-16 "mkdir -p $S"

python3 rail_sampler.py $S/${LABEL}_26.csv $RAILS & P=$!
ssh yotta-h200-16 "cd /raid/bo/dpep/v2; nohup python3 rail_sampler.py $S/${LABEL}_16.csv $RAILS >/dev/null 2>&1 & echo \$! > $S/${LABEL}_16.pid"
trap 'kill $P 2>/dev/null; ssh yotta-h200-16 "kill \$(cat $S/${LABEL}_16.pid) 2>/dev/null"' EXIT

./run_aiperf.sh "$LABEL" "$CONC" "$DURATION" > /dev/null 2>&1 || echo "aiperf exit $?"
kill $P; ssh yotta-h200-16 "kill \$(cat $S/${LABEL}_16.pid)"
scp -q yotta-h200-16:$S/${LABEL}_16.csv $S/
python3 summarize.py "$LABEL"
