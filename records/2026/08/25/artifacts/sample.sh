#!/bin/bash
. "$(dirname "$0")/env.sh"
N=${1:-200}
echo "ts kv_usage running waiting_cap apc_q apc_h ext_q ext_h preempt l1_obj l1_gb"
for i in $(seq $N); do
  m=$(curl -s http://127.0.0.1:$VLLM_PORT/metrics)
  g(){ echo "$m" | grep -m1 "^$1{" | awk '{print $2}'; }
  l1=$(curl -s http://127.0.0.1:$HTTP_PORT/status | python3 -c "import json,sys;d=json.load(sys.stdin)['storage_manager']['l1_manager'];print(d['total_object_count'], round(d['memory_used_bytes']/1e9,2))" 2>/dev/null || echo "NA NA")
  echo "$(date +%H:%M:%S) $(g vllm:kv_cache_usage_perc) $(g vllm:num_requests_running) $(echo "$m"|grep -m1 'waiting_by_reason.*capacity'|awk '{print $2}') $(g vllm:prefix_cache_queries_total) $(g vllm:prefix_cache_hits_total) $(g vllm:external_prefix_cache_queries_total) $(g vllm:external_prefix_cache_hits_total) $(g vllm:num_preemptions_total) $l1"
  sleep 5
done
