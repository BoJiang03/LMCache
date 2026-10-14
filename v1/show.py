import json, sys
d = json.loads(open(sys.argv[1]).read())
r = d['rails']
print(d['label'], 'rail7 tx/rx', r['rail7']['avg_tx_gbps'], r['rail7']['avg_rx_gbps'], 'peak', r['rail7']['peak_tx_gbps_500ms'],
      '| rail1 tx', r['rail1']['avg_tx_gbps'], '| rail8 rx', r['rail8']['avg_rx_gbps'])
for k in ('itl_rank0_ms', 'itl_rank1_ms'):
    print(' ', k, {x: d[k][x] for x in ('p50', 'p90', 'p99', 'p99.9')})
print(' ', d.get('p2p_req'))
