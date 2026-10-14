"""Per-second network sampler for the rails of the GPUs in use (+ an extra rail).

Each row: per rail, tx/rx Gbps and the per-second increase of the RoCE
congestion counters, plus this node's cumulative LMCache P2P load chunks.

  ecn       np_ecn_marked_roce_packets   packets that arrived ECN-marked (switch queue building)
  cnp       rp_cnp_handled               CNPs received -> this sender was rate-limited
  pause_tx  tx_pause_ctrl_phy            PFC pauses we sent (our receive side could not keep up)
  pause_rx  rx_pause_ctrl_phy            PFC pauses we received (we were told to stop sending)
  drop      rx_discards_phy + out_of_buffer
  retx      roce_adp_retrans + local_ack_timeout_err

  python3 rail_sampler.py <out.csv> <rail> [<rail> ...]     # runs until killed
"""
import subprocess, sys, time, urllib.request

HW = {"ecn": ["np_ecn_marked_roce_packets"], "cnp": ["rp_cnp_handled"],
      "drop": ["out_of_buffer"], "retx": ["roce_adp_retrans", "local_ack_timeout_err"]}
ETH = {"pause_tx": ["tx_pause_ctrl_phy"], "pause_rx": ["rx_pause_ctrl_phy"], "drop": ["rx_discards_phy"]}
LM_METRICS = "http://localhost:8180/metrics"


def ib_dev(netdev):
    out = subprocess.run(["bash", "-c", f"ls -d /sys/class/infiniband/*/device/net/{netdev}"],
                         capture_output=True, text=True).stdout.strip()
    return out.split("/")[4]


def read_rail(netdev, dev):
    port = f"/sys/class/infiniband/{dev}/ports/1"
    v = {"tx_bytes": int(open(f"{port}/counters/port_xmit_data").read()) * 4,
         "rx_bytes": int(open(f"{port}/counters/port_rcv_data").read()) * 4}
    for key, names in HW.items():
        v[key] = v.get(key, 0) + sum(int(open(f"{port}/hw_counters/{n}").read()) for n in names)
    eth = {}
    for line in subprocess.run(["ethtool", "-S", netdev], capture_output=True, text=True).stdout.splitlines():
        k, _, val = line.strip().partition(": ")
        eth[k] = val
    for key, names in ETH.items():
        v[key] = v.get(key, 0) + sum(int(eth.get(n, 0)) for n in names)
    return v


def p2p_chunks():
    try:
        for line in urllib.request.urlopen(LM_METRICS, timeout=1).read().decode().splitlines():
            if line.startswith("lmcache_mp_l2_prefetch_load_completed_chunks_total"):
                return line.split()[-1]
    except Exception:
        pass
    return ""


def main():
    out, rails = sys.argv[1], [f"rail{r}" for r in sys.argv[2:]]
    devs = {r: ib_dev(r) for r in rails}
    fields = ["tx_gbps", "rx_gbps", "ecn", "cnp", "pause_tx", "pause_rx", "drop", "retx"]
    with open(out, "w") as f:
        f.write("ts," + ",".join(f"{r}_{x}" for r in rails for x in fields) + ",p2p_load_chunks\n")
        prev = {r: read_rail(r, devs[r]) for r in rails}
        t_prev = time.time()
        while True:
            time.sleep(1 - (time.time() % 1))
            t = time.time(); dt = t - t_prev; t_prev = t
            row = [str(int(t))]
            for r in rails:
                cur = read_rail(r, devs[r]); p = prev[r]; prev[r] = cur
                row += [f"{(cur['tx_bytes'] - p['tx_bytes']) * 8 / dt / 1e9:.2f}",
                        f"{(cur['rx_bytes'] - p['rx_bytes']) * 8 / dt / 1e9:.2f}"]
                row += [str(cur[k] - p[k]) for k in fields[2:]]
            row.append(p2p_chunks())
            f.write(",".join(row) + "\n"); f.flush()


main()
