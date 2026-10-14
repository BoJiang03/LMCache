"""Summary of one aiperf agentx run (profiling phase).

  python3 summarize.py <label> [<label> ...]      # labels under /raid/bo/dpep/aiperf

Per run: TTFT, per-user decode tok/s (output tokens / decode time), total
output tok/s, uncached prompt tokens per request, and per
node (from aiperf's server-metrics export over the run): requests finished and
prompt tokens by source (local_compute / local_cache_hit / external_kv_transfer).
"""
import json, sys
from collections import defaultdict

ROOT = "/raid/bo/dpep/aiperf"


def pct(xs, p):
    xs = sorted(xs)
    return xs[min(int(p * (len(xs) - 1)), len(xs) - 1)] if xs else float("nan")


def summarize(label):
    ttft, user_tps, uncached = [], [], []
    out_tokens, t_lo, t_hi = 0, float("inf"), 0
    for line in open(f"{ROOT}/{label}/aiperf/profile_export.jsonl"):
        rec = json.loads(line)
        if rec["metadata"]["benchmark_phase"] != "profiling":
            continue
        m = rec["metrics"]
        if "time_to_first_token" in m:
            ttft.append(m["time_to_first_token"]["value"])
        osl = m.get("output_sequence_length", {}).get("value", 0)
        out_tokens += osl
        t_lo = min(t_lo, rec["metadata"]["request_start_ns"]); t_hi = max(t_hi, rec["metadata"]["request_end_ns"])
        if osl > 1 and m.get("decode_duration", {}).get("value"):
            user_tps.append((osl - 1) / (m["decode_duration"]["value"] / 1000))
        uncached.append(m["input_sequence_length"]["value"]
                        - m.get("usage_prompt_cache_read_tokens", {}).get("value", 0))

    per_node = defaultdict(float)
    metrics = json.load(open(f"{ROOT}/{label}/aiperf/server_metrics_export.json"))["metrics"]
    for name in ("vllm:request_success", "vllm:prompt_tokens_by_source"):
        for s in metrics.get(name, {}).get("series", []):
            host = s["endpoint_url"].split("//")[1].split(":")[0]
            if host == "127.0.0.1":  # the router's /metrics, proxied from one node
                continue
            per_node[(host, s["labels"].get("source", "requests"))] += s["stats"].get("total") or 0

    print(f"== {label}: {len(ttft)} requests")
    print(f"  TTFT ms   p50 {pct(ttft, .5):8.0f}  p90 {pct(ttft, .9):8.0f}  p99 {pct(ttft, .99):8.0f}")
    # slow users are the low tail, so report p50 / p10 / p1
    print(f"  tok/s/user p50 {pct(user_tps, .5):7.1f}  p10 {pct(user_tps, .1):7.1f}  p1  {pct(user_tps, .01):7.1f}")
    print(f"  output tok/s total {out_tokens / ((t_hi - t_lo) / 1e9):.0f}")
    print(f"  uncached  p50 {pct(uncached, .5):8.0f}  p90 {pct(uncached, .9):8.0f}  max {max(uncached):8.0f}")
    for host in sorted({h for h, _ in per_node}):
        row = {src: int(v) for (h, src), v in per_node.items() if h == host}
        print(f"  {host}: requests {row.pop('requests', 0)}  " + "  ".join(f"{k} {v}" for k, v in sorted(row.items())))


for label in sys.argv[1:]:
    summarize(label)
