# DPEP + LMCache P2P network contention (DeepSeek-V4-Flash)

Question: on two H200 nodes running DP attention + expert parallelism (EP all2all over NCCL) together with LMCache cross-node KV sharing (P2P over NIXL/UCX), does sharing the RoCE NICs make the two fight for bandwidth and hurt TTFT or per-user tok/s?

Answer (agentx, concurrency 200, 600 s, three runs): **no severe contention.** P2P on the same NIC as EP vs a separate NIC gives the same TTFT and tok/s within run-to-run noise. P2P bursts cause brief ECN marking on a shared NIC, without end-to-end impact. Numbers in `results/summary.txt`.

## Deployment

- Nodes yotta-h200-26 (head) and yotta-h200-16, GPUs 4 and 6 on each: DP4/EP4.
- vLLM flags aligned with perf-regression-testing (`config.toml [vllm]`), plus DSpark speculative decoding k=7.
- One LMCache MP server per node, coordinator on 26; P2P NIC chosen by `launch_lmcache.sh <rail>`.
- vllm-router consistent_hash (session-sticky via `X-Correlation-ID`).
- Workload: aiperf `inferencex-agentx-mvp` (weka cc-traces 256k), concurrency 200.

## Layout

| Path | What |
|---|---|
| `v2/common.sh` | paths, ports, NIC mapping |
| `v2/launch_lmcache.sh <rail>` | LMCache server (+ coordinator on 26), P2P bound to the given rail |
| `v2/launch_vllm.sh` | vLLM DP4/EP4; NCCL uses the GPUs' rails |
| `v2/launch_router.sh` | vllm-router |
| `v2/restart_all.sh <rail>` | restart everything and wait for health |
| `v2/run_exp.sh <label>` | one run: per-second NIC sampling on both nodes + aiperf + summary |
| `v2/rail_sampler.py` | per-second tx/rx, ECN, CNP, PFC pauses, drops, retx, P2P chunks |
| `v2/summarize.py` | TTFT, per-user tok/s, total tok/s, per-node token sources |
| `v2/decode_probe.py` | controlled decode probe (used while debugging the slow c32 config) |
| `v1/` | first-iteration scripts (old compilation config, superseded) |
| `results/` | summaries and per-second NIC CSVs of the three runs |

## Reproduce

```bash
# v2/ must exist on both nodes (on 16 at /raid/bo/dpep/v2)
cd v2
./launch_router.sh                 # on 26, once
./restart_all.sh 7                 # group A: P2P shares rail7 with EP
./run_exp.sh A_rail7_c200
# stop vLLM / LMCache, then
./restart_all.sh 1                 # group B: P2P alone on rail1
./run_exp.sh B_rail1_c200
```

Runs: `c200` (A, rail7), `A2_rail7_c200` (A repeat), `B_rail1_c200` (B, rail1).
