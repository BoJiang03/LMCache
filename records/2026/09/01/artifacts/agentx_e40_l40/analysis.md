# Agentx CONC=40 pair rerun on shipped PR defaults (2026-09-01)

Purpose: the PR body's CONC sweep was measured with the 08/29 recipe keys,
including lazy_offload_danger_floor_max_blocks=8192, which the shipped PR no
longer has. This pair reruns CONC=40 on the PR code (lazy_offloading_policy_pr
@ f2d0ab5f, the 1675-insertion flattened drain) with only shipped keys:
EVICTION_AWARE, horizon_steps 2.5, max_deferral_seconds 30. Everything else
matches the original sweep: Trinity-Large-Thinking-FP8-Block TP=4 GPUs 1-4,
fp8 KV, max len 262144, util 0.90, fresh MP server (250 GB L1, chunk 256,
LRU) and fresh engine per arm, aiperf inferencex-agentx-mvp on
semianalysis-cc-traces-weka-062126-256k, seed 1234, 1800 s + 600 s grace.

Code provenance: engine PYTHONPATH pointed at the PR worktree with the
sitecustomize import guard; worker log line numbers match the PR tree
(vllm_multi_process_adapter.py:1339 = "Registering kv caches"), and the lazy
arm logged the three-field LazyOffloadPolicyConfig at startup.

## aiperf (client view)

| metric | e40 eager | l40 lazy | lazy delta |
|---|---|---|---|
| TTFT avg (ms) | 5597 | 5391 | -3.7% |
| TTFT p50 | 2797 | 2549 | -8.8% |
| TTFT p90 | 14537 | 14057 | -3.3% |
| TTFT p99 | 28589 | 25340 | -11.4% |
| e2e avg (ms) | 70835 | 71362 | +0.7% |
| e2e p50 | 29984 | 31071 | +3.6% |
| e2e p99 | 716038 | 673703 | -5.9% |
| ITL avg (ms) | 62.5 | 63.5 | +1.6% |
| ITL p99 | 194.4 | 192.5 | -1.0% |
| output tok/s (total) | 233.9 | 227.2 | -2.9% |
| tok/s/user avg | 22.6 | 22.1 | -2.2% |
| requests completed | 517 | 509 | -1.5% |

## Cache side

Prompt token sources (vllm, whole arm): e40 external 20.37M (32.7%) +
local_cache 19.58M (31.4%) + compute 22.36M (35.9%) of 62.31M; l40 external
19.87M (32.5%) + local_cache 18.65M (30.5%) + compute 22.69M (37.1%) of
61.21M. Combined hit share 64.1% vs 62.9%.

L1 (MP counters): writes 709,792 vs 346,919 chunks (lazy writes 49% of
eager's volume), reads 1,249,209 vs 981,484, end usage 0.77 vs 0.65 of
250 GB (no L1 eviction pressure at this point).

Ledger (l40, closes: 2723+1115+4+9 = 3851): admitted=3851 emitted=2723
emitted_overdue=1824 dropped_evicted=1115 rejected_unhashed=107
rejected_prefix_broken=1102 dropped_on_request_drop=4 pending=9.
67% of emissions came from the 30 s deadline, in line with the 57-77%
range the original sweep reported; 29% of admitted ops were dropped to
eviction before coming due.

## Reading

At CONC=40 on shipped defaults the pair is close to a wash with a real TTFT
tail win: TTFT improves at every percentile (avg -3.7%, p99 -11.4%) while
throughput gives up 2-3% and e2e is flat. The half-volume L1 write stream is
the mechanism on both sides: less store traffic competing with loads
(TTFT), but at this concurrency L1 is not under eviction pressure, so the
saved writes buy no hit-rate and the dropped 29% of ops show up as slightly
lower external hit share.

The original table's stronger CONC=40 row (per-user +1.8%, ITL p99 -25%)
was measured with the danger floor holding 8192 blocks of headroom, i.e.
storing much earlier than shipped defaults do. That config no longer
exists; this pair is what the PR as shipped delivers at CONC=40. The
original sweep's larger wins were at CONC=48-72 (L1 churn territory),
which this rerun does not cover.
