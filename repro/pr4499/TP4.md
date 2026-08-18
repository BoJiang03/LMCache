# Tensor-parallel validation (TP=4)

This is a one-off four-GPU validation supplement for PR #4499. It is kept on
the reproduction branch and does not change the production PR diff.

## Environment and invariant under test

- Production commit: `8e4e851f91316bb7994be3d096966f0d1ef0b52b`
- Model: `Qwen/Qwen3-8B`
- GPU: 4 x NVIDIA H200 connected by NVLink
- vLLM: `tensor_parallel_size=4`, world size 4
- Lazy configuration: production default `EVICTION_AWARE`, horizon `2.5`

TP=4 increases the worker-receipt aggregation fan-in: all four MP worker ranks
must store and retrieve their own KV shard before the scheduler may interpret a
store as complete and release its pins. Every retained run starts four LMCache
worker adapters, registers four GPU KV caches, and cleanly unregisters all four.

## GSM8K retrieval correctness

The same 120 fixed questions were run cold and then cached at concurrency four.

| mode | cold strict | cached strict | cached coverage | cached time | cached TTFT p50 |
|---|---:|---:|---:|---:|---:|
| eager | 0.908 | 0.925 | 0.961 | 13.88s | 83ms |
| lazy | 0.925 | 0.925 | 0.961 | 14.03s | 93ms |

Cached strict score and coverage match exactly. L1 object counts scale with the
number of independently stored rank shards: the lazy cached runs retain about
`1452` objects at TP=1, `2904` at TP=2, and `5808` at TP=4. This linear scaling,
together with successful external retrieval, rules out a vacuous run in which
only one TP rank stores data.

The lazy ledger closes with `admitted=186`, `emitted=179`,
`dropped_evicted=6`, and `pending=1`. It reports zero failed stores,
request-drop losses, unhashed/prefix-broken rejections, and preemptions.

## Hot/cold performance

The single-GPU workload geometry was preserved: 20 GiB logical GPU KV pool,
40 GiB L1, three hot documents, eleven cold documents, 14 warmups, and 120
measured requests. Each mode was run twice on the same four GPUs.

| mode | wall-time runs | median wall | coverage runs | L1 eviction cycles | hot TTFT p50 median | cold TTFT p50 median |
|---|---:|---:|---:|---:|---:|---:|
| eager | 23.68s, 23.42s | 23.55s | 0.725, 0.725 | 14, 14 | 118ms | 422ms |
| lazy | 19.55s, 18.88s | 19.22s | 0.949, 0.947 | 3, 2 | 161ms | 204ms |

Median wall time decreases by **18.4%** (`1.23x` throughput-equivalent
speedup). The result is positive in both individual A/B repetitions. Lazy
trades about 43ms of hot TTFT for roughly 218ms lower median cold TTFT, while
reducing L1 eviction cycles from 14 to 2--3.

Both lazy performance runs report:

- four LMCache MP worker adapters and four registered/unregistered KV caches;
- no failed store, request-drop loss, unhashed/prefix-broken rejection, or
  preemption;
- no traceback;
- a closed policy ledger, including the explicitly retained pending gauge at
  shutdown.

The only warnings are vLLM all-reduce tuning warnings caused by FlashInfer not
being installed. They occur in eager and lazy TP runs and are not connector
failures.

## Reproduce

```bash
export HF_HUB_CACHE=/path/to/non-home/huggingface/hub
export SMOKE_GPU=0,1,2,3
export SMOKE_TP=4
export SMOKE_HORIZON=2.5
export REPETITIONS=2
./repro/pr4499/run_gsm8k.sh
./repro/pr4499/run_hot_cold.sh
```

Raw JSON and rank-registration evidence are in
[`results/tp4/`](results/tp4/).
