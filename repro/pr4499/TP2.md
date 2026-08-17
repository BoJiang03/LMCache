# Tensor-parallel validation (TP=2)

This is a one-off two-GPU validation supplement for PR #4499. It is kept on
the reproduction branch and does not change the production PR diff.

## Environment and invariant under test

- Production commit: `8e4e851f91316bb7994be3d096966f0d1ef0b52b`
- Model: `Qwen/Qwen3-8B`
- GPU: 2 x NVIDIA H200
- vLLM: `tensor_parallel_size=2`, world size 2
- Lazy configuration: production default `EVICTION_AWARE`, horizon `2.5`

TP=2 exercises a different correctness boundary from single-GPU runs: both MP
worker ranks must store and retrieve their own KV shard, while the scheduler
must keep blocks pinned until completion receipts from all ranks have been
aggregated. The retained startup evidence shows two LMCache worker adapters,
two registered GPU KV caches, and two clean unregisters.

## GSM8K retrieval correctness

The same 120 fixed questions were run cold and then cached at concurrency four.

| mode | cold strict | cached strict | cached coverage | cached time | cached TTFT p50 |
|---|---:|---:|---:|---:|---:|
| eager | 0.933 | 0.933 | 0.961 | 16.46s | 88ms |
| lazy | 0.925 | 0.933 | 0.961 | 16.55s | 87ms |

The cached strict score and coverage are identical. The TP=2 cached run creates
approximately twice the L1 object count of TP=1 (`2904` lazy objects versus
`1452` in the retained TP=1 run), as expected for two independently stored KV
shards. Retrieval is therefore non-vacuous on both ranks rather than silently
using one shard.

The lazy ledger closes with `admitted=196`, `emitted=189`,
`dropped_evicted=4`, and `pending=3`. It reports zero failed stores,
request-drop losses, unhashed/prefix-broken rejections, and preemptions.

## Hot/cold performance

The TP=1 workload geometry was preserved: 20 GiB logical GPU KV pool, 40 GiB
L1, three hot documents, eleven cold documents, 14 warmups, and 120 measured
requests. Each mode was run twice on the same pair of GPUs.

| mode | wall-time runs | median wall | coverage runs | L1 eviction cycles | hot TTFT p50 median | cold TTFT p50 median |
|---|---:|---:|---:|---:|---:|---:|
| eager | 29.88s, 29.41s | 29.64s | 0.725, 0.725 | 14, 14 | 113ms | 551ms |
| lazy | 24.48s, 20.38s | 22.43s | 0.848, 0.957 | 7, 3 | 149ms | 333ms |

Median wall time decreases by **24.3%** (`1.32x` throughput-equivalent
speedup). The result is positive in both individual A/B repetitions. Lazy
trades about 36ms of hot TTFT for roughly 218ms lower median cold TTFT and
reduces L1 eviction pressure.

Both lazy performance runs report:

- two LMCache MP worker adapters and two registered/unregistered KV caches;
- no failed store, request-drop loss, unhashed/prefix-broken rejection, or
  preemption;
- no traceback;
- closed policy ledgers apart from the explicitly retained `pending` gauge at
  shutdown.

The only retained warnings are vLLM all-reduce tuning warnings caused by
FlashInfer not being installed; they occur in eager and lazy modes and are not
connector failures.

## Reproduce

The reproduction harness accepts `SMOKE_TP` and records it in every result:

```bash
export HF_HUB_CACHE=/path/to/non-home/huggingface/hub
export SMOKE_GPU=0,1
export SMOKE_TP=2
export SMOKE_HORIZON=2.5
export REPETITIONS=2
./repro/pr4499/run_gsm8k.sh
./repro/pr4499/run_hot_cold.sh
```

Raw JSON and rank-registration evidence are in
[`results/tp2/`](results/tp2/).
