# Eager vs FIFO vs eviction-aware

This supplement compares the three store behaviors on production commit
`8e4e851f91316bb7994be3d096966f0d1ef0b52b` using Qwen3-8B on NVIDIA H200s.
It is retained on the reproduction branch and does not enter the production PR
diff.

`FIFO` below means the shipped legacy configuration, not a specially tuned
variant:

```text
lmcache.mp.lazy_offload = true
lmcache.mp.lazy_offload_policy = FIFO
lmcache.mp.lazy_offload_threshold = 100       # default
lmcache.mp.lazy_offload_select_count = 10     # default
```

## What the policies decide

- **Eager** submits every eligible store immediately.
- **FIFO** waits until 100 finished requests are queued, then submits the oldest
  ten. It does not inspect GPU eviction pressure while waiting.
- **Eviction-aware** observes the GPU free queue and submits a prefix near its
  last safe store opportunity.

All policies retain the same snapshot validation guard. If FIFO waits until a
block has been evicted/reallocated, the guard drops the stale operation and
logs `Block hashes missing or mismatched` rather than storing incorrect KV.

## GSM8K retrieval correctness (TP=1)

The same 120 fixed questions were run cold and then cached at concurrency four.

| policy | cached strict | cached coverage | cached time | L1 objects | stale-snapshot warnings |
|---|---:|---:|---:|---:|---:|
| eager | 0.908 | 0.961 | 21.72s | 1454 | 0 |
| FIFO (100/10) | 0.908 | 0.000 | 28.19s | 0 | 140 |
| eviction-aware | 0.908 | 0.961 | 22.33s | 1452 | 0 |

FIFO preserves answer correctness because stale operations are safely rejected,
but it never creates a usable lower-tier copy in this run. Its cached pass is
therefore another cold pass. Eager and eviction-aware both retrieve about 96.1%
of queried tokens and retain the same cached strict score.

## Hot/cold performance (TP=1)

The workload uses a 20 GiB GPU KV pool and 40 GiB L1 with three GPU-resident hot
documents, eleven rotating cold documents, 14 warmups, and 120 measured
requests. FIFO and eviction-aware have two retained runs; eager's representative
row is within its retained 41--43 second range.

| policy | wall time | total coverage | external hit rate | L1 eviction cycles | hot TTFT p50 | cold TTFT p50 | stale warnings |
|---|---:|---:|---:|---:|---:|---:|---:|
| eager | 43.12s | 0.725 | 0.000 | 14 | 130ms | 829ms | 0 |
| FIFO (100/10) | 41.86s, 42.02s | 0.725 | 0.057, 0.029 | 0 | 130ms | 785ms | 15, 15 |
| eviction-aware | 27.06s, 30.29s | 0.955, 0.911 | 0.838, 0.677 | 3, 5 | 156--163ms | 261--280ms | 0 |

Default FIFO is only marginally faster than eager because it submits very few
stores. Its zero L1 eviction cycles are not an efficiency win: it leaves only
234 objects in L1 and does not improve total coverage over vLLM APC. The stale
snapshot guard rejects the old operations after the 100-request delay.

Eviction-aware preserves the GPU hot set (`APC=0.725`), raises median total
coverage to `0.933`, and reduces median wall time to `28.68s`. Its median wall
time is about **31.6% lower than default FIFO**, while cold TTFT falls from about
`785ms` to `270ms`.

### Tuned FIFO control

To avoid attributing the result only to FIFO's large default threshold, FIFO was
also run with `threshold=10, select_count=10`:

| policy | median wall | median coverage | median L1 evictions | median hot TTFT | median cold TTFT | stale warnings |
|---|---:|---:|---:|---:|---:|---:|
| FIFO (10/10) | 33.11s | 0.715 | 3 | 157ms | 555ms | 14--17 |
| eviction-aware | 28.68s | 0.933 | 4 | 160ms | 270ms | 0 |

A smaller FIFO threshold helps, but remains about **15.5% slower** than
eviction-aware and still drains requests according to count rather than actual
block lifetime. Its coverage is both lower and less stable across repetitions.

## Hot/cold performance (TP=4)

The same logical workload was repeated with four H200s and
`tensor_parallel_size=4`. Each policy has two retained runs.

| policy | median wall | median coverage | L1 eviction cycles | hot TTFT p50 | cold TTFT p50 | connector-specific warnings |
|---|---:|---:|---:|---:|---:|---:|
| eager | 23.55s | 0.725 | 14 | 118ms | 422ms | 0 |
| FIFO (100/10) | 21.81s | 0.725 | 0 | 114ms | 369ms | 15/run |
| eviction-aware | 19.22s | 0.948 | 2--3 | 161ms | 204ms | 0 |

Eviction-aware is **11.9% faster than default FIFO** at TP=4 and provides much
higher coverage. FIFO's apparent hot-latency advantage comes from doing almost
no useful lower-tier work; it stores only 936 rank-sharded objects versus about
3,214 for eviction-aware.

All TP modes also emit the same environment-level vLLM FlashInfer/all-reduce
tuning warnings; those are excluded from the connector-specific warning column.
There are no tracebacks or failed stores in the retained runs.

## Conclusion

Default FIFO behaves as count-triggered delayed batching. On these workloads it
usually waits past the blocks' safe lifetime, so snapshot validation correctly
drops the stores. Lowering the threshold turns it into an approximate periodic
batcher but does not tell it which prefixes are actually near eviction.

Eviction-aware is the only policy of the three that simultaneously:

1. preserves GPU APC for the hot set;
2. creates useful lower-tier copies for the cold set;
3. avoids stale-snapshot warnings; and
4. improves total wall time consistently at TP=1 and TP=4.

Raw JSON and FIFO warning counts are in
[`results/policy_comparison/`](results/policy_comparison/).
