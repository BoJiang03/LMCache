# Hot-TTFT attribution controls

This supplement tests whether eviction-aware scheduler bookkeeping itself
causes the TP=4 hot-TTFT increase reported in the constrained hot/cold workload.
All runs use production commit `8e4e851f91316bb7994be3d096966f0d1ef0b52b`,
Qwen3-8B, and four H200s.

## Result

It does not. The increase requires concurrent cold retrieval plus the original
40 GiB L1 pressure. Disabling emitted stores, removing hot/cold concurrency, or
providing enough L1 capacity brings hot TTFT back close to eager.

| control | concurrency | L1 | query behavior | hot TTFT p50 | cold TTFT p50 | query L1 evictions |
|---|---:|---:|---|---:|---:|---:|
| eager baseline | 2 | 40 GiB | cold recompute; no external hits | 118ms | 422ms | 14 |
| eviction-aware, no stores | 2 | 40 GiB | policy active; 32K minimum rejects every 20K prefix | 120ms | 395ms | 0 |
| eviction-aware, mostly store-only | 2 | 40 GiB | 30 unique cold documents; negligible retrieval | 132ms | 394ms | 8 |
| eager retrieval control | 2 | 64 GiB | all cold KV retained; query object count unchanged | 133ms | 196ms | 0 |
| eviction-aware, unconstrained L1 | 2 | 64 GiB | cold retrieval plus 624 new rank-sharded objects | 136ms | 194ms | 0 |
| eviction-aware, serialized | 1 | 40 GiB | original reuse stream, no hot/cold overlap | 125ms | 183ms | 0 |
| eviction-aware, measured workload | 2 | 40 GiB | retrieval + drain/store + L1 pressure | 161ms | 204ms | 2--3 |

The no-store control still runs the eviction-aware admission, free-queue
observation, snapshot validation, deduplication, and economy gate on every
scheduler step. Its hot p50 is `120ms`, versus `118ms` for eager. This bounds the
policy decision-loop cost to noise at this scale.

The serialized control is stronger: it retains real external retrieval and
lazy stores, but prevents a hot request from sharing an in-flight window with a
cold request. Hot p50 falls from `161ms` to `125ms`.

## Request-position evidence

The measured sequence is fixed:

```text
hot0, hot1, hot2, cold, ...
```

At concurrency two, `hot2` is submitted with `cold`; `hot0` follows the previous
cold request. Splitting TP=4 TTFT by sequence position across the two measured
runs gives:

| request position | eager / 40 GiB | eviction-aware / 40 GiB |
|---|---:|---:|
| hot immediately after prior cold | about 110ms | about 155ms |
| middle hot control | about 125ms | about 115ms |
| hot submitted with cold | about 111ms | about 184ms |
| cold | about 422ms | about 204ms |

The middle hot request does not regress. Only hot requests sharing or following
the cold transfer path move upward.

## Mechanism

A TP batch that contains a cold LMCache hit must wait for the cold request's
rank-sharded CPU-to-GPU KV load before the batch can produce first tokens. The
hot APC-hit request in that same scheduling window therefore observes the cold
connector wait even though its own KV never leaves GPU. With a 40 GiB L1 near
its eviction watermark, concurrent retrieval, lazy drains, and L1
allocation/eviction extend that interference into the following request pair.

This is not evidence that eviction-aware offloaded the hot prefix:

- vLLM APC remains `0.725` in eager and eviction-aware;
- the non-adjacent hot position remains at baseline;
- active eviction-aware policy with zero emitted stores remains at baseline;
- real eviction-aware retrieval/stores at concurrency one remain near baseline.

The trade-off belongs to connector transfer scheduling and workload concurrency,
not to the free-queue decision algorithm. Potential follow-up work is per-request
transfer QoS or scheduling foreground GPU-hit requests separately from batches
performing lower-tier loads.

Raw JSON is in
[`results/hot_ttft_attribution/`](results/hot_ttft_attribution/).
