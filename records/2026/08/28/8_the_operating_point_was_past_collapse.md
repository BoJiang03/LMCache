# The operating point was past congestion collapse

R1-R4 all ran at `CONC=32`, which record 5 calibrated by driving GPU KV pool
occupancy into record 4's 0.7-0.8 target band. The question that broke it was
simple: nobody waits five minutes for an agent's first token, so why is TTFT
p50 265-304 seconds?

## 1. The check

TTFT by lane count, and by prompt size within each:

| lanes | TTFT p50 | p90 | by ISL bucket, TTFT p50 |
|---|---|---|---|
| 10 | **1.2 s** | 4.2 s | <50k=0.6s · 50-150k=1.0s · 150-400k=1.5s · >400k=4.3s |
| 32 | 154.7 s | 552.8 s | <50k=**287.5s** · 50-150k=**8.2s** · 150-400k=158.0s · >400k=478.6s |
| 64 | 290.0 s | 670.9 s | <50k=634.8s · 50-150k=288.2s · 150-400k=287.0s · >400k=447.5s |

At 10 lanes TTFT rises monotonically with prompt size, which is what prefill
does. At 32 it does not: a sub-50k prompt takes 287 s while a 50-150k prompt
takes 8.2 s. **Small requests 35x slower than large ones is not prefill, it is
a queue** -- they are stuck behind big ones.

Throughput says the same thing louder:

| lanes | req/min | input tok/s | TTFT p50 |
|---|---|---|---|
| **10** | **5.72** | **15 726** | 1.2 s |
| 32 | 2.70 | 7 355 | 154.7 s |
| 64 | 2.59 | 6 448 | 290.0 s |

R1-R3's six arms all landed at 6 283-6 698 input tok/s. **Raising offered load
3.2x cut throughput by 58 % and TTFT by 250x.** That is congestion collapse,
and every measured round sits past it.

The cause is record 4's target. With a mean context of 165 k tokens and a
186.6 GiB pool, roughly 12 requests fit at once; filling the pool to 74 %
necessarily carries a queue ~10 deep, and that is already over the knee.
**Pool occupancy is the wrong control variable for this workload.**

## 2. The corpus already knows the right answer

`traces.jsonl` is captured Claude Code traffic, and 56 432 of its 98 827
requests are streaming and carry the TTFT they actually experienced in
production, alongside `api_time` for all of them:

| input tokens | n | TTFT p50 | p90 | p99 | api_time p50 | p90 |
|---|---|---|---|---|---|---|
| <50k | 1 915 | 1.80 s | 4.59 s | 10.70 s | 5.91 s | 19.87 s |
| 50-150k | 14 549 | 2.16 s | 5.34 s | 11.53 s | 5.02 s | 20.34 s |
| 150-400k | 22 785 | 2.43 s | 6.87 s | 18.96 s | 7.48 s | 30.70 s |
| 400k-1M | 17 183 | 3.12 s | 9.23 s | 31.09 s | 10.43 s | 37.35 s |
| **all** | **56 432** | **2.64 s** | **6.98 s** | 22.90 s | **6.65 s** | 26.32 s |

So a reasonable TTFT for this workload is **p50 2.64 s, p90 6.98 s**, and it is
**nearly flat in context size** -- 1.80 s to 3.12 s across a 20x range of
prompt lengths, which is production's prefix caching working. This is not a
number anyone has to choose; it is recorded in the workload.

Against it:

| | TTFT p50 | p90 | latency p50 | p90 | shape |
|---|---|---|---|---|---|
| **production** | 2.64 s | 6.98 s | 6.65 s | 26.32 s | flat, monotone |
| ours, 10 lanes | 1.15 s | 4.18 s | **6.44 s** | 38.81 s | monotone |
| ours, 32 lanes | 154.74 s | 552.79 s | 296.93 s | 772.83 s | non-monotone |

**At 10 lanes end-to-end latency p50 is 6.44 s against production's 6.65 s** --
within 3 %, from Qwen3-Coder-30B-A3B on two H200s against whatever served the
original traffic. The serving configuration was never the problem. The load
was.

TTFT, not end-to-end latency, is the control target: TTFT is queue plus
prefill and is what the lane count moves, while end-to-end also carries decode
speed, which is a property of the model we deliberately did not change. By
that measure 10 lanes is *under*-loaded (1.15 s against 2.64 s), so the knee
is above 10 and below 32.

## 3. What survives from R1-R4

**Does not survive**: everything latency. A 2-5 % median TTFT improvement
measured on a 300-second, queue-dominated baseline says nothing about what a
user experiences. Record 6 section 2 has to be re-measured, not reinterpreted.

**Probably survives, but must be re-measured**: the store-side results. L1
churn is driven by request volume, not by GPU pressure -- at 10 lanes an eager
arm still writes ~1.4 TiB (94 requests x 165 k tokens x 98 304 B) into a
32-160 G tier. The mechanism behind `5.83x / 1.88x / 1.54x` less written and
`0.00 % -> 4.05 %` external hit rate does not depend on the queue.

**Survives outright**: the smoke gates, the pool sizing, the YaRN override,
the TP=2 double-counting fix, and the corpus fix from record 4 (ISL mean
53.8 k -> 165 k).

## 4. R4, and cross-round reproducibility

R4 ran before this was found, at `CONC=32`, so it carries the same caveat.
`off` versus lazy at 96 G: the `off` arm reached `kv_mean=74.8%`,
`running_mean=7.83`, `preempt_events=0`; lazy `kv_mean=76.1%`,
`running_mean=7.68`, `preempt_events=16`, `tokens_stored=10 998 656`,
`retrieves=14`, `l1_gib=75.71`.

Its lazy arm repeats R2's 96 G point on the same slot, which is the
cross-round check record 4 asked for:

| | R2 lazy | R4 lazy | spread |
|---|---|---|---|
| tokens_stored | 10.65 M | 11.00 M | 3.3 % |
| l1_gib | 73.68 | 75.71 | 2.8 % |
| kv_mean | 77.6 % | 76.1 % | 1.5 pt |
| preempted | 19 | 16 | 16 % |
| **retrieves** | **22** | **14** | **36 %** |

Volume metrics reproduce across rounds to within a few percent. **Retrieve
counts do not** -- 22 against 14 on identical settings. Record 6's
`10 -> 22 -> 34` trend spans more than that noise band, but no single round's
retrieve count should be read as a measurement.

## 5. The gate

`ttft.py` now runs on every arm and prints the production reference beside the
measurement, plus the by-ISL profile and an explicit verdict:

```
ttft_p50=1.15s (prod 2.64s)  ttft_p90=4.18s (prod 6.98s)  n=94
lat_p50=6.44s (prod 6.65s)  lat_p90=38.81s (prod 26.32s)
ttft_by_isl (ours/prod): <50k=0.6/1.8  50-150k=1.0/2.2  150-400k=1.5/2.4  400k-1M=4.3/3.1
ttft_shape=monotone-in-ISL (prefill-bound)
```

versus the 32-lane arm:

```
ttft_shape=NON-MONOTONE -- queue-bound, load is past the knee
```

No round gets reported again without this line.

## 6. Next

`calib2.sh` at 14 and 20 lanes, 900 s, targeting production TTFT rather than
pool occupancy. Then the L1 sweep (32 / 96 / 160 G, eager vs lazy paired)
re-run at whichever lane count lands closest to p50 2.64 s / p90 6.98 s with a
monotone profile.

Expect the re-run to show far less GPU-side eviction pressure -- at 10 lanes
the pool sat at 13 % -- so the lazy policy will be deferring nearly everything
and the story will be almost entirely L1-side. That is the honest version of
the result, and it is the one worth having.
