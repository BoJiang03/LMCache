# The miss is three populations, not one

Retracts record 1 and the quantitative half of 2026/08/28/15, then decomposes
where the recompute in `n60floor` actually comes from. Everything here is the
profiling window only, split out with `phase.py` or from aiperf's own
`server_metrics_export.json`, which carries a `warmup_metrics` block so every
Prometheus counter can be differenced across the phase boundary.

## 1. There is no stall

Record 1 measured 48-52 % of every profiling window as stalled, using
`prompt throughput < 100 tok/s AND generation < 10 tok/s AND Running > 0`.
The detector is wrong, not the engine.

`output_processor.py:628` flips `req_state.is_prefilling` to False on the first
output a request produces. `loggers.py:145` accumulates
`iteration_stats.prompt_token_stats.computed`, and `stats.py:365` only calls
`update_from_output` when `is_prefilling` is true. A chunked prefill emits no
output until it finishes, so the whole prompt is credited to the single
interval where the first token appears. Eighty seconds of `pre: 0.0` followed
by `pre: 82667.0` is deferred credit, not a stall and then a burst.

GPU telemetry bucketed by the same detector, both ranks, `n60floor`:

```
            util     power   mem activity   temp
"stalled"  100.0 %   690 W      16.3 %     64.3 C
normal      98.7 %   654 W      59.6 %     60.4 C
```

High compute with low memory traffic is prefill's GEMMs; the reverse is decode.
Power and die temperature are both higher inside the "stall". The engine was
never idle.

Consequences. Record 2026/08/28/15's decode-at-23.7 %-of-bandwidth and
prefill-at-37-62 %-of-compute both used wall clock that includes queueing as
the denominator; the GPU is at 92-100 % throughout. Both are withdrawn.

## 2. The tier ledger closes

`vllm:prompt_tokens_by_source` is a three-way split that sums exactly to
`vllm:prompt_tokens`, credited once per request. It is the only cache
accounting in this stack that balances, and it is what the rest of this work
uses. `vllm:prefix_cache_queries` does not: it reads 7.5e9 against a run that
presented 3.8e7 tokens, so the local hit *rate* derived from it is meaningless.
`ext_hit_mean` was retired earlier for averaging a rolling gauge across warmup.

`n60floor`, profiling window, end minus warmup:

```
presented                 25,833,741
  external (LMCache)      13,278,288   51.4 %
  local (GPU prefix)       3,523,840   13.6 %
  recomputed               9,031,613   35.0 %
```

## 3. Hits are all-or-nothing

93 retrieve operations (186 log lines, TP=2 duplicates) served 142 requests.
Retrieved p50 is 103,168 tokens against an ISL p50 of 107,246: when a lookup
hits it covers essentially the whole prompt. Prefix truncation is not the
failure mode. Roughly 49 requests get nothing at all, and 49 x mean ISL is
approximately the 9.03M recomputed.

## 4. Which 49, and why

Retrieves were matched back to requests by timestamp and size, and each request
paired with its own predecessor turn. The clock that matters is not the client
gap but the gap from the predecessor's completion to this request being
*scheduled*, since that is when the lookup happens (record 2026/08/27/7 found
the same for L1 residence; it applies to the GPU free queue too).

```
reuse clock       pred in warmup      pred in profiling        all
0-120 s            10 /  0 /  0 %      32 / 18 / 56 %     42 / 18 / 43 %
120-300 s           6 /  4 / 67 %      10 /  8 / 80 %     16 / 12 / 75 %
300-600 s           9 /  8 / 89 %      16 / 15 / 94 %     25 / 23 / 92 %
600-1200 s          2 /  1 / 50 %      32 / 26 / 81 %     34 / 27 / 79 %
>=1200 s           12 /  3 / 25 %       1 /  0 /  0 %     13 /  3 / 23 %
```

Three populations, two of them fixable and one not.

**Store had not landed yet (clock < 120 s).** 42 requests, 24 missed. Worst for
warmup predecessors: 10 of 10. Section 5 has the number.

**L1 evicted it (clock >= 1200 s, predecessor in warmup).** 13 requests, 10
missed. This retracts "L1 never filled, capacity and lookup are fine" from
2026/08/28/15. `server.log` carries 12 `L1 memory usage 0.8x above watermark
0.80; triggering eviction` lines, 7 of them during warmup. `config.py:576`
sizes the pool as `int(l1_size_gb * (1 << 30))`, so `--l1-size-gb 576` is 576
GiB and `eviction_controller.py:168` evicts above `used/total >= 0.80`, i.e.
460.8 GiB = 5,033,165 tokens at 96 KiB each. Warmup alone stored 11,269,376
tokens, 1.8x the pool. At the warmup store rate of 5,931 tok/s an entry lives
5,033,165 / 5,931 = 849 s, which is why conversations first scheduled 20 or
more minutes after their warmup turn find nothing.

**No predecessor in the run at all.** 12 requests, 10 missed. Twelve
conversations never appeared in warmup; their first turn has to be computed.

The middle band, 120-600 s, hits at 86 %. The system works there.

## 5. Deferral is the mechanism on the short end

`emitted_deferral_drains / emitted` = 2,106,569 / 953 = 2,211 drains, and
`drain_steps / span` = 20,642 / 2,320 = 8.9 drains/s, so the mean store lands
**248 s** after admission. Dropped operations waited 191 s before losing their
blocks. Against that, the client-side inter-turn gap has p50 2.6 s.

The 248 s is not a block sitting idle in the free queue. `admit()` stamps
`admitted_at_drain` while the request is still running, so most of it is the
request's own lifetime. Free-queue residency itself is much shorter: with 126,954
blocks total, `1 - kv_cache_usage` giving 18,698 free blocks at p50, and
consumption of 586 blocks/s, a block survives 32 s at p50, 77 s at p10 usage and
5 s at p90.

## 6. Why the rate model never sees it

`_rate_depth()` computes `max(blocks_per_step_ema, next_step_estimate) *
horizon_steps` and returns depth 0 below half a block. Per-step allocation is
extremely bimodal: `vllm:iteration_tokens_total` has p50 9.65 tokens and p99
18.5 tokens per step, 20,409 of 20,705 steps at or below 16 tokens, but 120
steps above 16,384 tokens and a request admission takes about 10,875 blocks.
With `_EMA_ALPHA = 0.3` a burst decays by 0.7 per step and reaches the
half-block floor in about 27 steps, roughly 3 s, while bursts recur every 70 to
170 steps. For most steps the policy computes that nothing is at risk and emits
nothing; when the burst arrives the free-queue head jumps past thousands of
blocks in one step and everything it passes is lost unseen. The
`danger_floor_max_blocks` docstring already names this failure.

Both safety nets were inactive. `announce_hits` was false (`announced_bursts`
0 in both phases), and it would not have helped: records 2026/08/27/2 sections
15 and 16 scored announce-then-admit and falsified 4 of 5 predictions, with
`announced_bursts` exactly equal to `retrieves` in both rounds because only hit
admissions announce. Cold chunked prefill never announces, and cold admissions
are 564k blocks of the allocation here. The danger floor raised once, to the
recent peak *step* allocation of 512-1024 blocks, an order of magnitude below
an admission burst.
