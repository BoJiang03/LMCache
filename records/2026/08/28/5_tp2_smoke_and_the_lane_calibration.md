# TP=2 smoke, and what a lane is worth

Record 4 settled the configuration on paper. This is the first round that ran
it. Two things came out: the configuration works, and the load setting in
record 4 was wrong by a factor of three.

## 1. The gates

Nothing proceeds until store and retrieve are known to work at TP=2 and the
lazy ledger is known to balance. Two probes, run on separate slots.

**Gate A -- store and retrieve, eager arm, GPU6+7.** One 14 009-token request,
then `POST /reset_prefix_cache` to destroy the GPU copy, then the same request
again:

```
before          stores=0 retrieves=0
req1 http=200
after_store     stores=4 retrieves=0      Stored 8192 + 5632  (per rank)
reset http=200
req2 http=200
after_retrieve  stores=4 retrieves=2      Retrieved 13824
l1_objects=108 l1_gib=1.266
```

8192 + 5632 = 13 824 stored, 13 824 retrieved. Nothing lost.

**Gate B -- ledger balance under load, lazy arm, GPU4+5, L1=96 G, 900 s.**

```
admitted=873  emitted=215  dropped_evicted=2  pending=656
215 + 2 + 656 = 873
```

Errors: `cudaMemcpy failed`, `AcceleratorError`, `cudaErrorInvalidValue`,
`Traceback` -- all zero on both server and engine.

**`preempt_events=0`.** Every round to date ran with a 24 GiB pool and logged
preemption constantly; record 4 diagnosed that as oversubscription rather than
a policy effect. At the real pool it is gone, which confirms the diagnosis.

## 2. The configuration, as measured rather than predicted

| quantity | record 4 predicted | measured |
|---|---|---|
| `max_seq_len` | 1 048 576 | 1 048 576 |
| KV pool | 1.99 M tokens / 181.8 GiB | **2 038 512 tokens / 186.6 GiB** |
| ISL mean | 270 k | **165 k** |
| ISL p50 | 202 k | **119 k** |
| ISL p90 | 625 k | **642 k** |

One correction to record 4's serving command. It gives `--rope-scaling`; on
this stack that flag does nothing, because transformers 5.15 has already
folded `config.json`'s (null) `rope_scaling` into `rope_parameters` by the
time vLLM patches it (`vllm/transformers_utils/config.py:492`, the v5 branch).
The override has to name `rope_parameters`:

```
--hf-overrides '{"rope_parameters":{"rope_type":"yarn","factor":4.0,
                 "original_max_position_embeddings":262144,
                 "rope_theta":10000000.0}}'
```

`hf_overrides_kw` is applied before `patch_rope_parameters`, so this lands.
Confirmed by the engine's own `max_seq_len=1048576`.

The ISL numbers are lower than record 4's corpus measurement but four times
i60N's (`isl_mean=53802 isl_p50=56733`). The corpus fix is real; record 4's
own figures were the corpus distribution, and what a 32-lane sample of it
actually draws is somewhat smaller.

## 3. A lane is not an in-flight request

`--concurrency 10`, which record 4 proposed as the starting point, produced
`inflight_mean=1.66` and 13 % pool occupancy. The reason is in the profiling
phase's dispatch:

```
PROFILING execute: resuming 10 trajectory sessions (spread; first-request spread 102607.8s)
...
sending complete | sent=94, completed=92 | sessions: sent=6, completed=2
```

Lanes are staggered across the trace timeline, and the spread here is 28.5
hours. In a 900 s window only 6 of 10 lanes ever opened. Concurrency buys
*candidate* sessions; how many are live at once is an outcome, not a setting.

So it has to be calibrated. One round, two slots, 32 lanes against 64.

The 64-lane arm spent 32 minutes in warmup before it reached a profiling
window at all. That warmup is worth a note on its own: `sent=69,
completed=42, in_flight=27` for twenty minutes, held up by one straggler at a
time. Warmup wall time is set by the largest few requests, not by the lane
count.

## 4. The measurement the calibration needed

`inflight_mean` from `profile_export.jsonl` counts client-side in-flight,
which includes requests queued at the scheduler holding no KV blocks. At 32
lanes that reads 15.78 against an engine that was running 6-8. The quantity
the 0.7-0.8 oversubscription target is about is pool occupancy, so it gets
read off the engine directly, over the profiling window only (warmup runs
~16 min at these lane counts and would drag every mean down):

| lanes | running_mean | waiting_mean | kv_mean | kv_p50 | kv_p90 | kv_max | preempts |
|---|---|---|---|---|---|---|---|
| 10 | 1.7 | -- | 13% | -- | -- | -- | 0 |
| **32** | **7.32** | **7.85** | **72.6%** | **80.9%** | **97.3%** | **100.0%** | **9** |
| 64 | 10.80 | 21.14 | 81.0% | 83.6% | 95.3% | 99.6% | 13 |

**CONC=32**, held fixed for every round so the L1 sweep is the only thing
that moves. 72.6 % sits inside the 0.7-0.8 target band, and the second point
shows why going further is not worth it: doubling the lanes buys 8 points of
occupancy and 169 % more queue. The engine saturates at a running set of
roughly 8; past that, lanes only wait.

Nine preemption events in 900 s at a p90 of 97 % is the workload occasionally
filling the pool, which is what an operating point in that band means. It is
not the old 24 GiB-pool regime.

The 64-lane arm also stored 14.4 M tokens against 32's 7.9 M and retrieved
**nothing** (32 retrieved 607 k). More pressure, more offload, no reuse --
consistent with lanes past saturation adding queue rather than work.

Two counting traps found while building this:

- **At TP=2 the MP server logs each store and retrieve once per rank**, so raw
  token sums are double. `arm.sh` now reports `tokens_stored_raw` alongside
  `tokens_stored = raw / TP`; the per-rank figure is the one that compares to
  served input tokens.
- **`Prefix cache hit rate` read from the last stats line is often 0**,
  because the engine emits a final line while draining. Reported as max and
  last.

One process lesson, from a snapshot that came back with `kv_mean=0.0%`:
**bash re-reads a script from a byte offset while it runs**, so editing
`arm.sh` mid-arm gets a mix of old and new. The arm that reported it had
started before the fix. Freeze the harness before a round, not during.

## 5. The lazy arm at this operating point

`cal_c32_s1`, lazy, L1=96 G, 900 s:

```
stores=175  tokens_stored=7 897 472        retrieves=18  tokens_retrieved=607 232
l1_objects=5492  l1_gib=64.36  l1_watermark_events=49
ledger: admitted=1901 emitted=1488 dropped_evicted=103 dropped_on_request_drop=7 pending=303
store_tokens  n=175 mean=90 257 p50=61 696 p90=212 992 p99=524 288 max=524 288
store_secs    sum=62.7 mean=0.358 p50=0.110 p90=1.023 p99=3.405 max=3.424
```

1488 + 103 + 7 + 303 = 1901. Balances.

Two things to carry into the sweep. L1 crosses its 0.80 eviction watermark 49
times inside 900 s at a 96 G budget, so the 32 G arm will be evicting almost
continuously -- which is what that point is for. And a single store is up to
524 288 tokens and 3.4 s; deferral is not free to get wrong at this size.

## 6. What the sweep can actually afford

Request volume is the binding constraint on latency statistics. At CONC=32 an
arm serves ~67 requests per 900 s, because the contexts are enormous. At
`DUR=1800` that is ~130 per arm, and paired medD only uses turns that appear
in both arms. Volume metrics -- `tokens_stored`, retrieves, `l1_gib`, ledger,
occupancy -- are unaffected; medD will be reported with its `n` and will be
thin. Extending `DUR` is the only lever, at ~30 min per arm per doubling.

Node 1's memory is the other constraint, and it moved since record 4. Another
user holds ~420 GB there (two `lmcache` processes at 210 GB RSS each), leaving
146 GB free with both 96 G arms up. **R3 as designed -- 2 x 256 G = 512 GB --
does not fit.** R1 (64 GB) and R2 (192 GB) do, and run first; R3 gets decided
against the box's actual state afterwards, with serial arms or a lower top of
the sweep as the options.

## 7. Running

```
R1  eager (GPU4+5) vs lazy (GPU6+7)   L1=32 G   CONC=32 DUR=1800 GRACE=600 SEED=1234
R2  same pair                          L1=96 G   same
```

Policy stays pinned to its slot across both rounds so the L1 trend within a
policy is read on one set of GPUs; the slots were measured equivalent in i60L
(92 ms spread between identical arms). R4 will put lazy on the other slot as
the check on that.
