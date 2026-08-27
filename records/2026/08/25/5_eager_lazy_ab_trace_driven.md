# Eager vs lazy A/B, scenario-native timing

Smoke-run A/B on the AgentX trace at the workload's own timing. Result is
negative for the branch in this configuration: lazy writes 2.4x more data and
TTFT is worse at every percentile.

## What was wrong with the first attempt

The first A/B used a hand-written probe: fixed concurrency, no arrival process,
then a replay of the same prompts in the same order with a ~0 s store->reuse
gap. It reported lazy favourably (phase B 29.5 s vs 96.7 s). That number should
be discarded. The defects, in order of severity:

1. **`--unsafe-override` suppressed the scenario's invariants.** The scenario
   `inferencex-agentx-mvp` declares `min_benchmark_duration_seconds=900`
   because it "requires duration >= 900s to reach steady state and trigger KV
   offloading" -- the exact mechanism under test. The run used 180 s.
2. **The timing was invented.** A policy whose whole decision is "will this
   block survive `horizon_steps`" was evaluated with no arrival process and a
   zero store-to-reuse interval.
3. **The reuse shape was wrong.** Real agentic reuse is a *growing prefix*
   within a session. The probe replayed whole duplicate prompts, which is a
   different access pattern -- and it landed on the LRU sequential-scan worst
   case, producing an eager 0% hit rate that is an artifact, not a property of
   eager.

The workload already carries its own timing; the scenario file states it:

```python
# aiperf/common/scenario/inferencex_agentx_mvp.py
timing_mode=TimingMode.AGENTIC_REPLAY,
forbid_ignore_trace_delays=True,
min_benchmark_duration_seconds=900,
default_benchmark_duration_seconds=1800,
system_idle_gap_cap_seconds=10.0,
forbid_trace_idle_gap_cap=True,
require_cache_bust=CacheBustTarget.FIRST_TURN_PREFIX,
```

`system_idle_gap_cap_seconds=10.0` removes dead air without distorting the
replay: when nothing is in flight, "all pending request timers shift earlier by
the same amount", so per-trace timing, timer order and relative spacing are
preserved. Two of my earlier flags were direct violations
(`--trace-idle-gap-cap-seconds`, and `--use-think-time-only` against
`forbid_ignore_trace_delays`), visible only because `--unsafe-override`
downgraded them to warnings.

One earlier attribution was also wrong: the 10,651.9 s "WARMUP spread" is not
caused by `--use-think-time-only` and is not the arrival rate. It is the
alignment of a phase's first requests on the trace timeline `t*`;
`--burst-phase-starts` collapses it, at the cost of being "throughput-oriented
rather than a faithful arrival replay".

## Configuration

Both arms identical except `lazy_offload`: GPU 2, TP=1, Qwen3-Coder-30B-A3B,
pool 16384 blocks (24 GiB), max model len 131072, L1 200 GB.

```
aiperf profile --scenario inferencex-agentx-mvp \
  --public-dataset semianalysis-cc-traces-weka-062126 \
  --max-context-length 100000 --num-dataset-entries 64 \
  --concurrency 8 --benchmark-duration 900 --benchmark-grace-period 120 \
  --random-seed 1234
```

No `--unsafe-override`; the run is scenario-valid. `--max-context-length` is
kept because the dataset's peak context reaches 996,579 tokens against a
131,072 window. Lazy policy: `horizon_steps=2.5, min_prefix_tokens=0,
max_drain_per_step=64`.

## The pair is comparable

| | eager | lazy | delta |
|---|---|---|---|
| requests | 249 | 247 | -0.8% |
| duration | 899.5 s | 900.6 s | +0.1% |
| total ISL | 13,427,915 | 13,282,900 | -1.1% |
| ISL p50 | 59,918 | 59,847 | -0.1% |
| theoretical prefix hit | 93.2% | 93.2% | 0.0% |
| effective concurrency | 2.37 | 2.42 | +2.3% |

Effective concurrency ~2.4 against `--concurrency 8`: the trace's arrival
process is the limit, not the concurrency cap. That is the real serving shape.

## Result

| metric | eager | lazy | delta |
|---|---|---|---|
| **TTFT avg** | 1012.8 ms | 1135.7 ms | **+12.1%** |
| **TTFT p50** | 581.7 ms | 605.1 ms | +4.0% |
| **TTFT p99** | 6598.7 ms | 7868.1 ms | **+19.2%** |
| request latency avg | 8588.4 ms | 8863.4 ms | +3.2% |
| request latency p50 | 4374.8 ms | 4555.4 ms | +4.1% |
| request latency p99 | 68031 ms | 67697 ms | -0.5% |
| request throughput | 0.2760/s | 0.2735/s | -0.9% |
| inter-token latency | 10.68 ms | 10.80 ms | +1.1% |
| **tokens stored** | 1,601,792 | 3,823,616 | **+138.7%** |
| store batches | 949 | 191 | -79.9% |
| tokens retrieved | 1,377,024 | 1,934,336 | +40.5% |
| **retrieved / stored** | **0.860** | **0.506** | **-41%** |
| L1 final | 146.65 GiB / 6257 obj | 139.36 GiB / 5946 obj | -5.0% |
| L1 watermark events | 0 | 0 | -- |
| errors (all patterns) | 0 | 0 | -- |

## Mechanism: lazy writes overlap data already resident

An earlier draft of this record claimed lazy "re-writes the whole prefix each
turn". That was wrong, read off the head of the batch-size distribution: the
first ~25 lazy stores are 57k-66k tokens, but those are cold-cache first-turn
stores, where a full-prefix write is correct. The *last* 25 are incremental
(256, 4096, 1792, 2304, 3840, ...). `PendingStoreOp` carries
`prefix_start_tokens` and `prefix_end_tokens`, and emission coalesces
contiguous pending ops into one store, so a large batch is by design one
transfer covering several not-yet-stored chunks -- not a re-send.

What the data does support: **lazy's writes overlap ranges already resident.**

| | eager | lazy |
|---|---|---|
| batches | 949 | 191 |
| tokens written | 1,601,792 | 3,823,616 |
| p50 batch | 256 | 7,168 |
| p90 / max batch | 8,192 / 10,752 | 60,672 / 92,928 |
| final L1 | 146.65 GiB | 139.36 GiB |
| L1 watermark events | 0 | 0 |

Lazy wrote 2.4x the tokens and finished with 5% *less* resident data, having
never crossed the L1 watermark. 3.82 M tokens is ~350 GiB at 96 KiB/token
against a 186 GiB L1, so the storage layer deduped by key: the excess writes
moved bytes that were already there. That transfer contends with prefill, which
is what the TTFT regression measures. `deduplicated=4` shows the policy itself
did not recognise the overlap.

Written volume by run quartile:

| quartile | eager | lazy |
|---|---|---|
| Q1 | 660,224 | 1,686,528 |
| Q2 | 364,288 | 1,271,040 |
| Q3 | 350,976 | 415,744 |
| Q4 | 226,048 | 441,088 |

Both arms front-load while the cache is cold, but lazy puts 77% of its volume
in the first half against eager's 64%: 1.94 M of the 2.22 M excess lands there.
At steady state (Q3+Q4) lazy still writes ~1.5x more (857 K vs 577 K).

## Root cause

The read path is fine, and that matters for stating the bug correctly. A
deferred offload means the blocks are still resident in the GPU pool, so a
follower turn over the same prefix hits vLLM APC and needs no load at all:
`needs_retrieve()` is `num_lmcache_hit_tokens > num_vllm_hit_tokens`
(`lmcache_mp_metadata.py:99`), which is false when APC already covers the
prefix. An earlier draft framed this as "the pending queue is invisible to
lookup", which misplaces the defect -- lookup should not be needed here.

The defect is on the **write** path, and it is an asymmetry in how a request
learns what is already safe in LMCache.

1. One tracker per request, `num_stored_tokens = 0`
   (`lmcache_mp_metadata.py:75-80`). Its own comment: "will be initialized
   when lookup the external hit tokens" -- the LMCache lookup result is the
   **only** cross-request source of "how much of this prefix is already
   stored".

2. `lmcache_mp_connector.py:806-816` -- on a lookup miss (`ret == 0`) the
   function returns before `increase_num_stored_tokens(ret)`, so
   `num_stored_tokens` stays 0.

3. An APC hit does not advance it. `num_vllm_hit_tokens` feeds only
   `computed_tokens = num_scheduled_tokens + max(num_vllm_hit_tokens,
   num_lmcache_hit_tokens)` (`lmcache_mp_metadata.py:216`), which raises the
   *upper bound* of what to store. So an APC hit is one-directional: **it
   enlarges the range to store and contributes nothing to the range
   considered already stored.**

4. `lmcache_mp_metadata.py:237-241` -- `start_token_idx =
   tracker.num_stored_tokens`. With a lookup miss and an APC hit, that is
   `start = 0` and an end near the full prompt: the follower stages
   `[0, full prefix)`.

Eager masks the asymmetry: the predecessor turn really did push to LMCache, so
the follower's lookup hits and `num_stored_tokens` jumps past the shared
prefix. Lazy still has the predecessor pending, so the lookup misses and the
follower starts from 0.

Re-staging is deliberate, and the guard for it is what fails. The comment at
`lmcache_mp_connector.py:791-796` keeps `num_vllm_hit_tokens` precisely so a
follower over a hot prefix can "re-buffer the prefix after those are dropped"
-- a real need, since `dropped_evicted=138` here. The redundant case is
supposed to be collapsed by dedup, and dedup cannot do it: `covering_op` is a
dict lookup on `_content_key = (cache_salt, prefix_end_tokens,
tuple(block_hashes))` (`eviction_aware.py:298-307, 505-507`), an exact match.
A follower range that strictly contains a shorter pending range has a
different `prefix_end_tokens` and a longer hash tuple, so it never matches.
Despite the name, no containment is computed. Both ops stay pending under
different keys, both are emitted, and the shared `[0, N_k)` is transferred
twice. `deduplicated=4` against `admitted=979`.

This matches the measured shape. The excess peaks while the cache fills,
because that is when consecutive turns arrive with the predecessor still
pending; as more lookups start hitting, the ratio falls from ~2.9x (Q1+Q2) to
~1.5x (Q3+Q4). It also explains why L1 does not grow: the re-staged bytes
carry keys already resident, so the storage layer drops them after the
transfer has been paid. Only 43 (eager) and 62 (lazy) retrieves across ~248
requests confirms most requests were served by APC and never needed a load.

Not a tuning problem: `min_prefix_tokens` changes which ops are held, not the
range an op covers, so gate 3 cannot fix it.

Fix directions:

- Advance the already-stored watermark from the pending store: when computing
  `start_token_idx`, account for the range a predecessor has already
  buffered, so the follower stages only the delta.
- Make the policy's dedup range-aware: compare intervals for containment
  instead of hashing `prefix_end_tokens`, and trim a longer follower range to
  the part not already pending.

Both are local to this branch, and both must keep the drop-recovery property
the current re-staging provides: if the predecessor's ops are dropped, the
prefix still has to be storable by someone. Neither needs lookup to consult
the pending queue.

A regression test has to cover multi-turn growing prefixes. The present suite
cannot see this: `min_prefix_tokens=0` plus single-turn synthetic prompts is
exactly the configuration that hides it.

## Caveats

- Smoke run at the scenario floor (900 s); the workload's own default is
  1800 s. Formal testing should use 1800 s.
- One run per arm. The 4% p50 deltas are within plausible run-to-run noise;
  +138.7% tokens written is not, and the TTFT direction is consistent across
  avg/p50/p99.
- **`min_prefix_tokens=0`, so gate 3 (economy) held nothing**: `held=0` and
  `rejected_short_prefix=0`. That is the gate meant to suppress unprofitable
  writes, and it is the newest commit on the branch (924e2c1c, "Move gate 3
  (economy) from emission to admission"). The obvious next experiment is a
  sweep of `min_prefix_tokens` above zero -- this result does not test it.
- `throttled_drains=2` (was 0 in the synthetic run), so `max_drain_per_step=64`
  is starting to bind.
- `dropped_evicted=138` of `admitted=979` = 14.1%.

Lazy ledger (final periodic line):

```
admitted=979 emitted=815 dropped_evicted=138 rejected_short_prefix=0
rejected_unhashed=0 rejected_prefix_broken=0 dropped_on_request_drop=8
dropped_failed_store=0 dropped_id_reuse=0 deduplicated=4 throttled_drains=2
drain_steps=74506 free_queue_blocks_read=2365287 requests_validated=1440
blocks_validated=2431034 pending=18 held=0
```

Closes: 979 == 18 pending + 0 held + 815 emitted + 146 dropped.

## Artifacts

- `artifacts/ab_trace.sh.txt` -- the scenario-native arm driver
- `artifacts/cmp.py.txt` -- the comparison
- aiperf exports under `smoke2/abt_{eager,lazy}/artifacts/`
