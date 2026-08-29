# The engine stalls half the window

**Retracted in full on 2026-08-29. See record 4 section 1.** There is no stall.
The intervals this record calls stalled are chunked prefill: vLLM credits a
request's prompt tokens only on the iteration that produces its first token
(`output_processor.py:628` flips `is_prefilling`, `loggers.py:145` sums
`prompt_token_stats.computed`), so a multi-step prefill reports zero prompt
throughput for its whole duration and then the entire prompt in one interval.
GPU telemetry split by the stall detector shows 100 % utilisation, 690 W and
16 % memory activity during the "stall" against 99 % / 654 W / 60 % outside it,
which is compute-bound prefill against memory-bound decode. Sections 2 to 8
below rest on the stall being real and do not survive it; section 1 (no
synchronised release) and the arrival measurements do.

Continues record 2026/08/28/15. Two more of its conclusions fall here, and the
thing that replaces them is not a configuration or a policy.

## 1. There is no synchronised release

Record 15 section 7a proposed a `CACHE_WARM` arm on the theory that profiling
releases all 60 lanes at once and the closed loop then locks into the congested
one of two basins. Measured from the per-request timestamps, the premise is
false.

First request of each conversation, offset from the profiling phase start:

```
                        n60pc12    n60floor
p50                      134.1s     115.2s
max                     1780.5s    1362.8s
conversations started within 30s   11/50      16/51
all requests within 30s            11/121     22/142
```

`agentic_replay.py` does what its docstring says -- "Default dispatch preserves
the stream's recorded offset from the replay boundary" -- and
`burst_phase_starts` is False. Arrivals are spread across the whole window. The
arm was cancelled before it ran.

## 2. There is also no second basin

The bistability claim used a decode time of 30 s, taken from `osl p50 546` times
`ITL p50 51 ms`. The measured mean `decode_duration` is 146 s: `OSL` mean is
1546, not 546, and long-output requests run at `ITL p90 168 ms`.

```
pool capacity in conversations = 2,029,760 / 176,380  = 11.4
pool residency per request     = decode 146s + prefill 14s = 160 s
capacity                       = 11.4 / 160  = 0.071 req/s
measured                                       0.0615 req/s   (87 % of it)
demand at N=60, Z=400s, no queue = 60/560     = 0.107 req/s
```

Demand exceeds capacity by 1.5x. One equilibrium, and it is the congested one.

Of the 160 s of residency, 146 s is decode. Even a perfect cache only removes
part of the 14 s prefill term, which caps throughput at 0.078 req/s -- still
short. The only term large enough to close the gap is `ITL`, at 49.6 ms against
a 22.9 ms bandwidth floor for batch 9.4 at 176k tokens.

So the question became: where does the 2.2x in `ITL` go. The standing answer --
chunked prefill stealing decode steps -- had never been measured.

## 3. Half the window the engine does nothing

Definition: an interval with prompt throughput < 100 tok/s **and** generation
throughput < 10 tok/s **and** Running > 0.

```
                span   stalled intervals   contiguous stall   KV fill during stall
n60pc12        2071s        105/208         990s  (48 %)          2,110 tok/s
n60floor       2401s        134/241        1241s  (52 %)          2,440 tok/s
```

The raw log, 80 seconds of it:

```
21:51:26  pre: 0.0  gen: 2.7  Running: 10  Waiting: 22  kv: 78.1%
21:51:36  pre: 0.0  gen: 2.7  Running: 10  Waiting: 23  kv: 79.4%
...
21:52:46  pre: 0.0  gen: 1.8  Running: 10  Waiting: 24  kv: 86.1%
21:52:56  pre: 82667.0  gen: 80.0  Running: 11  Waiting: 22  kv: 99.2%
```

Ten requests in Running, twenty-four queued, zero prefill, 2.7 output tokens per
second in total. Per-request that is 0.27 tok/s, so **the step is taking 3.7
seconds**. A normal step here is 23-50 ms. Two orders of magnitude.

KV pool occupancy climbs monotonically through the stall, so blocks are being
allocated the whole time. Nothing is being computed and something is being
filled.

## 4. It is not the storage tier

The server log times every transfer.

```
n60floor   188 retrieves   27,830,784 tokens   50.3 s total
           per-op tok/s  p50 = 1,006,132       aggregate = 553,362 tok/s
n60pc12    160 retrieves   23,144,960 tokens   27.8 s total
                                               aggregate = 831,894 tok/s
store side                                     aggregate = 194,997 tok/s
```

LMCache delivers 27.8M tokens in 50.3 seconds -- 2 % of the window. The engine
takes 1241 seconds to absorb about 3.0M tokens into the pool.

```
LMCache delivers        553,362 tok/s
pool absorbs              2,440 tok/s
                            227x
```

The time is not in the storage tier. It is between the `Retrieved` line and the
blocks becoming usable.

## 5. It is not the lazy drain either

```
                 free_queue_blocks_read   per drain step   stall share
n60pc12                    5,429,574            268           48 %
n60floor                 131,337,971          5,927           52 %
```

Enabling the danger floor multiplied the per-step free-queue scan by 22 and
moved the stall share by four points. Scanning the free queue is not the 3.7
seconds.

## 6. What this invalidates

Every quantitative conclusion of 2026/08/28 rests on the assumption that the
engine was working and the question was how to configure it. It was idle half
the time.

- Decode at 23.7 % of HBM bandwidth, prefill at 37-62 % of compute, the KV pool
  82 % full and not moving: all three are projections of the stall, not
  independent findings.
- The `lazy` drop rate of 29.8 % is measured under a pool that is being churned
  by whatever is stalling, not under normal pressure. It cannot be read as a
  property of the policy.
- The capacity arithmetic in section 2 above holds as a description of what the
  machine did, not of what it can do. `ITL` 49.6 ms, `decode_duration` 146 s and
  residency 160 s are all inflated by the stall.
- `theta = Z/OSL` and `k_max = theta/TPOT` from record 14 use a measured `TPOT`
  and inherit the same inflation.

No arm run to date is a valid measurement of the lazy policy. The eager/lazy
pair the work needs cannot be produced until this is fixed.

## 7. Next

Locating this needs a stack sample taken during a stall. `py-spy` is not
installed and the shared venvs are not to be modified.

1. Read the load path in `lmcache/integration/vllm/` from `start_load_kv` to
   blocks becoming usable, looking for a synchronous wait. The
   `multi_layer_block_kv_transfer mode: ptr` path is the first suspect.
2. If that does not name it, install `py-spy` with `pip install --target` into
   the scratchpad, run a short arm, and sample the EngineCore process during a
   stall.

Not started.

## 8. Corrections

- Record 15 section 7a, the synchronised-release theory and the `CACHE_WARM`
  arm derived from it: withdrawn, section 1.
- Record 15 section 7a, "two self-consistent solutions and both are feasible,
  2.1x margin": wrong, section 2. Decode residency was taken from a p50 where
  the mean is 2.7x larger.
- Every arm from 2026/08/28 measures a machine that is idle half the window,
  section 6.
