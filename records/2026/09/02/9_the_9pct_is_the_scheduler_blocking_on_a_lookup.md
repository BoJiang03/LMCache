# The common 9% is the vLLM scheduler blocking on an LMCache LOOKUP

2026-09-02, evening.  Arms 1j and 1k.  **This closes the question record 8 left
open.**  Read record 8 first for how the ground was cleared; this record is the
answer and the evidence for it.

## The answer in one paragraph

The +5.7 ms/step that LMCache MP and IP pay in common is the **vLLM EngineCore
scheduler thread sitting blocked on a synchronous round trip to the LMCache
server**, once per admitted request, inside `get_num_new_matched_tokens`.  It is
not CPU work, not the workers, not the KV pool, not vLLM's connector plumbing.
Measured: 7.36 ms/step of wall clock in that one call against **0.083 ms/step of
thread CPU** -- 98.9% waiting.  All eight TP workers together contribute
0.75 ms/step across every hook they have.

## How it was measured

`timedconn/timed_mp_connector.py` -- a subclass of `LMCacheMPConnector`
registered through vLLM's `kv_connector_module_path` (the route arm 1i proved
works), wrapping 20 hooks in a wall clock.  No LMCache source edit: the venv
installs LMCache editable onto this worktree, so patching `lmcache/` to add
timers would dirty the tree that carries the deliverables.

Two runs:

* **1j** -- every hook timed.  Locates the tax to a process and a hook.
* **1k** -- `get_num_new_matched_tokens` split into its four parts, each with
  `time.thread_time()` recorded beside `time.perf_counter()`.  Thread CPU counts
  only the calling thread, so wall-minus-CPU separates *computing* from
  *waiting* -- a distinction the process-wide `cpu_busy` field cannot make,
  because the EngineCore's other threads keep running while the scheduler
  thread blocks.

## The instrument is free (check this before reading anything below)

| arm | in-engine ms/step | p50 tok/s | cold duration |
|---|---|---|---|
| 1e  MP, uninstrumented | 91.0 | 89,990 | 686.0 s |
| **1j**  MP + 20 hook timers | **91.0** | **89,990** | 683.7 s |
| **1k**  MP + hook timers + 4 sub-timers | **91.0** | **89,988** | 681.9 s |

Digit for digit with 1e, and both instrumented runs are marginally *faster*.
Frozen in `engine_rate_c1000_with_1j_1k.txt`.

## 1j: which process, which hook

171 steady windows, 9 processes, scheduler wall 91.24 ms/step.

| process | hooks ms/step |
|---|---|
| **SCHEDULER (EngineCore)** | **8.50** |
| `get_num_new_matched_tokens` | **7.51** (~0.1 calls/step) |
| `build_connector_meta` | 0.83 |
| everything else it has | < 0.03 each |
| **each of 8 WORKERS** | **0.75** total |
| largest worker hook, `wait_for_save` | 0.53 |

The worker side -- `start_load_kv`, `save_kv_layer` and `wait_for_layer_load`
firing 36x per step, `get_finished`, the metadata bind/clear -- is 0.75 ms/step
all together.  That is the entire cost of LMCache on the GPU workers.

## 1k: which part of that hook, and is it computing or waiting

36 steady scheduler windows, wall 91.28 ms/step.

| sub-timer | wall ms/step | thread CPU ms/step | reading |
|---|---|---|---|
| `sub_submit_lookup` | **7.357** | **0.083** | 98.9% blocked |
| `sub_check_result` | 0.424 | 0.015 | blocked, but small |
| `sub_create_key` | 0.027 | 0.027 | pure CPU, negligible |
| `sub_get_tracker` | 0.001 | 0.001 | nothing |
| (hook total) | 7.929 | | residual 0.12 = `get_token_ids` etc. |

**A suspect of mine died here.**  `_create_key` builds `tuple(token_ids)` over
the whole 60,000-token prompt for every request, and I had flagged it in the
1k design as the likely CPU cost.  It is 0.027 ms/step.  Had I proposed it as
the mechanism instead of measuring it, it would have been the fourth wrong
guess in a row (record 7 lists the first three).

## The code

`lmcache/integration/vllm/vllm_multi_process_adapter.py`

* `maybe_submit_lookup_request` (:740) -- docstring: *"Sends a LOOKUP request to
  the server and blocks until a prefetch job ID is returned."*  The block is
  `fut.result(timeout=self._mq_timeout)` per server URL.
* `check_lookup_result` (:856) -- *"Sends a QUERY_PREFETCH_STATUS request to the
  servers and blocks."*
* `_create_key` (:1046) -- `token_ids=tuple(token_ids)`, the whole prompt.

`lmcache/integration/vllm/lmcache_mp_connector.py:851-857` calls the first, then
the second, inside `get_num_new_matched_tokens`.  vLLM's EngineCore loop is
sequential, so both waits are on the critical path of every engine step in which
a request is admitted.

Per-call cost is **~50-70 ms per admitted request**, as a range and not a
number: see the instrument gap below.

## Can the wait be removed?  Two arguments yes, one hard counter-example

**Yes-1: the value waited for is discarded.**  In `maybe_submit_lookup_request`
the result of `fut.result(timeout=...)` is not bound to anything.  The loop
exists only to catch `TimeoutError` and mark the server unhealthy.  The
scheduler blocks ~7.4 ms/step for an acknowledgement it does not use.

**Yes-2: vLLM's API already allows deferring.**  `get_num_new_matched_tokens`
may return `None`, meaning "ask me again later" -- that is what the `Deferred`
counter in the stat line is.  MP's `Deferred` is 0 in every block, so MP never
uses it.  IP does.

**The counter-example: IP already does this asynchronously and is slower.**
IP runs the same lookup through `LMCacheAsyncLookupClient` and sits at
97.5 ms/step against MP's 91.0.  Whatever IP's async path costs, it exceeds the
5.7 ms it saves.  Arm 1g also ruled out the backoff sleep as that cost.  So
"make it async" is **not** established as a win, and this record does not claim
it is.

The cheapest decisive test, not yet run: keep MP, remove only the blocking
`.result()` (fire the LOOKUP, keep the future, collect it on a later call),
everything else identical, instance-wrapped as before so LMCache source stays
untouched.  ~20 minutes.  Known risk: LOOKUP has a server-side side effect
(locking chunks) and the following QUERY_PREFETCH_STATUS could be processed
before it.  If ordering breaks, `check_lookup_result` returns None, the request
is deferred, and `Deferred` goes from 0 to positive -- visible in the log, so
the failure mode is self-announcing rather than silent.

## Mistakes and gaps in this session

1. **The wrapper broke a property.**  `transfer_intermediate_tensors` is a
   `@property` returning False from config; wrapping it into a method made it a
   bound method, which is always truthy, so the connector asked the server for
   the `TRANSFER_QUERY` feature it does not advertise and all 8 workers died at
   init.  Caught 90 s in.  Fixed structurally, not by exclusion: the wrapper now
   refuses anything that is not a plain function, and the pre-flight does a
   descriptor-level diff of the subclass against `LMCacheMPConnector` and aborts
   on any difference outside the timed hooks.  That guard immediately caught a
   second one (`build_kv_connector_stats` is a classmethod).

2. **The validity gate was wrong and cried wolf twice.**  It divided the bench
   JSON's end-to-end `total_token_throughput` into the 8192-token step and
   required 89.0-93.0 -- a band calibrated on `engine_rate.py`'s *in-engine*
   steady-state rate.  Different quantities: the end-to-end aggregate includes
   ramp and drain and reads ~2.7 ms/step higher in every arm.  1j scored 93.3
   and was declared VOID; **1e scores 93.7 by the same formula and would have
   failed its own gate.**  Both scripts now compare cold duration against 1e's
   686.0 s and name `engine_rate.py` as the authority.  1k's copy could not be
   fixed until it finished -- a running bash script must not be edited in place.

3. **Exact per-call cost was lost.**  The instrument dumped its JSON from
   `atexit`, and vLLM tears the EngineCore down with a signal it does not
   survive, so the one process that mattered never dumped.  Per-call cost is
   therefore a range (~50-70 ms) rather than a number.  Fixed: the dump now runs
   every reporting window and cannot be lost.

4. **`engine_rate.py`'s default `--min-outstanding=700` silently drops arms.**
   At ISL=60000 the API server's `Running + Waiting` peaks below 700 in 1d, 1e,
   1i, 1j, 1k, 1a_rerun and 1c_rerun -- exactly the arms being compared.  The
   earlier freeze `engine_rate_c1000.txt` contains those arms, so it was
   produced with a lowered threshold whose value was never recorded.  The new
   freeze records the full command in its header.

## Where the decomposition stands now

```
85.3 ms/step   no connector (1a x6, 1c x3) and NullConnector (1i)
               vLLM's whole connector path: +0.0
91.0 ms/step   LMCache MP (1d x2, 1e x2, 1j, 1k)
               +5.7 = scheduler blocked on the LMCache LOOKUP   <- SOLVED HERE
97.5 ms/step   LMCache IP (1b x6, 1f x2, 1g x1)
               +6.5 more, on top, still unexplained
```

## Open

* **IP's extra +6.5 ms/step.**  Same hook, different client
  (`LMCacheAsyncLookupClient`).  Not backoff (1g).  The same subclass trick
  applies to `LMCacheConnectorV1`; an IP twin of 1j/1k is the obvious next
  instrument if that question is reopened.
* Whether removing the blocking `.result()` actually pays -- see above.
* Why one LOOKUP takes ~50-70 ms server-side at all.  That is the server's
  work, and it is a separate question from whether the scheduler should be
  waiting for it.
* `1a@200` and `1c@200` were never run, so the whole c=200 column stays
  uninterpretable.
* Decode is unmeasured (OSL=1 is prefill only).
* Ask VAST for the `GPU KV cache size: N tokens` line from each of their runs.
