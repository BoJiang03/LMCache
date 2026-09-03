> **Corrected by record 9 (2026-09-03).** The LOSS 2 section below names the
> right line and the wrong mechanism.
>
> `slot_mapping.to(self.device)` is not an expensive copy. Synchronising the
> current stream immediately before it -- which changes nothing, because the
> pageable copy already waits for exactly that -- splits the 33.7 ms into
> **64.5 ms of draining the forward pass and 0.089 ms of actual DMA** for the
> 480 KB. The pre-allocated pinned buffer that the source's own TODO asks for,
> and that this record proposes, is worth **0.089 ms/call**.
>
> "The copy costs more than the transfer it prepares" is therefore false: the
> copy costs 0.089 ms against the transfer's 5.2 ms. What costs is the host
> BLOCK, not the bytes. Attempt 2's CUDA illegal access is also explained --
> `VLLMPagedMemGPUConnectorV2.from_gpu` runs its D2H on its own `store_stream`
> with no `wait_stream(current_stream)`, so today's pageable drain is the only
> thing ordering it against the forward pass.

# Both losses located: a per-submission stall, and one pageable H2D copy

2026-09-03, overnight, unattended.  A TP=4 lane on GPUs 0-3 with the KV pool
pinned identically in every arm.

**Read this before the numbers:** nothing here is comparable to phase1's
85.3 / 91.0 / 97.5 ms/step.  Those were TP=8.  Another tenant held GPUs 4-7 for
most of the night, twice taking them mid-launch, so the work moved to four free
GPUs and every comparison below is internal to that lane.  That turned out to
matter more than it cost -- see "the tax is not a fixed overhead".

## The lane

| | |
|---|---|
| model / TP | gpt-oss-120b, TP=4 on GPUs 0-3 |
| workload | 300 prompts x 60,000 tokens, OSL=1, c=300, cold pass |
| pool | pinned at 30,000 blocks = 1,920,000 tokens in EVERY arm |
| step | max_num_batched_tokens=8192, asserted from the log |

The pool is pinned with `--num-gpu-blocks-override` because on a shared box it
is otherwise not reproducible: the `mp` arm profiled 52.87 GiB of available KV
against the baseline's 92.48 GiB on GPU 0 alone, giving 3,080,064 tokens
against 5,387,584, and the comparability assert -- correctly -- threw the arm
away.  Pinning removes that whole class of failure.  The baseline is unchanged
by it: 296.9 s with a 1.92M pool against 295.7 s with a 5.39M one.

## The whole table

ms/step per worker from the step probe, steady state; end-to-end from the bench.

| arm | loop | exec | cpu | blocked | end-to-end |
|---|---|---|---|---|---|
| none | 131.78 | 81.82 | 81.82 | 0.00 | 296.9 s |
| nostore | 131.79 | 78.02 | 78.01 | 0.01 | 296.4 s |
| mp | 132.87 | 81.16 | 79.56 | 1.60 | 299.4 s |
| storeprobe | 133.00 | 81.30 | 79.60 | 1.70 | 299.9 s |
| nolookup | 133.06 | 83.63 | 82.07 | 1.56 | 299.5 s |
| bigl1 | 133.36 | 80.88 | 79.27 | 1.61 | 298.8 s |
| timedip | 143.28 | 140.39 | 140.18 | 0.21 | 326.1 s |
| ipstoreprobe | 143.64 | 140.47 | 140.25 | 0.22 | 326.0 s |
| ipprof (cProfile on) | 144.83 | 141.49 | 141.25 | 0.24 | 326.5 s |

`blocked` is exec wall minus exec thread-CPU: the time the worker's main thread
was not running bytecode.  **The baseline's exec EQUALS its cpu to two
decimals** -- the thread never blocks inside execute_model, it is busy
launching kernels for 62% of the step.  That zero is what makes the two
connectors separable, and it is the reason a probe that makes no CUDA call at
all can still tell "the GPU is doing more" from "the CPU is slower".

## The tax is not a fixed overhead

phase1 measured MP at +6.7% (85.3 -> 91.0) at TP=8.  Here MP is **+0.8%**.  A
fixed per-step or per-request cost would have shown the same absolute +5.7
ms/step at TP=4, which is +4.2% of a 136.5 ms step.  It is +1.1.

Every hypothesis this investigation has entertained -- the blocking lookup, the
KV pool, `_create_key`, the IP backoff, the connector plumbing -- had the shape
"find the one expensive call".  That shape was wrong from the start.

CAVEAT, not yet closed: phase1 was TP=8 AND c=1000; this lane is TP=4 AND
c=300.  Two variables moved.  A TP=4 / c=1000 pair is queued to separate them.

## LOSS 1 -- MP: the worker blocks per store SUBMISSION

No-op'ing one call, `worker_adapter.batched_submit_store_requests`, takes both
the blocking (1.60 -> 0.01) and the entire step-time cost (+1.09 -> +0.01) back
to baseline.  Removing the scheduler's LOOKUP round trip instead does nothing.

Three things then rule out every obvious fix:

* **It is not the copy.**  `bigl1` gives the server 500 GB with LRU eviction so
  every store lands: 8,400 stores succeed and 0 fail, against `mp`'s 56 and
  8,344.  150x the bytes -- ~1.27 TB instead of 8.5 GB -- for 0.01 ms/step
  more.  Cost is per submission, not per byte.
* **It is not backpressure.**  A store rejected instantly costs what a store
  that moves 151 MB costs, so telling a full server to stop being asked would
  buy nothing.
* **It is not in any hook.**  Sub-timers on the worker adapter: the entire
  submission is 0.454 ms/step and blocks for 0.03 s across the whole run, and
  the CUDA IPC event export -- the leading suspect -- is 0.038 ms/step.

So the blocking is caused by, but not located in, the submission: the LMCache
server acts on the request, on the same four GPUs and the same cores, while the
forward pass is running.  Submitting is cheap; being served is not.  Every rank
is delayed a little and a TP all-reduce waits for the slowest, which is the
shape that fits +6.7% over 8 ranks and +0.8% over 4.

`exec_wall - exec_cpu` cannot separate "waiting for the device" from
"descheduled while another process runs", so which of those it is remains open.
There is no code fix here yet, and this record does not claim one.

## LOSS 2 -- IP: one pageable host-to-device copy, 33.7 ms per call

IP costs +9.8% end-to-end where MP costs +0.8%, and its worker hooks say where:

| hook | calls | wall | thread CPU | ms/step |
|---|---|---|---|---|
| wait_for_save | 2,195 | 162.66 s | 162.66 s | **73.938** |
| save_kv_layer | 79,020 | 0.70 s | 0.71 s | 0.317 |
| wait_for_layer_load | 79,020 | 0.38 s | 0.31 s | 0.173 |

wall EQUALS CPU, so the thread is working, not waiting.  MP's same hook is
0.53 ms/step.  IP's scheduler hooks are trivial by comparison
(`get_num_new_matched_tokens` 1.805, `build_connector_meta` 1.048).

Splitting the body: `lmcache_engine.store()` inside it is only **4.953
ms/step**.  68 ms/step is the surrounding loop -- and `sub_ip_unpin`, the first
statement in that loop, fires 1.09 times per step, so the loop runs ONCE per
step.  67 ms in a single iteration, around a single 5 ms store.

Nothing visible in that function is plausibly 67 ms, so: cProfile inside a real
worker, steps 900..1100.

    ncalls  tottime  percall  filename:lineno(function)
      1016    7.404    0.007  {method 'to' of 'torch._C.TensorBase' objects}

    {method 'to'}  <-  217 calls  7.321 s  vllm_v1_adapter.py:1102(wait_for_save)

**33.7 ms per call**, ~37 ms/step, for a 60,000-element int64 tensor -- 480 KB,
which should move in ~50 us.  `slot_mapping` is a pageable CPU tensor, and a
pageable H2D copy is synchronous: the calling thread blocks until it completes,
and it completes only after every kernel already queued on the stream.  The
forward pass has just queued ~80 ms of them.  So the line is a synchronization
point in the middle of the forward pass, on the worker's main thread, and it
busy-waits -- which is why wall equals thread CPU throughout.

The line directly above it in LMCache's own source reads
`# TODO: have a pre-allocated buffer to hold the slot_mappings`.

cProfile did not distort the run it measured: `ipprof` finished in 326.5 s
against `ipstoreprobe`'s 326.0 s.

## For scale: the copy costs more than the transfer it prepares

From LMCache's own log, the same run:

    Stored 8192 out of total 8192 tokens. size: 0.1406 GB, cost 14.9143 ms,
    throughput: 9.4289 GB/s; offload_time: 14.7735 ms, put_time: 0.0926 ms

The real KV offload is 11-15 ms for 144 MB.  The `slot_mapping` copy that
precedes it is 33.7 ms for 480 KB -- more than twice the cost of the data
movement it is preparing for, for 0.3% of the bytes.

## Two fix attempts, both refuted

First hypothesis: hoist the `skip_leading_tokens == len(token_ids)`
early-continue above the copy, so the ~300 concurrent requests would not each
pay it.  Patched, rerun with the identical instrument:

    arm            wait_for_save   sub_ip_store   loop     end-to-end
    ipstoreprobe      72.921          4.953      143.64     326.0 s
    ipfixed           72.306          5.459      143.71     321.7 s

No change; reverted.  And the same timers refute the premise: `sub_ip_store`
runs 0.95 times per step, so the early-continue almost never fires and there
was never a storm of discarded copies to remove.  The copy is expensive, but it
does not hide behind that branch.

That is **six mechanisms proposed from reading source and six refuted by
experiment** in this investigation.  Reading has not once produced the answer;
the profiler produced it in one run.  The lesson is now costed: it took two
wasted arms tonight, and it should have been a profiler an hour earlier.

### Attempt 2: stage through a reusable pinned buffer.  Unsafe.

Copy into a reused pinned buffer, then `.to(device, non_blocking=True)`.  The
run died 62 s in with a CUDA illegal access on every worker, after a few
hundred successful stores.  REVERTED.

The bug is real, not incidental: LMCache creates its own CUDA stream per device
("Initialized cuda stream on device cuda:0"), so an async H2D issued on the
current stream is not ordered against the consumer's stream, and one reusable
buffer is overwritten on the next step while the previous copy may still be in
flight.  Making this safe needs an event handshake between the copy and
LMCache's stream -- a change in the transfer path, not a two-line edit.

**So loss 2 is diagnosed but NOT fixed here, deliberately.**  The line is
identified, the cause is measured, the author's own TODO anticipates the
remedy, and two naive implementations are now ruled out by experiment.  That is
where this hands off rather than shipping something unvalidated.

## Harness defects found, all now asserted against

* The lane's `mp` arm omitted `kv_connector_module_path`.  vLLM's factory maps
  the bare name `LMCacheMPConnector` to its OWN in-tree copy, a different class
  from the one every timed subclass inherits from.  Caught before it ran.
* Pinning `--max-num-batched-tokens 8192` -- the same value vLLM derives, and
  the value the log reports either way -- collapsed the pool from 13,724,416 to
  1,223,040 tokens (max concurrency 104.71x -> 9.33x) with identical 117.8 GiB
  available per GPU.  The sliding-window group's per-request budget comes from
  `scheduler_config.max_num_batched_tokens`.  No phase1 arm passed the flag.
* The lmcache server's HTTP port defaults to 8080, which another tenant took
  overnight.  The bind failure is FATAL and lands ~15 s AFTER the MQ port is
  already listening, so "the server is up" passed and the arm then died.  Now
  pinned to 8766, and the script greps the server log afterwards, because a
  listening socket is not proof the server lives.
* `n=$(grep -c ... || echo 0)` emits two lines on no match, so the following
  integer test errored instead of aborting.
* A CUDA-event version of the step probe killed Worker_TP1 on its first
  inference step, with no Python traceback -- a process death, which is what
  recording a timing event into a capturing stream does.  Rewritten to use only
  perf_counter and thread_time; it makes no CUDA call at all.
* Self-inflicted: `kill $(pgrep -u bo -f "chain7.sh")` matched the tool shell
  running that very command.  pgrep patterns must not match their own caller.

## Three things that were already true before any new run

Read out of logs that were already on disk, each changing how an earlier record
should be read.

1. **The store path was inert in every phase1 MP arm.**  `--l1-size-gb 8
   --eviction-policy noop` fills after 113 chunk allocations of 72 MiB; after
   that every store fails.  1d 221 stored / 117,836 failed; 1e 113 / 122,581;
   1j, 1k and 1l 113 / 55,887 each.  MP paid its +5.7 ms/step while
   successfully storing 0.2% of what it was asked to store.  `bigl1` has since
   shown this hid nothing, but no record should have rested on it unexamined.
2. **1l's null result was over-read.**  2026-09-03 record 1 concluded "the
   scheduler thread has slack".  1l's own timers show it ADDED 16.16 s of
   scheduler thread CPU (31,171 hook entries for 1,000 requests) while removing
   ~15.5 s of blocking.  Those nearly cancel.  What 1l established is that that
   async implementation broke even.  `nolookup` is the clean test and it agrees
   the lookup is not the cost -- but for a reason 1l could not show.
3. **1i (NullConnector) did not cover the whole connector path.**  It overrides
   neither `request_finished` nor `get_finished`, so it inherits False and
   (None, None): vLLM frees blocks immediately and the async-save accounting
   never runs.  LMCache returns True for every request.  "vLLM's connector path
   costs 0.0" is established only for the nothing-to-report case.

## Open

* Whether TP degree or concurrency drives loss 1's amplification (arm queued).
* Whether loss 1's contention is GPU time-slicing between two CUDA contexts or
  host descheduling.  `exec_wall - exec_cpu` cannot tell them apart.
* A TP=8 confirmation of everything here, once GPUs 4-7 are free.
* A safe implementation of the loss-2 fix: pinned staging plus a CUDA event so
  LMCache's stream waits on the copy, and a buffer per in-flight request rather
  than one reused buffer.
