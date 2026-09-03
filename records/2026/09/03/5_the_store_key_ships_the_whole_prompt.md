# Loss #1, named: every store request msgpack-encodes the whole prompt

`chain23` profiled `mp` and `nostore` over the identical step window
(2400:3000 of 7200) under phase1's FULL TP=8 protocol, all 8 workers, and
`scripts/prof_diff.py` summed the per-rank pstats.  Sanity check first: under
cProfile the arms are 709.5 s vs 660.3 s (+7.5%), against 686.0 / 639.0
(+7.4%) unprofiled -- the profiler did not distort the thing being measured.

## The one call that matters

    d_tot   calls/step   nostore     mp   function
    1.013     0 -> 5.8     0.000  1.013   <built-in method msgspec._core.msgpack_encode>
    0.243     0 -> 1.0     0.000  0.243   vllm_multi_process_adapter.py:2027(_create_key)
    0.32    2.1 -> 11.6    0.03   0.32    zmq/sugar (send / recv_multipart / poll)
    0.88   110 -> 110      1.89   2.77    <_launch_kernel>

(ms/step/worker.  Call counts here are small, so cProfile's per-call overhead
is negligible on these rows -- unlike the spin rows below, these numbers are
real.)

`_create_key` builds `IPCCacheServerKey`, whose `token_ids` field is
`tuple(op.token_ids)`.  `op.token_ids` is not the chunk being stored: it is
`RequestTracker.token_ids`, seeded at
`vllm_v1_adapter.py:198` as `prompt_token_ids[:num_tokens_to_compute].copy()`
and grown as the request advances.  `start` / `end` index into it.  So every
store request carries **the entire prompt prefix processed so far**, and both
the tuple build and the msgpack encode are O(prompt length) -- per step, per
rank.

Measured directly (`msgspec.msgpack.encode` on a real `IPCCacheServerKey`):

    N tokens   tuple() ms   encode ms      bytes
        8192        0.031       0.085      35669
       32768        0.125       0.370     142222
       60000        0.248       0.688     260915

The profiled `_create_key` cost, 0.243 ms/step, lands on the N=60000 row
(0.248).  These runs use 60k-token prompts.  So each store request is a
**255 KB** msgpack blob, built and serialised on the worker's model-execution
thread, and all 8 ranks build the same one -- `worker_id` is the only field
that differs between them.  ~2 MB/step of duplicated token ids to a single
lmcache server.

Direct cost per rank per step: 0.25 (tuple) + 0.69 (encode) + 0.32 (zmq)
= **~1.26 ms**, plus the rest of the LMCache Python, ~1.8 ms total.

## Where the other 4 ms goes

The largest tottime deltas in the diff are vLLM's shared-memory queue spin:

    d_tot   calls/step        function
    1.915   3167.5 -> 5115.3  shm_broadcast.py:176(wait)
    1.122      4.0 -> 4.0     shm_broadcast.py:657(acquire_read)
    1.120   3167.5 -> 5115.3  <built-in method posix.sched_yield>
    0.684   3177.4 -> 5125.3  shm_broadcast.py:60(memory_fence)
    0.606   3169.4 -> 5117.3  shm_broadcast.py:670(check)
    0.416   3167.5 -> 5115.3  utils.py:46(sched_yield)

Read the CALL COUNTS, not the times: at ~22k-36k calls/step these rows are
dominated by cProfile's own per-call overhead and their milliseconds are
inflated several-fold.  The counts are honest, and they say the workers spin
**61% more iterations** in `acquire_read`'s `sched_yield` busy-wait.  That is
why the missing CPU is `thread_time` and not `blocked`: `sched_yield` burns
real CPU on the calling thread.  Per rank the spin is asymmetric --
11.8-21.8 ms/step in mp against 5.9-13.6 in nostore -- the straggler shape.

Split by frame: `execute_model` cumtime is 107.36 (nostore) vs 112.72 (mp),
+5.36; total thread time is 115.15 vs 129.20, +14.04.  So roughly a third of
the profiled delta is inside `execute_model` and the rest is the worker
waiting for its next command.

Adding up against the step probe's +3.86 ms/step of worker CPU that `nostore`
removes:

    1.01  msgpack encode of the key
    0.51  LMCache Python (_create_key 0.24 + the rest)
    0.32  zmq send / recv / poll
    0.88  _launch_kernel (same 110 launches/step, each slower)
    ~1.2  residual spin
    ----
    3.9

## Why TP=8 and not TP=4

The per-rank cost is the same at both degrees; what changes is whether the
host thread has room to hide it.  At TP=4 the step is 136.54 ms and the
baseline main thread already burns 81.82 ms of CPU with `blocked` = 0.00 --
it is spinning, i.e. there is host slack.  `mp` there DISPLACES spin rather
than adding time (cpu 81.82 -> 79.56, blocked 0.00 -> 1.60) and the
steady-state loss is exactly zero: 136.54 against 136.54.  At TP=8 the same
model has half the GPU work per rank, the step falls to 85.34 ms, and the
~1.8 ms of per-step Python no longer fits in the gap.  It desynchronises the
ranks, and TP lockstep charges the whole step for it.

## What to fix

In order of value:

1. **Stop re-encoding the prefix.** `token_ids` is a growing prefix; only
   `end` moves. Ship it as a raw buffer instead of a msgpack array of ints and
   the encode becomes a memcpy (~0.02 ms instead of 0.69), with `tuple()`
   gone too. Same wire semantics, ~0.9 ms/step/rank recovered.
2. **Ship the delta, not the prefix.** The server hashes token_ids into
   prefix-chained chunk hashes, so it needs the prefix -- but it can keep a
   per-request token buffer and take only the new tokens. Turns an O(prompt)
   per-step cost into O(chunk).
3. **Deduplicate across ranks.** All 8 ranks send byte-identical token_ids;
   only `worker_id` differs. Register the token set once per request and
   reference it by id.

(1) is local and low-risk; (2) and (3) are protocol changes.

## Instrument note

The ownership-aware watchdog in `chain23.sh` is still wrong -- it calls a pid
ours iff it shares the script's session id, but `lane.sh` spawns the server
detached, so it flagged our own eight `VLLM::Worker_TP` processes as FOREIGN.
The genuinely foreign compute processes during this run were 1816225
(rui/lmcache), 3203853 and 3204611 (root/lmcache); 2939753 (bo/lmcache) is
our own lane's server. Ownership has to come from the lane's own pid list, not
from sid.
