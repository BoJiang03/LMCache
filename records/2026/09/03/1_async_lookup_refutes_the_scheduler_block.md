# The scheduler's blocking LOOKUP is a symptom, not the cost

2026-09-03, small hours.  Arm 1l (run 2026-09-02 20:29-20:49).

**This record corrects record 9 of 2026-09-02.**  Record 9's measurement stands;
its causal claim does not.  Record 9 named this experiment as the test that
would decide, and predicted nothing.  It has now decided: **no**.

## What was claimed, and what is now known

Record 9: the common +5.7 ms/step that LMCache MP and IP pay is the vLLM
EngineCore scheduler thread blocked on a synchronous LOOKUP round trip inside
`get_num_new_matched_tokens`.  Measured 7.36 ms/step of wall against 0.083 of
thread CPU -- 98.9% waiting.

That measurement is correct and reproduces.  The inference from it -- that the
waiting *is* the +5.7 -- is wrong.  Remove the wait and the step rate does not
move:

| arm | scheduler hooks ms/step | in-engine p50 tok/s | ms/step | Deferred max |
|---|---|---|---|---|
| 1d stock MP | -- | 89,991 | 91.0 | 0 |
| 1e stock MP | -- | 89,990 | 91.0 | 0 |
| 1j MP + hook timers | 8.417 | 89,990 | 91.0 | 0 |
| 1k MP + sub-timers | 8.722 | 89,988 | 91.0 | 0 |
| **1l MP + async lookup** | **6.372** | **89,989** | **91.0** | **563** |
| (reference) no connector | -- | 95,99x | 85.3 | 0 |

2.35 ms/step of scheduler hook time gone, of which ~2.1 ms/step is blocking
removed, and the engine produces tokens at the same rate to four significant
figures.  Frozen in `engine_rate_c1000_with_1l.txt`, which also records that the
script's `--min-outstanding` default silently drops every arm in this table.

**The scheduler thread has slack.**  While it sits blocked, the GPU is not
waiting for it.

## The patch, and the proof it took effect

`timedconn/async_mp_connector.py`, a subclass of the 1j/1k instrument, so every
hook number stays directly comparable.  One variable changed: when the scheduler
waits.

    stock:  send LOOKUP -> BLOCK for ack -> send QUERY -> BLOCK -> answer
    1l:     send LOOKUP -> return None (defer) -> a later call: ack arrived?
            -> send QUERY -> BLOCK -> answer

QUERY is never sent before the LOOKUP ack has been observed, so the server-side
ordering the stock code depends on (LOOKUP locks the chunks, QUERY reports them)
is preserved by construction.  That is the one correctness risk in the naive
version of this change, and it was designed out rather than hoped away.

Two independent witnesses that the patch was live, both required by the script
before it would run the benchmark:

* `Deferred` (vLLM's `num_skipped_waiting_reqs`) reaches **563**.  It is **0** in
  every block of every stock MP arm ever run here.
* `sub_submit_lookup` 7.357 ms/step -> `sub_async_submit` **0.001** ms/step.

## Where the wait went

Exact counts, from the scheduler's own dump (7,400 steps, 1,000 requests).  This
is the first arm to have them: the per-window dump added after 1k lost its
`atexit` dump to vLLM's shutdown signal.

| | calls | wall | thread CPU | per call |
|---|---|---|---|---|
| `get_num_new_matched_tokens` | 31,171 | 56.28 s | 16.16 s | 1.81 ms |
| `sub_check_result` (QUERY) | 995 | **38.32 s** | 0.12 s | **38.51 ms** |
| `sub_async_submit` (LOOKUP, fire) | 31,171 | 1.87 s | 1.83 s | 0.06 ms |
| `sub_async_defer` | 30,176 | 0.13 s | 0.11 s | 0.00 ms |
| `sub_async_collect` | 995 | 0.02 s | 0.02 s | 0.02 ms |
| `sub_create_key` | 1,000 | 0.28 s | 0.28 s | 0.28 ms |

Read it as:

* The blocking did not vanish, it **relocated**.  Stock MP blocks ~73 ms in
  LOOKUP and ~4 ms in QUERY per admitted request; 1l blocks 0 in LOOKUP and
  38.5 ms in QUERY.  Per-request scheduler blocking roughly halved, 77 -> 38.5 ms.
  The server's work is the same work; only who waits for it changed.
* Deferral itself is nearly free in wall clock (0.13 s over the whole run) but
  costs CPU through re-polling: the hook is entered 31,171 times for 1,000
  requests and burns 16.16 s of thread CPU, against ~0.1 s in stock MP.  At 2.3%
  of one core over a 691 s run that is not the bottleneck either.
* `_create_key` is called exactly once per request (1,000 for 1,000), so the
  early-return guard works and no LOOKUP is sent twice.
* End-to-end the arm is slightly *worse*: 701.7 s against 1e's 686.0 s (+2.3%),
  which is the deferral latency showing up in TTFT while the step rate is
  unchanged.

## So where is the +5.7 ms/step

Not in the scheduler's hooks: 6.4 ms/step, and 2.1 of that is now *proven* to be
free -- removing it bought nothing, so the rest of it is not obviously on the
critical path either.  Not in the workers' hooks: 0.851 ms/step across all eight
processes and all 20 hooks, unchanged from 1j's 0.746 and 1k's 0.855.

Neither side's hooks add up to 5.7, and the one part that was big has been shown
not to matter.  What is left is **off-hook**: work LMCache does outside any
KVConnector call.  The standing candidate is the store path -- the D2H copies
that push prefill KV to the server, plus LMCache's background threads -- eating
host CPU and PCIe against the forward pass.  Worker processes sit at
`cpu_busy` 1.87 (187% of a core each, times eight).  That is a hypothesis with a
number attached, not a finding.

## The mistake to learn from

Record 9 had a clean measurement and drew a causal conclusion from it without an
intervention.  The measurement said "the scheduler waits here"; the conclusion
said "therefore the waiting costs the throughput".  Those are different claims
and only the second one was interesting.  Record 9 did flag the claim as
unproven and named this experiment -- that is the only reason the error cost one
20-minute run instead of a wrong upstream patch.

Score so far: five mechanisms proposed from reading source, five refuted by
experiment (three in record 7, `_create_key`'s 60k-token tuple in record 9, and
now the scheduler block itself).  Reading the source has not once produced the
answer here; only intervention has moved the number.

## Harness bug found and fixed

`scripts/phase1l_async_lookup.sh` was generated from 1k by a Python transform
whose `s.index("PYEOF", i)` matched the heredoc's *opening* delimiter on the same
line, duplicating the result-summary block.  The bench, every assertion and the
timers were unaffected -- only the post-run summary print raised SyntaxError, and
the JSON it would have summarised was written and read back by hand.  Fixed and
`bash -n` clean.  Noted because "generate a script by string surgery on another
script" has now failed twice in this investigation.

## Where the decomposition stands

```
85.3 ms/step   no connector (1a x6, 1c x3) and NullConnector (1i)
               vLLM's whole connector path: +0.0
91.0 ms/step   LMCache MP (1d x2, 1e x2, 1j, 1k, 1l)
               +5.7 = UNEXPLAINED again.  Not the scheduler's LOOKUP wait (1l),
               not the connector plumbing (1i), not the KV pool (record 5),
               not _create_key (1k), not the IP backoff (1g).
97.5 ms/step   LMCache IP (1b x6, 1f x2, 1g x1)
               +6.5 more, on top, still unexplained
```

## Proposed next, not run

Two arms, ~20 minutes each, that partition the 5.7 between LMCache's two jobs.
Same instance-wrapping trick, so `lmcache/` stays untouched:

* **1m -- store off, lookup on.**  No-op `save_kv_layer` and `wait_for_save` on
  the workers.  The cold pass is all misses anyway, so nothing observable is
  lost.  If ms/step falls to 85.3 the cost is the store path.
* **1n -- lookup off, store on.**  The other direction.

If neither moves, the tax is the mere presence of LMCache's processes and
threads, and the next instrument has to be GPU-side (a profiler on the forward
pass), not another connector wrapper.

## Open

* The +5.7, again.
* IP's extra +6.5 ms/step.
* Why one LOOKUP costs the server ~40-75 ms at all.  1l measured it from the
  other side (38.5 ms per QUERY) without explaining it.
* `1a@200` and `1c@200` were never run, so the c=200 column stays
  uninterpretable.
* Decode is unmeasured (OSL=1 is prefill only).
* Ask VAST for the `GPU KV cache size: N tokens` line from each of their runs.
