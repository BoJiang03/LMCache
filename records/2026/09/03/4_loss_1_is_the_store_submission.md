# Loss #1 is the store submission, end to end

2026-09-03, TP=8, GPUs 0-7, gpt-oss-120b, 1000 prompts x 60,000 tokens, OSL=1,
c=1000, hybrid KV manager off, pool 13,724,416 tokens, `max_num_batched_tokens`
derived (not pinned) = 8192.  Results in `results/phase1_v2/`.

## The result

    arm            in-engine ms/step     end-to-end
    tp8_none            85.34             625.2 s
    tp8_nostore         85.34             639.0 s
    tp8_mp              91.04             686.0 s

`nostore` is stock MP with `worker_adapter.batched_submit_store_requests`
replaced by a counting no-op.  Nothing else changes: the LOOKUP still goes out,
every other hook still runs, the connector is still installed on all 8 ranks.

**The steady-state per-step tax returns to exactly the baseline.**  85.34 vs
85.34.  The whole +5.70 ms/step is the store submission.

## The environment reproduced phase1 to four significant figures

    tp8_none    95,992.8 tok/s   vs  phase1 1c   95,997   ->  85.34 / 85.3
    tp8_mp      89,986.9 tok/s   vs  phase1 1j   89,990   ->  91.04 / 91.0

This matters because GPUs 4-7 had been held for 3.5 h by another tenant's
SGLang (qwen3.5-2b, `--mem-fraction-static 0.80`, ~120 GB/GPU, 0% util, four
scheduler threads busy-polling 0.6-0.8 core each) plus its LMCache server.
Those were cleared with sudo at 07:02 (restart command lines saved in
`results/killed_neighbours/2026-09-03_sglang.txt`).  Four-digit agreement with
runs from a different session is the evidence that the box is the same one.

## Where the 7.97 ms/step goes

Step probe, steady state, 264 windows across 8 workers:

    arm              loop     exec      cpu   blocked
    tp8_none        83.94    68.76    68.75      0.00
    tp8_nostore     85.71    68.22    68.19      0.04
    tp8_mp          91.90    73.89    72.04      1.85

    vs tp8_none:
    tp8_nostore    +1.77    -0.53    -0.57     +0.03
    tp8_mp         +7.97    +5.13    +3.29     +1.84

`nostore` returns `exec`, `cpu` and `blocked` all three to baseline.  The
residual +1.77 on `loop` sits with exec/cpu/blocked at zero, so it is time
spent OUTSIDE `execute_model` -- the scheduler side (LOOKUP, connector
metadata), which `nostore` leaves running.  So

    +7.97  =  6.20  the store submission, all of it inside/around the worker
            + 1.77  scheduler-side plumbing

## Three instruments, three numbers, and they reconcile

    in-engine steady state   +5.70 ms/step x 7200 steps  =  41.0 s
    end-to-end               686.0 - 625.2               =  60.8 s
                                                   difference  19.8 s

The 19.8 s is ramp and drain -- connection setup, first-store latency, and
draining pending stores after the last request.  The probe's `loop` delta
(+7.97 x 7200 = 57.4 s) lands near the end-to-end figure because its windows
include some of that.  So:

  * **+5.70 ms/step (+6.7%)** is the steady-state per-step tax.  This is the
    number with 14 cross-session blocks behind it and the number for an issue.
  * **+9.7%** is what a user's job actually costs, and a third of it is ramp
    and drain rather than steady state.

VAST reported end-to-end, so their figure is the second one.  The distinction
had not been written down before.

## THE OPEN QUESTION, and it is a sharp one

`nostore` skips only ~0.48 ms/step of in-hook work:

    sub_store_key   (_create_key, hashing)   0.246 ms/step
    sub_store_event (IPC event export)       0.040
    the MQ send                              ~0.19

Yet the worker main thread's CPU falls by **3.86 ms/step** (+3.29 -> -0.57).

So ~3.4 ms/step of host CPU is burned outside every hook that has a timer on
it.  It is `thread_time` -- CPU the main thread really consumed -- not waiting,
so it is not GPU contention (that would land in `blocked`) and not core
starvation (160 cores, load average 11).  One MQ message per step goes out and
the worker process itself then does 3.4 ms/step more Python.

Next step: cProfile `mp` and `nostore` over a matched step window and diff.
That is function-level attribution, not another guess.

## What is now excluded for loss #1

  * client concurrency -- TP=4, pool 1.92M: c=300 +0.84%, c=1000 +0.51%
  * pool depth 13.7M <-> 25.8M -- phase1 got 85.3 / 91.0 at both
  * bytes copied -- `bigl1` (8400 stores succeeded, 500 GB L1) cost the same as
    `mp` (56 succeeded, 8344 allocation failures)
  * backpressure, the LOOKUP path, and every individual hook
  * batching more requests per submit -- `sub_store_submit` and `sub_store_one`
    both count 2096 over 2200 steps, so the loop body runs exactly once per
    call: one store request per step, not N.  There is nothing to coalesce at
    this workload.

## Instrument note

`stepprobe.txt` holds CUMULATIVE lines: `loop_ms/step` there is (wall since the
probe was installed) / steps, so it carries the whole model-load and idle-wait
gap.  At TP=8 that gap is 452 s; at n=7200 it inflates `loop` by 63 ms/step and
the raw line reads 146.8 instead of 83.94.  Steady state must come from
`scripts/probe_report.py`, which differences consecutive lines.  The frozen
`lane_table.txt` was always produced that way and is unaffected.

The GPU watchdog added to `lane.sh` this session is WRONG: it flags on free
memory and utilisation without checking process ownership, so it marks our own
TP=8 arms DIRTY.  Ownership must come from
`nvidia-smi --query-compute-apps=pid` filtered against our own pids.
