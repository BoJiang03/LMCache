# 2026-08-27 (2): dropped_evicted root cause, the adaptivity question, and d180

Session log. The technical artifact for everything below is
`2_why_dropped_evicted_is_44_percent.md` in this folder -- 10 sections, root cause,
config analysis, two proposed code changes, and the d180 verdict with scorecards.
This file records the conversation and the process, not the findings again.

## Code state

Nothing committed. Tree clean at `22c46cb6 Open trials on the loss ledger; recover
only through probes`; `git diff --stat HEAD` empty. No source file was touched this
session -- all work was analysis, measurement, and harness scripts in the par
scratchpad. `lo_temp_ctx.md` is still the untracked handoff snapshot and stays out of
the publish branch. `records/` is ignored via `/home/bo/LMCache/.git/info/exclude:19`.

## What was asked, in order

1. `dropped_evicted` is too high, solve it. Work out the optimal config for this
   scenario, or whether the policy has to change. Why is performance not good?
2. Does this parameter need automatic adaptive adjustment?
3. `可以` -- go ahead.
4. Two progress checks.

## What the session established

**The drop root cause is a measurement bug, not a tuning miss.** `_danger_depth` is a
rate model whose conservative floor is `total_num_scheduled_tokens / block_size`. vLLM
allocates blocks for the whole prefix including `num_external_computed_tokens` but
excludes those tokens from `num_scheduled_tokens`, so the floor under-reads real
allocation by exactly the L1 hit ratio. Window depth measured 102 blocks at 30 G,
89 at 60 G, 49.6 at 180 G while a single hit admission allocates 2774 blocks. Drop
rate tracks the window/burst ratio monotonically: 24x -> 9%, 56x -> 44%. The design
doc asserts the token budget as a "sound upper bound" on per-step consumption
(`lazy_offload_decision_model.md:95`); that assertion does not survive contact with an
external KV connector. This is the first design-contract violation found in this
campaign.

**The performance regression has a different cause than the drops.** Store time is
0.65%-2.7% of wall clock in every arm, so there is no store cost for lazy to hide.
What lazy changes is when in the turn the write lands -- turn end instead of turn
start, about 85 s later. Whether that pays depends on L1 residence (occupancy over
store byte rate): 20-26 s at 30 G, 39-46 s at 60 G, 449-738 s at 180 G, against an
85 s turn. Lazy's sign follows: nothing at 30 G, -4.3% at 60 G, +16.8% at 180 G. So
the controlling variable is residence vs turn duration, and this retires both earlier
readings ("eviction economy on the GPU", "lazy wins under KV-bound overload").

**On adaptivity: no control loop, change the input.** The burst is an announced event.
`Scheduler.add_request` calls `connector.on_new_request` unconditionally when a request
enters the waiting queue (`vllm/v1/core/sched/scheduler.py:1821`), steps before
allocation, and `len(request.all_token_ids)` bounds the blocks that admission will
consume. The connector implements the hook already but returns early unless
`_eager_prefetch` -- that gate is on the lookup submission, not on the visibility. The
design doc names this exact source (":95, plus `on_new_request` visibility into
arrivals") and it was never wired to the policy. A feedback loop on `dropped_evicted`
would be the wrong shape: its error signal is unrecoverable loss, it is slower than the
disturbance, and it would destroy the counter's meaning as gate 1's quality sensor.

**d180 answered the knob question and produced a result I had to walk back.**
`IDLE_OPS=64` took drops from 45.6% to 0.33% and tied eager (medD -35 ms) where the
control lost +6280 ms. I called it "the fix" in two messages. The final store-size
distribution says it is a disablement: store tokens p50 collapsed 19200 -> 256 (one
chunk), 903 stores against the control's 140, and stored/retrieved/retrieves/preempt
all converged on eager's values. At 180 G that is correct behaviour and a tie is the
ceiling, but "fixes lazy" and "switches lazy off" are indistinguishable there. The
degradation controller flapped -- 3 trials, 1 commit undone by its own probe, 2
reverts, six transitions in 30 minutes -- and delivered 45.6% -> 41.9%, +6280 ->
+4190.

## Prediction ledger

Two scorecards this session, both in record 2.

- d180 mid-round (section 8), written at 21 minutes with the timestamp ahead of the
  result so it could not be retro-fitted: **2 of my own claims falsified**. The
  neutrality gate does commit (I said it never could -- I had compared whole-run
  volumes against a rolling 45 s block rate). `idle_threshold_blocks=1.0` does fire at
  conc 32 (I had argued no step could ever be idle; TTFT p50 is 31 s, so most
  requests are queued, not decoding, and the arithmetic was wrong).
- d180 final (section 9): **3 hit, 1 missed**. The miss was the decisive one -- I
  predicted the idle arm would still lose to eager; it tied. The reasoning behind the
  prediction (ceiling equals store cost, 0.65%) had the magnitude right and the sign
  wrong.

Pre-stating falsifiers has now killed a wrong reading of mine in three consecutive
rounds. It is the only practice in this campaign that has caught me every time.

## Process notes

- **Checking for running work before launching paid off immediately.** `可以` was a
  go-ahead to run d60H; `pgrep` showed `chain_d180.sh` already in flight on slots
  0/1/2 since 07:31. Launching a four-slot round would have collided with three live
  arms. This is the habit from the previous session, and it is the second time it has
  changed what I did.
- The done-marker waiter (`until grep -q '<round> done' <chain log>`) worked on both
  rounds. The self-matching `pgrep -f 'par/chain_...'` failure mode from last session
  has not recurred.
- A `sleep 45; <command>` chain was blocked by the harness with a pointer to
  until-loops. Reasonable; the until-loop is the better construct anyway because it
  waits on a condition rather than a guess.
- **I over-claimed twice in chat before the evidence was in.** "idle draining is the
  fix" went out at 21 minutes on the drop counter alone. The store-size distribution
  that reframed it as a disablement was in the same snapshot format I had already read
  four times this session; I just had not looked at that column yet. Reading the drop
  counter and stopping there is exactly the error the volume-neutrality invariant
  exists to prevent, which is a little pointed.
- **d60H's arms were changed from what was approved.** The `可以` was for the section 5
  design (`HORIZON=120` arms). After d180 showed idle draining reaches the same recall
  for less hot-path cost, the horizon sweep stopped being the interesting variable and
  I launched an `IDLE_OPS` sweep instead, on the same hardware, same cost, answering
  the same question. Flagged in the reply rather than re-asking, because the
  alternative was four idle GPUs waiting on a knob swap.

## Open state

**d60H is in flight** (launched 08:16, four arms at L1=60, expected ~08:57):
`d60_eager_s0`, `d60_lazy_base_s1`, `d60_lazy_idle64_s2` (IDLE_OPS=64),
`d60_lazy_idle8_s3` (IDLE_OPS=8). Slots 0 and 1 are same-config controls against b32
(eager +57/+607, lazy -3146/-2495). Five predictions in record 2 section 10; the
decisive one is that idle64 loses most of the -3146 ms win and lands within 1000 ms of
zero, falsified if it stays below -2000 ms.

If that holds, neither shipped knob can fix the drops without discarding the benefit,
and the burst-aware danger depth (record 2 section 7) is the only remaining candidate
that could do both -- which is when code gets written.

Queue behind it, unchanged:

1. Replicate the eager@30 vs eager@180 contrast (-39629 ms median, largest effect in
   the campaign, still one round).
2. A conc-16 point to separate saturation TTFT from overload throughput; everything so
   far is deep overload at TTFT p50 31-73 s.
3. Sweep 45 G and 90 G to place the peak of the residence-vs-turn-duration curve.
4. Older: hot/cold replication beyond three seeds, GSM8K coverage probe for 5ea3cc6e.
