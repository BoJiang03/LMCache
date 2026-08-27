# agentx reconfigured: the load was missing, then the cache was, and lazy lost

Conversation log for 2026-08-27 (session continues from 2026-08-26 evening).
The technical detail lives in `records/2026/08/26/13_*.md`, which was written
and extended live during this session; this file is the narrative and the
process notes.

## Code state

Nothing committed. `git status --short` is `?? lo_temp_ctx.md`, HEAD remains
`22c46cb6 Open trials on the loss ledger; recover only through probes`. No
source file was touched this session -- all work was measurement, harness
scripts under the par scratchpad, and records (`records/` is gitignored via
`/home/bo/LMCache/.git/info/exclude:19`). `lo_temp_ctx.md` is the handoff
scratch file and stays untracked rather than joining the publish branch.

The only edits outside records were to the benchmark harness, in scratchpad:

- `par/arm.sh`: client parameters (`ENTRIES`, `CONC`, `DUR`, `GRACE`, `SEED`)
  made env-overridable with the previous literals as defaults, so old rounds
  stay reproducible; and a small fix so the snapshot header reflects a
  per-arm `L1_GB` override instead of printing env.sh's default (that label
  had been wrong since the y-series -- the *server* was always configured
  correctly, only the printed label lied).
- `par/chain_b32.sh`, `par/chain_c32L.sh`: the two new rounds.

## The chain of questions

The session opened with the user asking three things at once: how agentx is
currently invoked, what the correct invocation would be, and whether the
correct one would actually show lazy's advantage.

**1. Reading the harness and the scenario lock.** The invocation turned out
to be scenario-valid -- `inferencex-agentx-mvp` auto-injects `ignore_eos`,
`cache_bust=first_turn_prefix`, the trajectory ratios and a 10 s idle-gap cap,
and logged all of it with no violations. So "we are running agentx wrong
against its own spec" was not the problem.

**2. The problem was that nothing was loaded.** `effective_prefill_concurrency`
p50 was **0**; the waiting queue averaged 0.02; `tokens_in_flight` (134K avg)
fit inside the 262K-token GPU pool; the whole run spent **7.9 s of 900 s** on
stores. LMCache was a bystander serving 8.6% of input tokens against a 93.5%
theoretical reuse rate. Conclusion stated to the user: this is structural,
not statistical -- more replicates would only tighten the error bar around an
effect the configuration holds near zero.

**3. The one legal load knob is concurrency.** The scenario forbids
`--ignore-trace-delays`, `--inter-turn-delay-cap`, `--trace-idle-gap-cap`, and
refuses `--request-rate` / `--fixed-schedule`. Concurrency (trajectory lanes)
is what remains. A preflight confirmed 32 lanes fill from a 42-trace eligible
pool without wrap -- and incidentally that `--num-dataset-entries` was already
saturated at 64, so 42 traces is the hard ceiling and **concurrency 64 is not
reachable in this scenario**.

## b32: the first result that separated by config

concurrency 32, 1800 s, L1=60 G, knob off, four arms interleaved
eager/lazy/lazy/eager so both a same-config eager pair and a same-config lazy
pair fell out of the same round.

The load landed: prefill concurrency p50 0 -> 15-16, watermark events 6 -> 152-210.
Lazy beat eager by 3.4-4.3% on median paired TTFT against a **+57 ms**
eager-eager control -- 44-55x the floor, and consistent across median TTFT,
mean request latency and total throughput. First agentx result in the whole
campaign that separated by configuration rather than by slot.

Two things surfaced that had never been visible: lazy issued 1/9 the store ops
at 6.8x the payload and retrieved **4.9x more tokens** back out of L1; and
lazy caused **~50x more vLLM preemptions** (159/147 unique request ids vs 1/3).
The preemption count was checked against the source before reporting --
`_has_preemption_reqs` reads `scheduler_output.preempted_req_ids` inside
`build_connector_meta`, which runs every step in both modes, so it is a
physical count and not a lazy-only reporting artifact.

## "This improvement is far below my expectation"

The user's reaction, and it was the right one. Working the arithmetic
backwards found the cap: one average context is 55087 tok x 96 KB = **5.04
GiB**, so 32 lanes is a **161 GiB** working set against GPU 24 + L1 60 = 84
GiB. GPU prefix hit had collapsed to ~0% at this concurrency. Measured
retrieve runs at ~2.0M tok/s against ~78K tok/s of prefill -- **a hit is ~25x
cheaper than the recompute it replaces** -- so ~17M of 21.8M input tokens per
arm were being recomputed for want of cache.

The honest split offered at that point: raising L1 grows the *absolute*
number but probably *shrinks* the lazy-over-eager gap, because lazy's edge in
b32 looked like eviction economy (retrieval ratio 1.14x at idle 90 G, 4.9x at
loaded 60 G). Raising pressure was the lever for a bigger *lazy* number.

That framing turned out to be wrong, and the round designed to test it said so.

## c32L: L1 sweep, and the reversal

concurrency 32 fixed, L1 swept 30 vs 180 GB, launch order flipped between the
two L1 points so a launch-order artifact would show as an inconsistency
between the two contrasts.

**L1 capacity dominates everything else in the campaign.** eager@30 vs
eager@180: median paired dTTFT **-39.6 s**, TTFT p50 -56%, throughput +55%,
requests +52%. At 30 GB the cache does not function -- 23.5M tokens written,
**30K read back, three retrieve operations in a half-hour run**. At 180 GB the
same code serves 76.7% of input from L1 while storing a quarter as much.

**And lazy's sign flips with L1**: -0.05% at 30 G (a wash), -3.4/-4.3% at 60 G
(b32's win), **+16.8% against lazy at 180 G**. At the operating point that
actually matters, lazy costs 16.8% TTFT and 7.7% throughput.

Mechanism, from the ledger: `admitted=1560 emitted=835 dropped_evicted=690` --
**44% of admitted stores never got emitted** because the GPU blocks were
recycled first. At 30 G the same arm dropped 125 of 3858. The difference is
wall clock: at 180 G the system runs ~50% faster, so a deferred store has less
time to survive. **Deferral is a bet that the block will still be there later,
and the bet gets worse exactly as the system gets healthier.**

The pre-stated falsifier fired. "Lazy's edge is eviction economy" is dead; the
edge is not monotone in pressure, and b32's 60 G win is a local optimum
between "nothing survives to be cached anyway" and "deferral loses the race".

## What this session establishes

1. agentx as previously run measured nothing, for a nameable structural
   reason, and the fix is concurrency (the only knob the scenario leaves open).
2. L1 sizing relative to the live working set is the dominant performance
   lever for this workload by roughly an order of magnitude over lazy-vs-eager.
3. lazy vs eager is a small, sign-unstable effect whose sign depends on L1,
   and at a properly sized L1 lazy is the worse choice with the knob off.
4. b32's *measurement* stands (it had both control pairs); its
   *interpretation* has been narrowed to "true at L1 ~= 60 G with this working
   set".
5. The degradation knob has, for the first time, a documented failure mode it
   was built to repair: a 44% drop share clears `_MATERIAL_LOSS_SHARE = 0.25`
   by a wide margin, so at L1=180 the controller should open a trial and
   commit to immediate emission.

## Prediction scorecards

b32: 2 hit, 3 missed. The three misses were all calibration (noise floor
underestimated 12x on the trimmed mean; store-shape understated 3x; absolute
retrieval level predicted too high). The decisive one -- direction, with its
falsifier pre-stated -- hit.

c32L: 2 hit, 1 partial, 2 missed, **including the decisive one**. Predicted the
gap would shrink at 180 G and grow at 30 G; it vanished at 30 G and reversed at
180 G. Also predicted lazy's `dropped_evicted` would fall at 180 G; it went up
5.5x, and that turned out to be the finding of the round.

## Process notes

- Killed my own background waiter from the previous round: its
  `pgrep -f 'par/chain_b32.sh'` **matched its own command line**, so it looped
  forever and never fired while b32 had finished hours earlier. New waiters
  poll for a done-marker string in the log instead. The user caught this
  before I did.
- A 260 GB MP server on port 27137 (`vllm-mm` venv, GPU 3) belongs to the
  multi-modal validation line, not this session. Left alone; only counted
  against the host-memory budget for the 420 GB this round needed.
- A `records/2026/08/27/../26/...` heredoc redirect failed silently and the
  `&& echo "record appended"` after an unrelated `ls` printed success anyway.
  Caught and redone. Chained `&&` after a redirect is not evidence the
  redirect worked.
- Pre-stating predictions with explicit falsifiers is now two-for-two on
  catching a wrong conclusion: it killed the agentx sumD verdicts on 08-26 and
  killed the eviction-economy reading today.

## Open

1. `lazy@180 DEGRADE_SECS=450` vs `lazy@180` off vs `eager@180`, with a
   same-config control. Pre-committed success criterion: the knob should close
   most of the +5225 ms gap and drive `dropped_evicted` well below 690.
2. Replicate eager@30 vs eager@180 -- the largest effect in the campaign,
   currently resting on one round.
3. A mid-load point (concurrency 16) to separate saturation TTFT from overload
   throughput. b32/c32L both ran deep overload (TTFT p50 31-73 s).
4. Older queue, still untouched: hot/cold replication beyond three seeds, the
   GSM8K coverage probe for 5ea3cc6e.
