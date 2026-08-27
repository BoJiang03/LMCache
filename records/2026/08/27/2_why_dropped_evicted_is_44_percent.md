# Why dropped_evicted is 44%, what the best config is, and what the policy gets wrong

Question put to me: `dropped_evicted` is too high, solve it. Work out what config is
optimal for this scenario, or whether the policy itself has to change. What is the
reason performance is not good?

This is analysis only. No round was run. Everything below is read off the archived
c32L and b32 snapshots and the source.

## 0. The four measurements this is built on

All arms: conc 32, 1800 s, agentx MVP scenario, Qwen3-Coder-30B-A3B, 24 GiB GPU KV
pool (16384 blocks of 16 tokens, 1.5 MiB each), `DEGRADE_SECS=0`, `MAX_PENDING=0`,
`IDLE_OPS=0`, `BLOCK_CAP=0`, `HORIZON=2.5`, `min_prefix_tokens=0`,
`max_drain_per_step=64`. Every mitigation knob the policy has was off.

| arm | L1 | admitted | emitted | dropped | drop% | window depth (blk) | tokens stored | retrieves | preempt |
|---|---|---|---|---|---|---|---|---|---|
| c32_lazy_l30_s3 | 30 G | 3858 | 3654 | 125 | 3.2% | 102.5 | 17.32M | 16 | 194 |
| b32_lazy_s1 | 60 G | 3481 | 3017 | 340 | 9.8% | 89.2 | 16.84M | 94 | 170 |
| b32_lazy_s2 | 60 G | 3525 | 3107 | 297 | 8.4% | 89.2 | 17.18M | 77 | 157 |
| c32_lazy_l180_s1 | 180 G | 1560 | 835 | **690** | **44.2%** | **49.6** | 3.73M | 464 | 35 |

"window depth" is `free_queue_blocks_read / drain_steps`, i.e. the mean
`_danger_depth()` the policy actually ran at.

`throttled_drains = 0` in all four. `pending` p50 is 32 @30 G, 14 @180 G. So neither
the drain caps nor the backlog depth is the cause -- `max_drain_per_step`,
`max_drain_blocks_per_step` and `max_pending_ops` are all ruled out by the data
before any experiment.

## 1. Root cause of the drops: the danger window is 55x smaller than the burst

`_danger_depth()` (eviction_aware.py:2053) is

    per_step = max(EMA(new_blocks_allocated), est_next_step_blocks)
    depth    = ceil(per_step * horizon_steps)          # horizon_steps = 2.5

and `est_next_step_blocks` is fed from
`ceil(scheduler_output.total_num_scheduled_tokens / tokens_per_block)`
(lazy_offload_manager.py:537).

An op dies when a block it holds is reallocated before the drain sees it inside the
window. So the recall of the whole mechanism is `window / burst`: a single allocation
of B blocks sweeps the free-queue head past the window entirely, and any op sitting in
`[window, B)` is destroyed without ever being observed as due.

Measured burst size, from the retrieve ledger: at 180 G one retrieve loads
44389 tokens = **2774 blocks**, allocated in one scheduler step. Window is 49.6.
Ratio 56x. At 60 G a retrieve is 34400 tokens = 2150 blocks against a window of 89.2.
Ratio 24x. Drop rate tracks the ratio monotonically: 24x -> 9%, 56x -> 44%.

### Why the window *shrinks* as L1 grows

This is the part that makes it a bug rather than a tuning miss. In vLLM,

    num_computed_tokens = num_new_local_computed_tokens + num_external_computed_tokens
    num_new_tokens      = request.num_tokens - num_computed_tokens

(`vllm/v1/core/sched/scheduler.py:640,682`). Blocks are allocated for the **whole**
prefix, including `num_external_computed_tokens` -- the LMCache hit -- but the hit
tokens are excluded from `num_scheduled_tokens`. GPU-local prefix hits reuse existing
blocks and cost nothing; only the *external* hit allocates fresh blocks it will not
compute into.

So `est_next_step_blocks` under-reads real allocation by exactly the L1 hit ratio:

| L1 | retrieved/ISL | window depth |
|---|---|---|
| 30 G | 0.1% | 102.5 |
| 60 G | 14.8% | 89.2 |
| 180 G | 76.7% | 49.6 |

The token-derived term is what kept the window at 102 blocks at 30 G. It collapses
precisely when L1 starts serving the prefill. **The forecast is blindest exactly where
the product succeeds.**

The EMA term does see the burst -- `_count_new_blocks` counts new requests' full
`block_ids` -- but at alpha 0.3 and one step late, so it widens the window *after* the
burst has already destroyed the backlog. That is why 56% survive rather than the 2%
the window/burst ratio alone predicts: they are swept up by the post-burst wide window,
not protected by a pre-burst one.

### The design doc asserts the bound that vLLM does not honour

`docs/design/integration/vllm/lazy_offload_decision_model.md:70-75`:

> *One-step allocation feedforward*: the scheduler has already fixed the next step's
> token budget (`num_scheduled_tokens`), so next-step block consumption is
> near-deterministic; drain at least that many head blocks.
> Residual uncertainty: intra-step allocation bursts (bounded by one step).

and at :95

> only the per-step *cutoff* is uncertain, and it has a sound upper bound -- the
> scheduler's own token budget (`max_num_batched_tokens` / block_size ...)

That upper bound is unsound in the presence of an external KV connector. The doc
already names "intra-step allocation bursts" as the residual and calls them "bounded
by one step" -- true, but one step is 2774 blocks, 17% of the pool. This is a
design-contract violation, not a mis-tuned constant.

## 2. Root cause of the performance regression: it is not the drops, it is L1 residence

Separate question, and the answer changes what "optimal config" means.

Store time is negligible in every arm: eager spends 11.7 s @180 G, 46.6 s @60 G,
47.8 s @30 G of store time inside 1800 s (0.65% - 2.7%). **There is essentially no
store cost for lazy to hide.** Whatever lazy wins, it does not win by moving D2H
copies out of the way.

What lazy actually changes is *when in the turn* the write lands: eager writes turn
N's prefix during turn N's prefill; lazy writes it after the request's blocks are
freed, i.e. at turn end -- about 85 s later (b32 eager mean request latency
85464 ms). Whether that matters depends on L1 residence:

L1 residence = occupancy / store byte rate (steady state, 96 KiB per token):

| arm | occupancy | store rate | residence | vs 85 s turn | retrieves |
|---|---|---|---|---|---|
| eager @30 G | 24 GiB | 1.193 GiB/s | 20 s | << | 3 |
| lazy @30 G | 22 GiB | 0.881 GiB/s | 26 s | << | 16 |
| eager @60 G | 45 GiB | 1.146 GiB/s | 39 s | ~half | 20 |
| lazy @60 G | 39 GiB | 0.857 GiB/s | 46 s | ~half | **94** |
| eager @180 G | 124 GiB | 0.277 GiB/s | 449 s | >> | 505 |
| lazy @180 G | 140 GiB | 0.190 GiB/s | 738 s | >> | 464 |

That is the whole story of the L1 sweep:

- **30 G, residence 20-26 s.** Content dies mid-turn no matter who wrote it. Both
  arms retrieve nothing. Measured paired dTTFT: -36 ms (-0.05%). Nothing to win.
- **60 G, residence 39-46 s vs an 85 s turn.** Eager's write must survive the rest of
  its own turn plus the inter-turn gap; it does not. Lazy's write only has to survive
  the gap; it does. 4.7x the retrieves off 25% *less* stored volume. Measured
  -3146 / -2495 ms (-4.3% / -3.4%) in lazy's favour.
- **180 G, residence 449-738 s.** L1 never fills (140 of 180 GiB, 7 watermark events).
  Write timing is irrelevant; both survive. All that is left is lazy's costs: 44% of
  intake dropped -> 32% less stored -> 20% fewer tokens retrieved, plus 5x the
  preemptions. Measured **+5225 ms (+16.8%) against lazy**.

So the controlling variable is **L1 residence relative to turn duration**, and lazy's
value band is where they are comparable. This retires the earlier "eviction economy on
the GPU" reading for good; it also retires the b32 reading that lazy simply wins under
KV-bound overload.

### Second cost channel: the pin burst

Lazy coalesces: mean store 39822 tokens @60 G = 2489 blocks pinned at once, 15% of the
16384-block pool, versus eager's 367 blocks (2.2%). Preemptions: 170/157 vs 1/3 at
60 G, 35 vs 7 at 180 G. Roughly 0.4 preemptions per lazy store. This tracks the
*peak* pin, not the time-integral (integral ratio is only 2-3x), so the lever is
coalescing width, not `max_drain_blocks_per_step` (whose cap is soft and lets the
crossing op through whole).

## 3. What the degradation controller would do, and why it would be wrong

`DEGRADE_SECS=0` in every arm, so the controller has never run at this load. Reading
the machine against these numbers:

**At 180 G it would trial and then revert, forever.** `_loss_is_material` fires
immediately (44% >> `_MATERIAL_LOSS_SHARE` 0.25) even though residence 738 s never
crosses any sane threshold. A trial of immediate emission then emits at eager's rate.
The gate is

    trial_rate <= _NEUTRALITY_FACTOR * baseline        # 1.25

where the baseline is `_trailing_emitted_rate`, built from `_emitted_blocks_total`,
which `_note_emitted` advances **only for emitted ops**. Deferred baseline 3.73M
tokens/1800 s; immediate ~5.45M. Ratio 1.46 > 1.25 -> revert, 600 s cooldown, repeat.
`degrade_commits` would stay 0 and `degrade_reverts` would climb.

The bug: the gate reads a volume increase as "deferral was filtering stores out"
(a benefit worth preserving) when here it is "deferral lost 44% of them to eviction"
(a defect). The counter that separates the two, `dropped_evicted`, is used to *open*
the trial but never to *interpret* it. **The neutrality baseline should be intake --
emitted + dropped -- not emitted alone.** With drops charged, the 180 G baseline
becomes ~6.8M against an immediate 5.45M, ratio 0.80, and the trial commits.

**But charging drops alone would then also let it commit at 60 G**, where committing
throws away the only win we have (baseline 18.7M vs immediate 22.5M = 1.20 < 1.25).
It is saved there only by the trigger: 9.8% loss share does not reach 0.25, so a trial
opens at 60 G only if the *residence* trigger fires -- and residence there is 39-46 s,
so any `degrade_l1_residence_secs` above ~50 fires it. Given section 2, a residence
*floor* trigger has the polarity backwards for this workload: lazy wins at short
residence and loses at long residence. The residence trigger's stated rationale (pin
collision under churn) is real but is outweighed at 60 G by a factor of 4.7 in
retrieves.

## 4. Answers

**Why is performance not good?** Two independent reasons, neither of which is the one
the policy was tuned for:

1. At 180 G there is nothing to win. Store cost is 0.65% of wall clock, so lazy's
   ceiling there is +0.65%; its measured floor is -16.8%. No configuration fixes that.
   The right behaviour at large L1 is to not defer.
2. Where there *is* something to win (60 G), 9% of intake is still being thrown away
   by the same burst blindness, and each dropped op is cache content that would have
   been retrieved. Fixing it should *extend* lazy's win, not just tidy a counter.

**Best config for this scenario, no code change.** In priority order:

- `horizon_steps` 2.5 -> ~120. This is the one that matters. The window has to be
  sized against the burst, not against the step: 2774 blocks / 23 blocks-per-step
  = 120 steps. The usual objection -- draining earlier destroys the deferral -- does
  not apply here, and this is worth stating carefully because it is the crux. The
  window is 89 blocks ~ 3.7 steps ~ 105 ms wide today; at horizon 120 it becomes
  2760 blocks ~ 120 steps ~ 3.4 s. The write time that produces the 60 G win is set
  by when the request's blocks are *freed and recycled*, which is turn-scale (85 s),
  not window-scale. Raising the horizon 48x moves the write about 3 s earlier out of
  an 85 s turn -- 4% -- against a 46 s residence. The timing win survives; the recall
  loss does not.
  Cost to name: `_FreeQueueWindow.extend_to(depth)` walks the free queue every drain
  step, so this is a linear increase in scheduler-hot-path work (4.2M block reads
  today -> ~240M). Needs to be watched, and it argues for bounding the walk in the
  code fix rather than living on a large horizon forever.
- `idle_threshold_blocks` 1.0 -> ~4.0 with `idle_drain_max_ops` 8-16. The default is
  mis-sized for this concurrency: 32 decoding requests allocate ~32 tokens = 2
  blocks/step, above the 1.0 threshold, so **no step at conc 32 can ever be classified
  idle** and idle draining cannot engage even when enabled. This is the second lever
  that removes ops from the backlog before a burst can reach them.
- L1 = 60 G, not 180 G. And at 180 G run eager.
- `DEGRADE_SECS` = 0 at 60 G. Leaving it on with a threshold above ~50 s would open
  trials against the winning regime.
- `max_pending_ops`, `max_drain_blocks_per_step`, `max_drain_per_step`: leave alone.
  Ruled out by `throttled_drains=0` and by the pending depth already sitting at 14-32.

**Does the policy need to change?** Yes, two changes, both small and both correctness
rather than tuning:

1. **Burst-aware danger depth.** `_danger_depth` must not depend on a token-derived
   estimate that is structurally blind to external-hit allocation. Feed it a decaying
   running max of observed single-step gross allocation (`new_blocks_allocated` is
   already passed into `observe_step`) and take
   `depth = max(ceil(rate * horizon), recent_burst_max)`. This is what raising
   `horizon_steps` approximates by hand, but self-sizing and without paying the deep
   free-queue walk on quiet steps. It also makes the policy converge to near-eager
   behaviour under bursty hit-driven allocation *by construction*, which is the
   correct behaviour at large L1 and removes most of the controller's job.
   The design doc's gate-1 recall argument (:88-99) has to be amended with it.
2. **Neutrality gate charges dropped volume.** `_note_emitted` /
   `_trailing_emitted_rate` / `_regime_emitted_rate` compare emitted blocks; the
   invariant they defend is about *volume*, and dropped volume is lost volume. Baseline
   must be emitted + dropped. Without this the controller can never commit in the one
   situation it was built for -- deferral bleeding its own intake -- because the bleed
   makes deferral look cheap.

Both are independent of each other, and (1) largely obviates the need for (2) to fire.

## 5. Proposed next round, with the criterion written down first

Not started; needs a go-ahead.

**d60H** -- conc 32, 1800 s, L1 = 60 G, four arms, no code change:

    eager:d60_eager_s0
    lazy:d60_lazy_h120_s1:HORIZON=120
    lazy:d60_lazy_h120_idle_s2:HORIZON=120,IDLE_OPS=16,IDLE_THRESH=4.0
    lazy:d60_lazy_base_s3:HORIZON=2.5

Slot 3 is the same-config control against b32_lazy_s1/s2 (measured floor: medD -3146
and -2495 ms, so the noise band is about 650 ms from the b32 eager-vs-eager control).

Pre-stated predictions, falsifiers included:

1. `dropped_evicted` on the h120 arm falls below 100 (from 340/297, a 9.8%/8.4% share
   to under 3%). **Falsified if it stays above 200.** This is the decisive one -- if
   the window/burst ratio is not the mechanism, this number will not move.
2. `tokens_stored` on the h120 arm rises to 18.5-20M (from 16.84M), i.e. roughly the
   dropped volume comes back. **Falsified if it stays under 17.5M.**
3. `retrieves` on the h120 arm rises above 94 -- more surviving stores means more
   later hits. Confidence lower here (retrieves depend on trace structure, and 94 vs
   77 across two identical arms is already a 20% spread). **Falsified if under 77.**
4. Median paired dTTFT on the h120 arm beats b32's -3146 ms, i.e. more negative than
   -3500. Weakest prediction of the four; the drop recovery could be worth less in
   TTFT than the arithmetic suggests. **Falsified if it lands above -2000 ms.**
5. The idle arm does *not* separate from the h120 arm by more than the control band
   (650 ms). Idle draining and a deep window attack the same ops; I expect the deep
   window to have already taken them. **Falsified if the idle arm beats h120 by more
   than 1500 ms.**

Explicitly *not* predicted: that preemptions fall. Nothing in this round touches the
coalescing width, which is what drives the pin burst.

## 6. What this does not settle

- The eager@30 vs eager@180 contrast (-39629 ms median, the largest effect in the whole
  campaign) still rests on one round.
- Everything here is deep overload (TTFT p50 31-73 s). A conc-16 point is still needed
  to separate saturation TTFT from overload throughput.
- The residence-vs-turn-duration model in section 2 is inferred from six arms across
  three L1 sizes. It predicts a peak in lazy's advantage near residence ~= turn
  duration and no advantage on either side. That is a testable curve and nobody has
  swept it: 45 G and 90 G would place the peak.

---

## 7. Follow-up: does this parameter need adaptive tuning?

Asked after the analysis above. Short answer: no control loop, but the input has to
change. The reason `horizon_steps` looks like it needs to adapt is that it is
currently the only lever available to compensate for a missing measurement. Supply
the measurement and the knob goes back to being uninteresting.

### The cause is announced in advance, through a hook that already fires

`Scheduler.add_request` calls `connector.on_new_request(request)` unconditionally the
moment a request is enqueued into the waiting queue
(`vllm/v1/core/sched/scheduler.py:1821`) -- steps, often many steps, before it is
admitted and its blocks are allocated. At that point `len(request.all_token_ids)` is
known, so `ceil(num_tokens / block_size)` is an upper bound on the blocks that
admission will consume in one step. Requests leave the set at
`update_state_after_alloc` or on finish/abort.

LMCacheMPConnector already implements the hook (`lmcache_mp_connector.py:837`) but
returns early unless `_eager_prefetch`; that early return gates the *lookup
submission*, not the *visibility*. The arrival information is there for free.

So the burst is not a random variable that has to be estimated -- it is an announced
event. Feeding it forward turns gate 1 from prediction into arithmetic, which is the
stated design philosophy of the whole decision model ("each gate can be built so the
probability never has to be estimated", decision_model.md:185). The doc even names
this exact source: "plus `on_new_request` visibility into arrivals" (:95). It was
specified and never wired to the policy.

Caveats to state honestly:

- The bound over-reads: `num_tokens` includes tokens a GPU-local prefix hit will
  serve by reusing existing blocks, which allocate nothing. Tightenable later with
  the local hit count; over-reading is safe for recall and costs only filtering and
  free-queue walk.
- More than one request can be admitted per step, so a single max is not a bound on
  the step -- the sum over admissions is. The sum over the whole waiting queue is a
  bound but can reach the pool size and would disable deferral outright. Practical
  form: sum of the top-K waiting requests by size, K = plausible admissions per step.
  Multi-admission steps stay partly blind; that is a smaller residue than the one we
  have now.

### Why a feedback loop on the drop counter would be the wrong shape

The obvious adaptive design is AIMD on `dropped_evicted`: widen the window when drops
rise, narrow it when they fall. It should not be built.

- **The error signal is the damage.** You learn the window was too narrow only by
  irreversibly losing cache content. A controller whose error term is unrecoverable
  loss pays for every unit of learning in the currency it exists to protect.
- **The loop is slower than the disturbance.** Bursts arrive roughly every 137 steps
  at 180 G; convergence needs many bursts, so the loop would spend most of the run
  mis-sized in one direction or the other, and would oscillate against the EMA that
  already widens the window one step after each burst.
- **It would mask the sensor.** `dropped_evicted` is gate 1's quality readout and the
  `_loss_is_material` trigger. A loop that drives it to a setpoint destroys its
  meaning as evidence.

The acceptable reactive fallback, if the feedforward is too much for one PR, is a
decaying **high quantile** (not the max) of observed single-step gross allocation,
which is adaptation to a slow, stable property -- the trace's context-length
distribution -- rather than a fast one. A plain running max has a bad failure mode:
one pathological admission sets the floor to a large fraction of the pool and
suppresses deferral until it decays.

### Where adaptation genuinely is needed, and it is not this parameter

Section 2 established that the controlling variable for whether deferral is worth
anything is **L1 residence vs turn duration**, and that it flips lazy's sign
(-4.3% at 46 s residence, +16.8% at 738 s). That really does vary across deployments
and drifts within a run as L1 fills. That is the decision worth adapting, it already
has a home with a bounded trial/commit/probe protocol, and what it needs is not more
adaptivity but a corrected baseline (charge dropped volume, section 3) and a rethink
of the residence trigger's polarity.

There is also an interaction to watch: fixing the danger depth collapses
`dropped_evicted`, which removes the `_loss_is_material` trigger and leaves the
controller relying on the residence trigger alone -- the trigger whose polarity is
wrong for this workload. The two changes must be reasoned about together, not landed
as independent improvements.

### Sequencing

Validate the mechanism with a **static** horizon first (round d60H, section 5). If
prediction #1 fails -- `dropped_evicted` does not collapse at `HORIZON=120` -- then
the window/burst model is wrong and any adaptive machinery would have been built on
it. Adaptive control on an unvalidated mechanism is the worst of the available
outcomes. A static horizon is also reproducible, which matters while the campaign is
still running A/B arms against a measured noise floor.

---

## 8. Mid-round falsification (d180, 07:52, 21 min into a 33 min run)

Written before the round finished, so the scoring cannot be retro-fitted. d180 was
already in flight (launched 07:31) when the section 4-7 analysis above was written;
its live counters falsify two of that analysis's claims.

Three lazy arms at L1=180, conc 32, all else default:

| arm | admitted | emitted | dropped | drop share |
|---|---|---|---|---|
| d180_lazy_ctl_s0 (defaults) | 890 | 462 | 392 | 44.0% |
| d180_lazy_knob_s1 (DEGRADE_SECS=450) | 901 | 601 | 276 | 30.6% |
| d180_lazy_idle_s2 (IDLE_OPS=64) | 837 | 810 | **3** | **0.36%** |

The control reproduces c32L's 44.2% to within a tenth of a point, so the effect is
real and repeatable.

**Falsified claim 1 (section 3).** I predicted the degradation controller could never
commit -- that the neutrality gate would read immediate emission as a 1.46x volume
increase and revert forever, leaving `degrade_commits=0` and `degrade_reverts`
climbing. Measured: `degrade_trials=1 degrade_commits=1 degrade_reverts=0
degrade_probes=1`. It committed on the first trial.

The error: I compared lazy's whole-run emitted volume (3.73M tokens) against eager's
whole-run stored volume (5.45M). The gate does not compare those. It compares
`_regime_emitted_rate` -- blocks per second since the regime was entered -- against
`_trailing_emitted_rate`, a **rolling 45 s block rate** sampled at the instant the
trial opens. A whole-run average is not that quantity, and deferral's emission is
bursty (the post-burst wide window flushes in a clump), so the trailing window can be
sampled high. The section 3 conclusion that the gate is structurally blind to dropped
volume still stands as a code reading -- `_note_emitted` counts emitted blocks only --
but the claim that this blindness *prevents commitment in practice* is wrong.

Note what commitment bought: 44.0% -> 30.6%. The controller fires, commits, and does
not fix the problem. It is currently sitting in PROBE with `degraded_emitted=150` out
of 601, i.e. degraded for a minority of the run.

**Falsified claim 2 (section 4).** I argued `idle_threshold_blocks=1.0` is mis-sized
for concurrency 32, on the arithmetic that 32 decoding requests allocate ~32 tokens =
2 blocks per step, above the threshold, so no step could ever be classified idle.
Measured on the untouched default: `idle_drain_steps=348`, `idle_emitted=575`.

The error: TTFT p50 is 31 s at this load, so requests spend most of their life in the
waiting queue, not decoding. The number of *running* requests at any instant is far
below the 32 lanes held open, and plenty of steps schedule almost nothing. The
recommendation to raise `IDLE_THRESH` to 4.0 was built on that bad arithmetic and is
withdrawn.

**What survives.** The window/burst model of section 1 is supported, not challenged:
idle draining works by emptying the backlog in the gaps so a burst finds nothing to
destroy, which is exactly the mechanism the model predicts would help. But it changes
the recommendation. `idle_drain_max_ops=64` on stock settings takes the drop rate from
44% to 0.36% with **no code change and no deep free-queue walk**, where the
section 4/5 proposal (`HORIZON=120`) would have paid for the same recall with a
linear increase in scheduler-hot-path work (4.2M block reads per run -> ~240M).
Idle draining is the better answer and it was already in the tree.

**What this reopens.** 71% of slot 2's emissions happened on idle steps, i.e. the
stores no longer land at turn end -- they land in whatever gap comes first. At 180 G
that is free, because residence is 738 s and write timing is irrelevant (section 2).
At 60 G, where lazy's entire 4.3% win comes from writing ~85 s later than eager
against a 46 s residence, emitting early may destroy the win while fixing the drops.
That is now the question d60H has to answer, and it is a different question than the
one section 5 posed.

Predictions for d180's final numbers, stated now:

1. The idle arm still loses to eager@180 on median paired dTTFT -- fixing the drops
   does not make lazy worth running where store cost is 0.65% of wall clock and
   residence is 738 s. **Falsified if the idle arm beats c32_eager_l180_s2.**
2. The idle arm beats the control arm by a wide margin (control was +5225 ms against
   eager; the idle arm should recover most of that). **Falsified if the idle arm is
   within the 650 ms control band of the ctl arm.**
3. `tokens_stored` on the idle arm lands near eager@180's 5.45M, since almost nothing
   is dropped now. **Falsified if under 4.5M.**
4. The knob arm lands between the two, closer to the control. **Falsified if it beats
   the idle arm.**

---

## 9. d180 verdict (finished 08:12:42)

Baseline for the paired comparison is `c32_eager_l180_s2` (n=634, sumTTFT 20578.3 s,
TTFT p50 31163 ms). All three arms lazy at L1=180, conc 32, 1800 s.

| | eager@180 | ctl (defaults) | knob (DEGRADE=450) | idle (IDLE_OPS=64) |
|---|---|---|---|---|
| median paired dTTFT | -- | **+6280** | **+4190** | **-35** |
| pairs | 634 | 588 | 599 | 632 |
| admitted / dropped | -- | 1572 / 717 (45.6%) | 1643 / 688 (41.9%) | 1524 / **5 (0.33%)** |
| tokens stored | 5.45M | 3.64M | 4.13M | **5.37M** |
| tokens retrieved | 25.88M | 20.20M | 22.10M | **27.01M** |
| retrieves | 505 | 460 | 458 | 518 |
| preempt ids | 7 | 35 | 30 | 16 |
| store time (s) | 11.7 | 5.1 | 6.1 | 9.8 |
| store tokens p50 | 1536 | 19200 | 5376 | **256** |
| drain_steps | -- | 86605 | 72709 | **6180** |
| free_queue_blocks_read | -- | 4.16M | 3.75M | **2.04M** |

The control reproduces c32L's lazy@180 loss (+6280 vs +5225; the 1055 ms spread is
above the 650 ms eager-vs-eager control band, so run-to-run variance at this load is
larger than that band, but the sign and scale replicate).

### Scorecard on section 8's predictions: 3 hit, 1 missed

1. "The idle arm still loses to eager@180." **MISSED.** medD -35 ms -- a tie, well
   inside the control band, not a loss. The reasoning behind the prediction (store
   cost is 0.65% of wall clock, so the ceiling is a tie) got the magnitude right and
   the sign wrong.
2. "The idle arm beats the control by more than 650 ms." **HIT.** 6315 ms.
3. "`tokens_stored` on the idle arm near 5.45M, falsified under 4.5M." **HIT.** 5.37M,
   98.6% of eager's.
4. "The knob arm lands between the two, closer to the control, falsified if it beats
   the idle arm." **HIT.** +4190: 2090 from the control, 4225 from the idle arm.

### The degradation controller flaps and does not deliver

`degrade_trials=3 commits=1 reverts=2 probes=1 probe_recoveries=1`, six transitions in
30 minutes. Sequence: trial 1 committed -> DEGRADED -> a probe recovered it to NORMAL
-> trials 2 and 3 both reverted into cooldown. `degraded_emitted=272` of 924, so the
policy ran degraded for under a third of the run. Net effect on the drop rate:
45.6% -> 41.9%. Net effect on TTFT: +6280 -> +4190, real but a quarter of what is
needed to reach parity.

Section 3's claim that the neutrality gate can never commit was falsified (section 8);
the corrected reading is that it commits, is undone by its own probe, and then cannot
commit again. The `_note_emitted` blindness to dropped volume documented in section 3
remains true as a code reading and is the most likely reason trials 2 and 3 reverted,
but that is now an inference, not a measurement.

### What idle draining actually did, stated against my own framing

I called `IDLE_OPS=64` "the fix" in two messages before the round finished. The final
numbers say something narrower and it matters: **it fixed the drops by ceasing to
defer.** The evidence is the store size distribution, not the drop counter:

- store tokens p50: control 19200 -> idle **256**. One chunk. Mean 26000 -> 5949.
- 903 stores against eager's 1473, where the control managed 140.
- tokens stored, tokens retrieved, retrieves and preemptions all converge on eager's
  values (5.37M vs 5.45M, 27.01M vs 25.88M, 518 vs 505, 16 vs 7).
- `drain_steps` 86605 -> 6180, because `drain_steps` only counts steps with a
  non-empty pending set: there is almost never a backlog.

`idle_drain_max_ops=64` equals `max_drain_per_step`, so an idle step drains the full
per-step cap -- the most aggressive setting available. At L1=180 that is the correct
behaviour and a tie with eager is the ceiling (section 2: residence 738 s, store cost
0.65%, nothing to win). It also costs less on the hot path than the deferring arms:
half the free-queue reads, a twelfth of the block validation.

But "fixes lazy" and "switches lazy off" produce identical numbers at 180 G, and only
a measurement where deferral is worth something can tell them apart.

## 10. d60H launched 08:16

Same question at the L1 where deferral pays. Four arms, L1=60, conc 32, 1800 s:

    eager:d60_eager_s0                                  # control vs b32 eager (+57 / +607)
    lazy:d60_lazy_base_s1:DEGRADE_SECS=0                # control vs b32 lazy (-3146 / -2495)
    lazy:d60_lazy_idle64_s2:DEGRADE_SECS=0,IDLE_OPS=64  # the 180 G setting
    lazy:d60_lazy_idle8_s3:DEGRADE_SECS=0,IDLE_OPS=8    # a gentle setting

This replaces the `HORIZON=120` round proposed in section 5. The reason: section 8/9
showed idle draining reaches the same recall for less hot-path cost, so the horizon
sweep is no longer the interesting variable -- whether draining early is compatible
with the late-write mechanism that produces the 60 G win is.

Predictions, stated before the round lands:

1. `d60_lazy_idle64_s2` loses most or all of the -3146/-2495 ms win and lands within
   1000 ms of zero, because at 60 G "stop deferring" means "become eager" and eager is
   the zero. **Falsified if it stays below -2000 ms.** This is the decisive one.
2. Its store tokens p50 collapses from ~39000 toward eager's ~8192 or below, the same
   signature as at 180 G. **Falsified if p50 stays above 20000.**
3. `d60_lazy_idle8_s3` sits between base and idle64 on both dTTFT and store size --
   a gentle drain bounds backlog age without flushing it. **Falsified if it lands
   outside the interval spanned by the other two lazy arms on dTTFT.**
4. Drops: base ~8-10% (reproducing b32's 340/297), idle64 under 1%, idle8 in between
   but closer to idle64. **Falsified if idle8 exceeds 5%.**
5. `d60_eager_s0` and `d60_lazy_base_s1` reproduce their b32 counterparts, giving the
   round two same-config controls. **Falsified if the lazy base arm lands outside
   -1500..-4500 ms.**

If prediction 1 holds, the conclusion is that neither shipped knob can fix the drops
without discarding the benefit, and the burst-aware danger depth of section 7 -- which
widens the window only when a burst is announced, leaving quiet steps deferring -- is
the only candidate left that could do both. That would be the case for writing code.

## 11. The gap distribution, the scenario map, and why 4.3% is a fifth of the ceiling

Asked: which scenarios are favorable / unfavorable, and is 60 G's -4.3% reasonable.
Computed from `b32_eager_s0` / `b32_lazy_s1` profile exports (443/442 requests, 361
rehit opportunities, 20.1M opportunity prefix tokens).

### The inter-turn gap distribution is the scenario map

Gap (prev turn end -> next turn start), client-side: p50 **2 s**, p75 **16 s**,
p90 144 s. A turn itself runs ~85 s. So the survival requirement is wildly
asymmetric:

- lazy writes at turn end: needs residence >= gap, i.e. **~2-16 s** for most reuse;
- eager writes at turn start: needs residence >= rest-of-turn + gap, p50 **90 s**,
  p75 117 s.

Coverage of opportunity tokens vs residence (from the measured distribution):

| residence | lazy covers | eager covers |
|---|---|---|
| 20 s (30 G) | 76% | 2% |
| 46 s (60 G) | 80% | 7% |
| 105 s (~90 G lazy) | 87% | 63% |
| 450 s (180 G) | 98% | 97% |

Favorable band: residence between ~gap and ~turn+gap, i.e. **~16-120 s**, which at
~1 GiB/s store rate is roughly **L1 20-130 GiB at this load**. Below: both die
(30 G measured wash). Above: both survive, lazy is pure cost (180 G, +16.8% =
the drop bug; parity is the goal there). Caveat: store rate falls as hit rate
rises (dedup skips resident prefixes -- eager@180 stores at 0.277 GiB/s vs 1.146
@60), so success stretches everyone's residence and the band self-narrows from
above. Where the peak actually sits is the queued 45/90 G sweep's question.

Two parity edges already hold: 30 G measured -0.05%, and within b32 the **81
first-turn pairs (no reuse possible) show medD -57 ms** -- lazy's overhead on
no-reuse traffic is already zero. The goal's "unfavorable -> parity" is only open
at the big-L1 end, and that is the drop bug.

### Is -4.3% reasonable? No -- it is ~1/5 of what the timing edge should yield

- Whole-run medD -3146 dilutes: on the 361 rehit-opportunity pairs alone, medD is
  **-5505 ms (-6.5%)**, and the top 4 deciles win -8.8 s to -23 s. First turns are
  flat. Splitting by gap<=16 s vs >16 s changes nothing (-5505 vs -5934) -- within
  60 G, gap length does not discriminate, so eviction-before-reuse is not what
  separates hit turns from miss turns.
- Recall conversion is the leak: lazy retrieved 3.23M of 20.1M opportunity tokens
  (**16.1%**) against a timing coverage of 80%; eager 0.63M (3.1%) against 7%.
  The lazy/eager ratio (5.2x) roughly matches the coverage model (11x), but both
  arms convert only ~20-45% of covered tokens into retrieves. The missing 4/5 is
  not timing. Candidates: chain truncation at dropped chunks (9.8% drops at 60 G;
  retrieval stops at the first hole), watermark batch eviction (154 events =
  one purge per 11.7 s; batch dumps make effective survival << mean residence),
  lookup gating. 30 G is the smoking gun that the eviction dynamics can be far
  harsher than mean-residence arithmetic: 20-26 s residence against 2 s gaps
  should cover 76%, measured 3 retrieves total.
- Ceiling arithmetic: full 80% coverage at 60 G = ~16M retrieved tokens = ~200 s
  of prefill compute saved (hit is ~25x cheaper than recompute) = 11% of wall,
  before queue amplification (measured ~2x at small effect; the eager@180-vs-@30
  contrast shows 76.7% hit rate = -46% TTFT). Plausible ceiling at 60 G:
  **-15% to -30%**, against the measured -4.3%.

So the user's instinct is right. The -4.3% is real, controlled, and a fifth of
the prize. The order of attack it implies: (1) drops (burst-aware window -- also
what the parity-at-180 goal needs), (2) the conversion leak (watermark purge
dynamics / chain holes -- needs server-log forensics on which stored chunks were
gone at lookup time), (3) place the band peak (45/90 G sweep). d60H bears on (1)
directly: if idle64 collapses recall toward eager's 3%, timing is confirmed as
the whole separation.

## 12. The burst-aware window, designed (announce-then-admit)

Read the code paths while d60H runs. The naive wirings all lose the race:

- `on_new_request` (enqueue) is too early -- the request then waits ~30 s in
  queue, and the window would sit widened the whole time.
- Any "notify when the lookup result arrives" path is too late: the scheduler
  consumes the ready result inside `schedule()` and allocates the burst in
  that same call, *before* `build_connector_meta` runs the drain at the call's
  end. The drain that could have saved the endangered ops is the one at the
  end of the *previous* step, and no arrival-time callback can reach it.

The deterministic fix is a one-step delay owned by the connector.
`get_num_new_matched_tokens` (lmcache_mp_connector.py:750) already has a
"not ready" reply the scheduler fully supports: `(None, True)`. So:

1. First query at which the lookup result is ready with a hit: do NOT return
   it. Announce `ceil(hit_tokens / block_size)` blocks to the lazy manager,
   mark the tracker announced, return `(None, True)`.
2. The drain at this step's end sees the announcement:
   `_danger_depth() = max(rate_model, sum(announced))` -- and emits the
   endangered front. In-flight stores pin their blocks, so the next step's
   allocation cannot overwrite them (this is the existing pin-burst cost
   channel, now doing exactly its job).
3. Next query (next step): return the result normally; the manager clears the
   announcement when the request shows up scheduled (or on cancel/cleanup).

Cost: one scheduler step (~100 ms) of extra latency per hit admission,
against TTFT p50 of ~31 s under load. Width: exact burst size, held for
~1-2 steps, no estimator, no decay constant. Cold admissions stay on the
token-derived model, which section 1 showed is already correct for them
(chunked prefill allocates step-by-step).

Pieces: policy `announce_allocation(request_id, blocks)` /
`retract_allocation(request_id)` + depth max; manager forwards and clears on
scheduled/finished; connector implements announce-then-admit for lazy mode.
Tests at the policy level (announce widens, retract shrinks, endangered front
emits before a synthetic burst) plus a connector test for the one-step
contract. Implementation starts after d60H lands and is scored.

## 13. d60H scorecard and the verdict

Landed 08:55. Baseline d60_eager_s0 n=443, TTFT p50 72286.

| arm | medD | drops | retrieved | retrieves | store p50 | idle_emitted | preempt |
|---|---|---|---|---|---|---|---|
| lazy_base | **-7172** | 424 (12.8%) | 4.54M | 128 | 36864 | 0 | 157 |
| lazy_idle64 | **+245** | 3 (0.08%) | 0.78M | 19 | 256 | 3495/3790 | 15 |
| lazy_idle8 | **-189** | 10 (0.27%) | 1.06M | 27 | 256 | 3414/3721 | 16 |

Predictions:

1. **HIT (decisive).** idle64 lost the entire win: +245 ms, inside the control
   band. At 60 G "stop deferring" is "become eager", and eager is the zero.
2. **HIT.** Store p50 collapsed 36864 -> 256, the 180 G disablement signature.
3. **Letter hit, mechanism miss.** idle8 (-189) lies inside [base, idle64] as
   stated, but it is not intermediate -- it is idle64. The "gentle drain bounds
   backlog age without flushing" model was wrong: idle steps are plentiful
   (1183 of them), so even 8 ops/idle-step flushes the whole backlog
   (idle_emitted 3414 of 3721, retrieves 27, store p50 256). **There is no
   tunable middle ground in idle draining.**
4. **HIT.** base 12.8% (slightly above the predicted 8-10%), idle64 0.08%,
   idle8 0.27% -- ordering and magnitudes as stated.
5. **FALSIFIED.** The lazy base control did not reproduce inside -1500..-4500:
   it landed at **-7172**, double the b32 win, on +40% retrieved volume
   (4.54M vs 3.23M, 128 vs 94 retrieves). Same config, same seed; what
   differed is the neighbor mix on the box (b32: 2 eager + 2 lazy; d60H:
   1 eager + 3 lazy). The 60 G effect is large and round-unstable:
   the honest statement is "-3k to -7k", not a point estimate.

Verdict: the pre-committed criterion fired. Neither shipped knob fixes the
drops without discarding the benefit, and prediction 3's failure removes the
last escape hatch (a gentler idle drain is not a compromise, it is the same
disablement). Section 12's announce-then-admit window is the only standing
candidate that can hold both goal legs -- keep the -3k..-7k at 60 G and reach
parity at 180 G. Writing the code now.

## 14. Announce-then-admit implemented; e60A round predictions

Committed as 22be2125 (8 files, +358): policy `announce_allocation` /
`retract_allocation` + danger-depth floor + `announced_bursts` counter,
pending-store forwarders, manager `announce_hit_load` (tokens -> blocks) with
retraction on scheduled/finished/reset, connector announce-then-admit in
`get_num_new_matched_tokens` behind `lmcache.mp.lazy_offload_announce_hits`
(default on). 283 tests green (17 new). Harness gained `ANNOUNCE=true|false`.

Round e60A (after a single-slot smoke): four arms, conc 32, 1800 s:

    eager:e60A_eager_s0                                          # in-round baseline
    lazy:e60A_lazy_off_s1:DEGRADE_SECS=0,ANNOUNCE=false          # old behavior
    lazy:e60A_lazy_on_s2:DEGRADE_SECS=0,ANNOUNCE=true            # the fix, leg 1
    lazy:e180A_lazy_on_s3:L1_GB=180,DEGRADE_SECS=0,ANNOUNCE=true # the fix, leg 2

Predictions, stated before the round lands:

1. **Leg 1 (decisive): announce-on keeps the deferral win at 60 G.** medD vs
   the in-round eager stays below -2000 ms, in the lazy band (-3k..-7k seen
   across b32/d60H). **Falsified if it lands above -2000 ms** -- that would
   mean the one-step hold-back or the early emission it triggers costs the
   win the way idle draining did.
2. **Drops @60 collapse**: lazy_on dropped_evicted under 3% of admitted
   (lazy_off band: 8-13%), with announced_bursts on the order of the hit
   admissions (tens to ~150). Falsified above 5%.
3. **The fix is not a disablement**: lazy_on store_tokens p50 stays above
   20000 (turn-end signature; d60H base 36864). **Falsified below 8192** --
   the idle64 lesson, pre-committed this time.
4. **Leg 2: drops @180 collapse**: e180A_lazy_on dropped_evicted under 2%
   (d180 ctl/knob: 45.6%/41.9%), announced_bursts > 400.
5. **Volume @180 converges**: tokens_stored within 15% of eager@180's 5.45M
   (ctl was -33%); tokens_retrieved recovers toward the 25-27M band.
   TTFT vs the d180 eager baseline is cross-round and only directional:
   expected near 0, but not pre-committed.

Risks accepted: every hit admission pays one extra scheduler step (~100 ms
against 30-70 s TTFTs); the announced width can over-emit when the pending
backlog sits just past the burst depth (second-order, bounded by backlog
size ~900 blocks vs bursts ~2800).

### 14a. Launch amendments (09:23)

- The scripted smoke was refused by the scenario (duration >= 900 s), so the
  integration proof came from a hand-driven sequence on slot 0 instead:
  store A into a saturated pool, churn a full pool turnover, reset vLLM's
  prefix cache, re-request A. Result: **announced_bursts=1, retrieves=1,
  request completed** -- lookup hit -> announce -> one-step hold -> admit ->
  retrieve, live on a real engine. Two idle-pool false starts on the way
  are themselves confirmation of the mechanism's premise: in an
  unsaturated pool nothing is ever in danger, so nothing emits.
- GPU 5 was taken by the multi-modal line mid-morning, so slot 2 is
  unavailable. e60A runs the three 60 G arms on slots 0/1/3
  (`SLOTS` override added to round.sh); the 180 G leg
  (`e180A_lazy_on`) runs as a follow-on single-slot arm after the round.
  Predictions 4-5 unchanged, just deferred to that arm.
- Launched 09:23:51, expected done ~10:05. Liveness watch armed: the
  on-arm's ledger should show announced_bursts > 0 within ~20 min of
  traffic; if it stays 0 the announce gate is disconnected in real
  traffic and the round answers nothing about the fix.

### 15. e60A scorecard (landed 10:04:42)

Round: L1=60G, conc 32, 1800 s, seed 1234; three arms in-round on slots
0/1/3. Baseline eager n=446, TTFT p50 71629 ms.

| arm | pairs | medD | TTFT p50 | admitted | dropped_evicted | drop% | announced_bursts | retrieves | tok_retrieved | store_tok p50 |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| lazy_off (announce=false) | 446 | **-7155** | 63054 | 3526 | 285 | 8.1% | 0 | 84 | 3.50M | 37888 |
| lazy_on (announce=true) | 444 | **-730** | 68642 | 3535 | 200 | 5.7% | 79 | 79 | 2.72M | 39424 |

Predictions scored (pre-stated in section 14):

1. **FALSIFIED, decisively.** lazy_on medD = -730 ms, far above the
   -2000 ms line. The off-arm reproduced the d60H base win (-7155 vs
   -7172) in the same round, so the loss is attributable to the
   announcement mechanism itself, not round instability.
2. **FALSIFIED.** drop% = 5.66% (200/3535), above the 5% falsifier.
   Drops fell only ~30% (8.1% -> 5.7%): announcements protect against
   admission bursts, but at 60 G most drops come from steady churn the
   announcement never sees. The wiring itself is correct:
   announced_bursts=79 == retrieves=79, every hit admission announced.
3. **CONFIRMED.** store_tokens p50 = 39424 (> 20000): the deferral
   behavior is intact; the fix is not a disablement.

Verdict: at 60 G, announce-then-admit buys a 30% drop reduction at the
cost of ~90% of the TTFT win. As implemented, that trade is a loss and
the flag should stay off at 60 G.

Where the win went -- sensor evidence, hypotheses not conclusions:

- gain fell from -4917 s to -3039 s while lost rose only +173 s; the
  gain loss tracks tokens_retrieved (3.50M -> 2.72M, -22%) at ~2.4
  ms/token, i.e. the on-arm simply reused less, both externally
  (retrieves 84 -> 79, mean size down) and locally
  (covered_prefix_tokens_skipped 122k -> 82k, -33%).
- Storage-side content was *better* on the on-arm (stored 17.48M vs
  17.28M, fewer drops), so the leak is on the reuse side, not storage.
- free_queue_blocks_read 5.76M -> 27.97M (x4.9): an outstanding
  announcement widens the danger window to burst depth (~2800 blocks)
  every drain step. Leading hypothesis: the widened window's forced
  emissions pin the shallow front, the admission burst digs past the
  pins and evicts *deeper* GPU blocks -- which are other conversations'
  still-warm local prefix cache. Announce-on converts local hits (free
  in both arms of a pair) into misses, which is exactly the local
  coverage drop the counters show. Emission volume itself barely moved
  (emitted 3238 vs 3139), so it is the *eviction order shift*, not
  extra D2H, doing the damage.
- Alternative not excluded: earlier emission ages chunks earlier into
  L1's LRU, so the watermark purge removes them before reuse.

Implication for e180A (in flight): at 180 G the calculus can differ --
drops there are 44%, the retrieval upside is ~26M tokens, and L1 never
fills (no watermark confound). The digging cost remains, so predictions
4-5 stand as pre-stated. If e180A also loses its win, the one-step-hold
design is wrong in substance, not just mistuned, and the next move is
rethinking (e.g. announce without pinning: widen the window but release
holds lazily, or cap announce width) rather than a flag sweep.

### 15a. e180A launch (10:06:54)

Single arm `lazy:e180A_lazy_on_s1` on slot 1 (GPU 1), L1_GB=180,
DEGRADE_SECS=0, ANNOUNCE=true, same env/seed as e60A. TTFT comparison
vs c32_eager_l180_s2 is cross-round, directional only; the committed
sensors are the ledger counters (predictions 4-5, section 14).
Expected done ~10:45.

### 16. e180A scorecard (landed 10:46:53)

Single arm, L1=180G, announce=true, slot 1. Counters: admitted=1542,
emitted=1060, dropped_evicted=451 (**29.2%**), announced_bursts=468
(== retrieves), tokens_stored=4.26M, tokens_retrieved=21.6M,
store_tokens p50=13312, free_queue_blocks_read=143.1M,
throttled_drains=0, preempt_events=20. Cross-round vs c32_eager_l180_s2
(directional only): medD +5420, p50 38937 vs 31163.

Predictions scored (pre-stated in section 14):

4. **FALSIFIED.** drop% = 29.2%, nowhere near the < 2% target. The
   sub-clause held: announced_bursts=468 > 400, and again exactly one
   announcement per retrieve -- the wiring is live; the mechanism is
   just insufficient. Improvement over ctl/knob (45.6%/41.9%) is real
   but only a one-third cut.
5. **FALSIFIED.** tokens_stored 4.26M is -22% vs eager@180's 5.45M
   (outside the 15% band); tokens_retrieved 21.6M did not recover to
   the 25-27M band (ctl was 20.6M -- a 5% recovery, not a fix).

Combined verdict for announce-then-admit as implemented: **falsified on
4 of 5 predictions.** At 60 G it trades ~90% of the TTFT win for a 30%
drop cut; at 180 G it cuts drops 44% -> 29% and volume barely moves.
Per section 15's pre-commitment, this means the one-step-hold design is
wrong in substance, not mistuned. The flag default should flip to off
pending redesign (config key stays for A/B).

Residual-drop puzzle at 180 G: throttled_drains=0 says the per-step
volume cap never bound, so the surviving 29% were not emission-bandwidth
starvation at the capped drain. free_queue_blocks_read=143M (5x the 60 G
on-arm) says the widened window was scanned constantly. Candidates for
where the 451 drops actually happened, unranked: (a) bursts arriving
while a prior announcement's emissions still pin the front, digging past
protection; (b) chains whose blocks were already past burst depth at
announce time (window widens from the head; drops happen mid-queue);
(c) non-hit recycling (new-conversation prefills) that the rate window
underestimates at this hit rate. Needs per-drop rank forensics (log the
free-queue rank at drop time) before any redesign.

Next-step options (user decision, no further rounds launched):
- Per-drop rank logging + one replay round to locate the 451.
- Redesign candidates from section 15: multi-step hold until the front
  clears; announce-without-pin; allocation-side exemption (burst
  allocation skips blocks with pending stores instead of us emitting
  ahead of it).
- Or park the drop fix and attack the conversion leak (section 12
  queue), which the scenario map says bounds more upside at 60 G.
