# The reuse clock is queue-dominated, and it moves the favorable band

Asked again: which scenarios favour lazy, which do not, and is 60 G's -4.3%
reasonable. Section 11 of `2_*.md` answered "no, it is 1/5 of the ceiling, hunt
the conversion leak". That answer rested on one assumption that turns out to be
wrong. Recomputed here from the same `b32_lazy_s1` / `b32_eager_s0` profile
exports (pairing logic reproduces section 11 exactly: 361 pairs, 20.1M
opportunity tokens on the eager arm).

## 1. The assumption that was wrong

Section 11 measured the inter-turn gap **client-side**: previous turn's last
token to next turn's request submission, p50 1.5 s. It then required a stored
prefix to survive only that gap.

But the lookup does not happen when the request arrives. It happens when the
scheduler first considers the request, i.e. after it has cleared the waiting
queue. The KV must survive **gap + queue wait**.

Queue wait at conc 32, measured two independent ways:

| method | value |
|---|---|
| TTFT minus uncontended prefill (`isl / 15.2K tok/s`) | mean 63 s, p50 67 s |
| Little's law on vLLM's own waiting-queue log (mean depth 12.2, p50 14; throughput 0.237 req/s) | mean 51 s, p50 59 s |

The first is an upper bound (prefill under chunking is slower than the
uncontended rate), the second a lower bound (log sampling is coarse). Either
way the queue term is **35-45x the client gap** and it, not the gap, sets the
clock. The waiting queue is genuinely deep: of 32 lanes, ~4 running and ~14
waiting at p50.

## 2. Coverage, recomputed

Requirement per reuse pair: lazy writes at request end, so `gap + queue`; eager
writes at end of prefill, so `decode + gap + queue`. Coverage = share of the
21.0M opportunity tokens whose requirement is below L1 residence.

| residence | lazy (old clock) | lazy (real clock) | eager (old clock) | eager (real clock) |
|---|---|---|---|---|
| 20 s | 76.1% | 2.3% | 55.7% | 2.0% |
| 46 s | 80.6% | 13.8% | 75.0% | 8.0% |
| 90 s | 85.1% | 70.8% | 84.0% | 52.1% |
| 450 s | 97.8% | 97.3% | 97.8% | 97.3% |

The old clock is nearly flat in residence (everything is above a 1.5 s gap), so
it could not explain any of the L1 sweep; the leftover had to be called a
"conversion leak". The real clock has a knee, and the knee is where the sweep's
behaviour actually changes.

Validation against all four measured points (residences from `2_*.md` section 2):

| point | predicted coverage | measured conversion | verdict |
|---|---|---|---|
| lazy @30 G (26 s) | 3.3% | ~0 (16 retrieves) | fits |
| eager @30 G (20 s) | 2.0% | ~0 (3 retrieves) | fits |
| lazy @60 G (46 s) | 10.2% | **16.1%** | fits (bound is conservative) |
| eager @60 G (39 s) | 4.8% | 3.1% | fits |
| both @180 G (449/738 s) | 97% / 98% | 505 / 464 retrieves | fits |

The old model predicted 76% coverage at 30 G against ~0 measured. This one has
no such outlier.

## 3. Is -4.3% at 60 G reasonable? Yes -- the scenario is thin, not the code

At 60 G the ceiling is not 80% coverage, it is ~10%, and lazy converted 16.1%
of opportunity tokens against it. There is no 4/5 leak to hunt at 60 G: lazy is
already at or above its timing bound there. The lazy-minus-eager coverage
spread at 60 G is only **5.4 points**, and -4.3% whole-run / -6.5% on rehit
pairs is what 5.4 points buys.

The corollary is the useful part. Mapping L1 to residence by log-log
interpolation through the three measured L1 points, per arm:

| L1 | res eager | res lazy | cov eager | cov lazy | spread |
|---|---|---|---|---|---|
| 30 G | 20 s | 26 s | 2.0% | 3.3% | 1.3 pt (measured: wash) |
| 45 G | 30 s | 36 s | 3.1% | 4.9% | 1.8 pt |
| 60 G | 39 s | 46 s | 4.8% | 10.2% | 5.4 pt (measured: -4.3%) |
| 70 G | 55 s | 68 s | 9.9% | 31.0% | 21.2 pt |
| **80 G** | 74 s | 95 s | 26.9% | 74.9% | **48.1 pt** |
| 90 G | 96 s | 128 s | 59.0% | 82.6% | 23.7 pt |
| 110 G | 150 s | 213 s | 83.0% | 90.0% | 7.1 pt |
| 180 G | 449 s | 738 s | 97.0% | 98.3% | 1.3 pt (measured: +16.8%, all cost) |

The band is real but **narrow and centred well above 60 G** -- roughly 70-100 G
at this load, peaking near 80 G at ~9x the 60 G spread. 60 G sits on the low
shoulder. That is why the measured win is small, and it is also why the
announce-then-admit evaluation was run at two places where there was nothing to
win in the first place (60 G: 5.4 pt available; 180 G: 1.3 pt).

## 4. The band moves with load, which retires a loose end

Residence is a server property; the reuse clock is `gap + queue`, and queue is a
load property. So the favourable band's *location* is a function of load, not a
constant of the workload:

- idle / low concurrency: queue -> 0, the clock collapses back to the client gap
  (p50 1.5 s), both arms cover everything above ~20 s residence, and lazy has
  only its costs. **Lazy has no value at low load**, at any L1.
- heavy load: the clock stretches to tens of seconds and the band slides up to
  where only the later writer survives.

This explains the conc-8 agentx campaign (`26/12`, `26/13`) better than "the
load was missing" did: at conc 8 the queue term is small, so 60 G was already
above the knee for both arms, and there was no separation to measure regardless
of replicate count.

## 5. What this reprioritises

- **Drop the conversion-leak hunt at 60 G** (was option 3, described as the
  biggest bounded upside). Its premise -- 80% coverage, 16% realised -- was an
  artifact of the wrong clock. Watermark-purge and chain-truncation forensics
  are no longer justified by this arithmetic.
- **The band sweep becomes first priority** (was queued last). Points: 45 G as a
  counter-directional control (predicted *worse* than 60 G), 80 G as the
  predicted peak.
- **Re-site the 180 G parity work.** Parity at 180 G is still an open goal and
  the drop bug is still real, but the drop fix should be *evaluated* at the
  peak, not at 60/180 G.
- Unchanged: flip `lazy_offload_announce_hits` default to False.

## 6. Pre-registered predictions for the band sweep

Round: conc 32, 1800 s, seed 1234, four arms interleaved eager/lazy at 45 G and
80 G, config-to-slot rotated, knob off, announce off.

1. **80 G lazy retrieval share of opportunity tokens >= 40%** (60 G was 16.1%).
   Falsifier: <= 20%, which would kill the residence interpolation or the clock
   model.
2. **80 G medD more negative than -7000 ms** (the 60 G in-round reference was
   -7155). Falsifier: |medD| < 7000, i.e. no improvement over the low shoulder.
3. **45 G spread smaller than 60 G's**: 45 G medD less negative than -7000 ms
   and retrieval share <= 12%. This is the counter-directional test; if 45 G
   wins as much as 60 G, the knee is not where this model puts it.
4. **80 G drop share between 12% and 35%** (9.8% at 60 G, 44% at 180 G), since
   the danger window shrinks as hit rate rises. If drops at 80 G are already
   ~44%, the peak win will be suppressed and the drop fix moves back ahead of
   everything else.
5. Eager improves at 80 G too (predicted coverage 26.9% vs 4.8% at 60 G), so the
   eager baseline TTFT p50 should fall well below 71.6 s. If it does not, the
   residence mapping is wrong on the eager side.

## 7. Caveats

- No residence measurement exists between 60 G (46 s) and 180 G (738 s). The
  peak's *existence* and its being above 60 G follow from the clock plus the two
  bracketing points; its *location* is interpolation and is the softest number
  here.
- Queue wait is estimated, not instrumented. Both estimates agree to ~20%, and
  the coverage numbers use the pessimistic one.
- Coverage is an upper bound on retrieval, and points of coverage convert to ms
  non-linearly (queue amplification: better hit rate shortens prefill, which
  shortens the queue, which shortens the clock -- a positive feedback this model
  does not include, and which would sharpen the peak).
- Residence is itself hit-rate dependent (dedup cuts the store rate), so the
  mapping in section 3 is self-referential at the top end. Measured points, not
  the curve, are what settle it.

## 8. The parity guarantee: what it takes to never be worse than eager

User's requirement, stated after section 7: lazy must never lose to eager. This
section is the plan; it is not implemented.

### Where lazy is currently worse

One measured place, one unmeasured risk:

| scenario | status |
|---|---|
| L1 30 G (residence << clock) | parity, measured -0.05% |
| no-reuse / first-turn traffic | parity, measured -57 ms on 81 pairs |
| L1 60 G | wins, -4.3% |
| hot/cold 40 G long-doc | wins, cold TTFT 432-486 vs 811-813 ms |
| **L1 180 G (residence >> clock)** | **loses, +16.8%** |
| **low load (conc 8), any L1** | **unverified.** The model says lazy has no value there; the conc-8 campaign was noise-dominated (trimmed -6.4..+5.9 s) so it neither shows nor excludes a regression |

So the guarantee has exactly one known hole and one blind spot.

### Parity at 180 G is already proven achievable

`d180`'s `idle64` arm reached **medD -35 ms** -- a tie inside the control band --
with `tokens_stored` 5.37M vs eager's 5.45M (98.6%), retrieves 518 vs 505,
preemptions 16 vs 7. The mechanism was not a fix; it was cessation: store
tokens p50 collapsed 19200 -> 256, one chunk per op. At 180 G that is the
correct behaviour, because the ceiling there is +0.65% (store time as a share of
wall clock) and the floor was -16.8%.

The conclusion that matters for the guarantee: **we do not need a new mechanism
to reach parity. We need the existing switch to fire and stay fired.** "Stop
deferring" is parity by construction, and it is measured.

### Why the switch does not fire today

`d180` ran the controller: `trials=3 commits=1 reverts=2 probes=1
probe_recoveries=1`, degraded for only 29% of the run (272 of 924 emitted),
drop rate 45.6% -> 41.9%, TTFT +6280 -> +4190. It moves in the right direction
and stops a quarter of the way. Two defects, both already diagnosed:

1. **The neutrality baseline is blind to lost volume.** `_emitted_blocks_total`
   (`eviction_aware.py:1554`) advances only on emission; the trial gate
   (`:1413`) and the probe gate (`:1436`) compare emitted rates. At 180 G the
   deferred baseline is 3.73M tokens against an immediate 5.45M, ratio 1.46 >
   `_NEUTRALITY_FACTOR` 1.25, so a correct trial reads as a volume violation and
   reverts. The gate cannot distinguish "deferral filtered stores out" (a
   benefit) from "deferral lost 44% of them to eviction" (the defect it exists
   to catch). With lost volume charged, the baseline becomes ~6.8M against
   5.45M, ratio 0.80, and the trial commits.
   Note for implementation: `dropped_evicted` counts *operations* (`:1217`,
   `:2149`), not blocks. Charging volume needs a parallel block-level total,
   `sum(len(op.block_hashes) for op in dropped)`, alongside
   `_emitted_blocks_total`.
2. **The probe undoes a correct commit.** Recovery (`:1436-1439`) is verified on
   volume alone, so at 180 G a probe that emits at eager's rate reads as "the
   backlog is healthy again" and lifts a degradation that was right. Recovery
   must additionally require the windowed drop share to have fallen below
   `_MATERIAL_LOSS_SHARE`; a probe that recovers volume while still bleeding
   drops has not shown the danger passed.

### Why fixing this does not cost the 60 G win

The trial *trigger* is what protects 60 G, and it is untouched: windowed loss
share there is 9.8%, below the 0.25 line, and residence 46 s is below any sane
`degrade_l1_residence_secs`. No trial opens, so no gate change can reach it.
This was checked analytically in `2_*.md` section 3 and is worth re-checking in
the round, because it is the whole reason this fix is safe.

### Sequencing, and what it costs

1. Charge lost volume in the trial and probe gates; require a fallen drop share
   for probe recovery. Small, inside a tested component.
2. Flip `lazy_offload_announce_hits` to False (e60A: costs 90% of the 60 G win).
3. Verify at 180 G. Predictions: degraded coverage > 80% of emitted (was 29%),
   medD inside the +-650 ms control band, `tokens_stored` >= 95% of eager,
   `degrade_reverts` <= 1. Falsifier: medD > +2000 ms, i.e. the controller still
   cannot hold the degradation.
4. Verify at 60 G, same round if slots allow. Predictions: `degrade_trials` = 0,
   medD unchanged from -7155. Falsifier: any trial opens, or medD worse than
   -4000 ms.
5. Only then the 45/80 G band sweep (section 6).
6. Low load stays open: conc 8 at 60 G with a same-config control and >= 3
   replicates, to close the blind spot rather than assume it.

Honest statement of what this buys: parity by cessation, not by repair. Lazy
becomes eager wherever it cannot win, and keeps its win where it can. The drop
bug is still real and fixing it would *extend* the win at 60-80 G, but it is not
on the path to the guarantee.

## 9. Implemented as c59448fe; round f180V launched 11:47

Committed (local only, 40 ahead of origin/dev): 347 lazy tests pass, ruff
clean. `mypy` is not installed in this environment and was not run; the
changes add two int counters, a property, and two renamed helpers.

What went in:

- **Volume ledger charges losses.** `_volume_blocks_total` = emitted blocks +
  blocks lost to eviction, and the trial gate, the probe gate and the trailing
  baseline all read it. `_trailing_emitted_rate` / `_regime_emitted_rate` are
  now `_trailing_volume_rate` / `_regime_volume_rate`; `_note_lost` advances
  the lost half at both drop sites (`_promote_held`'s broken chain and
  `_drop_evicted_suffix`). Test
  `test_lost_volume_counts_against_the_deferred_baseline` is the mirror of the
  existing `test_volume_increasing_trial_reverts_and_cools_down`: the same
  backlog flushed by the same trial, differing only in whether the deferred
  window lost blocks first. It fails on the old emissions-only ledger
  (verified by reverting the property and re-running), which is the point of
  it.
- **The loss trigger is unconditional.** `degrade_l1_residence_secs <= 0` used
  to return before the regime machine ran at all, and it defaults to 0, so the
  shipped configuration had no volume guard whatsoever. Now the threshold
  gates only the residence trigger; the loss trigger always runs, and
  `wants_l1_pressure()` follows the policy rather than the knob.
- **Probe backoff.** Each consecutive failed probe doubles the spacing to
  `_PROBE_BACKOFF_MAX` = 8. Verified discriminating by pinning the factor to 1.
- **`lazy_offload_announce_hits` defaults to False.** The par harness's own
  `ANNOUNCE` default was flipped to match, since a harness default contradicting
  the shipped default is a trap.

Round f180V: L1 180 G, conc 32, 1800 s, seed 1234, two arms, lazy on slot 0
(GPU 0) and eager on slot 1 (GPU 1) -- rotated against the previous 180 G
round, which had lazy on slot 1 and eager on slot 2. `DEGRADE_SECS=0`, i.e.
the shipped default: the loss trigger is what has to fire. Only GPUs 0 and 1
were free, so there is no same-config control pair in-round; the effect under
test (+5225 ms) is an order of magnitude outside the +-650 ms band measured in
b32 and d60H. Predictions are pre-registered in section 8 and repeated in the
chain script header.

### 9a. f180V aborted 11:55, unscored

Killed on the user's instruction about 8 minutes in, during server startup /
early load. No arm reached the measurement phase, so nothing is scored and
section 8's predictions stay open and pre-registered for whenever the round is
re-run. Teardown: the launcher's session group was killed, `par/down.sh` ran on
slots 0 and 1, and both GPUs plus both LMCache servers were verified released.
No partial results are archived under `par/f180V_*`; treat them as absent
rather than as a short round.

Code state pushed to the fork for safekeeping: `BoJiang03/LMCache`, branch
`lazy-offload-publish` at `c59448fe`, 40 commits ahead of `origin/dev`. No PR.

### 9b. Fork branch scheme (settled 2026-08-27)

Each line on `BoJiang03/LMCache` has three branches, and only the dev one may
carry `records/`:

| role | lazy offload | multi modal |
|---|---|---|
| dev (carries session records) | `lazy-offload-policy` | `multi_modal` |
| PR / publish (code only) | `lazy-offload-publish` | `multi_modal_pr` |
| repro (code only) | `lazy-offload-policy-repro` | `multi_modal_repro` |

`.git/hooks/pre-push` enforces it: any ref whose history touches `records/` is
blocked unless it targets one of the allowlisted dev branches on the fork.
`records/` stays out of the publish branch through `.git/info/exclude:19`, so
the records commit is built with a redirected `GIT_INDEX_FILE` plus
`commit-tree` rather than by switching branches -- switching back would delete
the whole folder from the working tree.

`fork/lazy_offloading` is **not** the dev branch: it stopped on 2026-08-13 and
diverged 51/90 from the current line. Pushing to it would need a force push
that discards those 51 commits.

State at this point: `lazy-offload-publish` = `c59448fe` (code), local
`lazy-offload-dev` -> `fork/lazy-offload-policy` = `d2ae93a9` (code +
60 record files). No PRs.
