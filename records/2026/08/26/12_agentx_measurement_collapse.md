# The agentx measurement collapses; verdicts withdrawn

New session, handed off via `lo_temp_ctx.md` (written ~20:50 by the previous
session). Code state unchanged all session: commit 22c46cb6 on
lazy-offload-publish, 5 local commits ahead of origin, NOT pushed. Tree
clean apart from the untracked `lo_temp_ctx.md`; no code was written or
edited, so there is nothing to commit for this segment. records/ is
gitignored, so the record edits below never dirtied the tree.

Evidence, tables and prediction scorecards live in record 10's last three
sections ("a91: attribution + slot-swap round", "a91 verdict", plus the
consolidated hot/cold section). This file is the conversation log.

## What happened, in order

1. **Picked up the handoff.** Two rounds were in flight per
   `lo_temp_ctx.md`: the par chain (y3 at 60G then z91 at 90G) and
   hot/cold-40G v2.1 on GPU 2. hot/cold had already finished; y3 had
   already landed inside the chain.

2. **hot/cold v2.1 landed: PASS.** Zero trials, zero drops, cold TTFT
   roughly half of eager's, ext back in the pure-tail reference band.
   Found record 10 carried the SAME verdict twice, written minutes apart
   with different numbers (452/432/486 vs 397/276/473). Diagnosed: the run
   log prints p50, the json carries means; both were correct, the sections
   just quoted different statistics. Merged them into one section quoting
   both, and kept the honesty note about the GPU-7-vs-GPU-2 environmental
   gap.

3. **User asked how lazy compares to eager on agentx.** Reported y3 at 60G
   (-8.1/-12.1s, p99 -24%) and the pending z91. This answer was wrong, as
   step 7 shows -- not arithmetically, but because the metric it rested on
   cannot support the claim.

4. **z91 (90G) landed at 21:02: apparent big WIN**, -51.3/-54.9s against a
   -6.6s eager-eager spread, with 1 trial -> 1 commit -> 75% degraded and
   one non-recovering probe per seed. The pre-stated deciding question
   ("does the volume-neutral commit kill the tail win?") looked answered in
   the criterion's favour; record 10 got a verdict section and the
   acceptance arc was declared closed.

5. **User asked what could actually be concluded.** Listing the gaps
   honestly turned up two that mattered: (a) no knob-off lazy arm at 60G or
   90G, so -51s could not be split between the controller and the admission
   economy; (b) round.sh derives the GPU from the slot index and every
   agentx round had put eager on slots 0/2 and lazy on 1/3, so the
   eager-eager spread only ever measured slot0-vs-slot2 and a slot bias
   would be invisible. Proposed one round closing both.

6. **a91 launched** (user: "可以"): 90G, mapping inverted --
   slot0=lazy-on, slot1=eager, slot2=lazy-off (DEGRADE_SECS=0, a hard
   disable), slot3=eager. Four predictions pre-stated in record 10 before
   launch, including the falsifier "if on lands above -20s, z91 was
   slot-inflated and every agentx verdict needs the same check".

7. **a91 fired that falsifier.** The eager-eager pair -- identical config,
   same round, arms time-aligned within 7s -- came in at **+34.3s**, larger
   than z91's entire claimed effect. Both lazy arms sat between the two
   eagers (+22.0 on, +14.7 off). Per-minute decomposition found a ~4-minute
   window where three arms took 13-23s TTFT outliers and the baseline eager
   did not; excluding it collapsed every difference to +1.3/+5.1/+6.5.

8. **Re-scored the whole campaign on a trimmed statistic.** Raw sumD across
   all agentx rounds ranges over 89s (-54.9 to +34.3); 5%-trimmed, every arm
   -- lazy and eager control alike -- sits in -6.4 to +5.9s. z91's -51.3s
   has best10 = -54.4s: ten requests out of 273 are the whole win. y3's
   -8.1s trims to +5.9s, i.e. lazy slightly worse. Withdrew every agentx
   verdict in the campaign, in both directions: y60 parity, z90 -19s, y2
   +21s regression, y3 -8/-12s win, z91 -51/-55s win. That includes the y2
   "failure" that motivated v2.1 and the y3 "win" that motivated the
   deferral-wins-at-60G reframe.

9. **Explained the mechanism to the user.** Median paired difference is
   1-6ms on a 550ms median TTFT, so the systematic signal over 270 requests
   is worth 1-2 seconds; the trace's tail runs to 22s, so a single stall
   outweighs the entire signal ~15x, and the primary metric is a plain sum,
   which has no resistance to extremes. Stalls are not config-owned: four
   arms share the box with 4 x 90GB of host L1 plus neighbours' jobs.

## What survives

- **hot/cold 40G.** Broad-based, not a tail lottery: eager cold is a tight
  cluster (q1 784 / med 792 / q3 795, three seeds at mean 811/813/812) and
  lazy moves the whole lower half (min 167-197, q1 187-209, 10%-trimmed
  mean 445/421/484). Three seeds agree and the pre-knob tail reference
  lands in the same band.
- **Controller safety.** A ledger property, not a sumD property: 0 trials
  across three 40G seeds, every transition trial- or probe-verified, volume
  neutrality in every round.
- **The knob buys nothing measurable at 90G.** a91 is the first direct
  on-vs-off: +22.0 vs +14.7 raw, 3.1 vs 3.1 trimmed, and off dropped 75
  against on's 145. Defensible ship position: default off, documented
  opt-in.

## Prediction scorecard for a91 (1 hit, 3 misses)

1. on reproduces z91 at -40..-60s -- MISS (+22.0; -12.3 vs the other eager).
2. |on - off| < 15s -- HIT (7.3s apart, identical trimmed).
3. off drops >> on drops -- MISS, inverted (on 145, off 75).
4. eager-eager within +-8s -- MISS (+34.3s). This was the assumption every
   prior round rested on.

## Open, not started

- Settling agentx would take ~4 replicate rounds (25 min each) with rotated
  config-to-slot assignment, a robust primary metric (trimmed mean, median,
  or stall count), and possibly 2 arms per round at 90G to cut host-memory
  cross-arm coupling. User has not decided whether agentx is worth it given
  hot/cold already carries a clean win.
- DEGRADED -> NORMAL recovery has never fired in any measurement round
  (probe_recoveries = 0 campaign-wide); only unit tests cover it.
- Still unstarted from the earlier queue: load ramp (concurrency 2.4) and
  the GSM8K coverage probe for 5ea3cc6e.

## Process notes

- The duplicated hot/cold section in record 10 came from the previous
  session appending the same verdict twice with different statistics. When
  a section is re-appended rather than edited, quote which statistic is
  being used.
- Pre-stating predictions with explicit falsifiers is what caught this:
  prediction 4's +-8s band and prediction 1's stated falsifier both fired,
  and the falsifier named the follow-up check that produced the withdrawal.
  Without them the +34.3s eager-eager pair would have read as one odd round.
