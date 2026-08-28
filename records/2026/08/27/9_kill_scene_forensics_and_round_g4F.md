# Kill-scene forensics: what the logs already prove, and round g4F

Follow-up to record 8's discussion. The user set the frame: lazy must never
lose to eager, and before any redesign we need evidence for *why* it loses.
Chose option A (instrument the kill site, replay) over option B (redesign on
the mechanism map). This record holds the log-only attribution done first,
the instrumentation, and the pre-registered predictions for round g4F,
written before the round lands.

## 1. What existing logs attribute (no new run needed)

Method: drop lines in `vllm.log` ("blocks evicted before drain", with
timestamp, request id, per-op prefix offsets) aligned against retrieve
starts in `server.log` ("Retrieved N tokens in T seconds", start = log time
minus T) and against preempt/resume warnings. Null model: fraction of run
time within the window after any retrieve. Scripts: `dropforensics.py`,
`dropclassify.py` in this session's scratchpad.

| arm | ops dropped | <=0.5s after retrieve | null | resume-linked | unexplained (2s) |
|---|---|---|---|---|---|
| d180_lazy_ctl (no announce) | 717 | 56.5% | 8.5% | 1.5% | 35.4% |
| e180A_lazy_on (announce) | 451 | 31.7% | 9.0% | 1.3% | 63.0% |
| e60A_lazy_off (60 G) | 285 | 28.4% | 2.0% | 8.8% | 56.5% |

Established:

- **Hit-admission bursts are the dominant killer and the attribution is
  causal, not just correlational**: announce-then-admit (which protects
  exactly that event) removed ~65% of the tight-coupled mass (405 -> 143
  ops) and left the unexplained mass unchanged (265 -> 290).
- **Preemption resumes are negligible** (1-9%) despite re-allocating whole
  contexts. There are only 20-35 resume events per run.
- **The drop unit overstates the kill unit.** All 451 e180A drops come from
  the pending-validation site (`rejected_short_prefix=0`, held path never
  fired), in 79 events; a drop event drops the suffix from the first dead
  block on, so on average 5.7 ops fall per kill and most are prefix-closure
  collateral, not recycled blocks. Saving the first dead block saves the
  tail.
- **A third to two thirds of drops cannot be attributed from outside**: away
  from any retrieve or resume. Cold admissions are invisible in these logs
  (prefetch lines fire only on hits), which is also the leading suspect:
  the window's token floor is the *previous* step's scheduled tokens, and
  the drain runs after the step's own allocation, so the first prefill
  chunk of any newly admitted request (hundreds of blocks) is a sweep no
  part of the model sees. Record 2's "cold admissions are covered" claim
  holds only from the second chunk on. Consistency: e180A has ~140 cold
  admissions against ~40 unexplained kill events.

Mechanism map this leaves: steady churn (~2 blocks/step) can never cross a
~50-block window; **every kill is one step whose head advance outran the
danger depth**; sources are hit-admission bursts (proven), admission first
chunks (suspected, unproven), resumes (measured negligible).

## 2. Instrumentation (committed e76fc60b)

Timestamps ran out of discriminating power, so the kill step now names
itself. Two INFO lines, rare by construction:

- Policy, at `_drop_evicted_suffix`: dead-vs-collateral split of the
  dropped suffix, chain length, first-loss index, the last 4 per-step gross
  allocations, the last 4 danger depths, outstanding announcement width.
- Manager, any step allocating >= 64 blocks: total, per-admission
  `req_id=Nblk/Mhit_tok` (separates hit-carrying admissions from cold first
  chunks), remainder to running requests (resumes, decode growth).

Joining the two on time (same log, same process) dates each kill and names
the sweep. 282 tests green, ruff clean. Tests cover the dead/collateral
split, the ring content, the sweep threshold and the admission breakdown.

## 3. Round g4F, predictions pre-registered

Four arms, conc 32, 1800 s, seed 1234, shipped defaults on the lazy arms
(DEGRADE_SECS=0, ANNOUNCE=false -- loss guard live per c59448fe):

    lazy:g4F_lazy60_s0:L1_GB=60      forensics where deferral wins
    eager:g4F_eager180_s1:L1_GB=180  parity baseline
    lazy:g4F_lazy180_s2:L1_GB=180    guard parity + forensics
    eager:g4F_eager60_s3:L1_GB=60    60 G pair

Predictions, falsifiers included, stated before the round lands:

- **F1 (mechanism)**: >90% of kill lines show a recent step alloc exceeding
  the recent danger depths. Falsified if >25% of kills show no recent alloc
  above any recent depth -- that would mean kills without a sweep and the
  single-step model is wrong.
- **F2 (identity)**: joined to sweep lines within +-2 s, at 180 G the
  majority of kills join to hit-carrying admissions; at 60 G the majority
  join to admissions with hit_tok below 25% of blocks*16 (cold-dominated,
  hit rate there is ~15%). Falsified if either majority inverts.
- **G1 (guard parity)**: lazy180 medD vs in-round eager180 within +-1000 ms.
  Falsified above +2000 ms. (Reuses f180V's prediction; that round was
  stopped before producing data.)
- **G2 (guard ledger)**: degrade_commits >= 1, degrade_reverts <= 1,
  degraded_emitted > 50% of emitted. Falsified on reverts >= 2 or
  degraded share under 50% -- flapping would mean the charged ledger did
  not stabilise the commit.
- **G3 (volume)**: lazy180 tokens_stored >= 90% of eager180's. Falsified
  under 90%.
- **W1 (the win survives the guard)**: lazy60 medD <= -2000 ms vs in-round
  eager60. Falsified above -1000 ms -- that would mean the always-live loss
  guard costs the 60 G win (loss share there ran 8-13%, below the 25%
  material line, so it should never trial).

Explicitly not predicted: preemption counts, and where in [-2k, -7.5k] the
60 G win lands (round-unstable, band-only).

## 4. Round launch

Launched from the measurement scratchpad (`par/chain_g4F.sh`), ~35-40 min
expected. GPU check before launch: slots 1/2/3 idle, slot 0 carries 2.4 GiB
of resident processes from other users (rui, root) -- same state e60A ran
under this morning.

## 5. g4F scorecard (landed 15:50)

Round completed on all four slots. Paired TTFT: eager180 n=642 p50 31448;
lazy180 medD **+2506**. eager60 n=446 p50 70357; lazy60 medD **-20**.
Ledgers: lazy180 admitted=1402 dropped=44 (3.1%) commits=1 reverts=0
degraded_emitted=1044/1335 (78%); lazy60 admitted=3856 dropped=54 (1.4%)
commits=1 reverts=0 probe_recoveries=1 degraded_emitted=2907/3696 (79%).
Forensic lines: 11 kill events @180 (44 ops), 15 @60 (54 ops), sweeps
logged 1492/2912.

- **F1 FALSIFIED as stated.** 9 of 26 kill events (35%, above the 25%
  falsifier line) show no recent alloc above any recent depth. Two
  sub-classes, both real findings: (a) all-quiet rings -- the kill predates
  the 4-step ring, detection lagged (validation of a request with an
  in-flight store runs only at its receipt); (b) kills where the depth
  *covered* the recent allocs (512 vs 1280) and the tail ops died anyway --
  ops due-but-blocked behind their request's one in-flight store batch
  cannot be emitted, and die waiting. A mechanism nobody had on the list.
- **F2 HIT on both halves** (with 9/26 events unjoined). At 180 G every
  joined kill names a hit-carrying admission: 3375-6030 blocks with
  52k-95k hit_tok, against a previous-step depth of 3. At 60 G the joined
  kills split hit-bursts 4 / cold-or-running sweeps 7: cold first chunks
  (512blk/0hit_tok vs depth 3, four events) and running-request chunk
  restarts after a quiet step (1023-4151 blocks, zero hit_tok). The
  chunk-restart class extends the first-chunk hypothesis: any 512-block
  chunk step following a quiet step kills, not just the first.
- **G1 FALSIFIED.** +2506 ms, past the +2000 falsifier. The guard got
  volume to 90% but not TTFT to parity. Caveat: single round, and d180's
  replicate spread at this load was ~1 s; a replicate would separate
  guard-transition bleed from noise. Not launched.
- **G2 HIT.** commits=1, reverts=0, degraded share 78%. The charged ledger
  stabilised the commit -- no flapping, unlike d180's 6 transitions.
- **G3 HIT, barely.** tokens_stored 4.82M = 90.1% of eager's 5.35M.
- **W1 FALSIFIED, decisively, and this is the round's headline.** The
  always-live loss guard fired at 60 G and converted the -3k..-7k win into
  -20 ms. Timeline: burst cluster kills 19 ops 15:17:19-26, windowed loss
  share spikes past the material line, trial opens 15:17:46, commits
  15:18:38 (at 60 G immediate emission is volume-neutral, so the verdict
  cannot refuse it), probe 1 at 15:26:35 re-defers and bleeds 2 kills
  inside its own trial window and fails, probe 2 at 15:43:38 recovers,
  and deferral immediately starts bleeding again (8 kill events
  15:44-15:47) as the run ends.

## 6. What the evidence now says about why lazy loses to eager

Three killers, photographed:

1. **Hit-admission bursts.** One step allocates 2288-6030 blocks (the
   admitted request's whole matched prefix); the previous drain's danger
   depth was 3. The depth widens to ~burst size in the same step's drain --
   after the allocation already happened. Reactive by construction.
2. **Chunk-scale sweeps after quiet steps.** 512-block prefill chunks
   (cold first chunk, or a running prefill resuming after a scheduling
   gap) against depth 3. The token floor only protects steady chunk
   trains, where the previous step already scheduled 512.
3. **Blocked-op deaths and late detection.** A request with a store batch
   in flight cannot emit its remaining due ops; they die waiting for the
   receipt, even when the window saw them in time. ~a third of kill
   events.

One prior claim corrected by the data: every dropped op in all 26 events
had its own blocks recycled (`0 for prefix closure`). The e180A-era
inference that most of a dropped suffix is collateral and saving the first
block saves the tail is wrong -- a sweep that reaches a chain eats all of
it. The kills also concentrate on chain *tails* (first loss at 4-8),
i.e. the most recently freed content.

And one structural conclusion about the guard: **volume neutrality cannot
distinguish the 60 G win from the 180 G loss.** At 60 G deferral wins on
write timing at unchanged volume, so a degradation trial reads
volume-neutral and commits, destroying the win the guard was supposed to
leave alone. The verdict needs a benefit term the volume ledger does not
carry -- the obvious candidate is the retrieve rate, which collapses
(94 -> 29) when the 60 G win is destroyed and is unchanged at 180 G.
Shipped defaults after c59448fe are net harmful: they trade the only
measured win for a parity that did not materialise. Decision needed before
anything ships.
