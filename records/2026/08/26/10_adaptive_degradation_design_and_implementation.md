# 10. Adaptive degradation: design, implementation, y60 in flight

Segment: from the w60 verdict through shipping adaptive degradation.
Code state: `7fe73fce` (GET_L1_PRESSURE endpoint) + `190d3c97`
(degradation policy), both LOCAL ONLY -- user directive this segment:
no PRs, nothing leaves the machine. Details of w60/x60 verdicts are
also appended to record 8; this file is the segment log.

## Evidence chain that produced the design

1. **w60 (off x2 vs eager x2, 60G)**: off loses 230-250s sumTTFT (p90
   5722-6389 vs 2160-2577, 44 pairs out of window). Both branches of my
   decision rule were wrong -- L1 at 60G carries huge value; "store
   less" is dead.
2. **Snapshot probe across r/s/t/u/v/w60 (zero cost, no new round)**:
   - tokens_retrieved eager == tail at 60G (~1.0M each) -> the value-
     capture branch is dead too.
   - tokens_stored eager == tail (~2.0M) -> lazy filters ops (1053 ->
     225) but not bytes at 60G. Waiting buys nothing.
   - Total store wall time tail < eager (5.8s vs 9.1s) -> D2H time is
     not the residual either.
   - What differs: chunk shape (tail p99 65K tokens, single store max
     0.95s, pinning 25% of the pool) and preempts (4/2 vs 1/1).
   - retM pattern explained: head/idle arms retrieve MORE than eager
     (1.2-1.4M) as compensation for self-inflicted GPU misses.
3. **x60 (eager x2 vs tail+BLOCK_CAP=512 x2)**: cap did exactly what it
   was designed to do (p90 8192/p99 32000 chunks, 2.5 ops/batch, no
   receipt serialization, 16% fewer bytes at equal retrieval value) and
   the gap did not move (+14.0/+8.9s; preempts still 4/2). Pin
   dwell/chunk shape falsified. Emission-side design space closed twice.
4. **90G ledger pull (g/h/k/m rounds)**: lazy retrieves +0.4M tokens
   over eager at 90G (1.54-1.70M vs 1.18-1.23M) -- the 90G win is a
   retrieval-value dividend, worth ~80-100s, minus the same machinery
   cost both regimes pay. At 60G the dividend collapses to zero (L1
   churns before reuse) and only the cost remains. GPU-side ledgers are
   IDENTICAL across 60/90G (drop rate 12-18% both; the GPU pool is
   16384 blocks in both -- L1_GB is CPU-side capacity).
5. **Separation signal**: L1 eviction pressure. Watermark triggers per
   900s run: 60G 13-14, 90G 6, 30G 38, hot/cold 0. With eviction_ratio
   0.2: residence ~250-320s at 60G vs ~450-750s at 90G vs infinite at
   hot/cold. Threshold 450s splits with ~40% margin each side.

## Design (user approved "可以")

Degrade to immediate emission when L1 residence is short; equality with
eager then comes from removing the deferral, not from shaping it.

- **Rejected: true bypass to the eager store path.** Landmine found in
  the worker contract: `get_finished_with_lazy_offload` returns None
  for stores, so a degraded request following the eager contract
  (request_finished -> True) would never be released by vLLM. Mixing
  contracts means worker-adapter surgery. Not needed:
- **Chosen: immediate emission through the lazy pipe.** An op admitted
  while degraded is emitted on the first drain after admission, while
  its request is still running -- its blocks are not in the free queue,
  so the pins that protect the copy are inert bookkeeping. Chunk sizes
  match eager's own store shape (admission is per-step). y60 doubles as
  the machinery probe: any residual gap is the lazy pipe itself.

## Implementation (shipped, 281 tests green, ruff clean)

`7fe73fce` -- mechanism:
- L1Manager accounts (deleted_bytes, deleted_chunks) at its four
  deletion sites, public `deletion_totals()`;
  `StorageManager.get_l1_deletion_totals()` pass-through. Deliberately
  NOT an event-bus subscriber: the global bus defaults to disabled
  (event_bus.py:362) unless observability config enables it -- a
  functional signal must not depend on that. (First draft used the
  bus; reworked after finding the default.)
- `RequestType.GET_L1_PRESSURE` appended at the END of the enum
  (msgspec encodes ordinals; wire compat), controller protocol group,
  `L1PressureStats{total,used,evicted_bytes,evicted_chunks}` response,
  handler in ManagementModule.
- `LMCacheMPSchedulerAdapter.poll_l1_pressure(min_interval)`:
  threadless poll driven from the step path -- non-blocking future
  checks, per-server aggregation (sum), failed cycle dropped whole,
  stuck cycle abandoned after mq_timeout, no submission while
  unhealthy. Multi-server: summed rate over summed capacity.

`190d3c97` -- policy:
- `LazyOffloadPolicyConfig.degrade_l1_residence_secs` (float, 0=off).
- `EvictionAwareStoreQueue.observe_l1_pressure(t, capacity, evicted)`:
  baseline -> per-interval rate -> EMA (alpha 0.3, first observation
  taken as-is) -> residence = capacity/rate. Degrade below threshold,
  recover above 2x (hysteresis). Repeated timestamps ignored (caller
  re-feeds the latest sample every step); counter regression (server
  restart) re-baselines. Signal stays measurable while degraded --
  eager-pattern stores keep flowing -- so recovery needs no probes.
- `_drain_degraded`: full-FIFO flush replacing backlog+idle paths while
  degraded; validation, prefix closure, dedup-hole cut, economy
  backstop, one-in-flight-per-request, shared step budget all hold.
- Counters: degraded_emitted / degraded_drain_steps /
  degrade_transitions (in decisions()).
- Wiring: facade parses `lmcache.mp.lazy_offload_degrade_l1_residence_secs`,
  exposes wants_l1_pressure() (EVICTION_AWARE mode + knob only);
  manager passes through; connector polls every 10s
  (`_L1_PRESSURE_POLL_INTERVAL_SECS`) only when wanted, forwards
  (monotonic_time, total_bytes, evicted_bytes_total).
- Docs: l1_pressure_stats.md (new), eviction_aware.md "Adaptive
  degradation" section, lazy_offload.md config block + paragraph.

Test notes: two of my own test bugs fixed en route (stuck-cycle test
asserted no resubmission -- resubmitting after abandoning IS the
behavior; economy-backstop test used min_prefix_tokens=256 where the
truncated chain ends exactly at 256, not below). Connector test doubles
(_RecordingManager/_FakeSchedulerAdapter) extended for the new boundary.

## y60 in flight

eager x2 vs lazy tail+DEGRADE_SECS=450 x2 at 60G. Chain pid 781273,
early-check + completion watchers armed. At 60G, L1 fills ~5min in,
residence lands ~250-320 < 450 -> degrades for the remaining ~10min of
the 15min window; the paired eager covers the whole window. Read: the
cmp2 table plus ledger degraded_emitted/degrade_transitions (expect 1
transition, degraded_emitted the bulk of emitted after minute ~5) and
config echo `degrade_l1_residence_secs: 450`.

Interpretation rule, stated before results: parity (within eager's ~3-5s
band) => acceptance bar met at 60G by adaptive degradation; ship
recommendation is tail default + this knob. A residual gap with the
ledger showing the regime engaged => the cost is the lazy pipe itself
(receipt bookkeeping, manager overhead) -- next lever is worker-contract
work (true bypass), a bigger change.

## Standing constraints (unchanged + new)

- NEW this segment: no PRs, all operations stay local. Branch
  lazy-offload-publish is 3 commits ahead of origin, NOT pushed.
- Push only to fork BoJiang03/LMCache when told; user opens PRs.
- Authorship Bo Jiang <bo.jiang@temple.edu>, no Claude trailers.
- Shared box: no rebuilds of shared artifacts; writes in scratchpads
  (harness lives in old-session scratchpad par/; slot map restored to
  0/1/5/6 after GPUs 1/6 freed; GPUs 3/4 now held by another session).
- Remaining acceptance items after y60: 90G no-regression round with
  the knob at 450 (expect: never degrades, ledgers show transitions=0),
  hot/cold rerun, load ramp (concurrency 2.4) still unstarted, GSM8K
  coverage probe for 5ea3cc6e via hot/cold shape.

## y60 verdict (18:13): acceptance bar MET at 60G

Paired vs y60_eager_s0 (n=270, eager-eager spread sumD +6.4s):

    arm            medD   sumD   retM  drop   sto  preempts
    y60_eager_s2     1    +6.4   1.03     -  1055     1
    y60_adapt_s1     2    +2.0   1.08    12   720     1
    y60_adapt_s3     7    -4.2   1.03     5   726     1

Both lazy arms land inside (s1) or beyond (s3, negative = beats
baseline) the eager-eager noise band. Compare x60's +14.0/+8.9. The
discriminating sensor confirms: preempts back to 1/1, matching eager
(was 4/2 in every deferred variant). Stored bytes equal eager
(~2.03M vs ~2.01M tokens), retrieval equal-or-better (1.08/1.03M vs
1.00/1.03M), store shape eager-like (p50 256, p90 ~6K vs eager 8192),
dropped_evicted collapsed to 12/5 ops.

Regime engaged as designed in the large: degraded_emitted=748/766 of
emitted=1044/1042 (~72%), watermark events 14 = same as eager arms.

One wart, honestly noted: degrade_transitions=10/8, not the expected
1. Mechanism: eviction is bursty (watermark fires, evicts 20%, then
~60s quiet); between bursts zero-delta samples decay the rate EMA by
0.7 each, the residence estimate climbs through the 900s recover
line, then the next burst crashes it back under 450. Each flap
re-defers briefly, then the FIFO flush on re-degrade produces the
surviving large stores (max 87K tokens, 0.93/0.96s single store).
Performance met the bar anyway -- pins are inert while requests run,
preempts stayed 1 -- so this is a smoothness wart, not a correctness
one. If it matters for 90G (threshold 450 sits at the bottom edge of
90G's 450-750s residence band), candidate fixes: windowed rate over
a fixed horizon instead of per-sample EMA, or a minimum dwell time
per regime. Decide after the 90G no-regression round shows whether
flapping costs anything there.

Ship recommendation per the pre-stated rule: tail placement default +
degrade_l1_residence_secs=450. Emission-side story is complete: lazy
now beats eager at 90G (retrieval dividend +0.4M tokens) and matches
it at 60G (degradation removes deferral when the dividend collapses).

Next: 90G round with the knob at 450 (watch degrade_transitions --
expect 0 or few; if it flaps and costs, widen hysteresis or lower
threshold), then hot/cold rerun, load ramp, GSM8K probe. All local,
no push.

## z90 verdict (18:35): no regression at 90G -- both acceptance rounds green

Paired vs z90_eager_s0 (n=273, eager-eager spread sumD +5.2s):

    arm            medD   sumD   retM  drop   sto  preempts
    z90_eager_s2     4    +5.2   1.20     -  1021     1
    z90_adapt_s1    -2   -19.0   1.20   111   479     1
    z90_adapt_s3    10   -17.7   1.20    86   471     1

The win is preserved: -19.0/-17.7 sits in the same band as pure
tail's historical 90G wins (-18.5 record 7, -27.8 record 4 k-round).
Preempts 1/1 all four arms, watermark events 6 = calibration.

The regime tripped at 90G too: degrade_transitions=8 per arm (expected
0; the 450 threshold sits at the lower edge of 90G's 450-750s
residence band, and each watermark burst spikes the rate EMA under the
line before the quiet-period decay recovers it). degraded_emitted
407/401 of 894/922 emitted (~45%). Visible mix shift vs pure tail:
retrieval dividend not present this round (1.20M, equal to eager;
tail rounds showed 1.38-1.70M), dropped_evicted higher (111/86), but
covered_prefix skipping large (3.4M tokens) and store op count halved
(479 vs 1021) -- and the measured bottom line is unchanged. The flap
wart costs nothing measurable at either capacity; it remains a
smoothness issue only. Candidate fix if we ever want transitions<=1:
windowed rate over a fixed horizon (e.g. 120s) instead of per-sample
EMA, or a minimum dwell per regime.

Acceptance status: 60G parity (y60: +2.0/-4.2 inside eager noise,
preempts 1/1) AND 90G win intact (z90: -19.0/-17.7, preempts 1/1).
The knob at 450 meets the bar it was built for. Remaining items:
hot/cold rerun (0 watermark events -> residence infinite -> should
never degrade; confirms plumbing costs nothing), load ramp, GSM8K
coverage probe. All local, branch unpushed.

## hot/cold rerun (19:0x): the knob REGRESSES hot/cold at L1_GB=40

Setup note: first launch died (system python3 lacked torch/fastapi --
the harness needs SMOKE_PYTHON/SMOKE_VLLM pointed at the vllm-lazy
venv); relaunched correctly, all 6 fresh arms clean, config echo
verified in all 3 lazy vllm logs. Old lazy_tail rows below are from
the pre-knob run earlier today (13:52-14:06, code before 190d3c97),
kept in logs/ as the comparison target; fresh jsons archived the old
ones to logs/pre_degrade_baseline/ first.

Query-phase table (fresh eager and lazy=tail-default+450, x3 reps;
old lazy_tail x2 for reference):

    arm                 hot TTFT  cold TTFT   ext    apc
    eager (x3)          112-119   727-752   0.000  0.725
    lazy+450 (x3)       119-121   745-747   0.000  0.725
    lazy_tail old (x2)  149/190   299/639   0.581/0.458  0.725

The knob degraded (transitions=1 early in query phase, degraded_emitted
96 of 132) and the win evaporated: ext collapsed 0.5 -> 0.0, cold TTFT
back to eager's ~746ms, evictions 14 = eager's 14 (old tail: 5/8),
watermark 15-16 = eager's 15. Deterministic to the byte across reps.

Mechanism -- the signal is self-fulfilling here. At 40G the L1 barely
fits the 38.4G working set; residence is genuinely short (~60s scale,
watermark 15/run), so the policy reads "churn -> deferral pays nothing"
and emits everything immediately. But on this shape the deferral WAS
the value: pending ops died covered while hot docs stayed GPU-resident
(old tail query ledger: covered_prefix_advances=30 on admitted=33,
566K tokens skipped), lazy stored 27G vs eager's 31.9G, and that 5G of
withheld useless stores is what kept the cold docs alive in L1
(ext 0.5). Degraded emission forfeits the filtering window entirely
(fresh ledger: advances=0, skipped=0), recreates eager's churn, and
the churn then keeps residence low -- locked degraded, no recovery.

No fixed threshold separates the cases: hot/cold-40G needs <60s to
stay un-degraded, agentx-60G needs >320s to degrade. Residence alone
is the wrong discriminator. What DOES separate them in the ledgers is
the filtering dividend -- covered_prefix_advances/admitted:
hot/cold tail 91% (query phase), agentx tail 5.7% (y60), z90 11%.
Deferral that mostly filters is paying; deferral that mostly delays
is not. Candidate gate: degrade only if residence < threshold AND
recent covered-advance rate is low (e.g. <25%), both already visible
to the policy without new plumbing.

Acceptance state after this round: y60 parity (pass), z90 win intact
(pass), hot/cold-40G with knob = parity with eager but LOSES the
standing "stably beat on favorable" bar (knob-off tail beats eager
there). Options, user's call:
  a. Signal fix: add the covered-advance condition, rerun hot/cold +
     y60 (60G must still degrade) -- moderate work, closes all bars.
  b. Ship knob default-off, document it as opt-in for capacity-churn
     deployments; tail default alone already meets hot/cold + 90G,
     but 60G then needs the manual knob.
  c. Accept parity on hot/cold-40G (bar relaxation) -- not
     recommended; the win is real and cheap to keep via (a).

## Signal v2: volume-neutrality controller (commit 42dc3c81)

User picked option (a) with a directive: design generally, not against
our three workloads. The general principle shipped: **degrading may
change the timing of stores, never their volume.** A fixed residence
threshold cannot tell "churn regardless" from "churn because of me",
and the volume a regime WOULD produce is a counterfactual no passive
signal sees (the v1 trap) -- so the controller measures both sides by
briefly running them.

Machine: NORMAL -> (residence under threshold, windowed rate, outside
cooldown) -> TRIAL (45s immediate emission) -> commit to DEGRADED iff
trial emitted-block rate <= 1.25x the deferred trailing baseline, else
revert + 600s cooldown. DEGRADED lifts on residence recovery (2x) or
via periodic PROBE (45s deferred every 480s): probe rate dropping
below the degraded baseline by the same factor restores NORMAL.
Emission volume is the policy's own emitted-block ledger snapshotted
on the pressure heartbeat -- no protocol change, controller lives
entirely in the policy. All EMA estimators for L1 replaced by sliding
windows (120s, min span 60s) -- kills the y60/z90 flapping wart too.
Constants are measurement properties (window >= 2 burst cycles, trial
>= several store cycles, neutrality = short-window sampling noise,
probe/cooldown = duty cycle ~10%), none is workload-fit.

Knob semantics unchanged (threshold gates trial entry; 0 = off). New
counters: degrade_trials/commits/reverts/probes/probe_recoveries.
348 tests green (regime tests rewritten as controller walks: trial
opens not commits, neutral commits, volume-jump reverts + cooldown
holds, residence recovery, probe recovery, probe non-recovery via
pressure-path emission during probe, restart rebaseline, repeated
snapshots, zero capacity, disabled knob). ruff clean.

Predictions recorded before results:
- hot/cold-40G (v2 running, GPU 7): gate opens ~1-2 min into query,
  trial reads the ingest jump, reverts; one 45s degraded blip per
  phase, rest deferred. Ledger: trials=1, reverts=1, commits=0. Cold
  TTFT recovers most of the old tail win (blip costs the middle
  third of a 3-min phase; in a long-running deployment the duty is
  ~7%).
- y2-60G (running, slots 0/1/5/6): trial neutral -> commit ~min 7,
  probe ~min 15; degraded coverage ~55% of the window vs y60's 72%.
  Round measures whether that still buys parity. Ledger: trials=1,
  commits=1, reverts=0.

## y2-60G verdict (v2 controller, round done 19:43:59)

Prediction: trials=1, commits=1, reverts=0; degraded coverage ~55%; question was parity.

Actual (eager-eager noise band: y2_eager_s2 sumD = -6.3s):

| arm | sumD | p99 | drop | sto | retM | trans | trials | commits | reverts | degraded_emitted |
|---|---|---|---|---|---|---|---|---|---|---|
| y2_adapt_s1 | +21.0s | 10006 | 50 | 606 | 1.06 | 3 | 2 | 1 | 0 | 603/969 (62%) |
| y2_adapt_s3 | +6.3s | 9119 | 79 | 568 | 1.23 | 3 | 2 | 1 | 0 | 507/946 (54%) |

**FAIL.** s1 clearly outside noise; s3 at the edge. v1 reference (y60): +2.0/-4.2s,
drop=12/5, coverage 72%. v2 regressed the very workload v1 had already passed.

### Regime timeline (from counters heartbeats + eviction_controller lines, slot1)

- 19:27:11 run start (NORMAL, deferred).
- 19:30:31 first watermark burst (small early eviction already at 19:28:08).
- 19:32:25 trial opens (trans=1) -- **114s after churn onset**: the windowed
  estimator (120s window, 60s min span) needs ~2min of bursts before the
  average crosses the threshold. v1's EMA spiked through on the first burst.
- 19:33:16 commit (neutral, correct).
- 19:40:43 **direct residence-recovery lift** (trans=2): end-phase lull made
  burst spacing >120s, window saw ~zero evictions, estimate blew past the
  2x recovery line. Premature.
- 19:41:28 + 19:42:47 bursts return; 19:42:53 second trial opens (trans=3),
  unresolved at run end (hence trials=2, commits=1, reverts=0).

### Damage accounting (dropped_evicted change points)

- Onset window (churn start -> trial open): slot1 40/50 drops, slot3 48/79.
- Flap window (lift -> re-trial): slot1 10, slot3 31.
- Deferred-exposed time: ~120s at onset + ~130s in the flap, per arm.

Both windows are the same disease: **v2 verified the degrade direction with a
trial but left two unverified/slow paths** -- a slow attack (min-span honesty
delays protection exactly when the backlog is hostage to the first eviction
wave) and an unverified release (direct estimate-based lift, v1's flap bug
reappearing in the recovery direction, slower but longer-lived).

## Signal v2.1 design: fast attack, verified commitment, conservative release

General principle (not workload-fit): asymmetric evidence requirements.
A protective transition may be trigger-happy because the trial already bounds
the false-positive cost (45s + revert + cooldown); a risk-taking transition
must be verified because its false-positive cost (deferred backlog eaten by
the next burst) is unbounded by any machinery.

1. **Fast attack**: open the trial on per-heartbeat instantaneous residence
   (capacity / single-sample rate) < threshold, outside cooldown. Drop the
   min-span requirement for trial-opening only. This is v1's sensitive signal
   restored -- but it only opens a trial now, never commits. Hot/cold stays
   safe: its trials still revert on the volume check (cost: ~45s blip per
   cooldown period).
2. **Verified release**: remove the direct residence-recovery lift of
   DEGRADED entirely. Recovery happens only through a probe (45s deferred,
   volume-neutrality check). Windowed residence >= 2x threshold (min-span
   honesty kept for this direction) no longer lifts -- it *brings the next
   probe forward*. Probe false-positive cost is bounded: 45s exposure per
   probe interval, and the sensitive attack re-degrades ~10s after the next
   burst.
3. Constants unchanged; _PRESSURE_MIN_SPAN_SECS remains for release-direction
   estimates and baselines.

Expected on the three workloads: y60/y2 onset protection ~10s after first
burst (drops back to v1 levels), no unverified lift (end-phase lull becomes a
probe that may recover, re-degraded within ~10s of the next burst); hot/cold
trials revert as in v2; 90G tail unaffected (no churn crossings past the
sensitive gate... 90G DOES churn (6 watermark events) -- gate will open trials
there too; trial commits iff neutral, and at 90G immediate emission is
volume-neutral (nothing filtered), so it may commit and forfeit part of the
tail win. **Risk to watch in the z-round rerun**: if 90G commits degraded and
loses its -18s win, the neutrality factor or a tail-economy term must gate
commitment, not just volume. Record the prediction: z90-v2.1 is the round
that decides whether volume-neutrality alone is a sufficient commit criterion.

Status: hot/cold v2 still in flight (pid 956614, GPU 7); per the no-edit rule
nothing is implemented until it lands.

## hot/cold-40G v2 partial verdict (round crashed mid-way, rerun in flight)

Round shape this time: eager x3 + lazy(DEGRADE=450, v2) x3. Completed: eager
0/1/2 + lazy_0; crashed at lazy_1 startup -- teardown race, lazy_0's vllm had
not released GPU 7 yet (51.7 GiB free < 104.85 needed, engine core fails fast,
harness waits 900s anyway). GPU settled; lazy_1/lazy_2 relaunched with a
settle guard (rerun_lazy_12.sh, pid 1015102). The ld_L_lazy_h3c11_1/2.json
present during the crash window were stale v1-era files (18:57/19:00, ledger
lacks trials fields); the rerun overwrites them.

lazy_0 (v2), the completed seed:

- **Controller opened zero trials** (trials=0, transitions=0). The windowed
  gate never crossed: the query phase is ~35s, bursts are compressed, and the
  120s-window average keeps the residence estimate above threshold.
- Behavior therefore = pure tail: ext 0.353, cov 0.822, cold TTFT 711ms
  (eager 725/757ms), hot 188ms (eager 122/124ms), 9 evictions (eager 14).
- Tail win partially recovered vs v1's destruction (ext 0.000, cold 745-747)
  but weaker than the old pure-tail reference (ext 0.458-0.581, cold
  299/639ms). One seed; rerun will say if that gap is seed noise.
- **dropped_evicted=0.** Eviction at hot/cold only ever touches cold-doc
  blocks whose stores already emitted; the pending cohort (hot-set) is
  LRU-recent and never harvested.

## v2.1 attack signal revised: measured self-harm, not instantaneous residence

The instantaneous-residence fast attack written above is WRONG for hot/cold:
a single compressed burst yields instantaneous residence ~220s < 450 -> trial
opens -> the 45s trial spans the whole 35s query phase emitting hot-set
stores immediately -> the v1 regression returns through the trial door. The
lazy_0 result exposes the correct discriminator instead:

- y2 onset damage was entirely **pending ops evicted** (dropped_evicted 40/48
  in the gate-latency window).
- hot/cold has **zero** pending evictions ever; churn there only recycles
  already-emitted blocks.

So the fast attack trigger becomes: **the policy's own loss ledger**. Open a
trial when dropped_evicted increments between heartbeats (deferral observed
destroying its own backlog), outside cooldown. Keep the windowed-residence
gate (min-span honesty intact) as the slow path. Hot/cold then never opens a
trial even in fast mode (no drops), y2 opens one ~10s after the first burst
that bites pending (vs 114s), and the trial verdict still guards volume.

Caveat carried forward: at 90G, v1 saw dropped_evicted=111/86 -- drops there
are the tail-release economy working (blocks evicted before emission =
stores correctly avoided), not harm. Drop-triggered trials WILL open at 90G;
whether the volume-neutrality commit criterion then correctly reverts (or
commits harmlessly, as v1's 46%-degraded z90 still won by -19s) is exactly
what the z90-v2.1 round decides. If it commits and the win collapses, the
commit criterion needs an economy term, not the trigger.

Release side unchanged from the section above: probe-only recovery; windowed
residence >= 2x threshold schedules an immediate probe instead of lifting.

## hot/cold-40G v2 final verdict (rerun landed, all six runs clean)

Query-phase means (json-extracted; earlier console numbers were medians):

| run | hot_mean | cold_mean | ext | evict | drops | trials |
|---|---|---|---|---|---|---|
| eager_0/1/2 | 146/157/232 | 751/776/746 | 0.000 | 15/14/14 | - | - |
| lazy_0/1/2 (v2) | 289/259/275 | **572/605/585** | 0.353/0.358/0.391 | 9/9/8 | 0/3/4 | **0** |
| tail_0/1 (v1-era code, 13:58) | 226/268 | 456/524 | 0.581/0.458 | 5/8 | 0/0 | - |

**PASS.** Zero trials in all three seeds; the v1 destruction (ext 0.000,
cold = eager) is undone: cold mean beats every eager seed by ~20%. Hot mean
is worse than eager but matches old pure tail -- inherent to the family, not
a controller cost. Gap to old tail (50-120ms cold, 0.07-0.19 ext) is within
tail's own seed spread; different code era and time of day, not chased.

Two facts that reshape v2.1's fast attack:

1. Seeds 1/2 dropped 3-4 pending ops (seed 0: zero). "hot/cold never drops"
   is seed-dependent -- a bare any-drop trigger WOULD false-fire here.
2. Seeds 1/2 also show covered_prefix_advances=0 yet still win -- the win is
   partly timing-shaped (LRU-tail release order, fewer eviction cycles), not
   only volume filtering. The trial verdict remains the backstop if a trial
   ever does open here (v1 evidence: degraded emission stored +5G at hot/cold,
   so a trial would measure volume-increase and revert).

## v2.1 fast attack, final form: material loss share

Trigger comparison at first-drop time: y2 onset and hot/cold seeds 1/2 are
indistinguishable (one small drop event each). By the second event they
separate by an order of magnitude. Hence:

- **Fast path**: over the trailing _TRIAL_SECS (45s) window, open a trial
  when windowed dropped ops >= (_NEUTRALITY_FACTOR - 1) x windowed admitted
  ops (i.e. deferral destroying >=25% of intake -- the same 25% tolerance the
  commit criterion uses), with drops > 0. Cumulative admitted/dropped_evicted
  already live in the policy counters; snapshot them into _pressure_history.
  Replayed y2: triggers on the second drop event, ~24s after onset, ~14 drops
  eaten (v2: 114s, 40-48 drops; v1: ~12). Replayed hot/cold seeds 1/2: 5-10%
  share, silent. Margins: 40% vs 25% vs 10%.
- **Slow path** unchanged: windowed residence (min-span honesty) < threshold.
- **Release**: no direct residence lift. Windowed residence >= recovery line
  arms an immediate probe (minimum 4 x _TRIAL_SECS between probes so a failed
  probe is respected as evidence); recovery only via probe verdict.

Predictions to beat, recorded before implementation:
- y2-60G rerun: trials>=1 committing, drops <= ~20 per arm, no unverified
  lift (transitions even, matching trials+probe outcomes), sumD within
  eager-eager noise (~+-8s).
- hot/cold rerun: zero trials on seeds like these (share < 25%), cold win
  intact.
- z90 rerun: trials open (drops are structural at 90G); the round decides
  whether volume-neutral commit preserves the -18s tail win (v1's 46%-degraded
  z90 kept it) or the commit criterion needs an economy term.

## v2.1 implemented and committed: 22c46cb6

"Open trials on the loss ledger; recover only through probes." Pressure
history tuples now carry cumulative admitted/dropped op counts; a new
`_loss_is_material` (windowed dropped >= (_NEUTRALITY_FACTOR - 1) x windowed
admitted, over one _TRIAL_SECS window, with drops > 0) joins the NORMAL
gate; the DEGRADED direct residence lift is gone -- recovered residence
arms an early probe, spaced at least _PROBE_RETRY_MIN_SECS = 4 x
_TRIAL_SECS after the last one. Tests: direct-lift test rewritten as
probe-mediated recovery, two new material-loss tests, 327 passed /
13 skipped, ruff clean. Docs updated (eviction_aware.md loss gate +
recovery, lazy_offload.md knob paragraph, knob docstring).

Validation in flight (predictions in the section above):
- hot/cold v2.1 x3 seeds: pid 1075896 on **GPU 2** -- GPU 7 was taken by
  the other session's mm e2e suite between rounds (a leftover longdoc
  tree from a mis-launch was killed first; run_hot_cold.sh gained the
  settle guard).
- y3 (60G) then z91 (90G) chained on par slots 0/1/5/6: chain pid
  1067671, log v21chain.log.

### hot/cold v2.1 launch incident (no data lost, other session unharmed)

The first v2.1 hot/cold launch was killed seconds in to add the settle
guard, but killing the launcher shell orphaned its run_hot_cold.sh child
(same reparenting trap as the earlier longdoc orphan). The orphan --
still SMOKE_GPU=7 -- started its eager_0 the moment the other session's
mm e2e suite dipped between tests, collided with it on GPU 7, and raced
my GPU-2 launch for the same ld_L_eager_h3c11_0.json; the polluted json
failed validation and set -e ended the GPU-2 round after one run.
Verified afterwards: the mm suite finished naturally at 20:23 with 38/38
passing (junit suite_qwen2-vl-2b.xml), so the collision cost them
nothing. All hotcold trees killed (a pgrep pattern matching my own shell
self-killed one cleanup pass -- scope patterns to script paths, not
session ids), GPUs 2/7 confirmed empty, round relaunched at ~20:35 on
GPU 2 under setsid (whole tree in one session group for future
cleanups): pid 1098301, log hc_v21b.log. Lesson recorded: killing a
launcher must kill its session group, not the top pid.

## y3-60G verdict (v2.1, round done 20:39:58)

> **WITHDRAWN 2026-08-26 (a91).** The agentx `sumD` metric does not
> survive a same-config control: an eager-eager pair in the a91 round
> differed by +34.3s, larger than any effect claimed here, and 5%-trimmed
> every agentx arm sits in -6.4..+5.9s. See "a91 verdict" at the end of
> this file and `records/2026/08/26/12_*.md`. The hot/cold verdicts are
> unaffected.

| arm | sumD | p99 | drop | sto | retM | trials | commits | reverts | probes | degraded_emitted |
|---|---|---|---|---|---|---|---|---|---|---|
| y3_eager_s2 (noise) | +1.8s | 11169 | - | 1017 | 1.11 | | | | | |
| y3_adapt_s1 | **-8.1s** | 8540 | 106 | 310 | 1.15 | 2 | 1 | 1 | 0 | 175/903 (19%) |
| y3_adapt_s3 | **-12.1s** | 8615 | 117 | 311 | 1.16 | 2 | 1 | 1 | 0 | 176/901 (19%) |

**PASS -- and not parity but a win**, with better p99 than eager. Both arms
identical in shape: covered_prefix_advances=125/128 (5.0M tokens skipped,
vs y2's 2.7-3.4M), preempts 2/2, tokens_stored 1.92/1.90M.

Prediction scorecard:
- "trials open, one commits" -- HIT (trials=2, commits=1).
- "no unverified lift" -- HIT (probes=0, direct lift gone; transitions=3
  = trial1 in/out + trial2 in; trial2 committed near run end and stayed).
- "sumD within +-8s" -- EXCEEDED (-8.1/-12.1).
- "drops <= ~20" -- **MISS, and the harm model behind it was wrong**:
  drops were 106/117, higher than v2's 50/79, yet retM went UP and sumD
  won. These drops are the tail-release economy (blocks evicted before
  emission = stores correctly avoided), not damage -- the same pattern as
  90G. The y2 damage attribution (drops -> lost retrieval) was at best
  partial; the dominant y2 cost was the committed-degraded window
  disabling covered-prefix filtering for 62% of the run.

Mechanism, reconstructed from the ledger: the material-loss trigger
opened trial1 at churn onset, when the trailing deferred baseline was
still ~0 -- and the baseline-0 arithmetic reverted it (any trial emission
reads as volume increase). 600s cooldown then kept the run deferred
through the bulk of the churn; trial2 opened post-cooldown against a
flowing baseline and committed, covering only the last ~19%. The early
trigger changed the trial verdict from y2's commit to revert: an early
trial is judged against a strict (empty) baseline, a late one against a
loose one. The revert was CORRECT for this workload -- deferral at 60G
with the current code (economy admission, covered-prefix skips, caps,
idle drain) filters 5M tokens and beats eager outright; the v1-era
premise "60G needs degradation" described older code. What the knob now
buys at 60G is bounded trials that keep concluding "defer"; the y2
regression was the controller overriding a good default, not rescuing a
bad one.

Open question promoted by this round: is there any workload left where
committing DEGRADED is the right call? z91 (in flight) probes the 90G
side; the hot/cold reruns guard the other flank. If every trial should
revert, the knob's value is insurance, and the commit criterion's
baseline-sensitivity (strict when early, loose when late) is a feature
biasing toward deferral -- worth stating in the design doc once z91
lands.

ANSWERED by z91 below: yes -- at 90G the commit is right, and the win
grew rather than collapsed.

## hot/cold-40G v2.1 verdict (GPU 2, all eight runs clean)

Query phase. Both statistics are quoted because the run log prints p50 while
the json carries means -- an earlier pass of this section mixed the two:

| run | hot mean/p50 | cold mean | cold p50 | cold p90 | ext | evict | adv | drops | trials |
|---|---|---|---|---|---|---|---|---|---|
| eager_0/1/2 | 185/257/218, p50 114/131/115 | 811/813/812 | 792/791/790 | 807/816/811 | 0.000 | 14/14/14 | - | - | - |
| lazy_0/1/2 (v2.1) | 219/211/263, p50 139/138/172 | **452/432/486** | 397/276/473 | 782/748/784 | **0.555/0.584/0.536** | 6/6/7 | 23/34/46 | **0** | **0** |
| lazy_0/1/2 (v2, prior round) | 289/259/275 | 572/605/585 | - | - | 0.353/0.358/0.391 | 9/9/8 | 11/0/0 | 3/4 on two seeds | 0 |
| tail_0/1 (pre-knob reference) | 226/268 | 456/524 | 299/639 | 736/763 | 0.581/0.458 | 5/8 | 30/11 | 0 | - |

**PASS, full margin.** Cold mean 44-47% under eager, ext 0.54-0.58 back in
the pure-tail reference band (0.458-0.581) instead of trailing it, evictions
6-7 vs eager's 14, zero drops and zero trials in every seed. The
material-loss gate stayed silent because there was no loss at all, and the
residence gate never crossed. Hot mean stays above eager (211-263 vs
185-257), unchanged family cost, ranges overlap.

Prediction scorecard: "zero trials on seeds like these, cold win intact" --
HIT on both.

Honesty note: this GPU-2 round is also better than the v2 round's GPU-7
numbers (572-605ms cold, ext 0.35-0.39, drops 3-4, advances 0 on two seeds),
but both controllers ran zero trials, so NORMAL behaviour was identical and
that gap is environmental (GPU-7 neighbours / timing), not a v2.1
improvement. The claim v2.1 earns from this round is exactly: the
loss-gated controller does not disturb the favourable workload. Equally, the
pre-knob tail reference and v2.1 are statistically indistinguishable here,
which is the correct outcome -- on this workload the knob must be a no-op.

## z91 (agentx 90G) v2.1 verdict: the commit fired and the win grew

> **WITHDRAWN 2026-08-26 (a91).** The agentx `sumD` metric does not
> survive a same-config control: an eager-eager pair in the a91 round
> differed by +34.3s, larger than any effect claimed here, and 5%-trimmed
> every agentx arm sits in -6.4..+5.9s. See "a91 verdict" at the end of
> this file and `records/2026/08/26/12_*.md`. The hot/cold verdicts are
> unaffected.

Round done 21:02. Paired vs z91_eager_s0 (n=273, baseline sumTTFT 287.3s,
p50 581 p90 2161 p99 8781); eager-eager spread -6.6s:

    arm            medD    sumD   p50   p90    p99    max  retM drop  sto
    z91_eager_s2     -6    -6.6   560  1960   8824  12985  1.23    -  1026
    z91_adapt_s1    -10   -51.3   569  1683   7601   8516  1.33   18   731
    z91_adapt_s3     -8   -54.9   569  1747   7310   8571  1.30   19   732

Ledger, s1 / s3: admitted 979/980, emitted 961/961, dropped_evicted 18/19,
covered_prefix_advances 48/49 (2.47M tokens skipped), degraded_emitted
722/729 of 961 (75%), degrade_transitions 3, trials 1, commits 1, reverts 0,
probes 1, probe_recoveries 0.

Regime walk, identical on both seeds: NORMAL -> TRIAL (material loss at
onset) -> commit -> DEGRADED; one probe fired later, did not recover,
DEGRADED held to the end. Three transitions accounted for by one trial entry,
one commit and one probe excursion -- no unverified lift anywhere.

Prediction scorecard:
- "trials open, drops are structural at 90G" -- HIT, exactly one trial per
  seed, opened on the loss ledger.
- "the round decides whether the volume-neutral commit preserves the -18s
  tail win, or the criterion needs an economy term" -- RESOLVED for the
  criterion. It committed and the win grew from v1's -19.0/-17.7 to
  **-51.3/-54.9**, p99 7310-7601 vs eager 8781/8824, max 8.5s vs eager
  13.0s. No economy term needed; that work item is dropped.

Honest reading of where the -51s comes from. It is not mostly the
degradation. Against v1's z90 the same arms now drop 18/19 instead of
111/86 and store 731/732 instead of 479/471, with retM 1.30-1.33 against
eager's 1.23 (v1 was a flat 1.20 on all four arms) -- i.e. under the current
admission economy plus covered-prefix filtering the deferral itself retains
materially more than eager, which is the same reframe y3 forced at 60G. What
the commit adds is that the 75% degraded share stopped the structural
eviction loss from eating those stores, while volume stayed neutral (961 of
979 admitted emitted). The controller's contribution is correctly scoped:
detect real loss, verify that immediate emission does not change volume,
commit, and keep checking by probe.

## Acceptance arc closed at 22c46cb6 (v2.1)

> **WITHDRAWN 2026-08-26 (a91).** The agentx `sumD` metric does not
> survive a same-config control: an eager-eager pair in the a91 round
> differed by +34.3s, larger than any effect claimed here, and 5%-trimmed
> every agentx arm sits in -6.4..+5.9s. See "a91 verdict" at the end of
> this file and `records/2026/08/26/12_*.md`. The hot/cold verdicts are
> unaffected.

| workload | standing bar | v2.1 result |
|---|---|---|
| agentx 60G (unfavourable) | lazy >= eager | WIN -8.1/-12.1s vs noise +1.8s, p99 -24% |
| agentx 90G (favourable) | stable win | WIN -51.3/-54.9s vs noise -6.6s, p99 -13/-17% |
| hot/cold 40G (favourable) | stable win | WIN cold mean -44/-47%, controller a no-op |

Every regime change across the whole arc went through a bounded trial or
probe; zero unverified transitions in any round. Volume neutrality held
everywhere: 40G zero drops, 60G 106/117 drops with retM *up* (1.15/1.16 vs
eager 1.11), 90G 18/19 drops on 979 admitted with retM 1.30-1.33 vs 1.23.

Open items, none blocking: load ramp ("shang qiangdu", concurrency 2.4) and
the GSM8K coverage probe for 5ea3cc6e, both still unstarted.

## a91: attribution + slot-swap round at 90G (predictions pre-stated)

Two gaps the acceptance arc left open, closed in one round: (1) no knob-off
lazy arm at 60G/90G, so the -51s cannot be attributed between the controller
and the admission economy; (2) round.sh derives the GPU from the slot index,
and in y60/z90/y2/y3/z91 eager was ALWAYS on slots 0/2 and lazy ALWAYS on
slots 1/3 -- the eager-eager spread only ever measured slot0-vs-slot2, so a
systematic slot bias favouring 1/3 would be invisible and credited to lazy.

Design (L1_GB=90 throughout), config-to-slot mapping deliberately inverted:

    slot0 (gpu0) a91_on_s0      lazy  DEGRADE_SECS=450   <- was eager's slot
    slot1 (gpu1) a91_eager_s1   eager                    <- was lazy's slot
    slot2 (gpu5) a91_off_s2     lazy  DEGRADE_SECS=0     <- was eager's slot
    slot3 (gpu6) a91_eager_s3   eager                    <- was lazy's slot

DEGRADE_SECS=0 is a hard disable (`if threshold <= 0: return` before any
regime work), so off_s2 is pure deferral on identical code. Baseline for
cmp2 is a91_eager_s1; the eager-eager pair now measures gpu1-vs-gpu6.

Predictions, recorded before launch:
1. **on_s0 reproduces z91** in the -40 to -60s band despite sitting on
   eager's old slot. If it lands above -20s, z91's number was slot-inflated
   and every agentx verdict in this record needs the same swap check.
2. **off_s2 also wins big, -30 to -55s**, i.e. |on - off| < ~15s. Rationale:
   y3 already showed deferral alone winning at 60G, and z91's retention
   economy (retM 1.30-1.33, 2.47M tokens filtered) is a deferral property,
   not a degradation property. Confidence moderate -- this is the prediction
   most likely to miss, because 75% of z91's emissions were degraded.
3. **off_s2 drops >> on_s2's 18/19**, somewhere in the 60-150 band (v1's
   z90 saw 111/86 with the old economy); covered_prefix_advances stays
   ~45-50 on both, since filtering is upstream of the regime machine.
4. **eager-eager spread on gpu1-vs-gpu6 within +-8s**, matching the +5.2 /
   +1.8 / -6.6 seen on gpu0-vs-gpu5.

What each outcome would mean:
- off ~= on, both winning: the knob is insurance, not the source of the win.
  Default-off with documented opt-in becomes the defensible ship position,
  and the arc's -51s belongs to the admission economy.
- off materially worse than on (>20s): the knob earns its complexity at 90G
  and should default on.
- on collapses after the swap: methodology problem, not a code problem --
  every paired verdict in this record gets re-run swapped before anything
  ships.

Note GPU 0 carries ~1.6 GiB of three foreign small processes (other users);
they were present during z91's eager_s0 as well.

## a91 verdict: the agentx sumD wins do not survive a same-config control

Round done 21:43. Baseline a91_eager_s1 (n=270, sumTTFT 246.7s):

    arm            medD    sumD   p50   p90    p99    max  retM drop  sto
    a91_eager_s3      2   +34.3   543  2210   8370  21622  1.42    -   969
    a91_on_s0         1   +22.0   567  1917   8509  22866  1.42  145   304
    a91_off_s2        6   +14.7   563  1930   8750  13270  1.41   75   216

The eager-eager pair -- identical config, identical L1, same round --
differs by **+34.3s**, larger than z91's entire claimed -51s effect. Both
lazy arms sit *between* the two eagers. All four arms were time-aligned
(first request within 7s of each other), so the usual "different phase of
the trace" excuse does not apply.

Where the 34.3s lives: per-minute sums show a ~4-minute window (minutes 8-12
of the round, wall clock ~21:34-21:38) in which three arms took 13-23s TTFT
outliers and a91_eager_s1 did not (its max in that window is 1016ms against
21622/22866/13270 for the others). Excluding requests started in that window
collapses every difference:

    arm            sumD full   sumD excluding min 8-12
    a91_eager_s3      +34.3            +1.3   (n=191)
    a91_on_s0         +22.0            +5.1
    a91_off_s2        +14.7            +6.5

### Re-scoring every agentx round on a robust statistic

sumD is a sum over ~270 paired requests, so a handful of multi-second stalls
own it. 5%-trimmed sumD (drop the 13 most negative and 13 most positive
deltas) and median x n, computed over every round in this campaign:

    round          arm                sumD   trim5%   medD*n
    y60 60G v1     y60_eager_s2        6.4      2.9      0.3
    y60 60G v1     y60_adapt_s1        2.0      1.4      0.4
    y60 60G v1     y60_adapt_s3       -4.2      2.1      1.8
    z90 90G v1     z90_eager_s2        5.2      1.1      1.2
    z90 90G v1     z90_adapt_s1      -19.0      1.6     -0.4
    z90 90G v1     z90_adapt_s3      -17.7      5.4      2.6
    y2  60G v2     y2_eager_s2        -6.3      2.9      1.4
    y2  60G v2     y2_adapt_s1        21.0      3.5      1.4
    y2  60G v2     y2_adapt_s3         6.3     -2.0     -0.2
    y3  60G v2.1   y3_eager_s2         1.8      1.2      0.2
    y3  60G v2.1   y3_adapt_s1        -8.1      5.9      1.7
    y3  60G v2.1   y3_adapt_s3       -12.1      4.4      1.5
    z91 90G v2.1   z91_eager_s2       -6.6     -2.6     -1.7
    z91 90G v2.1   z91_adapt_s1      -51.3     -4.1     -2.8
    z91 90G v2.1   z91_adapt_s3      -54.9     -6.4     -2.3
    a91 90G swap   a91_eager_s3       34.3      1.3      0.6
    a91 90G swap   a91_on_s0          22.0      3.1      0.2
    a91 90G swap   a91_off_s2         14.7      3.1      1.5

sumD ranges over 89s (-54.9 to +34.3). Trimmed, every arm in the campaign --
lazy and eager control alike -- sits in -6.4 to +5.9s. Concentration check on
z91: its -51.3s has best10 = -54.4s, i.e. **ten requests out of 273 are the
entire win**; y3's -8.1s has best10 -45.5 and worst10 +30.2, trimming to
+5.9s (lazy slightly worse).

**Conclusion: no agentx round in this campaign resolves lazy vs eager.** The
harness's sumD is a rare-stall lottery at n~270, and a same-config control
draws tickets of the same size. Every agentx verdict in this record --
y60 parity, z90 -19s, y2 +21s regression, y3 -8/-12s win, z91 -51/-55s win
-- is withdrawn as unsupported, in both directions. That includes the y2
"failure" that motivated v2.1 and the y3 "win" that motivated the reframe.

### What still stands

- **hot/cold 40G.** Not a tail lottery: eager cold TTFT is a tight cluster
  (q1 784, median 792, q3 795, three seeds at mean 811/813/812) and lazy
  moves the whole lower half of the distribution (min 167-197, q1 187-209,
  10%-trimmed mean 445/421/484). Three seeds agree, and the pre-knob tail
  reference lands in the same band. This is a broad-based, reproducible
  ~45% cold-TTFT reduction and it is unaffected by everything above.
- **Controller safety.** Trials fire only on material loss (0 at 40G across
  3 seeds), transitions are all trial/probe-verified, volume neutrality held
  in every ledger. This is a property of the ledgers, not of sumD.
- **The knob buys nothing measurable at 90G.** a91 is the first direct
  on-vs-off comparison: +22.0 vs +14.7 full, 3.1 vs 3.1 trimmed. Off also
  dropped fewer (75 vs 145) and stored fewer (216 vs 304).

### Prediction scorecard (1 hit, 3 misses)

1. "on_s0 reproduces z91 at -40 to -60s" -- **MISS**. +22.0s against the
   baseline eager; -12.3s against the other eager. The stated falsifier
   ("if it lands above -20s, z91 was slot-inflated and every agentx verdict
   needs the same check") fired, and the check is the section above.
2. "off within ~15s of on" -- **HIT** (7.3s apart, identical trimmed).
3. "off drops >> on drops, 60-150 band" -- **MISS, inverted**: on 145,
   off 75. Degradation ran and *more* was lost, not less.
4. "eager-eager spread within +-8s" -- **MISS**, +34.3s. This was the
   assumption every prior round rested on, and it is the one that broke.

### What the harness needs before any agentx claim is made again

1. Primary metric must be robust: trimmed mean, median, or a stall-count
   (requests over 5s), with sumD demoted to a secondary. The current
   headline metric cannot see an effect smaller than its own tail noise.
2. A same-config control pair in **every** round is necessary but not
   sufficient -- prior rounds had one and it read +5.2/+1.8/-6.6, which is
   what made +34.3 unthinkable. Needs several rounds to characterise.
3. Config-to-slot rotation across replicates, since 4 arms x 4 configs
   confounds slot with config by construction.
4. Suspect cross-arm coupling: 4 arms x L1_GB=90 is 360 GB of host L1 plus
   the neighbours' jobs, and the minute 8-12 window looks like host-level
   contention. Consider 2 arms per round at 90G.

Estimated cost to actually settle agentx: ~4 replicate rounds at 25 min
each, rotating assignment, robust metric. Not started -- the user decides
whether agentx is worth that, given hot/cold already carries a clean win.
