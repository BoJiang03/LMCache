# 8. Idle-drain plan and the default flip

State at close of 2026-08-26. Evidence chain in record 7 (sections 1-12);
this file is the working plan the next session starts from. Tree clean at
`1c43ca02`, nothing committed today (measurement only).

## Decisions taken today

1. **Flip default `StoreReleasePlacement` to `LRU_TAIL`** (recommendation
   staged, user has not yet said go). Evidence: agentx 60G 2v2 recovers
   10-13.5 s, agentx 30G preview 30 s swing, hot/cold 2v2 parity with head
   and the ~10 s win over eager intact (dropped_evicted=0 there, head's
   premise never engages). Head's only observed edge: one arm, 14.5 s, at
   agentx 90G where both placements beat eager anyway. Keep `eviction_head`
   as a config value; document the trade. Ship as its own small PR.
2. **Idle-preferring drain + per-step byte cap** is the development item for
   the residual agentx gap (tail+hz10 still +9/+19 ms medians, 19 s config
   spread vs eager's 3 s). Separate PR after the flip.
3. **Pin-free optimistic emission rejected** (record 7 section 9): the
   corrupt-object visibility window needs a two-phase-commit MP protocol
   change or an upstream vLLM API; LRU_TAIL delivers most of the benefit.

## Acceptance bar (user, 2026-08-26 afternoon)

Lazy must at least match eager on unfavorable workloads and stably beat it
on favorable ones. Unfavorable here = agentx-shaped: reuse distance below
GPU pool turnover (gap p50 1.4 s vs pool turnover ~152 s), so APC is the
decisive resource and L1 only carries the tail.

## Idle-drain implementation checklist

- **Config** (`LazyOffloadPolicyConfig`, eviction_aware.py:366):
  `idle_drain_max_ops: int = 0` (0 disables), idle threshold on
  `new_blocks_allocated` (EMA-smoothed so the first step of a burst does
  not count as idle), `max_drain_blocks_per_step: int` byte cap. Validate
  in `__post_init__` like the existing knobs.
- **Signal**: already present -- `observe_step` receives
  `new_blocks_allocated` / `est_next_step_blocks` every token-producing
  step (lazy_offload_pending_store.py:374). No vLLM-side plumbing.
- **Policy** (`EvictionAwareStoreQueue.collect_due`): after the
  danger-depth pass, if the step is idle and budget remains, emit oldest
  pending ops FIFO up to `idle_drain_max_ops`, respecting
  `blocked_request_ids` and prefix-chain order. Byte cap truncates a
  request's due chunks at a prefix (`to_store` is chunk-granular; the
  remainder stays pending, one in-flight batch per request already
  enforced at lazy_offload_manager.py:518).
- **Counters**: `idle_emitted`, `idle_drain_steps` in the ledger,
  documented as effectiveness sensors per the convention at
  eviction_aware.py:413.
- **Tests**: layer-1 scenario with zero-allocation steps (idle emission
  fires, danger pass unchanged, FIFO mode untouched); unit tests for byte
  cap truncation and chain-order preservation.
- **Docs**: eviction_aware.md + the decision-model doc.
- **Acceptance rounds**: 60G eager vs tail+idle (target medD ~ 0, spread
  at eager's ~3 s), 90G no-regression round, hot/cold rerun.

## Measurement discipline (carried lessons)

- In-round paired 2v2 only; single arms are previews. Cross-round drift at
  60G reached 20 s on identical configs.
- `gpu_prefix_hit` (vLLM cumulative log line) is unusable: 8.3% vs 46.3%
  on identical arms. Use the retrieval/store ledgers.
- 30G is noise (spread 21-32 s); 60G is the trustworthy unfavorable point.

## Loose ends

- GSM8K coverage gap for 5ea3cc6e (record 5): hot/cold has the right shape
  (covered_prefix_advances 51/11 in the tail cells) -- add a correctness
  probe on that scenario rather than growing the GSM8K pool.
- QASPER re-cert with tail if the reviewers ask; hot/cold already covers
  the same failure mode (working set > pool, coverage as currency).
- The harness copy with the `lazy_tail` config key lives in this session's
  scratchpad `hotcold/` (CONFIGS whitelist lives in accuracy.py:98, not
  workload.py -- both patched there). Originals in lazy_offload_repro are
  untouched.

## Implementation and first acceptance results (added later the same day)

Both items shipped on `lazy-offload-publish`:

- `d03106da` flips the `StoreReleasePlacement` default to `LRU_TAIL`.
- `230d15bc` adds `idle_drain_max_ops` / `idle_threshold_blocks` (idle
  drain) and `max_drain_blocks_per_step` (block volume cap, soft bound,
  one `_DrainBudget` shared by the pressure/backlog/idle paths). Both off
  by default. 236 tests green, ruff clean.

**u60 (60G, eager x2 vs tail+IDLE_OPS=4 x2): idle drain judged harmful.**
medD +28/+29 ms, sumD +40.4/+41.0 s, p90 2903/3045 vs eager 2317/2327,
retM 1.19/1.28. The mechanism worked exactly as designed -- idle_emitted
956/1000, pending=0, dropped_evicted collapsed 180 -> 2 -- and that is the
problem: zero filtering plus ~4x the emissions turns lazy into delayed
eager whose pins and decode-phase copies all land on the serving path.
With cap8@90G (record 7 knob round) and hz10@60G (t60), this is the third
failure of the emit-earlier/smaller family. Config note: slots remapped
this round (GPU 1/6 held by another session): SLOT 1->GPU2, 3->GPU3.

**Next: v60** tests the untried lever -- BLOCK_CAP=64 on the pressure
path (tail placement, idle off), spreading a due burst at ~1K tok/step
(eager's own store shape) instead of one ~9K-tok contiguous copy in phase
with the prefill that triggered it. Extra drops from held-back tails are
free at 60G. Harness now carries IDLE_OPS/IDLE_THRESH/BLOCK_CAP env and
defaults STORE_RELEASE=lru_tail (matches the shipped default).

**v60 (60G, eager x2 vs tail+BLOCK_CAP=64 x2): block cap judged harmful,
worse than anything tried.** medD +49/+51 ms, sumD +102.0/+103.5 s, p99
11267/16021 vs eager 7745/7822. Drops 443/405 of ~970 admitted (46%),
throttled_drains 211/226, retM 1.07 (unchanged).

Attribution correction (my first read in chat was wrong): the drain-step
count does NOT explain it. Tail-only r60/s60 arms run ~81K drain_steps /
~2.6M free-queue blocks read as their normal state, v60's 70K/2.3M is in
family, and t60 (hz10) read 8.2M blocks while staying mixed -- per-step
machinery cost does not correlate with the losses across rounds. The
leading explanation is receipt serialization: with a 64-block cap and one
in-flight batch per request, a due ~9K-token chain needs ~9
emit-wait-receipt round trips; its uncopied remainder sits at the
eviction edge the whole time (46% died) and every fragment still lands
in phase with the burst -- the cap fragmented the in-phase volume without
reducing it.

Emission-side scoreboard at 60G, all against in-round eager pairs:
head placement -17s / tail placement -7..-10s / tail+hz10 mixed /
tail+idle4 -40s / tail+cap64 -102s. Every intervention that changes when
or how finely lazy copies has lost to plain tail waiting.

**w60 launched**: off x2 vs eager x2 at 60G -- the bounding measurement.
If off ~ eager, eager's store cost is already hidden under prefill
compute, lazy's ceiling at 60G is parity, and the residual 7-10s must be
found in the lazy machinery itself (pins, validation, store timing), not
in emission policy. If off beats eager, there is headroom to win by
storing less.

**w60 (60G, eager x2 vs off x2): off is far worse -- both branches of the
decision rule were wrong.** off sumD +248.5/+218.5 s vs in-round eager,
p90 5722-6389 vs 2160-2577, 49/53 pairs lost >1 s each, pair count fell
to 226/227 (43-44 requests drifted out of the pairing window). retM 0.00,
no stores, as expected.

The rule assumed off <= eager. Reality: L1 at 60G carries ~230-250 s of
TTFT value on agentx -- reuse below pool turnover still leaves a tail
that L1 serves, and eager banks all of it (~1050 stores, retM 1.0).
"Storing less" is dead as a direction, and "parity because stores are
free" is dead too: stores are hugely net-positive. Lazy tail already
captures ~97% of the L1 value and is 7-10 s short. The residual is now
squarely a value-capture or machinery question, not a store-volume one:
(a) lazy stores later and less (sto 330-840 vs eager ~1050;
dropped_evicted ops never reach L1), so some hits are smaller or missed
even at similar retrieval counts; (b) pins/copies on the serving path.
Next probe can start from existing r60/s60/w60 logs: compare retrieved
token volume per arm (not retrieval count) before spending another round.

**Post-w60 log probe (no new round, snapshot.txt across r/s/t/u/v/w60):**
the value-capture branch is falsified too. Retrieved token volume is
equal between eager and tail (s60: 974/978K vs 986/967K); stored token
volume is also equal (~2.0M both) -- at 60G lazy filters ops (1053 ->
225) but not bytes; everything is stored eventually, just consolidated.
Total D2H wall time is actually lower for tail (5.8 s vs 9.1 s). What
differs: chunk shape and pins. Tail stores p50 2.5K / p90 32K / p99 65K
tokens with single-store max 0.95 s (eager p50 256, max 0.08 s); a 65K
op pins ~4K blocks, 25% of the 16K pool, through copy+receipt on the
serving path. Preempts: tail 4/2 vs eager 1/1 in s60. Also explains the
cross-arm retM pattern: arms whose mechanism costs GPU blocks (head
placement 1.4M tok_ret, idle 1.19-1.28M) retrieve MORE than eager's
1.0M -- compensation for self-inflicted GPU misses, not extra value.
Residual hypothesis now: in-phase giant-chunk copies + pin dwell.
Direct lever untested on the emission side: cap that keeps chunks big
enough to avoid v60's receipt serialization but bounds pin dwell (e.g.
BLOCK_CAP 512, ~8K tokens = eager's p90), or split emission without
serializing receipts.

**x60 (60G, eager x2 vs tail+BLOCK_CAP=512 x2): cap512 does not move the
gap -- the pin-dwell/chunk-shape hypothesis is falsified.** sumD
+14.0/+8.9 s (plain tail's historical 7-10 s band), medD +16/+9 ms,
eager-eager spread -2.4 s. The cap did exactly what it was designed to
do: store chunks p50 ~2.9K / p90 8192 / p99 32000 tokens (tail was p50
2.5K / p90 32K / p99 65K), 2.5 ops coalesced per batch (no v60 receipt
serialization; drops 17% vs cap64's 46%), 16% fewer bytes stored
(1.69/1.70M vs 2.01M) at identical retrieved value (~1.0M). Soft bound
passed one whole 53K op (max store 0.97 s) as specified. And yet:
preempts 4/2 -- identical to uncapped tail, vs eager's 1/1 -- and the
deficit unchanged. Bounding pin duration and chunk size changes nothing.

Refined residual suspect: it is not how LONG pins live but WHEN they
appear. Danger-triggered emission fires, by construction, at allocation
bursts -- pins remove free blocks at exactly the moment the pool is
tightest, and preempt count (4/2 vs 1/1; each ~9K-context preempt costs
a multi-second recompute + queue cascade) is the one sensor that tracks
the deficit across every arm. Chunk shaping cannot fix a timing
coincidence that is the trigger condition itself. Emission-side design
space is now closed twice over.

Direction (proposed, awaiting user's word): adaptive degradation --
route ops to the plain eager store path at admission when the ledger
signal says filtering pays nothing (60G agentx: dropped_evicted 17-18%
and pool turnover far above reuse distance; 90G and hot/cold: ~0).
Equality with eager on unfavorable workloads then holds by construction
(same code path), and lazy keeps its wins where waiting filters.
Probe note: an emit-at-admission arm via HORIZON=huge is confounded --
danger depth would walk the whole 16K-block free queue every drain step
(~1.3B reads/run); FIFO policy is a finished-requests threshold-100
placeholder, not an immediate-emission probe. A clean machinery probe
needs a small temporary knob; skip it if degradation bypasses to the
eager path anyway.
