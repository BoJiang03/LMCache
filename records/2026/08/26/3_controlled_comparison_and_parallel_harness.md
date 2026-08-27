# Controlled lazy-vs-eager comparison, and the harness that made it possible

Follow-up to [2_lazy_tail_root_cause_and_backlog_cap.md](2_lazy_tail_root_cause_and_backlog_cap.md).
Task given: fix lazy being slower than eager before moving on to a load ramp.
Outcome: the deficit does not exist under a controlled comparison. Three
readings had to be retracted first, all for the same reason -- the effect is
~10% and every uncontrolled comparison carries at least that much drift.

## 1. Three retractions, one cause

| # | Reading | Basis | Defect |
|---|---|---|---|
| 1 | lazy loses ~20 s of tail at every L1 size | paired sweep | pairing key included `session_num`, a client session slot reassigned per run; matched 154/256, and which rows survived correlated with timing, so it kept losses and dropped the offsetting wins |
| 2 | total at parity, tail regression is real, not noise | key fixed to `(conversation_id, turn_index)`, 255/256; eager reproduced p99 to 1%, lazy's was +42% | lazy had n=1 |
| 3 | lazy is not reproducible, variance 18x eager's | lazy p99 11978 vs 8165 | those two runs were 14 h apart |

The same eager configuration, three times: 303.3 s (Aug 25 18:41),
298.8 s (Aug 26 08:47), 281.0 s (Aug 26 09:30, parallel slot). A 22 s
spread against a 24-29 s effect. **Any comparison that is not simultaneous
measures the drift.** That is the whole explanation for the flip-flopping;
the three readings do not contradict each other because they were not
measuring the same thing.

## 2. What the controlled round says

Four arms launched together, one eager control, three lazy replicates, one
GPU each (0/1/5/6), same trace/seed/duration. Paired on the in-round eager:

| arm | sum d | total | p50 | p90 | p99 | max | dropped_evicted |
|---|---|---|---|---|---|---|---|
| eager (control) | -- | 281.0 s | 547 | 2160 | 8575 | 12620 | -- |
| lazy a | **-28.8 s** | 251.7 s | 580 | 1979 | 7661 | 9183 | 132 |
| lazy b | **-26.4 s** | 254.1 s | 589 | 1952 | 7728 | 9165 | 164 |
| lazy c | **-24.1 s** | 256.4 s | 567 | 2046 | 7679 | 8920 | 135 |

Replicate-to-replicate noise, paired: +2.4 s and +4.7 s. The effect is 5x
that. Total -9~10%, p99 -10%, max -26~29%. Median delta is +11~12 ms: a
small consistent body cost, repaid many times over in the tail.

Remaining confound, not yet resolved: arm-to-slot mapping was fixed by
argument order, so eager always ran on GPU 0 -- the only one of the four
carrying other users' residents (1.6 GB: one `rui` process and two
`/opt/venv/bin/lmcache server`). A slot-rotated round (eager on GPU 0 and
GPU 6, lazy on GPU 1 and GPU 5) was in flight when this was written. Note
also that another of my own sessions (`multi_modal_verify`, `vllm-mm` venv)
held a transient 5.4 GB on GPU 1 at 10:16, inside that round's slot 1.

## 3. Candidate mechanisms, all closed

Every one of these was a hypothesis for "why lazy is worse", and each died
on a number:

- **Buffered-store loss.** `max_pending_ops=8` cut `dropped_evicted` from
  140 to 46 and retrieved the most tokens of any arm (1.78 M), and TTFT was
  still worse (+11.8 s paired, p50 651). Cutting the loss 3x bought nothing,
  which closes the loop opened in record 2. The knob does not ship; default
  stays 0.
- **Pin pressure.** Peak pin is 5808 blocks, 35% of a 16384-block pool, but
  the run spends 1.8 s of 900 s with >10% pinned and 0.1 s with >25%. A 0.2%
  coincidence probability cannot produce a dozen multi-second spikes.
- **Retrieve deficit.** Per-request retrieve volume inside each request's own
  TTFT window: of 23 pairs with |dTTFT| > 1 s, only 7 have the sign agreeing.
  The largest gains are cases where lazy retrieved *less*.
- **Preemption.** Real and directional (events per run: 90G 1 vs 2, 60G 1 vs
  3, 30G 0 vs 7) but it does not explain the tail: 0 of the 12 regressions at
  90G and 2 of 34 at 30G overlap a preemption window.
- **GPU prefix cache hit rate.** Retracted as a signal. Three
  identically-configured lazy replicates in one round gave 21.3% / 31.1% /
  16.0%, with the eager control at 20.1% in the middle. The earlier
  "12.5% vs 20.7%" was inside that spread.

What the tail actually is: the KV pool saturating (usage 89-97%, 4 concurrent
40-100k-token requests, waiting queue ~0), which both arms hit, amplified
per-conversation because a slow turn arrives later and stays in the congested
window. Worst conversation at 90G, turn by turn: lazy pays t6+t7 (+5.7 s,
+9.6 s) and wins t8 (-5.9 s); eager pays t7+t8 (5.9 s + 6.4 s). Arrival drift
inside the conversation grows 7 s -> 13 s -> 24 s. Same window, different turn
index caught in it.

## 4. Parallel-slot harness

`$S/par/{env,up,down,arm,round}.sh`. One slot = one idle GPU + its own port
triple and log dir; slots map to GPUs 0/1/5/6, deliberately avoiding 2 (this
session's serial queue) and 3/4/7 (other people). `down.sh` is slot-scoped:
only this slot's recorded pid trees and only our own uid's compute apps on
this slot's GPU. Throughput went from ~3.3 arms/hour to ~16.

The protocol lesson is now structural, not advisory: **the eager control has
to be one of the slots in the same round.** Cross-round numbers are only good
for sanity checks.

## 5. Faster workload: what is and is not allowed

The 900 s runs are mostly idle -- effective concurrency 2.42 against a cap of
8 -- because the scenario replays recorded think times. Two attempts:

- `--benchmark-grace-period` without `--benchmark-duration` is rejected. Four
  arms died in 1 s and ~13 min of GPU time was wasted; the flag combination
  should have been smoke-tested first.
- The scenario locks the timing knobs outright:
  `--trace-idle-gap-cap-seconds` must be None ("preserves original per-trace
  request timing and forbids timeline compression") and
  `--system-idle-gap-cap-seconds` must be 10.0.

`--unsafe-override` turns the lock into a warning. Only the *global* idle cap
is lowered (10 s -> 1 s): within a trace every recorded gap still replays, so
burst shape and KV reuse are untouched, and only time when the whole replay
would sit idle is removed. `--request-count 256` pins every replicate to the
same request set instead of a wall-clock truncation.

Caveat recorded in the script: numbers from this config are a compressed
variant and are **not** scenario-compliant agentx-mvp results. They compare
arms against each other, nothing more.

`--replay-speedup` and `--max-idle-gap-cap-seconds` are baseten_trace-only and
do not apply to the weka dataset we use.

## 6. Store-release placement is now configurable (1c43ca02)

An environment variable does not work for scheduler-side switches: vLLM starts
the EngineCore process with a 9-variable environment, and the lazy-offload
manager lives inside it. Two arms labelled "no-prepend" therefore ran as plain
lazy; they were re-read as extra lazy replicates.

Replaced with a real config key, `lmcache.mp.lazy_offload_store_release`, an
enum rather than a boolean per the coding standards: `eviction_head` (default,
unchanged) or `lru_tail`. Unknown values raise. 3 new tests, 231 lazy tests
green, design doc updated.

Whether `lru_tail` helps is now an open question rather than a fix for a
known deficit, since section 2 says there is no deficit to fix.

## 7. Housekeeping

Nothing leaked from this session; every mp-server/vllm/EngineCore/aiperf maps
to a live arm. Removed 77 of my own orphaned `/dev/shm` segments
(`torch_<dead pid>_*`, `sem.loky-<dead pid>-*`, 5 MB) and three `/tmp` smoke-test
artifacts. The 5.5 GB in `/dev/shm` is 95% other users' (root 1484 files,
weishu 1447, kuntai 158); my share is 308 files / 770 MB nominal, all live.
The three bisect MP servers noted in record 1 (ports 29435/29988/27555) are
gone.

## Still open

- Slot-rotated round to settle section 2's confound.
- Calibrate the compressed workload: replicate wall time, and whether it
  preserves the saturation windows and the drop rate.
- `lru_tail` vs `eviction_head`, in-round paired, now that the key exists.
- Re-test `max_pending_ops` under the controlled design; its negative verdict
  came from single runs and is no more trustworthy than the readings above.
- The load ramp, still not started (was stopped on request until the deficit
  question was settled).

## Artifacts

- Serial arms: `$S/sweep/l1_{90,60,30}_{eager,lazy}`, `$S/smoke3/abt_{eager,lazy}`,
  `$S/fix/{cap32_l1_90,cap8_l1_90,rep_eager_r2,rep_lazy_r2,tail_lazy_noprepend}`.
- Parallel round 1: `$S/par/{p_eager_a,p_lazy_a,p_noprep_a,p_noprep_b}`.
- Rotated round: `$S/par/g_{eager_s0,lazy_s1,lazy_s2,eager_s3}`.
- Analysis: `$S/cmp2.py` (paired on the stable key), `$S/hits2.py`,
  `$S/pins.py`, `$S/preempt.py`, `$S/pre2.py`.
- `$S` = `/tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/84352f47-e330-4d19-88ee-0abf7e23352a/scratchpad`
