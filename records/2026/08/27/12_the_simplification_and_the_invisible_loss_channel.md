# 12 — i60H scored, the simplification, and the invisible loss channel

Session continuing record 11. The user set the frame mid-session: "我们得简化
已有的lazy的实现……我们现在就是先专注 少存，晚存，不漏存" — strip the
optimizations that answer loss by giving up deferral, keep the ones that are
deferral itself. Code state: four commits on `lazy-offload-dev` after
`be97bf0a` (see §3); rounds i60H and i60I scored below, i60J in flight at
write time.

## 1. Round i60H (hold fix) scored: two falsifications, one culprit

60 G, conc 32, seed 1234, two floor arms (s1/GPU1, s3/GPU6, FLOOR=4096,
code be97bf0a) + eager control.

| | floor_a | floor_b | eager | verdict |
|---|---|---|---|---|
| Q1 drops ≤ 91 | **211 ✗** | 84 ✓* | — | falsified (>150) |
| Q2 raises < 15 | 19 | 11 ✓ | — | not falsified; hold works (i60F: 58) |
| Q3 stored ≤ 18 M | 18.6 M | **20.8 M ✗** | 22.1 M | falsified (>20 M) |
| Q4 medD ≤ −3000 | **−4140 ✓** | −357 ✗ | — | half |
| Q5 dIsl ≤ 1% | 0.0 ✓ | 0.0 ✓ | — | passed |

Mechanism, read off the periodic ledgers:

- The floor's first raise jumps straight to the recent peak allocation
  (~2–3 k blocks); the second doubles into the cap (4096). Both arms hit
  cap within ~2 raises, the stand-down expired, and the always-live loss
  gate opened a trial on each arm at ~7 min.
- The trial verdicts were a coin flip: floor_a reverted (stayed deferred,
  kept the win); floor_b **committed to DEGRADED** — 989 ops emitted
  immediately (27% of emissions), stored volume pushed to 20.8 M, the
  latency win given back, recovered only by a later probe.
- floor_b's Q1 "pass" is an artifact: while DEGRADED nothing is pending,
  so nothing can drop (drop counter frozen at 35 for minutes).
- Residual drops at cap: `store_tokens max = 98 k` ≈ 6100 blocks, so a
  whole-context retrieve exceeds a 4096 window in one step.

Same code, same workload — one 45-second trial verdict erased the floor's
entire benefit on one arm. That, plus the user's simplification directive,
set §3's scope.

## 2. Burst tuition, and why the floor stores *less*

Two user challenges worth recording:

- "burst 不能预测的话不是每次都得交学费?" — per-regime, not per-burst:
  the first burst of a cadence pays (reactive raise), the hold keeps the
  window up across the cadence, the raise-to-peak covers same-magnitude
  successors. Q1's ≤ 91 budgeted that residual tuition. The predictive
  complement is announce (§6).
- "拉高存储不会导致存多了吗?" — the floor moves *when*, not *whether*;
  its alternative (degradation) is full eager. Empirically the floor arm
  stores 20% less than eager (§4, §5): staying deferred lets coverage/
  dedup/hit-feedback keep filtering. Q3/R3 pre-registered exactly this
  risk and it passed in i60I.

## 3. The simplification: four commits

Criterion (user's): keep 少存/晚存 and their correctness apparatus; delete
every path that answers loss by giving up deferral.

- **`55612abb` Remove the adaptive degradation controller.** Regime
  machine, volume ledger, `_loss_is_material` (and the floor's stand-down
  guard, no longer needed), `observe_l1_pressure` and the entire feed
  chain: pending-store facade, manager `wants/on_l1_pressure`, connector
  poll, scheduler adapter `poll_l1_pressure` + `L1PressureSample`. Knob
  `degrade_l1_residence_secs` and 8 counters gone. Server-side
  `GET_L1_PRESSURE` endpoint kept as a generic observability probe
  (l1_pressure_stats.md rewritten to say so). Losses beyond the floor are
  now accepted and logged, never answered with immediate emission.
- **`e851a2ae` Remove the backlog cap (`max_pending_ops`).** Age-ordered
  early emission; superseded by floor (reactive) + announce (predictive).
- **`2e262727` Remove the idle drain (`idle_drain_max_ops`).** Early
  emission for timing smoothness, directly against 少存/晚存. Also removed
  the now consumerless `requests_in_admission_order`.
- **`1b9ddc3f`** Opportunistic: `announced_bursts` was missing from
  `decisions()` (record 11 §8 item); added with a test.

All three deleted features defaulted off, so no measured round is
invalidated. Policy file ~2420 → ~1650 lines; `collect_due` has one
emission path (pressure) plus the D2H shaping caps. 290 lazy tests green,
ruff clean. Kept and why: rate model + rank due-ness (晚存 itself), pin
cascade / prefix closure / dedup (correctness + queue boundedness),
gate 3 + economy gate (少存的"根本不该存"), announce (the only predictive
channel), the drain budget caps (cheap D2H shaping).

Gate 3 honesty note: **it has never been validated end to end** — every
GPU round ran `min_prefix_tokens=0`, `held=0` throughout; only the
placement (admission-side vs emission-side) was ever measured, and its
break-even calibration predates the current models. The user kept it for a
future value sweep, explicitly deferred until the store line is done.

## 4. Round i60I (simplified code, cap sizing): R1 falsified, deletion validated

Arms: floor4k (s1), eager (s2), floor8k (s3); 8192 = the full 131072-token
context in blocks, i.e. the largest single-request allocation burst.

| | floor4k | floor8k | eager |
|---|---|---|---|
| dropped_evicted | 223 | 153 | — |
| tokens_stored | 18.1 M | 18.7 M | 22.7 M |
| medD vs eager | **−5428 ms** | −1835 ms | — |
| retrieves | 93 | 67 | 13 |
| raises | 11 | 12 | — |
| dIsl | 0.0% | 0.0% | — |
| preempt_events | 129 | 133 | 2 |

- **R2 confirmed** (floor4k 223 > 150): i60H's drops were not the trials'
  doing. The deletion is validated: floor4k reproduces i60H floor_a's
  good arm deterministically — no coin flip — with the best medD yet.
- **R1 falsified** (floor8k 153 > 150, needed ≤ 91): doubling the cap to
  cover the largest single-request burst cut drops only 31%.
- R3, R5 passed; R4 half (8k arm −1835; also fewer retrieves than the 4k
  arm — unexplained, alongside the standing preempt_events ~130 anomaly).

## 5. Forensics: the invisible loss channel

Sampled the floor8k drop events: mid-request suffix chains (prefixes
32 k–68 k), several ops of one request dropped in a single drain — and the
worst case (10 ops, ~2100 blocks) belongs to a request **the server never
received a single store for** (zero hits in server.log). So the chain was
not lost while blocked behind an in-flight batch; its blocks went from
in-use (no rank — not in the free queue at all) to freed to reallocated
without surviving one drain in between. **No window width can see this**:
the rank signal only exists while a block sits in the free queue, and
these blocks' queue residence was shorter than one drain interval. This is
the channel that survives at any floor cap, and it caps what the
floor-alone approach can deliver at roughly 150–220 drops on this
workload (4–6% of admissions) vs the pre-floor baseline's 91.

## 6. Round i60J (announce × floor): in flight

The remaining untested lever aimed at the measured dominant loss mode
(losses 6–7× enriched within 1.5 s of a retrieve, records 3/5): announce
injects the imminent hit-load's block count into the danger depth *before*
the burst step, no queue residence required. Arms: two × (ANNOUNCE=true,
FLOOR=8192) on s1/s3 + eager s2. Pre-registered (chain_i60J.sh): S1 drops
≤ 91 both (falsifier > 150); S2 `announced_bursts` > 0 both (wiring); S3
stored ≤ 19.5 M; S4 medD ≤ −1000 both (the 不回吐 line); S5 dIsl ≤ 1%.

Interim at ~13 min of load: announced=45/62 (S2 will pass — the wiring is
alive), drops already 110/120 — **on pace to falsify S1**. If it lands
there, the conclusion is that the residual loss is not retrieve-driven
either: it lives in the free→realloc blind spot (§5), which neither floor
nor announce can reach, and the next move is a mechanism change (e.g.
emitting a finished request's pending chain on release, before its blocks
enter the queue) or accepting ~5% loss as lazy's structural cost and
saying so in the standard. Score in the next record.

## 7. Corrections this session

- Record 11 §2 said transfer latency is irrelevant to loss because
  emission pins in the same drain. Still true for *emitted* ops, but §5
  shows the pin protocol's contrapositive: blocks of ops never emitted
  have no protection at all during free→realloc races, and that channel
  was invisible until the floor removed the burst-window losses on top of
  it.
- My own "保留 gate 3" recommendation initially credited it with measured
  effect; corrected in-session — it has unit coverage and a placement
  measurement only (§3).
- i60I's R1 prediction assumed bursts are single-request; falsified.

## 8. Open items

- Score i60J (S1–S5); write the verdict.
- If S1 falsifies: design the release-time emission (or an equivalent
  answer to §5's channel) as the next pre-registered change; if it holds,
  sweep L1 points before any default flip (record 11 §8).
- preempt_events ~130 on all deferred arms (vs 2–3 eager) still
  unexplained; floor8k's lower retrieves (67 vs 93) unexplained.
- Gate-3 value sweep: deferred by the user until the store line closes.
- Records 11's remaining open items (L1-point sweep, announce×floor at
  other L1 sizes) unchanged.
- Harness: `up.sh` KV_ARGS no longer passes the deleted knobs;
  chain_i60I.sh / chain_i60J.sh added beside chain_i60H.sh in the
  session-`3d34c28a` scratchpad.
