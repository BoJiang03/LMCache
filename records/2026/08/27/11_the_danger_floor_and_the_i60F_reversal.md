# 11 — The danger floor: two commits, one reversal, one rollback

Session continuing record 10's close. The user restated the goal verbatim and
asked for it to be built: "我要的是一个自适应策略，能在每个step决定最少存多少
blocks。这样就能减少存储端对l1的压力，以及减少l0和l1数据的重复。" Standard set
later in the session: 少存保持、不丢 block、时延优势不回吐 ("我不管你怎么改，
反正我标准在那").

Code state: `9f95dd29` (loss-adaptive danger floor) and `be97bf0a` (hold fix)
on `lazy-offload-dev`, both after this session's rounds i60F (scored below)
and i60H (in flight at write time). 357 lazy tests green, ruff/isort clean.

## 1. Record 10's blocker resolved: `store_secs` is submit time

Record 10 §10 flagged `12.6 GB/s` (from `store_secs`) vs `46 s` L1 residence
(from watermark events) as mutually exclusive. Both chains re-derived from
g4F_lazy60_s0 raw logs:

- `store_secs` comes from `lmcache_driven_transfer.py:1216` ("Stored %d
  tokens in %.3f seconds"). The D2H copy is *submitted* to a CUDA stream
  (`transfer_kv_per_object_group`), the completion event is recorded on the
  stream, MP_STORE_END is published **on the stream** — and the timer stops
  without any synchronize. Sample: 74,752 tokens in 0.033 s = 217 GB/s.
  PCIe cannot do that. **`store_secs` measures submission latency, not the
  copy.** Every bandwidth inference built on it is dead, including record
  10's "only ~12% of nominal crosses the bus" and "L1 residence 383 s".
- The watermark chain is right and now closes: the L1 eviction controller
  (`eviction_controller.py:160`) wakes each second and, above watermark
  0.80, evicts `eviction_ratio = 0.2` of tracked keys. At the snapshot's
  1,888 objects × ~24 MiB that is ~9 GiB per event; 202 events over 1800 s
  ≈ 1.0 GiB/s evicted, balancing the nominal insert rate 21.5 M tokens ×
  96 KiB / 1800 s = 1.15 GB/s. **The full nominal volume lands in L1.**
  L1@60G residence ≈ 42–46 s. (KV = 96 KiB/token confirmed from the model
  config: Qwen3-Coder-30B, 2 × 48 layers × 4 KV heads × 128 dim × bf16.)

Consequence for the controller: `N_xfer = 15 steps` (built on the dead
chain) was discarded. It also turned out not to be needed — see §2.

## 2. Why the controller reduces to a floor under the danger depth

Facts assembled from the code before designing:

- Emission pins blocks out of the free queue **in the same drain** (the pin
  cascade), and blocks under transfer are ref-held by the manager until the
  receipt frees them (prepended, already stored). So an op the drain *sees*
  as due cannot be lost. Transfer latency is irrelevant to loss.
- Therefore loss ⇔ a single step's allocation crossed the whole un-walked
  window: exactly the burst mode records 3/5 measured (2,774-block bursts
  vs a 49.6-block window; 6–7× loss enrichment near retrieves).
- Stored blocks return to the queue *head* (`free_blocks(prepend=True)`),
  forming a shield that consumption eats first. Rank-based due-ness against
  `danger_depth` is therefore already the deficit rule "emit only what the
  forecast needs beyond the stored shield" — the per-step minimum the user
  asked for. What was missing is not the quota; it is the forecast's
  blindness to bursts, and the policy's only standing answer to that was
  the degradation controller flipping the run to eager (g4F/i60F lazy60:
  degraded_emitted = 79–82% of emitted). The user's target — 少存/晚存 —
  is destroyed by exactly that response.

So the adaptive piece is a **loss-adaptive floor under the danger depth**
(`9f95dd29`):

- A drain interval that lost ops to eviction raises the floor to the peak
  gross allocation of the last 8 steps, at least doubling on consecutive
  losses, capped at `lazy_offload_danger_floor_max_blocks` (0 = off,
  default). Loss-free drains decay it (0.999/drain). Depth =
  `max(rate model, floor, announced)`.
- While the floor is enabled and below its cap, the always-live loss gate
  of the degradation controller **stands down** — the floor is the
  graduated response to the same loss; the trial is the last resort once
  the floor is at cap and losses continue. Floor off ⇒ gate unconditional,
  bit-identical prior behavior.
- Sensor: `danger_floor_raises` (an event counter, outside the ledger
  equation, like `throttled_drains`).

## 3. Round i60F (floor, pure decay): the reversal

60 G, conc 32, 256 entries, 1800 s, seed 1234 (g4F parity). GPU 0 was
occupied by another user, so slots 1/2/3 (GPUs 1/5/6). Harness copied to
this session's scratchpad (`.../3d34c28a-*/scratchpad/par/`) with a new
`FLOOR` env → `lazy_offload_danger_floor_max_blocks`; FLOOR=4096.

| | eager_s1 | lazy_s2 (baseline) | floor_s3 (FLOOR=4096) |
|---|---|---|---|
| TTFT p50 | 72,005 ms | 72,467 ms | **65,876 ms** |
| medD vs eager | — | +224 ms | **−5,749 ms** (sumD −2,354 s / 444 reqs) |
| tokens_stored | 22.63 M | 20.56 M | **16.21 M** |
| stores (batches) | 3,840 | 1,631 | **421** (mean 38.5 K tok) |
| tokens_retrieved | 0.80 M | 1.68 M | **4.17 M** |
| retrieves | 25 | 43 | **114** |
| l1_watermark_events | 211 | 193 | **147** |
| dropped_evicted | — | 91 | **272** |
| degraded_emitted | — | 2,867 (82%) | **0** (0 trials) |
| danger_floor_raises | — | 0 | 58 |
| preempt_events | 3 | 32 | **135** |
| free-queue read/drain | — | 182 | 912 |

Scoring the four pre-registered predictions (chain_i60F.sh):

- **P2 (degrade suppression, <30% degraded): PASSED at 0%.** The loss gate
  stood down for the whole run; the arm stayed deferred end to end.
- **P4 (latency, within ±1000 ms of lazy baseline): PASSED beyond the
  scale asked** — −5.7 s median against both eager and lazy baseline.
- **P1 (drops ≤ half of baseline): FALSIFIED** — 272 vs 91. Mechanism read
  directly off the counters: decay half-life (~20 s) sits inside the burst
  inter-arrival, so the floor decays between bursts and every cycle's
  first burst takes the leading edge; 58 raises = 58 tuition payments.
- **P3 (volume parity ≥ 90% of eager): FALSIFIED in the direction the
  goal wanted** — 71.6% of eager, 79% of lazy baseline. The prediction's
  premise ("the floor re-times, never filters volume") was wrong: staying
  deferred lets coverage/coalescing/hit-feedback shrink what is admitted
  and stored. This is b32's mechanism (less volume → longer L1 residence →
  higher hit rate) reproduced at full scale, closing the loop the user
  named: 少存 → L1 少翻腾 (watermarks 193→147) → retrieves 2.5× → fewer
  prefills → less store pressure → TTFT p50 −9%.

Open anomalies from the round: preempt_events 135 (vs 32 baseline) —
unexplained, worth a forensic pass; per-drain free-queue read grew ~5×
(912 vs 182 blocks/drain) with no visible ITL cost (dIsl ≈ 0.0%).

## 4. The hold fix (`be97bf0a`)

P1's falsification has a clean mechanical fix: a raised floor now **holds
flat for two smoothed measured loss intervals** (EMA of drains between
losses; floored at 2,048 drains while only one loss has been seen) before
the decay may start, and *every* loss restarts the hold. A standing burst
cadence keeps the floor up; a workload that genuinely quiets waits two of
its own intervals and then decays exactly as before. Constants stay
properties of the measurement (the cadence is measured, not configured).

## 5. Round i60H (hold): in flight at write time

Same workload, floor+hold on **two** slots (s1/GPU1, s3/GPU6 — rotating
off i60F's s3-only placement) + eager control (s2/GPU5). Pre-registered
(chain_i60H.sh): Q1 drops ≤ 91 both arms (falsifier > 150); Q2 raises < 15
(falsifier ≥ 40); Q3 tokens_stored ≤ 18 M (falsifier > 20 M); Q4 medD ≤
−3000 ms both (falsifier > 0); Q5 dIsl within 1%. Score in the next
record.

## 6. The wrong-branch push, and the rollback

The user reported a push to the wrong branch. Verified before acting:
`fork/lazy-offload-policy` (the PR branch) pointed at `d2ae93a9` — a
records commit from the dev line — pushed 2026-08-27 12:23 per the
remote-tracking reflog; the pre-mistake tip was `d65765db` (2026-08-20).
Root cause found in branch config: `lazy-offload-dev` had
`fork/lazy-offload-policy` as its upstream, so a bare `git push` lands dev
commits on the PR branch.

Actions: force-pushed `d65765db` back to `fork/lazy-offload-policy`
(restoring the exact pre-mistake remote state; verified by `ls-remote`),
and `--unset-upstream` on `lazy-offload-dev` so a bare push now errors
instead of repeating this. Note: local `lazy-offload-policy` (a0c3d862)
has diverged from d65765db (4 commits each side of a dev merge); pushing
it was *not* the rollback and was not done — updating the PR branch to
a0c3d862 is a separate, deliberate act for the user to take when ready.

## 7. Corrections this session

- "12% of nominal crosses PCIe / L1 residence 383 s" — retracted (§1).
- "N_xfer = 15 steps" — retracted; transfer latency does not bound loss at
  all (§2), the whole time-budget frame was unnecessary.
- P3's premise "storing later cannot store less" — falsified by i60F; the
  volume channel and the timing channel are coupled through the hit-rate
  feedback loop.
- P1's implicit premise "a reactive floor with fixed decay can hold a
  cadence" — falsified; fixed by measuring the cadence (§4).

## 8. Open items

- Score i60H (Q1–Q5). If Q1 holds, the 标准 (少存 + 不丢 + 赢面) is met at
  60 G; then sweep the other L1 points (30 G choke, 90–100 G band edge,
  180 G quiet) before any default flip.
- preempt_events 135 on the floor arm: find the mechanism (likely the
  retrieve traffic tripling GPU allocation pressure; could also be the
  held pins). `records`-worthy on its own if it survives i60H.
- `announced_bursts` is missing from `LazyOffloadCounters.decisions()`
  though its docstring claims everything but the five cost sensors is
  included — pre-existing discrepancy noticed while adding
  `danger_floor_raises`; fix opportunistically.
- The floor and `announce_hits` have not been measured together
  (ANNOUNCE=false in all arms so far, matching the shipped default).
- Harness for i-rounds lives in this session's scratchpad
  (`/tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/3d34c28a-*/
  scratchpad/par/`), with cmp2.py path fixed and the FLOOR env added.
