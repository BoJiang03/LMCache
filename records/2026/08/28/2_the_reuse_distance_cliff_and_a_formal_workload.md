# 2 — The reuse-distance cliff: 60 G was the worst possible operating point

Continues record 1. The user pushed back on the round-shortening proposal:
"你缩时间的标准是不是按照什么时候无命中来的？但是这个时间本来就和 l1
有关吧？我要的是一个尽可能正式的 workload，然后我去优化 lmcache 的性能。
真实同时足够快能跑完。" The objection was correct and it led to the
largest result of the whole line.

## 1. The circularity, and the L1-independent replacement

Record 1 section 9's proposal sized `DUR` by when retrieves stop. But the
time at which hits die is a function of L1, so calibrating the benchmark
to it welds the (arbitrary) 60 G choice into the benchmark. Withdrawn.

The correct criterion is **stationarity of the scored ratios**, and by that
test the configuration we had been using for weeks fails at every
duration: utilization drifts monotonically 65% → 40% → 28% → 22% → 18.6%
and never settles (measured on i60L pair_a by replaying the archived
ledger and server logs).

The L1-independent property that explains it is the **LRU reuse distance**:
for each consecutive turn pair of a conversation, the KV tokens written by
every request in between. Computed from i60L's eager arm
(`conversation_id`, `turn_index`, `request_start_ns`, ISL; 364 turn pairs,
23.84 M tokens written):

| cache | tokens | full-prefix hit rate |
|---|---|---|
| 15 GB | 164 k | 0.5% |
| 30 GB | 328 k | 1.4% |
| **60 GB** | 655 k | **4.7%** |
| 120 GB | 1311 k | 72.3% |
| 240 GB | 2621 k | 90.4% |
| 480 GB | 5243 k | 96.7% |

p50 reuse distance = 1038 k tokens = **95 GB**; p50 time gap = 89 s.

**60 G sits on the vertical part of the curve, just below the knee.** Its
steady-state hit rate is ~5%; the 89 retrieves we measured there are
almost entirely the cache-*filling transient*. Every headline number in
records 1–12 was measured at the worst point on the curve.

## 2. Round i60N (L1 = 30 / 120 / 240 G) — the cliff is real

Policy fixed at i60L's (FLOOR=8192, ANNOUNCE=false), code `0b75fa3b`.
60 G column from i60L; eager column from i60M.

| | eager | 30 G | 60 G | 120 G | 240 G |
|---|---|---|---|---|---|
| tokens_stored | 22.21 M | 17.75 M | 18.2 M | **5.42 M** | **3.67 M** |
| 少存 vs eager | — | −20% | −18% | **−75.6%** | **−83.5%** |
| tokens_retrieved | 0.81 M | 0.39 M | 3.21 M | 19.22 M | 23.25 M |
| retrieves | 27 | 14 | 89 | 421 | 498 |
| read/write ratio | 0.04 | 0.02 | 0.18 | **3.54** | **6.34** |
| TTFT p50 | 72,025 | 72,829 | ~67,800 | **40,079** | **33,227** |
| medD vs eager | — | +786 | ~−3,500 | **−29,407** | **−35,469** |
| l1_watermark_events | 207 | 317 | 163 | 21 | 4 |
| preempt_events | 4 | 145 | 122 | 25 | **14** |
| dropped_evicted | — | 92 | 204–254 | 442 | 287 |
| dIsl | — | 0.0 | −0.0 | −0.0 | −0.0 |

X1 confirmed (utilization monotone 2.2 → 17.6 → 354 → 634%). X2 confirmed
(240 G is 1.79× the 120 G value, under the 2× falsifier — returns are
diminishing). X3 confirmed far beyond its bar (3.67 M against a ≤17 M
prediction). X4 **falsified**: drops are non-monotone (92 → ~230 → 442 →
287). X5 not falsified once converted to wall clock — deferral is 16.5 /
18.2 / 15.0 s across the three L1 sizes (21% spread, under the 40% bar),
confirming record 1 section 3: the wait is set by GPU block residence, not
by anything cache-side. X6 confirmed: the 60→120 G jump (+336 points)
exceeds every other gap combined. X7 confirmed (section 3).

Three consequences worth stating plainly:

- **The 少存 ceiling is not −18%, it is −75% to −83%**, and it is unlocked
  by cache sizing, not by policy cleverness. Storing less and hitting more
  are the same event: a prefix already resident is a prefix you do not
  re-store, so `covered_prefix` and server-side dedup collapse the write
  volume as soon as the cache actually holds things.
- **The preemption anomaly is solved.** 145 → 122 → 25 → 14 as L1 grows.
  It was never about pinning (record 1 section 10 computed lazy at only
  1.8× eager in block-seconds); it was prefill pressure, which collapses
  when the cache hits.
- **The read/write ratio crosses 1.0 between 60 and 120 G.** Below the
  knee the system writes 5–50× more than it reads back and the cache is a
  cost; above it, each stored token is read back 3.5–6.3 times.

## 3. Stationarity: only above the knee

Retrieve events per 300 s bucket over the load:

```
 30 G  [  8,   2,   3,   1,  0,  0, 0, 0]   dies immediately
 60 G  [ 10,  37,  20,  11, 10,  1, 0, 0]   a transient
120 G  [ 82,  86,  66,  65, 60, 54, 8, 0]   sustained
240 G  [104, 105,  83,  76, 70, 52, 8, 0]   sustained
```

X7 confirmed. **Only at and above the knee is the workload stationary**,
i.e. only there does a measurement mean anything independent of how long
you ran it.

## 4. The formal workload

`agentx-std`, everything below fixed; only the thing under test varies.

- model `Qwen/Qwen3-Coder-30B-A3B-Instruct`, TP 1, block-size 16,
  `--num-gpu-blocks-override 16384` (24 GiB pool), gpu-mem-util 0.60
- aiperf scenario `inferencex-agentx-mvp`, dataset
  `semianalysis-cc-traces-weka-062126`, `--max-context-length 100000`,
  `ENTRIES=256`, `CONC=32`, `SEED=1234`
- **`L1_GB=120`** — just past the p50 reuse distance (95 GB). 72% of reuse
  events fit, the hit rate is stationary, and the cache is still under
  real pressure (89 of 120 GB used, 21 watermark events, 442 drops), so
  eviction and replacement behaviour is still exercised. 30/60 G measure a
  thrashing cache whose steady-state hit rate is ~0; 240 G is
  over-provisioned (4 watermark events) and stops testing eviction at all.
- **`DUR=900`, `GRACE=120`** — round wall clock ≈ 20 min against today's
  41 (bringup is only 2 min; aiperf is 85% of the round). The
  justification is not "hits stop" but effect size: at 120 G the 900 s
  window already gives medD −23,004 ms (78% of the full-run −29,407),
  against −3,500 ms for a full 1800 s run at 60 G. **An 8× larger effect
  is what buys the shorter run**, non-circularly.
- compared arms must sit on the same NUMA node (record 1 section 1)
- every round reports medD at the 600 s and 900 s windows plus the full
  number, and the retrieve-per-300 s vector as the stationarity check
  (`par/early.py`, `par/converge.py`)

Known residual: at 120 G the cache is still filling at the end of the run
(89 of 120 GB), so medD grows until ~1500 s. A fixed `DUR` keeps runs
comparable, but a warm-cache preload phase would make the benchmark
genuinely steady-state and is the obvious next harness improvement.

## 5. What this does to the three goals

Record 1's ceiling table was computed entirely below the knee and its
少存 row is now superseded.

| goal | at 60 G (records 1–12) | at 120 G | status |
|---|---|---|---|
| 少存 | −18.5% | **−75.6%** | ceiling was never a policy property; it is a cache-sizing property |
| 晚存 | δ = 17.2 s, write path worth 0 | δ = 18.2 s | unchanged and still capped — δ does not depend on L1 |
| 不漏存 | 214 ops / 4.5% of volume | 442 ops / 22% of admitted volume | rate rises with hits, as the r=0.70 coupling predicted; net value still overwhelmingly positive (19.22 M retrieved against 1.52 M dropped) |

## 6. Open items

- Re-run the policy comparisons that matter (eager vs lazy vs off) at
  120 G on `agentx-std`. Everything scored so far was measured at 60 G,
  where the cache barely functions; the *relative* verdicts may not
  survive the move above the knee. This is now the top priority.
- Warm-cache preload phase in the harness, so the benchmark is
  steady-state rather than still-filling.
- X4's falsification: drops peak at 120 G (442) and fall at 240 G (287).
  Unexplained; likely the 240 G arm admits far less (1240 vs 1723) because
  more prefixes are already covered.
- `max_drain_blocks_per_step` still never tried (record 1 section 10),
  though its motivation weakens now that preemption is explained.
- Gate-3 `min_prefix_tokens` sweep, still deferred by the user.
