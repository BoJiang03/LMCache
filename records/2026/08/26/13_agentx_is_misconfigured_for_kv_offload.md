# agentx: how we run it, how it should be run, and whether that would show lazy

Written 2026-08-26 after the a91 collapse. Companion to `12_*.md` (which
established that the agentx numbers were noise) and `10_*.md` (verdict log).
`12` said the measurement failed. This one says *why the workload itself
cannot resolve the question as configured*, and what a correct configuration
would be.

## 1. How agentx is currently invoked

Harness: `par/round.sh` -> `par/arm.sh` (slot -> GPU 0/1/5/6) -> `par/up.sh`.

Client (`par/arm.sh`):

```
aiperf profile --model agentx --endpoint-type chat --streaming
  --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct
  --scenario inferencex-agentx-mvp
  --public-dataset semianalysis-cc-traces-weka-062126
  --max-context-length 100000 --num-dataset-entries 64 --concurrency 8
  --benchmark-duration 900 --benchmark-grace-period 120 --random-seed 1234
```

Server (`par/up.sh` / `par/env.sh`):

- Qwen3-Coder-30B-A3B-Instruct, TP=1, `--max-model-len 131072`, block 16
- `--gpu-memory-utilization 0.60`, `--num-gpu-blocks-override 16384`
  = 262144 tokens = 24 GiB of GPU KV pool
- LMCache MP server, `--l1-size-gb 60` or `90`, LRU
- eager arm: plain `LMCacheMPConnector`; lazy arm: same plus
  `lazy_offload=true, policy=EVICTION_AWARE, horizon=2.5, degrade_secs=<N>`

Metric: `cmp2.py`, paired TTFT delta keyed on `(conversation_id, turn_index)`,
reported as `sumD` over the ~262-273 requests that pair.

The invocation is scenario-valid. The lock auto-fills what we omitted, and
logs it:

```
Scenario 'inferencex-agentx-mvp': setting timing_mode=agentic_replay
  injecting extra_inputs.ignore_eos=true (was absent)
  auto-set --cache-bust=first_turn_prefix (was at default none)
  auto-set --trajectory-start-min-ratio=0.0 / max-ratio=1.0
  auto-set --system-idle-gap-cap-seconds=10.0
```

No violations, no `--unsafe-override`, `submission_valid` intact. So the
problem is not that we are running agentx wrong against its own spec.

## 2. Why this configuration cannot resolve lazy vs eager

From `a91_eager_s1` (aiperf aggregate + vLLM stats + MP server snapshot):

| what | value | reading |
|---|---|---|
| `effective_concurrency` | 2.2 (asked 8) | lanes idle in recorded think time |
| `effective_prefill_concurrency` | avg 0.2, **p50 0.0** | GPU is not prefilling most of the time |
| vLLM `Waiting` queue | mean 0.02, max 1 | never backs up; no queueing anywhere |
| vLLM GPU KV usage | p50 49.6%, mean 43.2% | pool half empty |
| `tokens_in_flight` | avg 134K, p90 232K | vs 262K pool: **the working set fits on GPU** |
| `theoretical_prefix_cache_hit` | 93.5% | the dataset offers plenty of reuse |
| LMCache retrieves | 35 of 262 requests | ...but only 13% of it is asked of LMCache |
| tokens retrieved / total ISL | 1.24M / 14.4M = 8.6% | LMCache serves under a tenth of input |
| server store time, whole run | **7.9 s of 900 s** (lazy 5.7 s) | under 1%, and asynchronous |
| L1 occupancy at 90G | 70.5 GiB, 6 watermark events | essentially no eviction pressure |

The GPU prefix cache absorbs the reuse the dataset offers, so LMCache is a
bystander. Everything lazy offload can change -- when the store is emitted,
whether a covered store is skipped at all -- lives inside that <1% slice, and
that slice is not on the critical path because the GPU has prefill capacity
idle more than half the time.

Against that, a91 measured an **eager-vs-eager pair differing by 34.3 s** of
summed TTFT. Request-latency p99 is 65.6 s and TTFT max is 22 s, so a handful
of stalls is worth more than the entire mechanism under test. The median
paired difference across every arm in the campaign is 1-6 ms on a 553 ms
median TTFT.

This is structural, not statistical. More replicates would tighten the error
bar around an effect that this configuration holds near zero by construction.

## 3. What the correct configuration is

The scenario locks the timing shape hard: `forbid_ignore_trace_delays`,
`forbid_inter_turn_delay_cap`, `forbid_trace_idle_gap_cap`. Think time may
not be compressed. It also refuses `--request-rate` / `--fixed-schedule`
(`_CONFLICTING_PHASE_TYPES`). **Concurrency is the one load knob the lock
leaves open** -- under `AGENTIC_REPLAY` it is the number of trajectory lanes
held open, each recycling into a fresh root when its tree drains.

Levers, in order of effect:

1. **`--concurrency 8 -> 32`, `--num-dataset-entries 64 -> 256`.**
   (`--allow-dataset-wrap` defaults False, so entries must exceed
   concurrency.) 32 lanes x ~55K avg context is ~1.76M tokens of live reuse
   working set against a 262K-token GPU pool: ~15% can live on GPU, the rest
   must come from L1 or be recomputed. That is the regime the feature exists
   for. Sizing check: current `input_token_throughput` is 16.0K tok/s and
   `active_prefill_throughput` is 78K tok/s, so 4x load lands near prefill
   saturation without deep queueing -- concurrency 64 would over-saturate.
2. **`--benchmark-duration 900 -> 1800.`** 900 is the scenario *minimum*;
   `default_benchmark_duration_seconds` is 1800. Combined with (1) this moves
   the paired sample from ~270 requests to ~2000.
3. **L1 sized for pressure.** 90G holding 70G with 6 watermark events gives
   the eviction-aware policy nothing to decide. 60G under a 4x working set,
   or a lower point, puts admission economy under test.
4. **GPU KV pool as a second, cheaper axis.** Lowering
   `--num-gpu-blocks-override` (24 GiB -> 6-8 GiB) emulates the same pressure
   at concurrency 8. It is the honest stand-in for "many more concurrent
   users than 2", not a rigged knob -- but (1) is the faithful version and
   should be preferred when the GPU-hours are affordable.

Leave alone: `--max-context-length 100000` is already correctly matched to
`--max-model-len 131072`; admitting the 256k traces would need a larger
`max-model-len`, which costs GPU memory and shrinks the KV pool anyway.

Measurement changes that must accompany it (from `12_*.md`):

- primary metric is a trimmed mean or median of paired TTFT plus a stall
  count, never `sumD`
- rotate the config-to-slot assignment every round
- always carry a same-config control arm
- >= 3 replicates before any verdict

## 4. Would the correct configuration show lazy beating eager?

It would show *whether* it does. That is the honest claim. Two separable
mechanisms, and the current config triggers neither:

- **Interference avoidance** -- the deferred store stays off the critical
  path. Requires a prefill-saturated GPU. Today `effective_prefill_concurrency`
  p50 is 0, so there is nothing to interfere with; lever (1) is what turns
  this on.
- **Admission economy** -- covered-prefix skipping (4.39M tokens skipped in
  `a91_on_s0`) and not storing what is about to be evicted. Requires L1 under
  pressure; lever (3).

Note that the second mechanism is exactly what the hot/cold harness was built
to isolate, and lazy wins there: cold TTFT 432-486 ms vs eager 811-813 ms
across three seeds, distribution-wide rather than tail-driven. So "the most
correct agentx configuration" is largely agentx retuned into the regime
hot/cold already occupies -- which is a reason to expect a positive result,
and also a reason to admit that agentx would then be confirming hot/cold
rather than adding an independent axis.

Prior expectation, stated before any such round is run: lazy >= eager on
median TTFT at concurrency 32, with the margin coming from L1 hits that eager
also gets (so a modest win), plus a tail improvement from fewer, larger store
ops (a91 lazy: 304 stores at mean 5456 tokens; eager: 975 stores at mean
1755). A *negative* result is the genuinely informative outcome: it would mean
deferral loses turns to eviction under real agentic inter-turn gaps, which no
harness we have run so far could have detected.

Not started. Cost is roughly 40 min per 4-arm round at 1800 s, >= 3 rounds.

---

## b32: the first properly-loaded agentx round (predictions pre-stated)

Launched 2026-08-26 ~22:40. Config below; predictions written before any arm
reported.

Client (changed from every prior round): `--concurrency 32`,
`--num-dataset-entries 256`, `--benchmark-duration 1800`,
`--benchmark-grace-period 180`. Everything else identical, scenario lock
intact.

Server: unchanged except `L1_GB=60` on all four arms. GPU KV pool stays at
16384 blocks / 24 GiB deliberately -- the load, not a shrunken pool, is what
creates the pressure.

Arms, config-to-slot interleaved so neither config owns the early or the late
launch position:

| slot | gpu | tag | config |
|---|---|---|---|
| 0 | 0 | `b32_eager_s0` | eager, L1=60 |
| 1 | 1 | `b32_lazy_s1` | lazy, L1=60, DEGRADE_SECS=0 |
| 2 | 5 | `b32_lazy_s2` | lazy, L1=60, DEGRADE_SECS=0 |
| 3 | 6 | `b32_eager_s3` | eager, L1=60 |

The degradation knob is OFF on both lazy arms. Round 1's question is the base
feature -- deferral vs eager store -- because there is no point testing the
controller in a regime where nothing was measurable. Knob on-vs-off is round 2,
and only if round 1 shows signal above its own control delta.

Preflight (22:36-22:37): composed the dataset at the new settings and confirmed
`built 32 trajectories from 42 traces`, no wrap needed, mmap cache now warm for
all four arms. Note `--num-dataset-entries 256` turned out to be a no-op:
the pool is 153 unique conversations / 42 eligible root traces either way,
already saturated at 64. It is kept only so the request cannot bind. **42
traces is the hard ceiling on concurrency without `--allow-dataset-wrap`, so
concurrency 64 is not reachable in this scenario as configured.**

Predictions:

1. **The load lands.** `effective_concurrency` >= 8 (was 2.2) and
   `effective_prefill_concurrency` p50 >= 1 (was 0). If this misses, the round
   failed at the same place a91 did and nothing downstream is interpretable.
2. **LMCache stops being a bystander.** Retrieves per arm >= 150 (was 35-39);
   tokens_retrieved / total ISL >= 25% (was 8.6%); L1 watermark events well
   above 6.
3. **The noise floor is reported on a robust metric, and shrinks.** The
   eager-eager control delta (`s0` vs `s3`) stays under 30 ms on median paired
   TTFT and under 50 ms on the 5%-trimmed mean. No prediction is made about
   `sumD`; it is not the primary metric any more.
4. **Direction: lazy <= eager on median paired TTFT, by 0-15%.** Falsifier: if
   lazy is *worse* than eager by more than the eager-eager control delta on the
   trimmed mean, deferral is genuinely losing turns to eviction under real
   agentic inter-turn gaps, and the feature's premise fails at this load. That
   outcome is more informative than the win.
5. **Store shape repeats a91.** Lazy issues roughly a third of eager's store
   ops at roughly triple the mean payload.

---

## b32 verdict: the first agentx result that separates by config, not by slot

Round finished 23:19:30. All four arms clean.

### The load landed

| | a91 (conc 8) | b32 (conc 32) |
|---|---|---|
| `effective_concurrency` | 2.2 | 17.5-17.8 |
| `effective_prefill_concurrency` | avg 0.2, p50 0 | avg 14.4-14.7, **p50 15-16** |
| vLLM waiting queue | mean 0.02, max 1 | mean ~12 |
| L1 watermark events | 6 | 152-210 |
| L1 occupancy (of 60G) | n/a (70 of 90G) | 39-45 GiB, actively evicting |
| `dropped_evicted` | 145 / 75 | 340 / 297 |
| server store time | 7.9 s / 900 s | eager 45-47 s, lazy 16-17 s / 1800 s |

### Results (baseline `b32_eager_s0`, n=442 paired, baseline TTFT p50 73521 ms)

| arm | median dTTFT | trim5 mean | trim10 mean | mean TTFT | request latency | tok/s |
|---|---|---|---|---|---|---|
| `b32_eager_s3` (control) | **+57** | +607 | +575 | 66711 | 86179 | 11020 |
| `b32_lazy_s1` | **-3146** (-4.3%) | -4406 | -4505 | 62570 | 80940 | 11612 |
| `b32_lazy_s2` | **-2495** (-3.4%) | -3177 | -3285 | 63732 | 82355 | 11427 |
| `b32_eager_s0` (baseline) | -- | -- | -- | 66227 | 85464 | 11109 |

The thing a91 destroyed is finally satisfied: **the effect exceeds the
same-config control by 44-55x on the median and 5-7x on the trimmed mean.**
The two eager arms sit within 484 ms of each other on mean TTFT, the two lazy
arms within 1162 ms, and the gap between the groups is ~3300 ms. Requests
completed: eager 411/408, lazy 427/421. Groups separate; slots do not.

Consistent across three independent framings: median paired TTFT -3.4/-4.3%,
mean request latency -4.4/-5.3%, total token throughput +3.7/+4.5%.

### Mechanism

| | eager s0 / s3 | lazy s1 / s2 |
|---|---|---|
| store ops | 3838 / 3761 | 423 / 432 |
| mean store payload (tokens) | 5872 / 5888 | 39822 / 39759 |
| tokens stored | 22.5M / 22.1M | 16.8M / 17.2M |
| tokens retrieved | 0.63M / 0.72M | **3.23M / 2.62M** |
| retrieved / total ISL | 2.9% / 3.3% | **14.2% / 11.7%** |
| server store time | 46.6 s / 45.0 s | 15.8 s / 17.0 s |

Lazy issues **one ninth** the store ops at ~6.8x the payload, stores 25%
fewer tokens (covered-prefix skipping), spends a third of the store time --
and retrieves **4-5x more tokens back out of L1**. Coalesced, prefix-filtered
stores make L1 content that is actually worth hitting; eager's 8192-token
dribble does not.

### The cost nobody had seen: preemption

Unique preempted request ids, from `scheduler_output.preempted_req_ids` read
in `build_connector_meta` (which runs every step in **both** modes, so this is
a physical count and not a lazy-only reporting artifact):

| eager s0 | eager s3 | lazy s1 | lazy s2 |
|---|---|---|---|
| 1 | 3 | **159** | **147** |

Deferral holds GPU KV blocks that cannot be freed until the tail is emitted,
which raises GPU KV pressure until vLLM preempts. ~50x more preemption,
cleanly separated by config, and it never showed up before because no prior
round made GPU KV the binding constraint. Lazy wins anyway here -- the L1-hit
gain outweighs it -- but this is a real cost that could flip the sign at a
smaller GPU pool or higher pressure, and it should be measured, not assumed
benign.

### Honest caveat on the regime

TTFT p50 is 73.5 s. Concurrency 32 overshot: this is deep overload, not
saturation. In an overload regime TTFT is mostly queueing, so the ~4% is a
throughput difference expressed through the queue. The defensible claim is
**"lazy delivers ~4% more throughput under KV-bound overload, visible as
~4-5% lower TTFT and latency"** -- not "lazy cuts TTFT by 4%" at a normal
operating point. A mid-load point (concurrency 16) is needed to separate the
two, and this needs replication before it is a verdict rather than a result.

### Prediction scorecard: 2 hit, 3 missed

1. **Load lands** (eff conc >= 8, prefill conc p50 >= 1) -- **HIT**, decisively.
2. **LMCache stops being a bystander** (retrieves >= 150, retrieved/ISL >= 25%)
   -- **MISS**. Actual retrieves 77-94, retrieved/ISL 11.7-14.2%. The absolute
   level stayed lower than predicted; what I failed to anticipate is that the
   *ratio between configs* would be the signal (lazy 4-5x eager), not the
   absolute level.
3. **Control delta < 30 ms median, < 50 ms trimmed mean** -- **MISS**. Actual
   +57 ms median (2x my bound) and +607 ms trimmed mean (12x my bound). I badly
   underestimated the noise floor at high load. The conclusion survives only
   because the effect is 5-55x the floor, not because the floor was where I
   said it would be.
4. **lazy <= eager by 0-15% on median paired TTFT** -- **HIT**. -4.3% / -3.4%,
   control +0.08%. The falsifier (lazy worse than eager by more than the
   control delta) did not fire.
5. **Store shape: ~1/3 ops at ~3x payload** -- **MISS**, understated by a
   factor of three in both directions: 1/9 the ops at 6.8x the payload.

Three of five missed, all of them calibration rather than direction. The one
that decided the question, #4, hit, and it hit with its falsifier pre-stated.

### Next

- replicate b32 at least twice more with rotated slot assignment
- a mid-load point (concurrency 16) to separate saturation from overload
- only then knob on-vs-off, which now has a plausible mechanism to test:
  degradation should reduce the preemption count by emitting sooner

---

## c32L: L1 sweep at fixed load (predictions pre-stated)

Launched 2026-08-27 06:25, after b32 showed the 4% gain was capped by a
starved cache rather than by the feature.

The diagnosis b32 forced: one average context is 55087 tok x 96 KB =
**5.04 GiB**, so 32 lanes is a **161 GiB** live working set against GPU 24 +
L1 60 = 84 GiB. GPU prefix hit rate collapsed to ~0% at this concurrency
(42-64% at concurrency 8), and LMCache recovered 14.2% (lazy) / 2.9% (eager)
of a 92.5%-reusable workload. Server-side retrieve measures ~2.0M tok/s
(~180 GB/s) against ~78K tok/s of prefill -- a hit is ~25x cheaper than the
recompute it replaces -- so roughly 17M of 21.8M input tokens per arm were
recomputed at 25x cost for want of cache.

Round: concurrency 32, 1800 s, unchanged otherwise; L1 swept 30 vs 180 GB.
180 GiB is chosen to just exceed the 161 GiB working set; 30 GB doubles the
pressure b32 ran at.

| slot | gpu | tag | config |
|---|---|---|---|
| 0 | 0 | `c32_eager_l30_s0` | eager, L1=30 |
| 1 | 1 | `c32_lazy_l180_s1` | lazy, L1=180 |
| 2 | 5 | `c32_eager_l180_s2` | eager, L1=180 |
| 3 | 6 | `c32_lazy_l30_s3` | lazy, L1=30 |

No same-config control this round; b32's measured floor (median +57 ms,
5%-trimmed mean +607 ms) is the yardstick. Launch order deliberately flips
between the two L1 points -- eager launches first at 30 G, lazy first at
180 G -- so a launch-order artifact would show up as an inconsistency between
the two contrasts rather than as a uniform bias. Host memory: 420 GB pinned
for this round on top of another session's 260 GB MM server, against 1180 GB
available.

Predictions:

1. **At 180 G the cache stops thrashing.** tokens_retrieved / total ISL rises
   above 50% on *both* arms (from 14.2% / 2.9%); `dropped_evicted` and L1
   watermark events fall sharply from b32's 340/297 and 152-210.
2. **Absolute performance jumps at 180 G**: total token throughput more than
   30% above b32's ~11.1K tok/s, TTFT p50 well under 73.5 s.
3. **The lazy-over-eager gap shrinks at 180 G and grows at 30 G.** At 30 G the
   median dTTFT (lazy - eager) is more negative than b32's -2495/-3146 ms; at
   180 G it is less negative, plausibly inside the +/-607 ms trimmed-mean
   floor. **Falsifier for the whole "lazy's edge is eviction economy" reading:
   if the gap instead *grows* at 180 G, the edge is the store path, not
   admission economy, and the a91-vs-b32 retrieval-ratio trend (1.14x at 90 G
   idle, 4.9x at 60 G loaded) was a coincidence.**
4. **Preemption stays lazy-heavy at both L1 points.** It is a GPU-side
   consequence of holding un-emitted blocks and should not care about L1 size.
   If lazy's preemption count falls at 180 G, the mechanism attributed to it
   in the b32 verdict is wrong.
5. **Retrieve ratio lazy/eager** above 4x at 30 G, below 2x at 180 G.

---

## c32L verdict: L1 sizing dominates, and lazy loses where it counts

Round finished 07:06:19. All four arms clean.

### L1 capacity is the lever. It is not close.

Both arms eager, concurrency 32, everything else identical:

| | eager @ L1=30 | eager @ L1=180 |
|---|---|---|
| requests completed | 411 | **626** (+52%) |
| TTFT mean | 70730 ms | **33449 ms** (-53%) |
| TTFT p50 (paired) | 71501 ms | **31163 ms** (-56%) |
| request latency | 85614 ms | 44070 ms (-49%) |
| total token throughput | 11121 | **17235** (+55%) |
| retrieve ops | **3** | 505 |
| tokens retrieved | 0.03M | 25.88M |
| retrieved / total ISL | 0.1% | **76.7%** |
| tokens stored | 23.5M | 5.4M |
| L1 occupancy | 23.7 GiB | 124.4 GiB |
| watermark events | 442 | 13 |

At 30 GB the cache is so oversubscribed it does not function: 23.5M tokens
written, **30K read back**, three retrieve operations in a half-hour run. At
180 GB the same code serves 76.7% of input from L1 and stores a *quarter* as
much, because content survives long enough to be reused instead of
re-written. Median paired dTTFT between the two eager arms is **-39629 ms**.

This is the answer to "the gain is far below what I expected". The 4% b32
found was measured inside a cache that barely worked. Sizing L1 to the
working set is worth an order of magnitude more than anything the lazy/eager
choice does.

### lazy vs eager is small and its sign flips with L1

| L1 | median dTTFT (lazy - eager) | relative | reading |
|---|---|---|---|
| 30 G | -36 ms | -0.05% | wash, inside b32's +/-57 ms floor |
| 60 G (b32) | -2495 / -3146 ms | -3.4% / -4.3% | lazy wins |
| 180 G | **+5225 ms** | **+16.8%** | **lazy loses** |

At the operating point that matters -- L1 sized for the working set -- lazy is
worse by 16.8% on median TTFT, 7.7% on throughput (15908 vs 17235) and 49
requests (577 vs 626).

### Why lazy loses at 180 G

The lazy ledger at 180 G: `admitted=1560 emitted=835 dropped_evicted=690`.
**44% of admitted stores never got emitted** -- the GPU blocks were recycled
before the deferred tail was written. At 30 G the same arm dropped only 125 of
3858. The difference is wall-clock: at 180 G the system runs ~50% faster, so a
deferred store has proportionally less time to survive. Deferral is a bet that
the block will still be there later, and the bet gets worse exactly as the
system gets healthier.

Consequence: lazy stored 3.73M tokens against eager's 5.45M, and retrieved
**20.60M against eager's 25.88M**. The retrieve ratio lazy/eager, which was
13.7x at 30 G and 4.9x at 60 G, is **0.80x** at 180 G.

Preemption, unique request ids: eager 5 (@30) / 7 (@180), lazy 185 (@30) /
34 (@180). Lazy stays 5-27x eager, but its count falls 5.4x going to 180 G --
consistent with the held-block mechanism, since blocks are held for less
wall-clock when the system is fast.

### This is the first workload with a real job for the degradation knob

Both lazy arms ran `DEGRADE_SECS=0` -- the controller was off. A 44% drop
share is exactly what `_loss_is_material` exists to catch: the gate fires when
windowed drops exceed `_MATERIAL_LOSS_SHARE = 0.25` of windowed admissions.
44% clears it by a wide margin, so at L1=180 the controller should open a
trial, find the deferred baseline losing, and commit to immediate emission.

Every previous round left the knob with nothing to do (0 trials at hot/cold
40 G; no measurable difference at a91 90 G). This one gives it a documented
failure mode to repair, with a pre-committed success criterion: **lazy@180
with the knob on should close most of the +5225 ms gap to eager@180, and its
`dropped_evicted` should fall well below 690.** If it does not, the controller
does not do the job it was built for.

### Prediction scorecard: 2 hit, 1 partial, 2 missed -- including the decisive one

1. **At 180 G the cache stops thrashing** (retrieved/ISL > 50% both arms,
   drops and watermarks fall) -- **PARTIAL**. retrieved/ISL 76.7% / 66.2% and
   watermarks 442/305 -> 13/7, both hit. But lazy's `dropped_evicted` went the
   other way, 125 -> 690, which is the finding of the round and I predicted its
   opposite.
2. **Absolute jump at 180 G** (throughput > +30%, p50 well under 73.5 s) --
   **HIT**: +55% and 31.2 s.
3. **Gap shrinks at 180 G, grows at 30 G** -- **FALSIFIED, both halves.** At
   30 G it vanished (-36 ms); at 180 G it reversed to +5225 ms against lazy.
   The stated falsifier fired: "lazy's edge is eviction economy" is dead. The
   edge is not monotone in pressure at all -- b32's 60 G win sits at a local
   optimum between "nothing survives to be cached anyway" (30 G) and
   "deferral loses the race against block recycling" (180 G).
4. **Preemption stays lazy-heavy and L1-independent** -- **MISS** on the
   independence clause: lazy's count fell 5.4x at 180 G. The held-block
   mechanism survives, but it scales with how long blocks are held, not with
   L1 size.
5. **Retrieve ratio > 4x at 30 G, < 2x at 180 G** -- **HIT** in direction,
   understated in magnitude: 13.7x and 0.80x. I did not anticipate it dropping
   below 1.

### What this does to the b32 verdict

b32's measurement stands -- it had both control pairs and the effect was 5-55x
the floor. Its *interpretation* does not. "Lazy delivers ~4% more throughput
under KV-bound overload" is true only at L1 ~= 60 G with this working set, and
must now be stated with that qualifier. At a properly sized L1 the same code
costs 16.8% TTFT and 7.7% throughput.

### Next

1. `lazy@180 DEGRADE_SECS=450` vs `lazy@180 off` vs `eager@180`, with a
   same-config control arm. The knob's first real test.
2. Replicate the eager@30 vs eager@180 contrast -- it is the largest effect in
   the whole campaign and currently rests on one round.

---

## d180: attacking `dropped_evicted` at the properly-sized L1 (predictions pre-stated)

Launched 2026-08-27 ~07:5x.

### Diagnosis

Drop share rises as the system gets healthier:

| L1 | admitted | emitted | dropped_evicted | share |
|---|---|---|---|---|
| 30 G | 3858 | 3654 | 125 | 3.2% |
| 60 G (b32) | 3481 | 3017 | 340 | 9.8% |
| 180 G | 1560 | 835 | **690** | **44.2%** |

The mechanism is the one the code documents against `max_pending_ops`: the
danger depth is `ceil(max(EMA, next_step_estimate) * horizon_steps)`, a
forecast of per-step allocation, and "it cannot see a single admission that
consumes thousands of blocks at once -- the eviction that destroys a waiting
operation and the allocation that pays for the forecast are the same event."
One agentx context is 55087 tokens = ~3400 blocks at block size 16, and at
L1=180 the engine turns requests over ~50% faster, so those bursts are both
bigger and more frequent relative to the steady-state EMA.

Note what the c32L ledger rules out. `pending=8` at end of run and
`throttled_drains=0`: the backlog is shallow and the per-step caps never bound
anything. So `max_pending_ops`, whose job is to bound a *deep* backlog, is not
the lever here -- ours is not deep. And raising `horizon_steps` cannot reach:
covering a 3400-block burst from an EMA of a few blocks per step would need a
horizon in the hundreds, which is "emit immediately" spelled expensively. That
leaves the two mechanisms that do not depend on forecasting the burst.

**All the mitigations are off by default, and the harness matched the
defaults**: `max_pending_ops=0`, `max_drain_blocks_per_step=0`,
`idle_drain_max_ops=0`, `degrade_l1_residence_secs=0`. No round in this
campaign has ever exercised any of them. That is a finding about the shipped
defaults, not a harness bug.

### Round

L1=180, concurrency 32, 1800 s. Primary metric is `dropped_evicted`, a ledger
counter -- b32 showed those reproduce within ~13% across same-config arms, so
this round can be read in-round and against c32L's 690.

| slot | gpu | tag | config |
|---|---|---|---|
| 0 | 0 | `d180_lazy_ctl_s0` | lazy@180, all defaults (control) |
| 1 | 1 | `d180_lazy_knob_s1` | lazy@180, `DEGRADE_SECS=450` |
| 2 | 5 | `d180_lazy_idle_s2` | lazy@180, `IDLE_OPS=64` |
| 3 | -- | unused | see below |

**Slot 3 is deliberately left idle.** Another session's multi-modal validation
has grown to GPUs 2/3/4/7 and two 280 GB MP servers, leaving 877 GB of host
memory. Four 180 GB arms would want ~560 GB actual (c32L's 180 G arms expanded
to 124-140 GiB each, not the full cap) and 720 GB worst case. Taking that while
a five-hour-old run next door is still allocating risks OOMing *their* work, so
this round runs three arms and takes the eager@180 TTFT reference from c32L,
with the control arm establishing whether that cross-round reference is valid.

### Predictions

1. **Control reproduces**: `dropped_evicted` between 600 and 780, `admitted`
   between 1400 and 1700. If it does not, c32L's 690 was not a stable property
   and nothing else in this round is interpretable.
2. **Knob arm fires and helps**: `degrade_trials >= 1` and
   `degrade_commits >= 1` (a 44% loss share clears the
   `_MATERIAL_LOSS_SHARE = 0.25` gate by a wide margin),
   `degraded_emitted > 0`, and `dropped_evicted < 300`.
3. **Idle arm fires and helps**: `idle_emitted > 0`, `idle_drain_steps > 0`,
   `dropped_evicted < 400`.
4. **Drops are the cause of the retrieval deficit, not a symptom.** Whichever
   arm cuts drops most also moves `tokens_retrieved` closest to eager@180's
   25.88M (lazy@180 default managed 20.60M). **Falsifier: if drops fall but
   `tokens_retrieved` does not rise, the 20.6M-vs-25.9M gap has a different
   cause and fixing drops will not recover the 16.8% TTFT loss.**
5. **Neither treatment fully closes the gap to eager.** The best lazy arm is
   still at or above eager@180 on median TTFT, because degrading to immediate
   emission is at best eager-equivalent minus the machinery's overhead.
