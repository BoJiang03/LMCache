# Session log: where the engine's time goes, why L1 misses, and a 2.7k PR

Conversation record for 2026-08-31, branch `lazy_offloading_goodput_check`.
Two threads. Bo asked whether the deployment's numbers are reasonable, which
turned into a time-budget and cache-residency audit of the existing sweep
artifacts (no new runs). Then the colleague's review came back -- "too much
code" -- and the second half is the cut plan.

Analysis script: `artifacts/goodput/goodput_decompose.py`. It reads only
`<arm>_samples.log` and `<arm>_server.log`, both already archived, so every
number below is reproducible from what the sweep wrote.

## 1. The engine's time budget

Counter differencing on the 15 s scrapes, then least squares of the scrape
interval on (computed prefill tokens, decode steps, externally transferred
tokens). Unit costs come out stable across every arm -- 51 to 60 us per
computed prefill token, 10.6 to 16.4 ms per decode step -- which is the
check that the decomposition means anything.

| arm | prefill | decode | ext transfer | unattributed |
|---|---|---|---|---|
| e32 / l32 | 38.3% / 43.0% | 41.2% / 38.7% | 3.7% / 2.7% | ~16% |
| e40 / l40 | 47.8% / 49.7% | 26.9% / 27.9% | 8.5% / 8.7% | ~15% |
| e48 / l48 | 55.5% / 52.7% | 20.9% / 22.6% | 9.4% / 8.4% | ~15% |
| e72 / l72 | 61.1% / 56.2% | 20.5% / 21.5% | 3.6% / 8.0% | ~14% |

Independent check on the step accounting: decode steps (from the log's
generation rate over Running) plus prefill chunks (computed tokens over the
8192 budget) give 44.7k for l40 against vLLM's own
`iteration_tokens_total_count` of 45.5k, 2% apart.

**Correction.** I first told Bo the deployment was prefill-bound already at
CONC=32. It is not: at 32 prefill and decode are within a few points of each
other, and prefill only becomes the majority between 40 and 48. What is true
at 32 is the other two claims below.

## 2. The decode step curve, and what 50 tok/s/user costs

From windows where prompt throughput is exactly 0 and Running is stable
across two consecutive log lines:

    B=6  14-16 ms    B=9  19-21 ms    B=10 20 ms    B=11 22 ms
    B=13 24 ms       B=15 25 ms       B=17-19 28 ms B=24 38 ms

That is the deployment's real tpot curve and it matches the cost model in
`deployment_candidate.md` Part 4. Decode is not the problem. The ITL users
see is 1.8x (CONC=32) to 3.3x (CONC=72) that curve, because decode steps are
interleaved with 8192-token prefill chunks costing ~410 ms each.

Two things that are true at every tested concurrency:

- The throughput knee is at or below 32. Delivered prompt token rate is
  29.3k (32), 28.3k (40), 28.2k (48), 24.2k (72) -- flat then falling --
  while TTFT goes 3.4 s, 5.1 s, 8.0 s, 40 s. Nothing below 32 was ever run,
  so where the knee actually sits is unknown.
- R1 (50 tok/s/user) is met nowhere. Best is 31.5 at CONC=32.

Calibrating `E = 2.4 s prefill + 1095 x t_dec(B)/B` against the measured
0.25 req/s (model says 0.23) and the measured per-user rate (model runs 25%
low, corrected below):

| target B | ~CONC | per-user tok/s | req/s vs today |
|---|---|---|---|
| 13 | 32 | 23 | baseline |
| 8 | ~20 | 33 | -20% |
| 5 | ~12 | 45 | -35% |
| 4 | ~10 | 51 | -41% |

So 50 tok/s/user is purchasable, at about 40% of the throughput. Halving the
prefill work per request (section 3) moves the same target to B=6 at about
-15%, which is the cheaper path.

## 3. L1 holds fewer tokens than the GPU pool

    L0 (GPU pool)   3,250,930 tokens          (vLLM's own log line)
    L1 (250 GiB)    268e9 / 122,880 = 2.18M tokens capacity, 1.5-1.7M resident

Verified on this run rather than taken from the 08-30 spot measurement: in
the pre-eviction growth phase, delta(l1_memory_usage) over delta(write
units) is 3.93 MB in every arm, times 8 units per 256-token chunk is
**122,880 B/token** -- all 60 layers, dense. Retrieve needs the 15 full
attention layers (30,720 B/token) plus the SWA suffix (~3.4k B/token
amortised over a 112k sequence), about 34,000. The store is 3.6x what the
read needs, and the second cache tier ends up holding half as many tokens as
the pool it backs.

Residency, from write rate against resident bytes:

| arm | wrote | resident | residency |
|---|---|---|---|
| l32 | 12.7M tok / 2170 s | 1.61M tok | 276 s |
| l40 | 13.1M tok / 2383 s | 1.72M tok | 313 s |
| l48 | 11.8M tok / 2320 s | 1.51M tok | 297 s |
| l72 | 13.9M tok / 2800 s | 1.67M tok | 336 s |
| e40 | 22.4M tok / 2459 s | 1.50M tok | 164 s |

Lazy keeps entries about twice as long as eager at the same concurrency,
which is the residence mechanism the earlier records only had two lookup
points for.

**Correction.** `artifacts/ab_analysis_snapshot.md` said "neither arm filled
L1, so no eviction pressure this round", read off the end-of-run usage ratio
(0.69, 0.73). Wrong: usage rises to ~0.80 by t+250-400 s in every arm and
then sawtooths between 0.65 and 0.84 for the remaining ~1700 s. That is the
LRU watermark, not headroom. Patched in place.

## 4. The hit rate is a warm-phase average

Instantaneous split of prompt tokens over 200 s windows, e40:

    t+ 400s   L0 66.7%  L1  2.9%  recompute 30.4%
    t+ 800s   L0 70.0%  L1 14.8%  recompute 15.3%   <- 84.7% hit
    t+1000s   L0 58.0%  L1 26.7%  recompute 15.4%
    t+1200s   L0  8.2%  L1 52.8%  recompute 39.0%
    t+1600s   L0  1.4%  L1 40.1%  recompute 58.4%
    t+2200s   L0  6.3%  L1 42.8%  recompute 50.9%

Every arm has this shape: 78-85% total hit while the working set still fits,
then L0 collapses from ~68% to 0-6% and L1 -- smaller than the pool --
cannot absorb it, so recompute settles at 40-60%. The arm averages (l40: L0
29.2%, L1 34.3%, recompute 36.5%) are the mean of a warm phase and a
saturated phase; the saturated phase is the steady state.

The 84% peak is the useful number: the corpus does have the reuse, the cache
just cannot hold it. Two levers, both untested:

1. `--l1-size-gb 1000`. Pure config; the box has 2015 GB with 1398 free
   (shared, needs Bo's call). L1 goes from 2.18M to 8.7M tokens, 2.7x the
   pool.
2. Stop storing the 45 SWA layers' out-of-window blocks: 122,880 ->
   ~34,000 B/token. Needs code.

Either invalidates the existing A/B (both arms would have to rerun), and a
non-evicting L1 plausibly weakens lazy's skip-never-reused advantage while
leaving its bandwidth advantage intact. Unknown until measured.

Unconfirmed lead: `l1_read_chunks` is exactly 8x the number of hit chunks,
suggesting retrieve pulls all eight object groups per hit chunk rather than
only the SWA suffix `lazy_offload.md` describes. If real, the read side
wastes at the same ratio as the write side.

## 5. The PR is 1,520 lines of code

The colleague's review: too much code, move all but the core tests to dev,
delete the over-designed mechanisms. Measuring first, with docstrings,
comments and blanks excluded:

    +9,611 total diff
      1,520  non-test code
      2,300  docstring / comment / blank in the code files
      3,233  test code
      2,214  docstring / blank in the tests
        703  design and configuration docs

Per file: `eviction_aware.py` 744 code, `lazy_offload_pending_store.py` 306
(upstream 50), `lazy_offload_manager.py` 290, `lazy_offload_state.py` 96,
connector/adapter/metadata/fifo/types/base 41, L1 pressure stats 43.

Docstring density is most of the complaint. `eviction_aware.py` carries 852
docstring plus 145 comment lines against 744 of code, ratio 1.34. Comparable
upstream files: cache_engine 0.40, local_cpu_backend 0.38, vllm_v1_adapter
0.45, controller_manager 0.32, token_database 0.65, l1_manager 0.70 --
aggregate 0.47. `LazyOffloadPolicyConfig` spends about seventy lines
documenting six fields, each with a sizing-sensor paragraph.

## 6. What the ledger says to delete

Four lazy arms, 33 min each, 14,799 admissions:

| mechanism | counter | l32 | l40 | l48 | l72 |
|---|---|---|---|---|---|
| min_prefix gate + held two-stage admission | `rejected_short_prefix` / `held` | 0 / 0 | 0 / 0 | 0 / 0 | 0 / 0 |
| announce_allocation | `announced_bursts` | 0 | 0 | 0 | 0 |
| content deduplication | `deduplicated` | 1 | 0 | 0 | 0 |
| covered-prefix advance | `covered_prefix_advances` / tokens | 101 / 181k | 31 / 55k | 32 / 50k | 5 / 6k |
| danger floor | `danger_floor_raises` | 6 | 3 | 3 | 1 |
| block volume cap | `throttled_drains` | 0 | 0 | 0 | 0 |
| **deferral deadline** | `emitted_overdue` / `emitted` | **77%** | **64%** | **64%** | **57%** |

The first six are dead. Covered-prefix skips 0.1% of admitted tokens while
paying 450k-650k `covered_blocks_probed` per arm; the danger floor is the
most intricate code in the file (peak tracking, two-interval hold, decay)
and fires 1-6 times in 35k-58k drain steps. The deferral deadline is the
opposite: it releases more than half of all emissions and is where the win
comes from.

Code lines inside those spans: 148, plus scattered counter fields, config
fields, enum members and call sites, call it 210.

**Correction.** I told Bo the wiring commit would "harden FIFO's drain by
routing it through revalidation". Wrong: upstream `lmcache_mp_connector.py`
already revalidates block hashes before the store
(`old_block_hashes == new_block_hashes`). The manager moves that check, it
does not introduce it.

## 7. The plan

One PR, not a ladder. Arithmetic:

    1,520  code today
      -43  L1 pressure stats split out (nothing in the PR consumes it)
     -210  six dead mechanisms
    ------
    1,267  code
           x1.47 at the repo's docstring norm, plus blanks  -> ~1,995
           + core tests (~350 code lines)                   -> ~  500
           + lazy_offload.md update, eviction_aware.md slim  -> ~  250
    ==================================================================
                                                             ~2,750

Bo approved 2.7k. Pushing to 2.1k would mean deferring `lazy_offload_state.py`
(id reuse and preemption correctness), cutting the counters from 20 fields to
6, and thinning the tests -- and the counters are the only reason section 6
could be written at all, so that trade was declined.

Keeping the manager in the same PR also means `store_release=lru_tail` ships
with the policy, so the sweep's numbers reproduce from the PR as written; a
policy-only PR would have needed a fresh l40 arm to confirm the win survives
without it.

Order of work: split out L1 pressure -> delete the six mechanisms -> bring
docstrings to the repo norm -> move all but the core tests to dev -> diffstat
for Bo -> build, unit and gsm8k gates.

Tests kept: EA's ConfigValidation, Admission, PressureTrigger, PrefixClosure,
EmissionContiguity, DrainOrderingAndCap, StoreFailure, DeferralDeadline; the
manager's epoch-reset and hash-revalidation drop; the connector's lazy switch
and `request_finished` returning False. `lazy_offload_decision_model.md` (159
lines, selection rationale rather than an interface contract) moves to dev.

## 8. Corrections made this session

- "prefill-bound at CONC=32" -> prefill and decode are within a few points
  there; the crossover is between 40 and 48 (section 1).
- `ab_analysis_snapshot.md` "neither arm filled L1, so no eviction pressure"
  -> L1 fills at t+250-400 s and evicts for the rest of every arm
  (section 3, patched in place).
- "the wiring commit hardens FIFO's drain" -> upstream already revalidates
  (section 6).

Deployment facts confirmed on the box: arcee-ai/Trinity-Large-Thinking-FP8-Block,
afmoe, 398B/A13B, 60 layers (15 full attention, 45 SWA 4096), 48 query heads
at head_dim 128, hidden 3072, 256 experts top-4, TP=4 on H200 141 GB,
`max_num_batched_tokens` 8192, GPU KV pool 3,250,930 tokens.
