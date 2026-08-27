# Lazy beats eager once the comparison is controlled

Date: 2026-08-26 (afternoon session)
Branch: `lazy-offload-publish`, code at `1c43ca02`
Workload: aiperf scenario `inferencex-agentx-mvp`, dataset `semianalysis-cc-traces-weka-062126`,
64 entries, concurrency 8, seed 1234, `--max-context-length 100000`, 900 s benchmark duration.
Model Qwen3-Coder-30B-A3B-Instruct, TP=1, pool 24 GiB = 16384 blocks, L1 = 90 GiB.

## 1. Verdict

On the scenario-compliant workload at this operating point, lazy offload beats eager on every
aggregate that matters, and the margin is an order of magnitude larger than the measurement
floor. The earlier "lazy is worse than eager" readings were all measurement defects; none of
them survived a simultaneous, correctly-paired comparison.

Four plain-lazy replicates against an in-round eager control, across two rounds and four GPUs:

| round | lazy arm | GPU | sumTTFT delta | share |
|---|---|---|---|---|
| p_* | p_lazy_a | 1 | -28.8 s | -10.2% |
| p_* | p_noprep_a | 5 | -26.4 s | -9.4% |
| p_* | p_noprep_b | 6 | -24.1 s | -8.6% |
| k_* | k_lazy_s0 | 0 | -42.3 s | -14.8% |

4/4 win. The slot (GPU) effect, measured independently in each round by running two eager arms
on different GPUs, is 1-2%: -3.4 s in the k_* round and +5.5 s in the g_* round. It cannot
account for a 24-42 s difference.

`p_noprep_a` / `p_noprep_b` were *intended* as `lru_tail` arms but the env var never reached
EngineCore (see section 3), so they are plain-lazy replicates. That accident is what gave the
p_* round three independent lazy replicates instead of one.

## 2. Where the "eager is better" readings came from

Every one of them was mine. Four independent defects, each fixed:

| bad reading | defect | fix |
|---|---|---|
| "lazy loses ~20 s of tail at every L1 size" | pairing key included `session_num`, a client session slot reassigned per run. Matched only 154/256 requests, and *which* requests survived the match correlated with timing. | pair on `(conversation_id, turn_index)` -> 254-255/256 matched |
| "lazy is worse than eager" | serial runs compared across hours on a shared box. The same eager config gave 303.3 / 298.8 / 281.0 s at three different times -- 22 s of drift against a 24-42 s effect. | 4-slot parallel harness, all arms simultaneous. Eager then reproduces to 1.7% (compliant: 281.0 / 285.8 / 282.4 s across GPU 0 / 1 / 6). |
| "the regression is real, not noise" | n=1. One lazy run treated as a mean. | the next replicate came in at -18.6 s |
| "lazy only ties on the fast workload" | the compressed workload is my own speedup hack, and it removes lazy's benefit rather than adding cost (section 4) | discarded for eager/lazy conclusions; smoke-test only |

The parallel harness is the single fix that mattered most: serial measurement on this box cannot
resolve a 30 s effect, and no amount of replication fixes that, because the drift is common-mode
within a run and differential across runs.

## 3. The env-var bug that voided two arms

`LMCACHE_LAZY_STORE_FREE_PREPEND=0` reached the `vllm` parent process but not
`VLLM::EngineCore`, which is started with only 9 environment variables. Two arms labelled
`lru_tail` therefore ran as plain lazy, silently. Fixed in `1c43ca02` by moving the switch to
`kv_connector_extra_config` as `lmcache.mp.lazy_offload_store_release`, which is the only channel
that reaches the scheduler side. Any future scheduler-side switch must go the same way.

## 4. The compressed workload is not a valid proxy

Built to cut a 20-minute arm to 14 minutes by capping the global idle gap 10 s -> 1 s
(`--system-idle-gap-cap-seconds 1 --unsafe-override`) and switching to `--request-count 256`.
It runs faster and it changes the answer. Decomposing sumTTFT into the requests that got slower
(`lost`) and faster (`gain`):

| workload | lost | gain | net |
|---|---|---|---|
| compliant, 3 lazy arms | 30.5 / 33.5 / 32.6 s | **-59.3 / -59.9 / -56.7 s** | -28.8 / -26.4 / -24.1 s |
| compressed, 5 lazy arms | 31.3 / 34.9 / 35.6 / 35.6 / 61.9 s | **-24.6 / -34.8 / -35.7 / -39.6 / -25.7 s** | +6.7 / -0.8 / +0.8 / -4.1 / +36.3 s |

`lost` is essentially unchanged (~33 s both ways). `gain` collapses from -58 s to -33 s. So
compression does not make lazy more expensive; it removes the thing lazy is good at. Mechanism:
with 1 s instead of 10 s between turns, the session's next turn arrives while its prefix is still
resident on the GPU, so there is nothing for the lower tier to serve.

Compression also made lazy erratic in a way it is not on the compliant workload: five compressed
replicates spread 40 s (+36.3 to -4.1) against 18 s for the four compliant ones. `g_lazy_s1`'s
+36.3 s outlier was checked against slot contamination and cleared -- an eager arm run on that
same GPU in the next round gave 247.6 s against 247.3 s on GPU 0, i.e. 0.1%.

Conclusion: the compressed configuration stays in the harness (`par/farm.sh`) for smoke-testing
flag combinations, and its numbers are never quoted as agentx-mvp results.

## 5. `eviction_head` vs `lru_tail`, in-round paired

First valid measurement of this knob (`k_*` round, both arms in the same round as the eager
control):

| arm | placement | sumD | p90 | p99 | max | GPU prefix hit | tokens retrieved | dropped_evicted |
|---|---|---|---|---|---|---|---|---|
| k_lazy_s0 | `eviction_head` | **-42.3 s** | **1630** | 7700 | **8770** | **34.5%** | **1.70 M** | **116** |
| k_tail_s2 | `lru_tail` | -27.8 s | 2098 | 7775 | 9013 | 13.8% | 1.38 M | 151 |
| k_eager_s1 | (eager, baseline) | 0 | 2137 | 8898 | 12942 | 14.4% | 1.18 M | -- |

`eviction_head` wins by 14.5 s and dominates on every secondary metric. The default is correct.

The result also resolves what looked like a contradiction: requeueing just-stored blocks at the
*eviction head* produces a **higher** GPU prefix-cache hit rate (34.5% vs 13.8%) than requeueing
them at the LRU tail. The reason is the one already written into the `StoreReleasePlacement`
docstring, now with numbers behind it: those blocks have a copy in L1, so spending them first
spares the blocks that do not. `lru_tail` instead pins large just-stored prefixes (mean 7.8 k
tokens per store) in the pool on behalf of conversations that may not return, and ends up
delivering less cached content from *both* tiers.

## 6. Signals that never changed sign

- **Median cost is positive in 10/10 lazy arms**: +2, +9, +11, +11, +12, +13, +13, +13, +13, +14 ms.
  The sign is consistent, so the body cost is real. The magnitude is less certain than the
  spread suggests: the eager-vs-eager reference itself moves by 9 ms (`k_eager_s3` medD = -9 ms
  against `k_eager_s1`), which is the same order as the effect. Call it "roughly +10 ms, sign
  certain, magnitude within a factor of 2".
- **Tail is better, consistently**: p99 -10 to -13%, max -26 to -32%, across both compliant rounds.
- **GPU prefix hit rate is 2-7x eager's**, in-round: lazy 34.5 / 39.2 / 35.5 / 25.4 / 22.7 / 18.0%
  vs eager 5.0 / 5.2 / 8.7 / 14.4 / 18.7%. This signal was retracted in record 3 as noise; that
  retraction was based on a cross-round comparison and was itself wrong. In-round it is robust.
- **Store count and volume**: lazy issues ~4.5x fewer stores (222-231 vs 1007-1026) carrying
  slightly more tokens (1.69-1.83 M vs 1.70-1.73 M), i.e. mean 7.8 k tokens per store vs 1.7 k.

So the honest one-line characterisation of lazy on this workload: **it costs roughly 10 ms at the
median and returns 13-32% at the tail.** The aggregate sumTTFT is tail-dominated, which is exactly
why it was the quantity that kept flip-flopping under noisy measurement.

## 7. Not a complete win, and not a general result

- The median regression is real, and 5-10 requests per run of 273 get more than 1 s slower
  (against 10-14 that get more than 1 s faster). Net gain with losers, not a Pareto improvement.
- **One operating point.** One model, one pool size (24 GiB), one concurrency (8, effective 2.4),
  one L1 size (90 GiB).
- **The L1 30 G and 60 G points must be re-measured.** Their numbers come from the broken
  serial + `session_num` regime and carry no weight.
- **The load ramp has not been run at all.** This is the most likely place for the picture to
  change: lazy's benefit depends on the pool being saturated, while lazy's cost (blocks pinned
  out of the free queue) grows with concurrency. At effective concurrency 2.4 the pin high-water
  mark is 35% of the pool but time above 10% is only 1.8 s out of 900 s.
- **`dropped_evicted` is 116-151 per run**, 12-15% of admitted stores lost to eviction before
  emission. Unfixed.

## 8. In flight

Compliant knob sweep launched 11:12 (`m_*` round): eager control on GPU 0, plain lazy on GPU 1,
`max_pending_ops=8` on GPU 5, `horizon_steps=10` on GPU 6. Both knobs previously got negative
verdicts from single serial runs; this is their first in-round paired test.

## 9. Housekeeping

The `rep3` serial queue finished at 10:34 and exited clean. GPU allocation checked on request:
this session held GPU 0/1/5/6 (four arms), the user's other task held GPU 2/7 and later
consolidated onto GPU 3, root held GPU 4. No collision, no idle reservations, no leaked
processes from this session.

## 10. Artifacts

Scratchpad `.../84352f47-.../scratchpad/par/`:

- `env.sh` (SLOT->GPU 0/1/5/6, ports 27100+SLOT*10, `HORIZON`, `MAX_PENDING`, `STORE_RELEASE`),
  `up.sh`, `down.sh`
- `arm.sh` (compliant, 900 s + 120 s grace), `farm.sh` (compressed, smoke-test only)
- `round.sh` / `fround.sh`, spec `"cfg:tag[:ENV=V,...]"`, 4 slots, 20 s stagger
- `chain_k.sh` -- waits on one round's log then launches the next, so GPUs do not idle between rounds
- rounds: `round1.log` (p_*), `ground1.log` (g_*), `hround.log` (h_*), `kchain.log` (k_*), `mround.log` (m_*)
- per-arm `snapshot.txt` (stores, retrieves, tokens, watermark/preempt events, GPU prefix hit,
  effective concurrency, L1 objects/GiB, lazy ledger, store token and latency distributions)
- `../cmp2.py` -- paired comparison on `(conversation_id, turn_index)`
