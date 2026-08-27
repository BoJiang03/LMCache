# 7. Placement verdict: EVICTION_HEAD is the bill

Continues record 6. The puzzle there: at L1=60G lazy loses ~11s of sumTTFT to
eager, yet 80% of requests never touch LMCache's data path, nothing queues,
step rate is unchanged, prefill volume is equal. Six mechanisms falsified; the
one unexamined family was "lazy changes GPU-side shared state that every
request sees." That family is now confirmed, and the specific member is the
completed-store block release placement.

## 1. Why this workload is decided on the GPU, not in L1

Two structural differences against the PR_INFO workloads (QASPER, SWE-agent
replay), both quantified from the n60 arms' per-request exports
(`gaps.py`, 45 conversations, 226 turns with a predecessor):

- **Reuse distance vs pool size.** Inter-turn gap: p50 = 1.4 s, p75 = 5.1 s,
  p90 = 25 s. (Record 6 and earlier chat said "10 s scenario gaps"; measured,
  the median is 1.4 s -- an agent tool loop, not a think-time gap.) New-KV
  rate is ~1.7 ktok/s against a 262K-token GPU pool, so pool turnover is
  ~152 s. Only 7 of 226 turns (3% of reuse-carrying ISL) return beyond pool
  turnover; ~90% of reuse mass is servable by the GPU prefix cache. The PR
  workloads were constructed so that 100% of reuse falls outside GPU
  residency (QASPER: 384K tokens flow past a 143K-token pool in the 16 s
  return gap; SWE-agent: 36.7 GiB live against a 20 GiB pool, off reaches
  only 0.308 coverage from APC alone).
- **Nothing to save on the L1 axis.** tokens_stored and l1_watermark_events
  are point-for-point equal between eager and lazy here (60G: 2,009,856 vs
  2,009,344; 14 vs 13 watermark events). Lazy merges ~1050 small stores into
  ~220 large ones at equal bytes. On QASPER/SWE-agent, lazy cut L1 eviction
  cycles 3x and held coverage where eager's collapsed to zero; that leg does
  not exist on agentx.

L1 still carries the tail: retrieve events are 36-78 of ~270 requests, but a
single retrieve pulls ~27K tokens and the top conversations reach 91-95K
final ISL (8.3 GiB of KV each -- 2-3 of them fill the 24 GiB pool). The
30G loser-board conversations from record 6 (`14129d47`, `12bd4c7c`,
`259d1cc3`) are exactly the top-3 final-ISL conversations.

## 2. The three channels that touch every request

Even a request that never stores or retrieves shares the GPU with:

1. **Store-release placement.** Default `StoreReleasePlacement.EVICTION_HEAD`
   requeues a completed store's blocks at the eviction head
   (`free_blocks(prepend=True)`): "their content has a copy below the GPU, so
   spending them first spares blocks that do not." The enum's own docstring
   names the collision: "in a multi-turn workload the just-stored prefix is
   what the session's next turn asks for." With a 1.4 s median gap, the
   just-stored prefix is the next APC hit, and the default donates it to the
   allocator.
2. **Emission timing.** Lazy emits when eviction danger peaks, i.e. exactly
   when a large prefill is allocating; eager stores at request completion,
   which under gapped load tends to be quiet. Same bytes, opposite phase.
   (The decode-rate probe cleared only the scheduler step path, not
   prefill-window interference.)
3. **Dropped coverage.** 62-182 ops per run are dropped_evicted; their KV has
   no L1 copy, so medium-gap reuse that leaks past APC recomputes.

## 3. q30 preview (30G, one arm each -- noisy point, direction only)

| arm | gpu_prefix_hit | retrieved | sumD vs eager | p99 |
|---|---:|---:|---:|---:|
| eager | 86.2% | 627K | -- | 8738 |
| lazy default (head) | 7.9% | 995K | +11.4s | 12550 |
| lazy lru_tail | 25.7% | 703K | **-18.5s** | 9004 |
| lazy cap64 | 33.6% | 977K | +41.9s | 9044 |

Changing only the release placement flips the sign, removes the +368K excess
retrieval, and cuts p99 by 3.5 s. cap64 being worst is channel-2-consistent:
forcing emission at pressure moments (333 stores, 156 via backlog) amplifies
the collision. gpu_prefix_hit remains an untrustworthy instrument (see 5).

## 4. r60 verdict (60G, 2 default-lazy vs 2 lru_tail, one round)

| arm | sumTTFT | retrieved | preempt | dropped_evicted |
|---|---:|---:|---:|---:|
| r60_lazy_s0 | 298.4s | 1.40M | 3 | -- |
| r60_lazy_s3 | 277.1s | 1.40M | 3 | 84 |
| r60_tail_s1 | **263.6s** | 1.02M | 1 | 182 |
| r60_tail_s2 | **267.3s** | 1.08M | 1 | 179 |

- Every tail arm beats every lazy arm. Against the better lazy arm the
  recovery is 10-13.5 s -- the size of the n60 deficit to eager (+11.2 s).
  (Within-lazy spread is 21.3 s this round, s0 high, so the conservative
  reading is against s3.)
- The mechanism signature replicates q30: the ~0.35M excess retrieved tokens
  vanish under tail (1.40M -> 1.02/1.08M, eager's level), preemptions 3 -> 1.
- The trade is visible: tail doubles dropped_evicted (84 -> ~180). Head
  placement does protect unstored blocks -- it just protects them by spending
  the blocks the next turn needs. At 90G that trade wins (k-round: head beat
  tail by 14.5 s sumD, drop 116 vs 151); at 60G and 30G it loses. The default
  is L1-size- and reuse-structure-dependent.

**Record 6's prediction resolved: the cost is the placement, not the deferral.**

## 5. Instrument note

gpu_prefix_hit (vLLM's cumulative "Prefix cache hit rate" log line) read
8.3% and 46.3% on the two *identical* plain-lazy arms of r60. It is unusable
for cross-arm comparison; the retrieval ledger (tokens_retrieved) is the
reliable signal and is what carried both verdicts. This subsumes record 6's
APC/EXT-vs-throughput contradiction: no claim should rest on the APC line.

## 6. The fix ladder (difficulty assessment, pre-implementation)

Any placement choice perturbs vLLM -- head and tail each win one workload.
Three tiers were assessed:

1. **Flip/auto-select the default** (hours; mechanism exists since
   `1c43ca02`). Treats the symptom; the trade stays and must be documented.
2. **Idle-preferring drain** (days). Emit pending ops when the engine is
   idle, keep the eviction trigger as backstop. Removes channel 2 on gapped
   load; no behavior change under constant load.
3. **Pin-free optimistic emission** (about a week; no vLLM changes). Do not
   touch the free queue at all: copy D2H with the blocks still queued,
   revalidate the block-hash snapshot at completion, discard on mismatch via
   the existing dropped_failed_store path. vLLM resets a block's hash at
   allocation before writing new KV, so torn reads and reallocation are
   detected at completion; a same-hash ABA means identical content and the
   store stays valid. Cost: an occasional wasted copy, and drops race the
   allocator (horizon_steps=10, record 5's free win, adds runway). This makes
   lazy read-only toward vLLM: the placement question disappears, channels 1
   and 2 both close.

Recommendation staged with the user: confirm with s60, then implement tier 3
with tier 2 as its complement; skip tier 1.

## 7. In flight

- `s60` round launched 13:07 (`chain_s.sh` -> `schain.log`): 2 eager
  (slots 0,2) vs 2 lru_tail (slots 1,3) at 60G. r60 had no eager arm and
  cross-round drift (r60's lazy arms sit 7-28 s above n60's) makes the
  eager comparison unsafe; this is the direct verdict on "does tail-lazy
  reach eager where default-lazy lost." Due ~13:27.

## 8. s60 verdict (added 13:25): tail alone does not reach eager

2 eager vs 2 lru_tail at 60G, one round:

| arm | sumTTFT | retM | preempt |
|---|---:|---:|---:|
| s60_eager_s0 | 258.0s | 0.97M | 1 |
| s60_eager_s2 | 263.6s | 0.98M | 1 |
| s60_tail_s1 | 275.1s | 0.99M | 4 |
| s60_tail_s3 | 267.0s | 0.97M | 2 |

Both tail arms lose to both eager arms; group means differ by 10.2 s
(within-eager spread 5.6 s). The placement fix did what r60 said it does --
the excess-retrieval signature is gone (retM equal across all four arms) and
pooled across rounds tail (268.3 s, n=4) recovers about half of head's
deficit (278.5 s, n=4; eager 259.4 s, n=4). But a ~9 s residual remains, and
its shape is channel 2: preemptions 2-4 on tail vs 1 on eager (emission still
fires in the pressure phase; the transient pin and the D2H burst land on
someone's large prefill), and the median still pays (+5/+16 ms).

Section 6's framing is revised accordingly: placement is half the bill, not
the bill. Config alone has one shot left -- horizon=10 (record 5's free win:
emits earlier and spread out, drop 127->63 at 90G). t60 round launched 13:25
(`chain_t.sh` -> `tchain.log`): 2 eager vs 2 tail+hz10, due ~13:46. If
tail+hz10 does not close the residual, the user's bar (lazy >= eager on
unfavorable workloads) requires the idle-preferring drain (tier 2), which
becomes the critical-path development item.

## 9. t60 verdict (added 13:55): config-only is exhausted, idle-drain is the item

2 eager vs 2 tail+horizon=10 at 60G:

| arm | sumTTFT | medD | preempt | drop | sto |
|---|---:|---:|---:|---:|---:|
| t60_eager_s0 | 266.6s | -- | 1 | -- | 1055 |
| t60_eager_s2 | 263.6s | +3 | 1 | -- | 1055 |
| t60_thz_s1 | 281.3s | +19 | 1 | 67 | 247 |
| t60_thz_s3 | **262.0s** | +9 | 1 | 84 | 242 |

hz10 fixed every emission-phase observable: preemptions 4->1 (all arms now
1), dropped_evicted halved (122-138 -> 67-84), retrieval equal. And the
latency is now *mixed*: thz_s3 beats both eager arms, thz_s1 loses to both by
15-18 s; within-config spread 19.3 s against eager's 3.0 s. Both thz medians
are still positive (+9/+19 ms) with every data-path counter clean.

Pooled 60G picture (group means across rounds): eager 261.3 s (n=6, three
rounds), head-lazy 278.5 s (n=4), tail 268.3 s (n=4), tail+hz10 271.7 s
(n=2, dominated by the 281.3 outlier). Config work moved lazy from "loses
11-20 s consistently" to "sometimes wins, sometimes loses 15 s" -- short of
the acceptance bar the user set this afternoon: lazy must at least match
eager on unfavorable workloads and stably beat it on favorable ones.

Remaining suspect for the residual median tax: emission *byte granularity*.
Lazy's coalesced ops average ~9K tokens ~= 850 MB of contiguous D2H per
emission against eager's ~1.9K-token trickle; hz10 changes when batches
emit, not how big they are. The development item is the idle-preferring
drain (tier 2), with byte-level throttling of a single drain's submission;
the pin-free tier 3 stays rejected (see section 6 revision: completion-side
hash validation cannot close the corrupt-object visibility window without a
two-phase-commit MP protocol change or an upstream vLLM API).

## 10. In flight

Hot/cold placement A/B launched 13:52 on GPU 0 (copied harness at the
session scratchpad `hotcold/`, originals untouched; `lazy_tail` config key
added to the copy). eager / lazy(head) / lazy_tail, 2 reps each,
interleaved; SMOKE_REPO points at this worktree. This is the PR-side
regression gate for shipping any placement change: PR_INFO's numbers were
measured with head behavior (verified: df199979 already had prepend=True).
Due in roughly 40-60 min.

## 11. Hot/cold verdict (added 14:10): tail passes the PR-side gate

All six cells clean (zero tracebacks/warnings), copied harness, GPU 0,
Qwen3-8B, L1=40G, horizon 2.5, SMOKE_REPO at this worktree:

| config | query wall | hot p50 | cold p50 | evictions | dropped_evicted |
|---|---:|---:|---:|---:|---:|
| eager x2 | 40.0 / 40.6s | 130 / 137ms | 770 / 775ms | 14 / 13 | -- |
| lazy head x2 | 29.6 / 31.4s | 172 / 189ms | 314 / 511ms | 5 / 7 | 0 / 0 |
| lazy tail x2 | 29.6 / 32.3s | 149 / 190ms | 299 / 639ms | 5 / 8 | 0 / 0 |

Tail is indistinguishable from head within rep drift (rep 1 is worse for
both) and keeps the full ~10 s win over eager. The explanation is the
ledger: dropped_evicted is 0 in all four lazy cells, so EVICTION_HEAD's
premise -- spend backed-up blocks first to protect unstored ones -- never
pays here. Where the premise does engage (agentx 60G, drops 84 vs 180),
what it protects is worth less than the APC entries it spends.

Bonus: covered_prefix_advances fires on this workload (51 and 11 in the two
tail cells, 615K/159K tokens skipped) -- hot/cold has the shape needed to
close record 5's GSM8K coverage gap for 5ea3cc6e.

## 12. Where this leaves the day

- **Recommendation: flip the default StoreReleasePlacement to LRU_TAIL.**
  Evidence: agentx 60G 2v2 (tail recovers 10-13.5 s), agentx 30G preview
  (30 s swing), hot/cold 2v2 (parity with head, win over eager intact).
  Head's only observed advantage anywhere is a single-arm 14.5 s at agentx
  90G where both placements already beat eager comfortably. Keep
  eviction_head as a config value and document the trade.
- **The user's bar (lazy >= eager on unfavorable, stably > on favorable) is
  not yet met at agentx 60G**: tail+hz10 still shows +9/+19 ms medians and
  19 s config spread. The development item is the idle-preferring drain
  with a per-step byte cap; the idle signal (new_blocks_allocated) already
  reaches the policy via observe_step, so the change is contained in
  lazy_offload_policy/ plus config, counters, layer-1 scenarios, docs.
  Estimate 2-3 days.
- Pin-free emission stays rejected (section 6 revision).
