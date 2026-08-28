# Gate 1 is a constant: the free-queue signal, and what survives

Conversation log for 2026-08-27 (afternoon/evening, continues from records 7-9).
No source file was touched this session -- everything here is analysis,
re-measurement of logs that already existed, and round g4F landing mid-session.
The user drove the whole thread by challenging each conclusion in turn; several
of my own numbers were falsified along the way and are retracted below.

## Code state

HEAD `9cb9a47d`, `git status --short` is `?? lo_temp_ctx.md`. The three
working-tree modifications noted in the record-8 handoff (lost-volume ledger,
`wants_l1_pressure`, `announce_hits` default) are now committed as `c59448fe`
and are the code under test in g4F. Record 9 was still uncommitted at the start
of this session and is committed together with this file.

## 1. Round g4F: both pre-registered predictions falsified

g4F landed at ~15:45. Config: conc 32, 1800 s, seed 1234, four arms; lazy60 on
slot 0, eager180 on slot 1, lazy180 on slot 2, eager60 on slot 3.

| prediction | pre-registered | measured | verdict |
|---|---|---|---|
| **W1** 60 G win survives the guard | medD <= -2000 ms (falsifier > -1000) | **-20 ms** | **FALSIFIED** |
| **G1** 180 G parity with eager | within +-1000 ms (falsifier > +2000) | **+2506 ms** | **FALSIFIED** |
| G2 degrade switch fires and holds | commits >= 1, reverts <= 1, degraded > 50% | commits=1, reverts=0, **78%** | passes |
| G3 lazy180 stored >= 90% of eager180 | -- | 4.82 M / 5.35 M = **90.1%** | passes (barely) |

Ledgers:

```
g4F_lazy180_s2  admitted=1402 emitted=1335 dropped_evicted=44 (3.1%)
                degraded_emitted=1044 (78% of emitted) commits=1 reverts=0
                probes=2 probe_recoveries=0  preempt=18  gpu_prefix_hit=2.8%
g4F_lazy60_s0   admitted=3856 emitted=3696 dropped_evicted=54 (1.4%)
                degraded_emitted=2907 commits=1 reverts=0
                probes=2 probe_recoveries=1  preempt=39
g4F_eager180_s1 stores=1517 preempt=14   TTFT n=642 p50=31,448 ms
g4F_eager60_s3  stores=3780 preempt=8    TTFT n=446 p50=70,357 ms
```

**The parity plan succeeded on its own terms and lazy still lost.** Drops fell
by an order of magnitude (180 G: 44% -> 3.1%; 60 G: 9.8% -> 1.4%), the switch
committed and never reverted, 78% of emissions were immediate -- and the 180 G
arm lost 2.5 s while the 60 G win collapsed from -4.3% to a -20 ms tie.

This is consistent with the rest of the session: drops were never the main
term, because the gate the machinery evaluates turned out to carry no
information.

`degrade_probe_recoveries=0` on lazy180 confirms record 7 section 8's "defect 2"
is closed by the lost-volume ledger; lazy60 shows 1 recovery, so the mechanism
is live rather than wedged.

## 2. The central measurement: gate 1 has no discriminating power

Method, no new run required. A block leaves the GPU free queue by exactly one
of two paths: **touch** (vLLM's own prefix cache hits it -- storing it was
waste) or **eviction** (storing it was necessary). Therefore

```
store_necessity = E / (E + T)  >=  1 - X      X = vLLM local prefix cache hit rate
```

X was in every log all along. Everyone had been reading `External prefix cache
hit rate` (LMCache's own, 13-75%); the local one is the separate
`Prefix cache hit rate` field, and its median is **0.00%**. Weighted by prompt
throughput over active intervals:

| arm | L1 | GPU local hit | **necessity >=** | ceiling on wasted stores |
|---|---|---|---|---|
| b32_eager_s0 | 60 G | 0.052% | **99.95%** | 0.05% |
| e60A_eager_s0 | 60 G | 0.158% | **99.84%** | 0.16% |
| b32_lazy_s1 | 60 G | 0.228% | **99.77%** | 0.23% |
| b32_lazy_s2 | 60 G | 0.306% | **99.69%** | 0.30% |
| d180_lazy_ctl | 180 G | 0.266% | **99.73%** | 0.84% |
| d180_lazy_idle | 180 G | 0.406% | **99.59%** | 0.67% |
| c32_eager_l180 | 180 G | 0.752% | **99.25%** | 1.27% |
| b32_eager_s3 | 60 G | 0.753% | **99.25%** | 0.76% |

The eager arms are the clean control (no pinning to suppress local hits) and
land in the same range, so this is not an artifact of lazy's pin.

"Ceiling on wasted stores" is a true upper bound: it assumes every single local
hit landed on a block we had already stored. `snapshot.txt` has carried
`gpu_prefix_hit` all along; nobody had read it as a signal-quality number.

**Consequence.** Gate 1 returns true >= 99.2% of the time -- H(0.992) ~ 0.067
bits. The 2248-line machinery computes a constant, and pays 9.8-44% of intake
to do it.

Against the cost of waiting the extra distance:

| | benefit ceiling | measured cost |
|---|---|---|
| 60 G | <= 0.3 pt | drops 9.8%, preempt 170 vs eager 1-3 |
| 180 G | <= 1.3 pt | drops 44% |

## 3. Why: the free queue is a conveyor, not a cache

Free-queue residence, computed as `free_blocks / allocation_rate`:

| arm | mean usage | free queue | alloc | **residence** |
|---|---|---|---|---|
| b32_eager_s0 (60 G) | 72.4% | 4528 blk | 712 blk/s | **6.4 s** |
| b32_lazy_s1 (60 G) | 75.2% | 4066 | 528 | **7.7 s** |
| d180_lazy_ctl (180 G) | 76.8% | 3801 | 357 | **10.7 s** |
| c32_eager_l180 (180 G) | 77.3% | 3722 | 298 | **12.5 s** |

Reuse clock is ~60 s (client gap p50 1.5 s + queue wait 51-67 s, record 7).
Blocks die 5-10x before the reuse arrives, which is why X ~ 0. This is
structural, not incidental.

Correcting a framing error of mine that the user caught: **entering the free
queue is a supply event, not a pressure event.** `BlockPool.free_blocks` only
decrements `ref_cnt` and appends those that reach 0 -- no pressure test
anywhere. Eviction happens in `get_new_blocks`, on the allocation path.
Pressure does not push blocks into the queue; it pulls them out.

And `is_free()` (`eviction_aware.py:243`, `ref_cnt == 0 and not is_null`) is
**exactly** "the last request holding this KV finished". A block shared by
three requests enters the queue only when the third releases it. The signal the
user wanted already exists, is O(1), and is strictly better than a
`request_finished` hook because preemption and abort also route through
`free_blocks`.

vLLM's L0 eviction is already correct for this access pattern:
`ordered_blocks = reversed(req_blocks)` (tail first, so a chain's prefix
outlives its suffix) plus `free_blocks(uncached_blocks, prepend=True)`. A
hypothesised prefix-orphaning inefficiency does not exist.

## 4. The residence ladder and the binding-band rule

Same yardstick applied to every tier, `residence / reuse clock`:

| tier | residence | ratio | does policy bind? |
|---|---|---|---|
| L0 free queue | 6.4-12.5 s | **0.11** | no -- nothing survives, Belady would also fail |
| **L1 @ 60 G** | 39-46 s | **0.77** | **yes -- marginal entries decided by policy** |
| L1 @ 180 G | 422-707 s | **11.8** | no -- everything survives, policy irrelevant |

L0's ceiling is capacity, not policy: the **whole** GPU pool is 16384 blk /
712 blk/s = **23 s** of time-capacity against a 60 s requirement, 2.6x short.
The pool is a harness parameter (`--gpu-memory-utilization 0.60
--num-gpu-blocks-override 16384`, POOL_GIB=24 on a 143 GB card); raising
utilisation to ~0.95 would give ~52,000 blocks = 73 s and push L0 past the
clock for the first time -- at the cost of moving value back from L1 to L0.

**This band rule is the single most operationally important result of the
session:** L1 eviction-policy work cannot be validated at 180 G, because
capacity does not bind there and any policy ties with LRU. Validation must run
at 60-100 G.

## 5. Request shape, and the two segments of a block's GPU life

From `aiperf.log` (b32_lazy_s1, 427 requests):

| | mean | p50 |
|---|---|---|
| Input Sequence Length | **53,303 tok** | 55,197 |
| Output Sequence Length | **540 tok** | 256 |
| Request Latency | **80.9 s** | 82.2 s |
| Time to First Token | **66.6 s** | 71.0 s |
| Inter Token Latency | 27.4 ms | 23.0 ms |
| Tokens In Flight | 948 K | **1,018 K** |

Decomposition of the 81 s: queue ~52-60 s (**~72%**), prefill ~5-15 s,
**decode 14.8 s** (540 x 27.4 ms, and independently 80.9 - 66.6 = 14.3 s).

A block's GPU life splits into two segments with **categorically** different
risk:

| segment | duration | why |
|---|---|---|
| **1**: allocation -> ref_cnt 0 | ~20 s | ref-held; eviction is *physically impossible* |
| **2**: in the free queue | 6.4-12.5 s | racing every other allocation |

Segment 1 is 72% of the duplication window and is **risk-free by construction**.
Segment 2 is 28% and carries the entire 44%/9.8% bill.

Where the three policies act:

| | acts at | later than eager by |
|---|---|---|
| eager | during prefill | 0 |
| emit-on-free-queue-entry (untested) | end of segment 1 | **+14.8 s** |
| current lazy (danger depth) | end of segment 2 | **+22.5 s** |

## 6. Transfer measurement

`GPU KV cache size: 262,144 tokens` from 23.39 GiB confirms **96 KB/token**.

Store-time scatter (b32_lazy_s1, 423 ops) has a clean lower envelope that
recurs *exactly* across a 300x size range:

| span | secs | implied |
|---|---|---|
| 256 | 0.002 | **12.6 GB/s** |
| 512 | 0.004 | **12.6 GB/s** |
| 32,512 | 0.253 | **12.6 GB/s** |
| 77,824 | 0.606 | **12.6 GB/s** |

So full-transfer bandwidth is **12.6 GB/s = 128 K tok/s**, and fixed cost is
**~0** (256 tok in 2.0 ms is exactly 256/128K, with no headroom). Implied
bandwidth p50 = 119 GB/s means the median op only moves ~10.6% of its nominal
span -- `Stored N tokens` reports the *span*, not the bytes moved.

Retrieve (g4F, 180 G): n=500, 24.8 M tokens, **11.9 s total**, mean 24 ms,
p50 23 ms, max 84 ms, 2.09 M tok/s. Against a TTFT p50 of 31,448 ms that is
**0.07%**. Prefetch is dead -- there is nothing on the critical path to move.

Step quantum: `drain_steps` 63,645 / 1800 s = 35.4 steps/s = **28.2 ms/step**.
Therefore:

> **A store is 15 steps long. The burst that kills it is 1 step wide.**

No per-step control can react to that; you either hold the block or start early
enough that the expected number of bursts during the transfer is ~0. Two
independent routes agree on the resulting exposure: `transfer/residence` =
0.4-5.5%, and `per-step death rate x 15 steps` = 0.54%.

## 7. Corrections made this session

Recorded because several are load-bearing and appear in earlier records.

1. **"decode ~ 85 s" was wrong; decode is 14.8 s.** 85 s is the whole turn.
   Lazy's clock channel is therefore ~15 s, not 85 s, and the benefit split is
   clock:capacity ~ **2:1**, not 12:1.
2. **"37% of L1 duplicates GPU" retracted.** The method (retrieve rate x window)
   double-counted repeated reads of the same entry. The valid bound is
   `duplication <= GPU pool = 262 K tok` = 56% of L1@60G, 21% of L1@180G.
3. **"fixed store cost 4.8 ms" retracted; it is ~0.** The 4.8 ms came from
   percentile-pairing across a dedup-contaminated distribution. There is
   consequently **no break-even prefix length** -- storing is proportionally
   worthwhile at any size.
4. **beta corrected 89 -> 10.6** (128 K tok/s store vs 12,127 tok/s prefill).
   Gate 3's reuse threshold moves 1.1% -> **9.4%**. Still a constant for agent
   trajectories, but the margins quoted earlier were 8x too generous.
5. **"single-object guarantee helps eager equally" retracted for the store
   side.** Store-side exclusivity (do not place in L1 while GPU holds it) *is*
   lazy offload; eager structurally cannot do it. Only the retrieve side is
   tier-agnostic.
6. **"storing less buys nothing" retracted.** `T = C/R`: storing 23% less lifts
   60 G residence from 46 s past the 60 s clock. b32 measured exactly this
   mechanism -- lazy stored 25% less, residence 39 -> 46 s, hit rate
   2.7% -> 13.2%.
7. **"eviction-side filtering strictly dominates admission-side" retracted.**
   LRU grants every new entry a full residence T regardless of merit, so
   deferring the decision to eviction filters T seconds too late.
8. **"don't take segment 2" softened.** Segment 2 is takeable *by a clock*, not
   by a rank; see section 9.
9. **Record 7's "the filtering it bought was the 9-22 ops still pending at
   shutdown"** overstates: those ops are *unresolved at truncation*, not proven
   filtered. True filtering is currently unmeasured (no `gpu_served` counter).

## 8. The three-way split

The store decision, correctly costed, contains one decision and no more:

| category | content | nature |
|---|---|---|
| **invariant** | one object system-wide | lookup, always right, no probability |
| **constraint** | prefix closure, block integrity (A4) | structural, not a choice |
| **the only decision** | will it be reused | prediction, largely unknowable |

Gate 1 and gate 3 both collapse to constants. Gate 2 is the only one with
content, and its two implementations are an explicit client signal
(session/task end -- needs a new interface) or the next tier's residence
(already running, never evaluated as such).

Gate 2's implicit form is itself band-limited: it needs
`S = L1 residence / trajectory lifetime >= 1`, and trajectory lifetime is
~10 requests x 81 s = **810 s**. At 60 G S = 0.057 (dead content is evicted
long before it dies); at 180 G S = 0.87. Threshold is `C > R x 810 s` =
1.64 M tok ~ **205 GB**. Below that there is nothing for a reuse predictor to
filter, however accurate.

## 9. What survives, and the proposed controller

Killed by measurement this session: gate 1, gate 3, gate 2 below ~205 GB, L0
eviction policy, L1 eviction policy at 180 G, prefetch, and the deferral
machinery itself (drops 44% -> 3.1% and still lost).

Alive:

| | magnitude | confidence |
|---|---|---|
| single-object guarantee (store side = lazy; retrieve side untested) | 60 G ceiling 42% -> 67% | high |
| L1 eviction, **only in the 60-100 G band** | 13.2% -> ~20% | medium |
| cache-aware scheduling | 13% -> 60-80% | low, needs upstream |

The last is the only lever that moves the *denominator* of
`R = residence / (gap + queue)`, of which 97% is queue. Its evidence is that
the same code at two L1 sizes sits at two equilibria: **eager180 TTFT p50
31,448 ms with 642 requests completed, eager60 70,357 ms with 446** -- 2.24x
TTFT and 1.44x throughput, from hit rate alone.

### Proposed controller

The user's stated goal: adaptively decide the minimum blocks to emit per step,
to cut store-side L1 pressure and L0/L1 duplication. Steady state forces
`emit rate = arrival rate`, so per-step *volume* is not reducible; what is
adjustable is *phase*. The two goals therefore need two mechanisms:

- **store pressure** <- lookup-before-emit (skip chunks already in L1; this is
  the single-object guarantee at chunk granularity, and it also yields an exact
  byte budget, removing the 14x uncertainty in transfer time)
- **duplication** <- deferral, window `27.7 s -> alpha x residence`

```
each drain step:
  R = EMA(blocks allocated this step)        # exists
  D = num_free_blocks()                      # exists, O(1)
  residence = D / R                          # observed, in steps
  T_safe = alpha * residence - N_xfer        # N_xfer = 15 steps
  due = [op for op in backlog if op.age_steps >= T_safe]   # EDF by age
  for op in due:
      absent = [c for c in op.chunks if not l1.contains(c)]
      emit(longest prefix of absent fitting the byte budget)
```

Three properties that distinguish it from danger depth:

1. **Age-triggered, not rank-triggered.** This is the actual fix for the 44%.
   A rank predicate is a *conditional* trigger: with a 49.6-block window and a
   2774-block burst, blocks pass from outside-the-window to evicted without ever
   being observed as due -- they are **skipped, not merely late**. An age
   deadline fires unconditionally; every block gets its shot.
2. **alpha is the single monotone knob.** alpha -> 0 is emit-on-entry (loss ~
   the 15-step transfer exposure, 0.5%); alpha -> 1 is today's behaviour.
3. **The loop closes on measured loss, not predicted danger.** AIMD on
   `dropped_evicted / admitted` against a 0.5% target: back off x0.7, probe
   x1.05. This is what makes it work "under all conditions" without any burst
   model -- 180 G converges to a small alpha on its own, 60 G climbs.

Delete with gate 1: `_FreeQueueWindow`, `_danger_depth`,
`free_queue_block_ids`, `est_next_step_blocks`, `announce_allocation`, pin
cascade, `_DegradeRegime` and its trial/probe/backoff, the loss ledger,
`observe_step`'s rate model. Keep: `is_free()`, `_snapshot_intact`, prefix
closure, batching, epoch/failure handling, dedup, and
`max_drain_blocks_per_step` as a PCIe-spike guard only (`throttled_drains=0`
today -- the caps have never bound; emission is 0.047 ops/step against a cap
of 64).

Note on volume: `emitted/admitted` is the clean discriminator between deferring
and dropping. b32_lazy_s1 is at **86.7%** -- the current design has been doing
unintentional "store less" via drops, and part of the 60 G win came from that.
The deliberate replacement should be **cold insertion / segmented LRU**, not
dropping: it captures nearly all of the capacity benefit while the cost of a
wrong call is ~0 (a wrongly-cold entry can still be hit and promoted), which
matters precisely because no accurate reuse predictor exists. Its own operating
band is residence in [1.5, 3] x clock ~ **80-160 GB**, which coincides with
record 7's 70-100 G favourable band.

## 10. Open items

- **Unresolved inconsistency, blocks the byte budget.** `store_secs` implies
  only ~12% of nominal volume actually crosses PCIe (199 GB moved vs 1.62 TB
  nominal), which would put distinct L1 insertion at ~1123 tok/s and L1
  residence at **383 s**, not 46 s. But `l1_watermark_events=154` implies an
  eviction rate of ~7,385 tok/s, consistent with 46-58 s. **12.6 GB/s and 46 s
  cannot both be right as used above.** Either `store_secs` does not cover the
  full copy, or the dedup-fraction inference is wrong. Resolve before trusting
  `N_xfer = 15 steps` in the controller.
- `gpu_served` counter (pending op's block leaves the free queue with its hash
  intact) is still not implemented. It would turn the >= 99.2% bound into a
  point estimate. Not worth a run at this X; worth it on a workload where X is
  non-trivial.
- `f` (share of L1 held by trajectories not extended for > 900 s) unmeasured.
  It is the ceiling on any gate-2 work and needs no NLP or new interface.
- `fifo` + `lazy_offload_threshold=1` (emit at request end, zero code change)
  still never run at conc 32. It is the cheapest probe of the alpha -> 0 point.
- The text signal the user proposed for gate 2 (output ends in a tool call vs a
  final answer) **cannot be validated on this harness**: the scenario injects
  `ignore_eos:true`, so there is no real finish_reason or tool-call structure.
- Screening metric for a new scenario, before building anything:
  `G = X / (beta * d)`, build only if `G > 1`, with a prior absolute gate of
  `X > 5%`. Measured: 60 G G = 2.6e-4, 180 G G = 1.9e-4, g4F lazy180 G = 1.0e-2.
  Diagnostic companion `R = residence / clock` decomposes *why*.
