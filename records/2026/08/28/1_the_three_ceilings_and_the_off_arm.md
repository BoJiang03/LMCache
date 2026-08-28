# 1 — The three ceilings: 少存 has headroom, 晚存 is capped at zero, 不漏存 is at its floor

Continues record 12. The user set this session's frame with one question:
"少存，晚存，不漏存 这三个目标我需要你给出 上限在哪，不然我怎么知道该
优化到多少才停?" Four rounds later two of the three ceilings are measured
and one of them is zero. Code on `lazy-offload-dev`: `0436a131`,
`aed6c7bc`, `0b75fa3b` on top of record 12's `80cca26f`.

## 1. Rounds i60J–i60M scored

All at 60 G, conc 32, 256 entries, 1800 s, seed 1234.

**i60J (announce × floor 8192), S1–S5.** S1 falsified (drops 149 / 189,
falsifier 150); S2 passed (announced_bursts 55 / 73 — the wiring is
alive); S3 half (20.02 M / 18.72 M against a 19.5 M target, 20.5 M
falsifier); S4 half (medD −214 / −2305 ms); S5 passed. Per deferred op the
loss was 4.1% / 5.4% against i60I floor8k's 4.2% with announce **off** —
the announce lever does nothing. Retired.

**i60K (paired identical arms + token weighting), T1–T5.** T1 not
falsified: `dropped_evicted_tokens` came to 2.75% / 4.44% of stored volume,
mean 3.3–3.7 k tokens per dropped op — 2.6% of a full context, so the
residual loss is a tail of short suffixes, **not** destroyed full-length
chains. That killed the release-time-emission proposal's volume
justification. T2 and T3 falsified: two arms of an identical config gave
drops 148 vs 250 and medD −3128 vs −1099 ms.

**i60L (NUMA fix), U1–U5.** `nvidia-smi topo -m` explained T2: slot1 is
GPU1 on NUMA node 0, slots 2 and 3 are GPU5/GPU6 on node 1, so the eager
control — 3741 store batches, the round's heaviest host-memory consumer —
had been sharing node 1 with exactly one lazy arm in every round ever
scored. Reassigning (eager alone on node 0, both lazy arms on node 1) cut
the medD spread from **2029 ms to 92 ms**. U1 not falsified, U3/U4/U5
passed. U2 falsified: drops did not fall to the quiet-slot number, they
rose to 204/254.

i60L is the best round measured: stored 18.23 / 18.16 M against eager's
22.37 M (**−18.5%**), medD **−3520 / −3428 ms**, retrieves 89 vs 22,
utilization 17.6 / 17.8% vs 2.9%, dIsl −0.0%, both arms agreeing.

**i60M (the ceiling probe), W1–W5.** Arms: FLOOR=16384 (the entire GPU
block pool) on the quiet node, eager and `off` symmetric on node 1.

| | wide (FLOOR=16384) | eager | off |
|---|---|---|---|
| tokens_stored | 18.11 M | 22.21 M | 0 |
| retrieves | 87 | 27 | 0 |
| dropped_evicted | 214 | — | — |
| medD vs eager | −2536 ms | — | **+1477 ms** |
| TTFT p50 | 68,709 ms | 72,025 ms | 72,646 ms |

W1 confirmed, W2 falsified, W3 falsified in the opposite direction, W4/W5
passed.

## 2. The off arm: the write path has no net cost

**Not offloading at all is 1477 ms slower than storing eagerly.** Eager's
3759 synchronous stores cost less on the critical path than the 27
retrieves they buy back. So `TTFT(eager) − TTFT(off)`, which record 12's
successor framing had nominated as the ceiling of 晚存, is **negative**:
there is no store-path interference to reclaim, and every millisecond lazy
takes off eager comes from the read side.

The consequence is conceptual, not just numeric: **晚存 and 少存 are not
two goals.** They are two names for one quantity — hits returned per token
written. Deferral is valuable only because it raises that ratio. Any
future argument of the form "deferral moves the copy off the critical
path" is now falsified on this workload.

Three arms in one round: off 72,646 → eager 72,025 → lazy 68,709 ms, i.e.
lazy is **−4013 ms (−5.5%)** against no cache at all.

## 3. Deferral measured for the first time (`0b75fa3b`)

Nothing had ever measured what the policy exists to buy. Each op is now
stamped with the drain counter at admission and the elapsed drains are
accumulated at both exits.

i60M's wide arm: `emitted_deferral_drains / emitted` = 616 drains =
**17.2 s** mean deferral; `dropped_deferral_drains / dropped_evicted` =
563 drains = **15.7 s**.

Two readings, both load-bearing:

- **The danger window is a nearly empty knob.** 17.2 s of deferral with
  the window opened to the *entire pool*, which should emit at the first
  sight of a block. The wait is dominated by the block still being in use
  by its running request; the window governs only the sliver after it goes
  free. This is why FLOOR from 4096 to 16384 moves nothing, and why the
  wide arm stored *less* (18.11 M) than the 8192 arm (19.80 M) instead of
  collapsing to eager as W2 predicted.
- **Dropped ops are 91% as old as emitted ones.** Not corpses that waited
  too long, and not ops that died young: they die in a one-step race at
  the moment they come due. Record 12 section 5's "invisible channel" is
  real but is the same event seen from the other side — the block's queue
  residence is shorter than one drain. Release-time emission cannot reach
  them (the request has not finished when they die); that proposal is
  withdrawn.

## 4. Why the loss cannot be zeroed cheaply

Verified in code, not from memory. `pool.touch()` runs only at emission,
on the batch being stored (`lazy_offload_manager.py:566`); pending ops are
**unpinned**, so vLLM may recycle their blocks at any time. A drop is
`_snapshot_intact` failing — the block's hash changed because it was
evicted and reallocated. The data is gone from the GPU; this is never a
"too slow to copy" failure.

Pinning at admission would make loss structurally impossible. The cost:
mean op = 18.23 M / 3193 = 5710 tokens = 357 blocks, and pending sits at
39 ops, so it would hold ~13,900 of the pool's 16,384 blocks — **85%** —
on a pool already so oversubscribed that lazy shows 116–132 preemptions
against eager's 3–4. Zeroing the loss costs more than the loss is worth.

Its worth: i60L dropped 791 k tokens; at that round's 17.6% utilization
their expected value is ~139 k tokens of hits, **5.4% of the 2.56 M extra
hits the same arm gained**.

## 5. Loss is coupled to success

Across all eight deferred arms measured so far, retrieves against drops:
55→149, 61→148, 67→153, 69→250, 72→189, 89→254, 89→204, 93→223.
**r = 0.70, slope 2.2 drops per extra retrieve, intercept 36** (n=8, so
suggestive rather than settled). Mechanism: a retrieve loads a large
prefix, allocates hundreds of blocks in one step, recycles the free queue,
and kills whatever was waiting. Preemptions move the same way (eager 22
retrieves / 4 preemptions; lazy 89 / 122).

So drops are substantially a *byproduct of hits*. Driving them toward zero
drives hits down with them, which is why "not losing blocks" is a bad
objective function on its own; the net figure in section 4 is the right
one.

## 6. The e2e value of the whole line

i60L, both arms, against the in-round eager:

- extra hit tokens 2.56 M ÷ aggregate prefill throughput 12.4 k tok/s =
  **206 s of prefill work removed from a 1800 s run = 11.4% of the
  machine's prefill capacity**
- observed total TTFT gain **6.2% / 7.3%**, median TTFT 71.5 → 67.8 s
- decode untouched (dIsl −0.0%), storage loss 4.0–4.3% of written volume

Roughly 60% of the freed capacity shows up as TTFT; the rest is eaten by
queueing non-linearity and the retrieves' own cost.

## 7. Instrumentation added this session

- **`0436a131`** `dropped_evicted_tokens` — losses weighed by token range
  at both drop sites. The op count could not distinguish a tail of short
  suffixes from a few destroyed contexts; this settled it (section 1, T1).
- **`aed6c7bc`** stop tracking `lo_temp_ctx.md`, committed by accident in
  `55612abb`; added to the worktree's local exclude.
- **`0b75fa3b`** `emitted_deferral_drains` / `dropped_deferral_drains` —
  the deferral itself (section 3). Both are weights, outside the ledger
  equation, alongside `covered_prefix_tokens_skipped`.

All three are pure observability; no policy behaviour changed, so no
earlier round is invalidated. 225 lazy tests green, ruff clean.

## 8. Where the three goals stand

| goal | today | ceiling | status |
|---|---|---|---|
| 少存 | −18.5% vs eager | fixed point of `V* = V_eager · U(L1/V*)` | **the only lever with headroom**; i60N in flight |
| 晚存 | δ = 17.2 s | write-path cost = **0**, measured negative | capped; its gains are read-side and already counted |
| 不漏存 | 214 ops / 4.5% of written volume | window width cannot improve it; pinning costs 85% of the pool | at its mechanism floor |

## 9. Round i60N (the L1 sweep): in flight

The last unmeasured ceiling. Store volume and L1 size enter the system
only through residence `T = L1_tokens / store_rate`, so sweeping L1 at a
fixed policy traces `U(T)` directly, and `U(T)`'s knee is the stopping
rule. Arms: L1 = 240 G (s1/GPU1, node 0), 30 G (s2/GPU5), 120 G (s3/GPU6);
60 G is already measured. Predictions X1–X5 in `chain_i60N.sh`: monotone
utilization; the knee inside the sweep; stored ≤ 17 M at 240 G; drops
monotone in utilization; and deferral within 20% across all three points
(a free re-test of section 3's claim that the wait is physics, not
policy). Caveats recorded in the script: no in-round eager control, and
the 240 G arm sits alone on node 0.

## 10. Open items

- Score i60N (X1–X5); the U(T) knee is the answer to the user's question.
- `max_drain_blocks_per_step` has been 0 (uncapped) in every round. Lazy
  pins 2515 blocks per in-flight store (15.4% of the pool) against eager's
  512. By block-seconds lazy is only 1.8× eager (41 k vs 23 k), which does
  not explain 30× the preemptions — but the knob exists to cap exactly
  this footprint and has never been tried.
- Preemptions 116–132 on every deferred arm against eager's 3–4, still
  unexplained; section 5 suggests it tracks retrieves rather than pins.
- Gate-3 `min_prefix_tokens` value sweep: still deferred by the user until
  the store line closes. That line is now effectively closed on two of
  three goals.
- Harness: `chain_i60K/L/M/N.sh` added; the slot-to-NUMA map is now a
  first-class experimental variable (section 1) and every future round
  must place compared arms on the same node.
