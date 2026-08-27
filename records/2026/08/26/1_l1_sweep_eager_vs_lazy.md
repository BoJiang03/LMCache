# L1-size sweep: eager vs lazy across the pressure regime

Session continuation from 2026-08-25 (records/2026/08/25/7). Code is
unchanged from commit 5ea3cc6e; this session is measurement and analysis.

## Question

After the write-amplification fix reached parity at L1=200G (no pressure),
does lazy's structural benefit (exclusive GPU/L1 layering) convert once L1
is smaller than the working set? Swept L1 down through the pressure regime.

## Setup

Same workload as the 200G regression pair, one variable moved:
scenario-native timing (`inferencex-agentx-mvp`), 64 entries, 900 s,
seed 1234, concurrency 8, Qwen3-Coder-30B-A3B TP=1, GPU pool 24 GiB,
chunk 256. Working set ~147 GiB; 900 s writes ~147 GiB, so L1=90/60/30
fills at ~55%/37%/18% of the run (120G skipped: pressure only in the last
quarter of the window). `env.sh` made `L1_GB` overridable; each arm
archives its server/vllm logs before teardown (`up.sh` truncates per arm).
All 6 arms zero errors; load matched (theoretical_prefix_cache_hit 93.2%
every arm, eff_conc 2.4-2.6, total_isl ~13.3M).

## Results (eager / lazy)

| metric | 200G | 90G | 60G | 30G |
|---|---|---|---|---|
| retrieved Mtok | 1.34 / 1.46 | 1.39 / 1.43 | 1.01 / 1.30 (+29%) | 0.51 / 1.24 (+143%) |
| stored Mtok | 1.61 / 1.61 | 1.65 / 1.61 | 2.00 / 1.99 | 3.24 / 2.48 (-23%) |
| yield (retr/stored) | 0.83 / 0.91 | 0.84 / 0.89 | 0.50 / 0.65 | 0.16 / 0.50 (3.2x) |
| store secs sum | 7.7 / 4.8 | 8.1 / 5.4 | 9.1 / 5.5 | 9.8 / 5.1 |
| watermark evictions | 0 / 0 | 6 / 5 | 14 / 12 | 57 / 39 |
| TTFT avg ms | 1037 / 1071 | 1067 / 1061 | 1159 / 1091 (-5.9%) | 1425 / 1424 |
| TTFT p90 ms | 1967 / 1922 | 2290 / 1931 (-16%) | 2709 / 2160 (-20%) | 3646 / 3920 |
| TTFT p99 ms | 9378 / 11009 | 7141 / 11336 | 8806 / 7198 | 12289 / 7317 (-40%) |

Caveat carried from record 7: `tokens_stored` counts the requested range,
not chunks copied. Under eviction, requested != copied is possible for
both arms, but re-stores of evicted keys are real copies (the key is
gone, so `mode="new"` admits them). `store_secs sum` is the physical
cross-check: eager's copy time grows with pressure (8.1 -> 9.8 s), lazy's
stays flat (~5.1-5.5 s).

## Findings

1. **The exclusivity benefit converts exactly as predicted.** Hit volume
   +2.5% (90G) -> +29% (60G) -> +143% (30G). Eager's L1 mirrors recent
   traffic and overlaps the GPU pool, so at 30G its reach barely exceeds
   the GPU itself; lazy's L1 holds the complement (effective ~GPU+L1).
2. **A new mechanical disadvantage for eager under pressure: eviction
   destroys the server-side dedup.** Once a key is evicted, the next
   request over that prefix re-stores it for real. Eager's stored volume
   balloons 1.65M -> 3.24M tokens; its L1 churns harder (57 vs 39
   watermark events), which evicts more, which re-stores more.
3. **`dropped_evicted` is flat across L1 (114-140)**, confirming it is
   GPU-side pressure, independent of L1 size.
4. **TTFT: parity everywhere, real edge at 60G (avg -5.9%, p90 -20%),
   p99 -40% at 30G.** p99 remains noisy (eager's own p99 moved +42%
   between identical runs earlier), but the 30G p99 direction matches the
   mechanism: a cold return (~50k tokens) is seconds of recompute vs a
   PCIe retrieve, and that is exactly the tail.

## Why the latency win is small (the conversion-rate analysis)

TTFT conversion = (cold-return traffic share) x (compute cost per token)
x (queueing multiplier). All three take low values here:

- 93% theoretical hit is mostly within-session turn reuse served by GPU
  APC identically in both arms; L1 only serves cold returns (4-10% of
  tokens).
- A3B active params make prefill cheap (~30-60k tok/s): the extra 0.73M
  retrieved tokens at 30G save only ~12-25 s of GPU time total, ~0.05-0.1
  s per request on avg -- inside noise.
- Arrival-limited (eff_conc ~2.5): saved prefill becomes GPU idle, not
  queue relief; no compounding.

The 2.4x cache-efficiency advantage is hard; the missing part is the
conversion channel. To see avg TTFT separate: saturate the GPU (higher
arrival rate) or use a compute-dense model (dense large model, longer
contexts).

## Also this session

- **Why lazy cannot lead in every scenario (first-principles answer):
  lazy is a trade, not a pure optimization.** It saves copy volume
  (bounded by GPU/working-set, ~16% ceiling here) and L1 capacity
  (exclusive layering), and pays with worse timing (copies at the
  eviction edge = deadline work instead of idle work), a visibility
  window, and drop risk. The trade only pays where the saved resource is
  the binding constraint. A deferred-payment strategy cannot beat a
  prepaid one when the prepayment window is free (idle bandwidth, ample
  L1); best case there is parity, which is what we now measure.
- **`dropped_evicted` tunability**: horizon_steps up / max_drain up
  reduce it but converge toward eager; irreducible component is a burst
  of allocations outrunning any horizon (vLLM has no block-level pin for
  connectors; request-level delay-free only). On these runs drops did not
  cost hits (yield favors lazy at every point); leave the knobs alone
  unless the benefit regime shows drops correlating with misses.
- **Resource cleanup**: the leaked pytest MP server (port 25050) exited
  on its own. Found ~4.5 GB of leaked /dev/shm segments (psm_* transfer
  buffers 4.2G + dead-pid cuda.shm/torch ~340M, dated Aug 6-22, from
  crashed earlier runs). Bulk rm was blocked by the auto-mode
  classifier; wrote `scratchpad/clean_shm.sh` (triple-guarded: owned by
  me + live-process whitelist rebuilt at run time + not created today)
  for the user to run. The three bisect MP servers (ports
  29435/29988/27555, other session, likely live multi_modal work) left
  untouched.

## New clue: the ~1.0 s store stall is a timeout, not a copy

Every lazy arm shows exactly one store at ~1.000 s (0.999 / 1.003 /
1.002 s) regardless of L1 size or store size. A duration that precise is
a 1-second timeout or poll interval somewhere in the store path, not
data movement. This also explains record 7's "1.002 s store not
explained by size". Not yet investigated.

## Still open

- Locate the 1 s wait in the store path (grep for a 1-second
  timeout/poll in the MP store path).
- p99 comparisons need repeats before any claim; single runs move +/-40%.
- Higher-arrival-rate run at L1=30-60G to show TTFT conversion once the
  GPU saturates.
- `max_drain_per_step` counts ops not work (needs own justification);
  residual write amp components not separated; DeepSeek-V4-Flash KV
  width unconfirmed (record 1).

## Artifacts

`scratchpad/sweep/` (session 84352f47): `run_sweep.sh`, `sweep_arm.sh`,
`agg.py`, `l1_{90,60,30}_{eager,lazy}/` each with `snapshot.txt`,
`errors.txt`, `aiperf.log`, `artifacts/`, and gzipped server/vllm logs.
200G point read from `scratchpad/smoke3/abt_{eager,lazy}/`. `env.sh`
(session 7445f449) now takes `L1_GB` from the environment, default 200.

## Follow-up: why lazy is not unconditionally >= eager (paired analysis, 90G)

Paired the two 90G arms per-request on (conversation_id, turn_index, session_num): 154 pairs.

- Median paired TTFT diff: +0.8ms. The distribution body is exact parity; no systematic overhead.
- Tail: 6 pairs >+1s worse, 3 pairs >1s better. The two worst (+4.4s each: conv 14129d47/26 isl=95371, conv 3735de09/7 isl=49604) share one event at trace offset t~677s.

Event anatomy (t~677s, lazy wall clock 19:01:47-59):
- Both arms stall at the same trace offset (prompt tput 0.0, Waiting: 2, pool ~91%): the workload forces it.
- Eager: victim's 93440-token L1 retrieve fires at arrival (18:43:30.5); the same window carries 12 store commits at 3-13ms each, no contention. Copy debt was paid during decode.
- Lazy: same request's retrieve fires 11s after arrival (19:01:58.574), coinciding exactly with dropped_evicted bursts 109->123 (19:01:58.7, 19:01:59.4) and a +14k free_queue_blocks_read jump. The eviction wave's pending-covered blocks had to drain-or-drop before reuse.
- Recovery prefill tput: eager 5181/4260 tok/s vs lazy 421 tok/s in the matching windows (suggestive; window alignment confounds, the solid number is the 11s retrieve delay).

Mechanism (first principles): lazy moves copies from store-time (idle, amortized) to the
eviction edge, and the eviction edge is by definition the allocation-pressure peak -- the
same arrival burst both causes the evictions and waits for the blocks. Debt gets margin-called
at the worst moment. This is why lazy cannot unconditionally dominate. It is also why lazy
wins at high pressure: at 30G eager's own debt (write-amp 3.24M vs 2.48M stored tokens) makes
eager's storms worse (the 3 reverse pairs at 90G are eager's storm victims, e.g. -5.4s).

Notes ruled out while tracing:
- request_finished(): lazy returns False (instant free, blocks stay observable); EAGER is the
  one that returns True (delay-free until async saves land). Block-holding at finish is not
  lazy's problem.
- need_flush_before_forward -> flush_inflight_stores() only triggers on real preemptions
  (none in the event window; 2-3 preemption events per lazy run vs 1 in eager).
- The ~1.0s max store: "Stored in X s" spans prepare_store->commit_store server-side
  (engine_driven_transfer.py:472), i.e. includes the worker's gather wait (_event.wait on the
  forward stream), gather_done.synchronize(), and the serialized _commit_lock -- a measurement
  span, not necessarily a 1s copy.

Fix directions that reclaim the p99 cells without giving up dedup/exclusivity:
1. Idle trickle pre-drain: above a pool-pressure threshold, pre-pay LRU-tail ops so the
   eviction edge finds the debt mostly settled.
2. Prefer drop over drain when requests are waiting: drops cost no hits on this workload
   (yield leads everywhere), defaulting is cheaper than queueing the debt.

Next (pending user confirmation): higher arrival rate to saturate the GPU (eff_conc ~2.5 now),
L1=30-60G, 3 repeats for p99 (known +/-40% noise).
