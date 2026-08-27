# Lazy TTFT tail: root cause, and a backlog cap

Follow-up to [1_l1_sweep_eager_vs_lazy.md](1_l1_sweep_eager_vs_lazy.md).
Question asked: lazy should never be worse than eager -- find out why it is.
Answer, after a measurement correction: on this workload it is not worse.
Total TTFT is at parity at every L1 size; what is left is a noisy p99 that
swings both ways, plus a real but small content-loss mechanism (sections
2-4) that the attempted fix (sections 5-6) makes worse, not better.

## 1. Measurement correction: parity, not a regression

The first version of this record reported a constant ~20 s aggregate tail
cost at every L1 size. That was a pairing artifact and it is retracted.

The paired key included `session_num`, which is the client's session slot,
reassigned run to run by arrival timing (it differs on 101 of 255 rows
between the 90G eager and lazy arms). It matched only 154 of 256 rows, and
which rows survive correlates with timing, so the subset kept regressions
and dropped the offsetting gains.

The stable key is `(conversation_id, turn_index)` -- the request's identity
in the trace. It matches 254-255 of 256 rows, and matched rows are the same
request: |d isl|/isl is 0.0000 at p50, 0.0001 at p90, 0.0074 at max. (The
dataset is multi-turn and each turn's input embeds the previous turn's
generated output, so isl is not bit-identical across runs -- which is why
the old key looked plausible: 84 of 255 rows do match exactly.)

Re-paired on `(conversation_id, turn_index)`, lazy against eager:

| L1 | pairs | med d | sum d | vs eager total | >+1s | <-1s | sum d>0 | sum d<0 |
|---|---|---|---|---|---|---|---|---|
| 200G | 255 | -2 ms | +7.1 s | 296.8 s | 8 | 7 | +29.9 | -22.8 |
| 90G | 255 | -0 ms | -1.6 s | 303.3 s | 12 | 11 | +49.4 | -51.0 |
| 60G | 255 | +3 ms | -16.4 s | 324.9 s | 13 | 17 | +63.0 | -79.4 |
| 30G | 242 | +5 ms | +8.4 s | 377.9 s | 34 | 27 | +121.9 | -113.6 |

Unpaired percentiles, lazy / eager (ms):

| L1 | p50 | p90 | p99 | max |
|---|---|---|---|---|
| 200G | 601 / 593 | 2384 / 2321 | 11696 / 11028 | 15428 / 15062 |
| 90G | 604 / 602 | 2364 / 2676 | 11978 / 8414 | 15459 / 14994 |
| 60G | 616 / 584 | 2690 / 3189 | 9389 / 9285 | 16784 / 16627 |
| 30G | 636 / 624 | 4213 / 4292 | 8384 / 12292 | 11150 / 15860 |

- Total TTFT is within +/-5% at every point, with no consistent sign
  (-16.4 s at 60G, +8.4 s at 30G). The body is exact parity: median delta
  is <=5 ms everywhere.
- p90 is better under lazy at 90G and 30G, worse at 60G and 200G.
- p99 swings hard in both directions: -3.9 s at 30G, +3.6 s at 90G. Both
  the worst regression (+9.6 s) and the biggest gain (-7.7 s) at 90G belong
  to the *same* conversation, which is the signature of scheduling luck,
  not of a mechanism.

There is one run per point, so none of the tail numbers have an error bar
yet. Getting one is the next task, and it is the same experiment as the
requested load ramp.

## 2. The cost is missing cache content

The parity above is a sum of real losses against real gains, so the loss
side still has a cause worth naming. For each request that regressed by
>0.6 s at 90G, the retrieve volume inside its own TTFT window (server
log):

| dTTFT | isl | eager retrieved | lazy retrieved |
|---|---|---|---|
| +1086 | 72381 | 13056 | 9216 |
| +1015 | 48772 | 26880 | 12288 |
| +767 | 93452 | 39424 | 3328 |
| +670 | 91191 | 13312 | 12032 |

Lazy retrieved less than eager in every measurable bad event. The cleanest
one: a 36096-token shortfall, which at the ~47k tok/s prefill rate this
model sustains is 0.77 s -- the observed delta to the millisecond. It is
not copy cost and not pin contention (the worst 90G event had `emitted`
flat at 500 and no store in flight for the whole 24 s stall). Lazy is
prefilling tokens eager loads.

## 3. Why the content is missing

`danger_depth = max(EMA_0.3(gross alloc/step), next-step estimate) x
horizon_steps`. A request whose prefix comes back from L1 has blocks
allocated for the *whole* external hit in one scheduler step -- a 73216-token
hit is 4576 blocks at once, against an EMA of single digits while the engine
was decoding. The step that would have to be predicted is the step that
produces the signal, so the window is ~0 exactly when thousands of
free-queue blocks are consumed from the eviction head.

Measured, per lazy arm: fraction of dropped ops falling within 1.5 s after a
retrieve, against the baseline probability of a random instant landing in
such a window.

| L1 | drop events | ops dropped | after a retrieve | after a >=20k retrieve | random baseline |
|---|---|---|---|---|---|
| 90G | 29 | 140 | 84 (60%) | 49 (35%) | 11% |
| 60G | 31 | 130 | 104 (80%) | 35 (27%) | 10% |
| 30G | 27 | 114 | 80 (70%) | 27 (24%) | 11% |

6-7x enrichment. Direct instance: retrieves of 73216 and 49408 tokens at
19:01:58.574/.597, drops at 19:01:58.7 (4 ops) and 19:01:59.4 (10 ops).

Amplification: `covered_prefix_tokens_skipped` is 4.26M / 4.23M / 5.06M
tokens at 90/60/30G, against 1.61M/2.0M/2.5M actually stored. Requests skip
re-staging content a pending op already covers, so a dropped cover also
orphans everything that skipped on its behalf, and breaks the prefix chain
for the skipper's own suffix. That is how ~130 dropped ops turn into
individual +1 to +4 s spikes. What section 1 corrects is only their sum:
they are offset by gains of the same size, not additive on top of parity.

## 4. The wait is net-negative here

The ledger equation closes exactly: `admitted = emitted + dropped_evicted +
dropped_on_request_drop + pending` (958 = 788+140+8+22; 986 =
835+130+8+13; 1028 = 896+114+9+9).

So the delay filters nothing that shows up as an outcome. What the wait
actually buys is the content never evicted from the GPU while the engine
runs -- anything evicted later is stored either way, just later -- which is
the 9-22 ops still pending at shutdown. Against 114-140 lost. (This says
the *depth* of the queue is not paying for itself; it does not say lazy
loses, which section 1 settles.) Lazy's real
wins are elsewhere and need no queue depth: covered-prefix skipping, L1
exclusivity, and not re-storing after an L1 eviction destroys the server's
dedup.

Pending-depth distribution (all lazy arms): p50 27, p75 48-56, p90 66-96,
max 131-137.

## 5. Fix: `max_pending_ops`

Committed as 66c64116. What cannot be forecast can be bounded: above the cap
the oldest pending ops are emitted regardless of free-queue rank, at
`max_drain_per_step` per step, only down to the cap. Age is the ordering
because prefix closure already forces front-first within a request, and a
request's front ops are both its oldest and the ones whose blocks reached the
free queue first. Prefix closure, dedup-hole cut, loss check and
one-batch-per-request are the pressure path's; requests with a batch in
flight or already emitted this drain are skipped. Default 0 = unbounded =
today's behaviour. New counter `backlog_emitted` (subset of `emitted`) is the
activity sensor; `dropped_evicted` is the sizing sensor.

210 lazy tests green (21 new). Both design docs updated. Config parse
verified end to end; counter verified live in an MP run.

## 6. Validation in flight (incomplete at time of writing)

Two arms at L1=90G, cap 32 then cap 8, against the recorded 90G eager/lazy
pair (same trace, seed 1234, so paired per-request). Scripts:
`$S/fix/{env,up,arm,run}.sh`, comparison `$S/cmp.py`.

Interim, cap=32 at admitted=787: `emitted=627 backlog_emitted=289
dropped_evicted=119 pending=32`. The cap is timing 46% of stores, and drops
are only ~8% below the baseline's ~130 at the same admission count. So at 32
the cap moves *when* half the stores happen without preventing much loss --
a real negative signal, to be confirmed against the final numbers and cap=8.

Two candidate reasons to check if cap=8 also fails to move drops:
- one-batch-per-request means a cap-driven emission blocks that request from
  responding to genuine pressure for a step or two (`blocked_request_ids`
  skips it entirely in `discover`), so a small `overflow` can emit 1-2 ops
  and then block the request through the burst that kills the rest;
- age ordering may not target the exposed set well enough; the exposed set is
  every pending op with a block in the free queue, because a single big
  admission can consume the whole queue. `is_free` is O(1), so preferring
  ops with a free block is affordable if age proves too blunt.

## Still open

- Pick the shipped default for `max_pending_ops` once the two arms land.
- 60G/30G re-validation after a default is chosen.
- "Up the intensity" (next task): raise arrival rate to saturate the GPU
  (eff_conc is only ~2.5 today, so nothing compounds), keep L1 at 30-60G,
  3 repeats for p99 (+/-40% run-to-run noise).
- Carried over from record 1: the ~1.0 s max store span is
  prepare_store->commit_store server-side, so it includes the worker's
  gather wait and the serialised commit lock; not shown to be a copy.

## Artifacts

- Baseline arms: `$S/smoke3/abt_{eager,lazy}` (200G),
  `$S/sweep/l1_{90,60,30}_{eager,lazy}`.
- New arms: `$S/fix/cap32_l1_90`, `$S/fix/cap8_l1_90`.
- Analysis scripts: `$S/paired.py`, `$S/events.py`, `$S/hits.py`,
  `$S/drops.py`, `$S/cmp.py`.
- `$S` = `/tmp/claude-1016/-home-bo-LMCache-worktrees-lazy-offloading/84352f47-e330-4d19-88ee-0abf7e23352a/scratchpad`
