# A deadline on deferral

Record 2 section 5 leaves the policy holding a store for a mean of 248 s while
the workload comes back for it in seconds. Every knob the policy has is a
spatial signal about the free queue, and none of them can express a deadline.
This adds one, and it is the first change all session whose predictions all
held.

## 1. The change

`LazyOffloadPolicyConfig.max_deferral_seconds`, default 0.0 (off). A request
whose oldest surviving pending operation has waited longer than the bound is
due regardless of where its blocks sit, is decided before the window-driven
candidates (`_OVERDUE_RANK = -1` sorts ahead of every free-queue rank and is
always below the emission threshold), and releases its whole surviving front.
The deduplication-hole cut, gate 3 and the drain budget all still apply, and it
fires on a step whose danger depth is zero without walking the free queue.

Seconds, not drains. Drain rate is a property of the engine, not the workload:
it measured 8.9/s in `n60floor`, 15.1/s in `n24floor` and 20.9/s in
`n24defer30`. A bound in drains means a different wall-clock deadline at every
operating point, which is the same category of mistake as using free-queue
proximity to track a reuse interval.

The policy still reads no clock. `PendingStoreOp.admitted_at_time` is stamped
from the last timestamp the caller passed to `observe_step`, and
`lazy_offload_pending_store.py` passes `time.monotonic()`. Unit tests stay
deterministic.

`LazyOffloadCounters.emitted_overdue` records how many operations the deadline
released rather than the window. Read against `emitted` it says which of the
two clocks is binding.

Config key `lmcache.mp.lazy_offload_max_deferral_seconds`. Harness knob
`DEFER_SECS` in `par/env.sh`, wired into the connector string in `par/up.sh`.

11 new tests; 261 lazy-offload tests green; ruff clean.

## 2. Choosing 30 s

`n24floor`'s reuse clock, predecessor completion to this request's scheduling:
p25 9 s, p50 235 s, strongly bimodal. A 30 s bound covers 71 % of reuse pairs
and a 10 s bound covers 73 %, so there is nothing to buy below 30. The roughly
27 % that return inside 10 s cannot be served by any store deadline and belong
to the GPU prefix cache.

## 3. Result

`n24defer30` against `n24floor`, same config apart from `DEFER_SECS=30`.
Predictions were written to `n24defer_predictions.md` before launch.

```
                     n60floor   n24floor   n24defer30
mean deferral          248 s      200 s       25.0 s
emitted_overdue / emitted  -          -      932 / 1013
drop_rate              32.0 %     45.2 %      12.8 %
recompute share        35.0 %     23.4 %      11.9 %
external share         51.4 %     53.8 %      73.8 %
TTFT p50                272 s       83 s      44.2 s
requests in the window    142        121         220
0-120 s bucket hit       43 %       25 %        70 %
```

Q1 (emitted_overdue > 0), Q2 (drop below 30 %), Q3 (0-120 s bucket above 45 %)
and Q4 (recompute below 18 %) all held.

The cleanest statement: recompute barely moved, 5,069,172 to 5,167,056 tokens,
while presented prompt went from 21.6M to 43.6M. The same prefill compute
served twice the traffic.

92 % of emissions were deadline-driven, which says the free-queue signal almost
never fires first at this operating point.

Q5, the stated risk, came true without binding. Profiling L1 eviction triggers
went 1 to 3. `l1_gib` ended lower (378.19 against 446.65) and Q4 passed anyway.
The working set grew because throughput grew: 29 conversations at a mean
context of 209,820 against 26 at 184,783.

## 4. What this arm does not establish

`n24defer30` completed 82 % more requests than `n24floor`, so only shares are
comparable between them, not absolute volumes. A strict pair needs the request
count aligned.

The short end still misses 30 %. The 0-120 s bucket is now 164 of 220 requests
and hits at 70 %; the remainder return inside about 10 s, which no store
deadline reaches.

And with a 25 s mean deferral the policy is behaving almost eagerly. It
performs best in the configuration closest to eager, which is a result about
where the value is, not a vindication of deferral.
