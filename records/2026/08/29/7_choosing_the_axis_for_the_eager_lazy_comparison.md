# Choosing the axis for the eager/lazy comparison

Record 6 closed the tuning line and froze a base configuration. The next
question is the one the whole project exists to answer, whether deferring
stores buys anything against storing eagerly. This record is the design, not
a result. Nothing here has been run.

## 1. A launch that should not have happened

I built a matrix and started two arms on it before the design had been
discussed. The user stopped it. Both arms were killed during bringup, both
slots were taken down clean (GPUs back to 4-8 MiB, no ports held), the two
output directories were removed, and the waiting task was stopped. No arm
reached a profiling window, so nothing was measured and nothing was
discarded.

Recording it because the same reflex produced the c60/c84 launch earlier in
the day, which was also unasked. The pattern is starting work during the
window where the design is still being decided. The cost here was 3 minutes
of GPU time; the cost of being wrong about the design is 2 hours.

## 2. The base, frozen

From record 6 section 8: fp8, native 262,144, the 256k corpus, TP=2,
CONC=72, `BLOCK=64 FLOOR=2048`, HORIZON 2.5, 1800 s, no `--unsafe-override`.
`f8k256c72b64` is the lazy `DEFER_SECS=30` cell of the matrix and does not
need rerunning.

## 3. Vary L1, not CONC

Both are available. They are not equally useful.

CONC is a confound, not a variable. Record 6 section 1 measured a 40-point
jump in ext hit between 1.49x and 1.78x slots. Any policy difference placed
inside that jump is unreadable, because lazy occupying fewer GPU blocks
feeds back into the branch itself. A win on the CONC axis cannot be
attributed to the policy.

L1 is the clean axis, on two grounds.

It does not move the branch. In the bf16 era `n14L576` and `n14L256` sat on
the same branch with L1 cut by more than half, so changing L1 changes what
the policy acts on without changing the regime it acts in.

It is the axis the mechanism lives on. The claim being tested is reduced L1
write pressure and reduced L0/L1 duplication. With L1 unbounded, storing
everything is free and selectivity is worth nothing by construction. With L1
against its working set, eager writes entries that LRU evicts before they
are read, and selectivity is worth something. L1 tightness is the
independent variable of the effect.

## 4. The levels

L0 is fixed at 4,077,968 tokens x 49,152 B = 186.67 GiB of fp8 KV. The
current 320 GiB is already loose: its 0.80 watermark is 256 GiB and c72
ended with 226-233 GiB of live objects, touching the watermark 9 times. So
the levels go down from there, and all three stay inside the `[1,3]` ratio
the user asked for.

| L1 | ratio to L0 | 0.80 watermark | against the 226-233 GiB working set |
|---|---|---|---|
| 192 GiB | 1.03 | 154 GiB | well under, genuinely tight |
| 256 GiB | 1.37 | 205 GiB | at it, critical |
| 320 GiB | 1.71 | 256 GiB | over, loose |

Six cells, eager and lazy at each level, minus the one already measured, so
five arms in three rounds of about two hours. Both slots run the same L1
within a round, which caps pinned host memory at 640 GiB against 1,482 GB
available.

Not sweeping both axes. Twelve cells is over four hours to re-measure a
branch structure record 6 already has. If load sensitivity turns out to
matter, the cheap version is one extra round at the tightest L1 with
CONC=60, which asks whether the effect survives off the knee.

## 5. What the comparison has to measure

The standard is the one set when the line opened: store less, lose no block
that is later needed, and do not give back the latency advantage.

The headline is tokens written to L1 during the profiling window.
`f8k256c72b64` wrote 7,802,880 over 2450 store events.

That number needs splitting, because eager and lazy do not have the same
store semantics. Lazy declines to write for two different reasons and the
ledger separates them: `covered_prefix_tokens_skipped` (1,193,728 in the
reference arm) is a write avoided because a longer prefix already covers it,
which is duplication genuinely removed. `dropped_evicted` and
`dropped_deferral_drains` are writes that never happened, which is not the
same virtue. Eager has neither mechanism. Reporting one combined "stored
less" figure would let the second kind masquerade as the first.

Latency is judged on `waiting_mean` and `tpot`, whose run-to-run floor
record 6 section 8 put near 2 percent. Not on TTFT p50, which moved 9.23,
6.43, 8.50 s across three replicates of one configuration.

## 6. Predictions, drafted and not yet scored

Written before the cancelled launch, kept in
`eager_vs_lazy_predictions.md`. E1 eager writes at least 30 percent more
than 7.8M. E2 eager's ext hit is at most 5 points above lazy's, and more
than that means lazy is dropping blocks that were going to be used. E3
eager's TTFT is no better beyond the noise floor. E4 is the falsifier for
the line: if eager stores no more than lazy and matches it on ext and
latency, deferral buys nothing here. E5-E7 covered a `DEFER_SECS` sweep,
which the L1 axis displaces; they are superseded, not scored.

E2 and E4 survive the axis change unchanged and are the two that matter.

## 7. Blocking the launch

Two questions the design cannot settle on its own.

Whether this round is a conclusion or a baseline. Record 6 section 7 item 3
notes that no arm to date has touched the exclusive move-not-copy retrieve
path, which is the second half of the method. If this comparison exists to
give that change a baseline, the axis stays the same, because exclusive move
is precisely a reduction in L1 occupancy, but `l1_gib` and an explicit
L0/L1 duplication figure have to be collected per arm rather than derived
afterwards, and that means adding to the snapshot before the first arm runs.

Whether `off` earns a slot. `arm.sh` supports it and it gives the floor,
what LMCache buys at all. It is 40 minutes and it answers a different
question than the one asked.
