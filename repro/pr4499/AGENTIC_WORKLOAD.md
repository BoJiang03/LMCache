# Agentic session replay: real SWE-agent traces

## Purpose

The reported hot/cold and QASPER results cover a synthetic capacity stress
test and a real long-context workload where a returning user re-sends one
*fixed* prefix. Agent serving is a third shape, and the one that puts the
sharpest question to a lazy-offload policy: a session's prompt **grows
monotonically**, each step extends the previous step's prompt with an action
and a tool observation, and the session then idles while the tool runs. Every
step's KV is written once and read again a few seconds later -- exactly the
window in which the policy must decide whether a lower-tier copy is worth
making.

No production source was changed. The harness is
[`agentic/`](agentic/README.md) in this reproduction package.

## Dataset and workload

The source is the published `nebius/SWE-agent-trajectories` dataset: real
SWE-agent runs against real GitHub issues, recorded as the full conversation
(system prompt, issue text, then alternating actions and command output).

- shard: `data/train-00000-of-00012.parquet`, read in file order;
- cohort: the first 48 trajectories that satisfy every selection guard, one
  per issue;
- cohort SHA-256:
  `04243c35b216d300894f2a07ad01ad74d6cd4aa04e787065be3d85a5d2e6cf51`;
- 12 replayed steps per session; final-step prompt 8024--19937 tokens
  (median 9981); 508786 tokens over the full 48-session cohort.

Selection rejects a trajectory whose role pattern is irregular, whose
step-12 prompt falls outside the 8K--22K token window, or whose steps are not
token-prefix-stable under the serving chat template. **That 12-step cap and
token window turn out to decide the latency result** -- they put the median
request below the length at which a cache can pay for itself. Sections 1--3
report them as run; [section 4](#4-the-same-trajectories-untruncated)
removes both caps and is the result to read for whether the policy is worth
its cost. The engine's own
prompt-token count matched the cohort's tokenizer count for **every** step of
every run reported here, so the replay sent exactly the prompts the selection
reasoned about.

The engine's answer is discarded and the *recorded* action is appended, so
the two policies replay byte-identical request streams; a policy A/B is then
a comparison of two runs of the same workload, not of two sampling paths.

## Load model

Session `s` releases its step `k` at `t0 + (s + k * sessions) / RATE`:

- the aggregate step rate is `RATE` (2/s unless stated) at **every** cohort
  size, so the offered load is fixed while the working set is swept;
- a session's own gap between steps is `sessions / RATE` -- the agent's
  tool-execution time, 4 s at 8 sessions and 24 s at 48;
- a step that could not be released on schedule is recorded as lag. Lag p90
  was 0.0 ms in every run reported here, so no run was reshaped by
  saturation.

## Fixed settings

- production commit `8e4e851f91316bb7994be3d096966f0d1ef0b52b`;
- Qwen3-8B, TP=4 on NVIDIA H200;
- 20 GiB GPU KV pool and 40 GiB CPU L1, matching the hot/cold and QASPER runs;
- `lazy_offload_horizon_steps = 2.5` (the production default);
- 32 generated tokens per step;
- distinct KV working set = the sum of every session's final-step prompt:
  10.8 GiB at 8 sessions, 23.7 at 16, 36.7 at 24, 47.4 at 32, 69.9 at 48.

Every point boots a fresh MP server and engine. Runs are on a shared node;
each run records the compute processes on its GPUs before and after, and no
reported run had a foreign process on its GPUs. One 32-session eager run was
discarded and rerun because a neighbouring tenant took one of its GPUs
between runs; its counters matched the rerun to within 0.003 external hit
rate.

## 1. What the policy decides, and what it costs

Both repetitions run every cohort size; repetition 1 reverses the policy
order. Coverage is the fraction of queried prompt tokens served by the GPU
prefix cache or LMCache. Latency is over continuation steps (step > 0), the
steps a cache can actually serve.

| sessions | working set | coverage eager | coverage lazy | external hit eager | external hit lazy | L1 eviction cycles eager | lazy |
| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 8 | 10.8 GiB | 0.868 | 0.868 | 0.000 | 0.000 | 0 | 0 |
| 16 | 23.7 GiB | 0.861 | 0.854--0.858 | 0.196 | 0.295--0.308 | 0 | 0 |
| 24 | 36.7 GiB | 0.777--0.824 | 0.842--0.847 | 0.602 | 0.709 | 3 | 1 |
| 32 | 47.4 GiB | 0.558--0.561 | 0.781--0.801 | 0.315 | 0.682 | 16 | 4 |
| 48 | 69.9 GiB | 0.385--0.392 | 0.598--0.638 | 0.141 | 0.434 | 39 | 16 |

The cache result is unambiguous and monotone in pressure. Eager's coverage
collapses from 0.868 to 0.385 as the working set grows, because it writes
every step's KV -- including the KV the GPU is still serving -- and the
resulting churn evicts the copies that would have been reused. Eviction-aware
holds 0.598 at the same point, with 16 L1 eviction cycles against 39.

At 8 sessions the two policies serve **identical** coverage, and the ledger
shows why: eviction-aware admitted 95 operations, emitted none, and wrote
0.0 GiB to L1, while eager wrote 10.0 GiB that was never read back. That is
the intended behaviour, stated as a measurement.

The latency it costs, at the default `max_drain_per_step`:

| sessions | rep | TTFT p50 | TTFT p90 | E2E p50 | E2E p90 |
| ---: | --- | ---: | ---: | ---: | ---: |
| 8 | 0 / 1 | -8.9% / -7.0% | -8.7% / -6.8% | -61.9% / -48.3% | -86.7% / -65.6% |
| 16 | 0 / 1 | -5.6% / -14.4% | -14.4% / -7.1% | -84.4% / -66.6% | -106.1% / -91.3% |
| 24 | 0 / 1 | -16.7% / -9.0% | -2.8% / +1.0% | -54.7% / -37.2% | -46.7% / -39.6% |
| 32 | 0 / 1 | -5.9% / -8.8% | **+16.0% / +14.5%** | -31.9% / -35.9% | -6.5% / -7.2% |
| 48 | 0 / 1 | -6.8% / -0.9% | +3.7% / +10.1% | -22.6% / -20.6% | -5.5% / -2.5% |

Positive means eviction-aware is faster. The two repetitions agree
throughout, so the regression is not a warm-machine artifact of policy order.
TTFT p90 turns positive once the working set clears the L1 budget -- the
retrieval win is real and grows with pressure -- but E2E stays negative at
every size. §2 shows why, and §3 shows that it is fixable.

## 2. Attribution: the cost is the free-queue read, not the deferral

At 8 sessions the GPU pool holds every session, so no store is ever due: the
pending queue only grows and the drain never fires. Whatever separates these
variants is decision-loop cost. Decode time is split into the four quarters
of the run, because a cost that scales with queue depth must rise *within* a
single run.

| variant | TTFT p50 | decode p50 | decode Q1 | Q2 | Q3 | Q4 | E2E p50 | pending at end | L1 written |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off (no connector) | 72.5 | 98.6 | 101.1 | 99.0 | 98.1 | 96.2 | 170.9 | -- | 0.0 GiB |
| eager | 85.0 | 97.2 | 98.6 | 98.0 | 96.3 | 96.0 | 183.1 | -- | 10.1 GiB |
| lazy (`max_drain_per_step`=64) | 89.3 | 181.8 | 98.4 | 112.4 | 192.7 | 235.8 | 270.5 | 96 | 0.0 GiB |
| lazy, `=256` | 90.1 | 171.3 | 100.3 | 105.0 | 186.1 | 225.7 | 265.5 | 96 | 0.0 GiB |
| lazy, `=4` | 87.5 | **97.2** | 100.3 | 97.7 | 97.0 | **94.8** | 184.9 | 96 | 0.0 GiB |

Reading down the decode column: the connector itself is free (eager equals
`off`), and deferral itself is free (`=4` equals eager, with the same 96
operations buffered and the same zero bytes written). What is not free is the
default drain budget. `collect_due` reads the free queue to a depth of
`danger_depth + max_drain_per_step x largest pending operation`, capped by
the total pending blocks; at 64 that read grows with the queue and doubles
decode time by the last quarter of the run, and at 4 it disappears. Raising
the budget to 256 does not double the cost again, which is the cap binding.

This is a scheduler-thread cost paid **per step**, not per store, so it lands
in full at agentic concurrency, where few requests are in flight to amortize
it. Running the same 32-session cohort at four times the step rate confirms
that reading: the lazy decode penalty falls from +20% (118 ms vs 98 ms at
2 steps/s) to +13% (137 ms vs 121 ms at 8 steps/s), and E2E p90 flips from
-7% to **+41%**.

## 3. The same policy with a cheap drain budget

Re-running the three pressure points with `max_drain_per_step = 4` and
nothing else changed:

| sessions | run | coverage | external hit | TTFT p50 | decode p50 | E2E p50 | L1 cycles |
| ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: |
| 24 | eager | 0.777 | 0.602 | 86 | 83 | 170 | 3 |
| 24 | lazy, budget 64 | 0.847 | 0.709 | 100 | 140 | 264 | 1 |
| 24 | lazy, budget 4 | 0.845 | 0.713 | 97 | 100 | 202 | 0 |
| 32 | eager | 0.561 | 0.315 | 96 | 98 | 192 | 16 |
| 32 | lazy, budget 64 | 0.801 | 0.682 | 102 | 118 | 253 | 4 |
| 32 | lazy, budget 4 | 0.790 | 0.662 | 106 | 99 | 210 | 3 |
| 48 | eager | 0.385 | 0.141 | 109 | 100 | 213 | 39 |
| 48 | lazy, budget 64 | 0.598 | 0.434 | 117 | 117 | 261 | 16 |
| 48 | lazy, budget 4 | **0.657** | **0.516** | 114 | 96 | **212** | 15 |

The cache benefit survives the smaller budget at every size (coverage within
0.011 at 24 and 32, and *better* at 48, where a smaller per-step batch keeps
the drain closer to the eviction front). The decode tax is gone. At the
heaviest pressure the policy is now free: 212 ms E2E p50 against eager's
213 ms, with coverage 0.657 versus 0.385 and 15 L1 eviction cycles versus 39.
At 24 and 32 sessions a residual 9%--19% E2E gap remains, so the tuning does
not make the policy uniformly free -- it removes the part of the cost that
scaled with the pending queue.

These are single runs per point, not two-repetition results like §1.
`throttled_drains` was 2, 2 and 7 at 24, 32 and 48 sessions, so a budget of
4 did bind occasionally -- rarely enough that coverage did not suffer, and
at 48 sessions it improved, but it is not the "never binding" case. The
ledger's `throttled_drains` counter is the sensor for a cohort whose
per-step operations are larger.

## 4. The same trajectories, untruncated

Sections 1--3 cap every session at 12 steps and keep only trajectories whose
final prompt lands in an 8K--22K token window. That cap decides the result.
The paired per-request analysis of the 48-session point shows why: the
eviction-aware policy costs a fixed ~21.7 ms per request and saves ~4.27 ms
per 1000 prompt tokens, so it breaks even at about 6000 tokens -- and the
capped cohort's median continuation prompt is 5635 tokens, just below it.
More than half of every run sat in the regime where retrieving a prefix
cannot beat recomputing it, which is why coverage doubled with no median
latency to show for it.

The cap was the harness, not the data. Re-scanned with the serving
tokenizer, the dataset's trajectories run to a median of 22 steps and a
maximum of 158, and their final prompts reach 34591 tokens -- inside
Qwen3-8B's native 40960 context. Every trajectory in the file passes the
role, prefix-stability and context guards: nothing has to be dropped.

### Replaying whole sessions at constant load

Trajectory length varies by a factor of forty, so one session per trajectory
would leave the run trickling: the short ones finish in minutes and the
offered load decays six-fold before the long ones are done. Instead the run
is `slots` concurrent slots, each replaying a queue of *whole* trajectories
back to back until it has issued `AGENTIC_SLOT_STEPS` steps. When a
trajectory ends the next starts on the following tick -- a new session on
the same slot, its first step cold, the finished session's KV now dead
weight in the cache. Concurrency and step rate stay constant for the whole
run and no session is shortened; only the last trajectory of a slot can be
left unfinished, identically under both policies. Queues are packed longest
trajectory first, which is what lets a 74-trajectory pool fill 14 slots of
158 steps.

- pool: every usable trajectory in the file, 74 of them -- the dataset has
  1500 rows but only 74 distinct issues, and one trajectory is kept per
  issue so sessions do not share content;
- cohort SHA-256:
  `628ef0c75745371f61dc3c02886fd8579257bdb5fff73d71eac7db3c25b4e131`;
- steps per trajectory 4--158 (median 22); final prompt 2532--34591 tokens
  (median 17471);
- 14 slots x 158 steps = 2212 requests per run, over 71 of the 74
  trajectories, 12 of which are cut by the slot budget;
- continuation prompts p25 5637, **p50 11175**, p75 18889, p90 25334 --
  within 0.4% of the whole population's percentiles, so the packing
  introduces no length bias;
- 36.7 GiB live at once, 182.6 GiB of distinct KV pushed through the cache
  over a run;
- `--max-model-len 40960`; everything else as in "Fixed settings".

Pressure is swept with the L1 budget rather than the session count, because
74 trajectories cannot fill more slots. Holding the workload fixed and
moving the budget also means every point compares the same requests.

### Result

Repetition 1 reverses the variant order; both repetitions of the two valid
budgets agree to within 0.013 coverage and 2.5 ms of mean paired E2E.

| L1 | live/L1 | variant | coverage | external hit | L1 cycles | TTFT p50 | p90 | E2E p90 | paired E2E vs eager |
| ---: | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| 40 GiB | 0.9x | eager | 0.763 | 0.658 | 132 | 153 | 366 | 477 | -- |
| | | lazy, `=4` | 0.895 | 0.850 | **42** | 159 | 285 | **377** | **-24.6 ms** |
| | | lazy, `=64` | 0.899 | 0.855 | 42 | 155 | **281** | 399 | -0.8 ms |
| 20 GiB | 1.8x | eager | 0.310 | **0.004** | 714 | 189 | 523 | 624 | -- |
| | | lazy, `=4` | **0.626** | 0.462 | 322 | **170** | 428 | **522** | **-39.4 ms** |
| | | lazy, `=64` | 0.623 | 0.458 | 290 | 170 | **422** | 529 | -23.3 ms |
| 10 GiB | 3.7x | eager | 0.308 | **0.000** | 954 | 190 | 509 | 595 | -- |
| | | lazy, `=4` | 0.398 | 0.111 | 459 | 179 | 500 | 589 | +10.9 ms |
| | | lazy, `=64` | 0.425 | 0.102 | 313 | 178 | 493 | 586 | +22.8 ms |

Negative means eviction-aware is faster; the paired column is the mean over
2141 requests whose prompts are token-identical between the two runs.

**The policy now wins on latency, not only on coverage.** At a 20 GiB
budget it doubles coverage (0.626 against 0.310), halves L1 eviction cycles,
cuts TTFT p90 by 18% and E2E p90 by 16%, and saves 74.7 s of TTFT across the
run. The gain is concentrated exactly where the model predicts -- by prompt
size, TTFT moves +11 ms at 2760 tokens, +1 ms at 9674, and **-112 ms
(-28%)** at 19134 and **-159 ms (-29%)** at 26231.

**Eager's failure mode is now unmistakable.** At 20 GiB its external hit
rate is 0.004 and at 10 GiB it is 0.000: its whole 0.31 coverage is the GPU
prefix cache, and not one token comes back from LMCache. It writes every
step of 182.6 GiB of KV into a budget that cannot hold it, and its own next
writes evict what it just stored -- 714 and 954 eviction cycles against the
policy's 322 and 459.

**The policy has a working range, and it is bounded on both sides.** At
40 GiB the budget nearly holds the live set, eager already reaches 0.763,
and the win narrows to the tail (TTFT p90 -22%, p99 -28%). At 10 GiB the
budget cannot hold even what is worth keeping: eviction-aware reaches only
0.398 coverage, its long-prompt advantage disappears (-0.3% at 19134 tokens
against -28% at a 20 GiB budget), and it loses on E2E. Deferring a store
helps when there is somewhere worth deferring it to.

**`max_drain_per_step` is still worth lowering, for a different reason than
in §2.** Here concurrency is deep enough to amortize the per-step read --
the default no longer doubles decode -- but it still costs the whole
end-to-end gain at a 40 GiB budget: identical coverage and TTFT to `=4`
(0.899 against 0.895, p90 281 against 285) and a paired E2E of -0.8 ms
against -24.6 ms.

### Why it loses where it loses

Five controls at the 20 GiB budget, each a full 2212-request run of the same
workload, separate the causes. `off` runs the engine with no KV connector at
all and is the baseline every number below is measured against; the three
`max_drain_per_step` values and the two gate-3 settings vary one thing each.
Paired means are over the 2141 requests common to every run.

| variant | TTFT | decode | E2E | coverage | throttled | dropped_evicted |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| off | -- | -- | -- | 0.31 (APC only) | -- | -- |
| eager | +27.2 | +30.5 | **+57.7** | 0.310 | -- | -- |
| lazy, `=64` | -6.2 | +40.6 | +34.4 | 0.623 | 0 | 138 |
| lazy, `=4` | -7.7 | +26.0 | +18.3 | 0.626 | 134 | 350 |
| lazy, `=1` | -2.1 | **+2.8** | **+0.7** | 0.600 | 731 | 1017 |
| lazy, `=4`, `min_prefix_tokens=6000` | -5.9 | +22.9 | +17.0 | 0.624 | 118 | 262 |
| lazy, `=4`, `min_prefix_tokens=12000` | -8.8 | +25.5 | +16.8 | 0.612 | 95 | 242 |

**Eager is worse than having no connector at all here, at every prompt
length.** Its TTFT cost against `off` rises from 6 ms at 2760-token prompts to
66 ms at 26231-token ones -- it writes in proportion to what it reads in -- and
its external hit rate is 0.004, so none of it comes back. That is 57.7 ms per
request of pure loss, and it is the baseline the rest of this document
compares eviction-aware against.

**The short-prompt penalty is not unprofitable retrieval.** Gate 3 is the
direct test, because it acts on the store side: with
`min_prefix_tokens = 12000` the policy refuses 1108 operations and emits 1548
instead of 2522, so a request below 12000 tokens provably has nothing of its
own trajectory in L1 to retrieve. Its penalty against `off` at 2760 tokens is
+16 ms -- exactly what it was without the gate. The hypothesis that short
requests lose by retrieving what they could have recomputed is refuted.

What gate 3 does instead is protect the *long* prefixes, by not spending L1 on
short ones: at `min_prefix_tokens = 12000` the 26231-token TTFT gain grows from
93 ms to **135 ms** and the 19134-token gain from 64 ms to 93 ms, for 0.014 of
coverage. That is the opposite of the effect it was reached for.

**The decode cost is the free-queue read, and `max_drain_per_step` sets its
depth.** Decode split into run octiles shows the shape:

| variant | O1 | O2 | O3 | O4 | O5 | O6 | O7 | O8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 99 | 95 | 95 | 94 | 95 | 93 | 97 | 98 |
| eager | 95 | 108 | 109 | 107 | 103 | 106 | 104 | 97 |
| lazy, `=64` | 233 | 89 | 91 | 89 | 99 | 90 | 123 | 212 |
| lazy, `=4` | 98 | 89 | 92 | 89 | 99 | 91 | 126 | 202 |
| lazy, `=1` | 98 | 92 | 93 | 90 | 92 | 90 | **92** | **92** |

For three quarters of the run eviction-aware decodes *faster* than no
connector; then it doubles. `collect_due` reads the free queue to
`danger_depth + max_drain_per_step x largest live operation`, and dropping the
multiplier from 4 to 1 flattens the curve completely -- 202 ms to 92 ms in the
final octile, +2.8 ms of decode against `off` over the whole run. Pending depth
is not the cause: `=1` ends with 81 pending against `=4`'s 101. The last-octile
prompts are also the run's *shortest* (median 7056), so it is not context
length either.

**`=1` is not free, and the trade is the point.** It throttles 731 drains and
loses 1017 admitted operations to eviction -- 28% of everything it admitted,
against 12% at `=4` -- which is the failure mode `max_drain_per_step`'s
documentation names for a cap below the number of concurrently prefilling
requests. The cost lands on exactly the requests the policy exists for: the
26231-token TTFT gain falls from 93 ms to 25 ms, and E2E p99 rises to 1031 ms
against `=4`'s 905 ms and `off`'s 933 ms. What `=1` buys is uniformity, not
depth -- its E2E against `off` stays inside -30 to +13 ms at every prompt size,
where `=4` runs +42 ms at short prompts and -74 ms at long ones.

The two are not naturally in tension. One parameter is setting both the drain
rate, which decides how much KV survives to be written, and the depth of a
read paid on every scheduler step. Bounding that read independently of the
drain budget -- by the pending operations' actual block count rather than
`budget x largest op`, or by a separate cap -- should give the 100 ms
long-prompt gain of `=64` and the flat decode of `=1` at once. Section 5 is
that change, measured.

### What did not pass

Three of the nine runs failed the preemption guard: 12 preemptions in
`lazy, =64` at 10 GiB, and 2 each in `lazy, =4` at 10 GiB and `lazy, =64` at
20 GiB. All nine eager runs and both repetitions of `lazy, =4` at 20 and
40 GiB preempted zero times. The counts are small against 2212 requests, but
they appear only under the deferring policy and only under budget pressure,
which is a mechanism rather than noise -- and it is the 10 GiB row, where
the policy loses, that carries the largest count. That row should be treated
as unconfirmed until the preemptions are explained; the 20 and 40 GiB rows
are clean in both repetitions.

## 5. The same panel, with the read decoupled from the drain budget

Section 4 ends on a prediction: `max_drain_per_step` was setting two unrelated
quantities, and separating them should keep `=64`'s long-prompt gain while
removing the decode cost. `collect_due` now opens the free-queue read at
`danger_depth` and widens it only by the blocks an emission has *already*
pinned out of the queue, instead of prepaying `max_drain_per_step x largest
pending op`; whether a pinned block was in the queue is asked of the pool
(`ref_cnt == 0`) rather than of the window, so a pin deeper than the window
still counts. Same workload, same cohort, same 14 slots x 158 steps, L1 =
20 GiB.

| vs `off` | TTFT | decode | E2E | coverage | read/step | evicted | throttled |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| eager | +27.8 | +29.6 | +57.4 | 0.310 | -- | -- | -- |
| lazy `=64`, before | -6.7 | +39.2 | +32.4 | 0.623 | ~backlog | 138 | 0 |
| lazy `=4`, before | -8.1 | +25.2 | +17.1 | 0.626 | ~backlog/16 | 350 | 134 |
| **lazy `=64`, after** | **-5.9** | **-1.9** | **-7.8** | 0.614 | **83.3** | 149 | 0 |
| **lazy `=4`, after** | **-7.9** | **-0.6** | **-8.5** | 0.622 | **79.4** | 333 | 125 |

The decode cost is gone, and E2E crosses zero: the connector is now cheaper
per request than running without one, which no configuration in section 4
managed. Decode by run octile shows it directly -- the post-change curves sit
on top of `off`, including the noise at O3 and O5, which `off` has too:

| variant | O1 | O2 | O3 | O4 | O5 | O6 | O7 | O8 |
| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |
| off | 98 | 95 | 120 | 94 | 119 | 93 | 97 | 99 |
| eager | 97 | 124 | 189 | 115 | 177 | 122 | 121 | 106 |
| lazy `=64`, before | 233 | 89 | 91 | 89 | 99 | 90 | 123 | 212 |
| lazy `=64`, after | 97 | 97 | 107 | 96 | 115 | 98 | 95 | 94 |
| lazy `=4`, after | 98 | 102 | 119 | 96 | 106 | 98 | 97 | 94 |

**The long-prompt gain survives and the short-prompt penalty does not.** By
prompt length, against `off` (TTFT/E2E, ms):

| variant | 0-4k | 4-8k | 8-16k | 16-24k | 24k+ |
| --- | ---: | ---: | ---: | ---: | ---: |
| eager | +10/+8 | +9/+16 | +24/+41 | +47/+64 | +57/+217 |
| lazy `=64`, before | +16/+73 | +19/+103 | -0/+42 | -35/-26 | -45/-54 |
| lazy `=64`, after | +12/+7 | +14/+9 | +1/-1 | -29/-22 | -40/-48 |

The +73 and +103 ms E2E penalties on prompts below 8000 tokens -- the reason
the policy could look worse than eager on short requests -- were the per-step
read, charged to every request in the run whether or not it had anything to
gain. They fall to +7 and +9 while the 24k+ gain moves only from -54 to
-48 ms, inside the drift floor measured below.

**Why the read was so overcharged.** The new cost sensors put a number on it:
across 70096 drains the policy emitted 2794 operations, **0.04 per drain**,
while the old bound sized every drain's read for 64. The read is now 83.3
blocks per step; the old bound was `min(pending blocks, 64 x largest op)`,
which under this backlog is the whole pending set -- thousands of blocks. That
figure is an estimate from the pending-op count and op sizes, not a
measurement: the pre-change runs have no counters.

**The budget is now a single-purpose knob.** `=4` and `=64` read to the same
depth (79.4 against 83.3 blocks per step) and decode the same (-0.6 against
-1.9 ms), while they still differ in exactly what the cap is for: `=4`
throttles 125 drains and loses 333 admitted operations to eviction, against 0
and 149. Before the change the same pair differed by 14 ms of decode. There is
no longer a *latency* reason to lower it from the default: `=4` buys 0.7 ms
of E2E, inside the noise floor, for 184 more lost operations. (What the cap
still prices -- the drain's transient pin under budget pressure -- is
measured in §7.)

**Drift control.** The panel spans hours, so `off` was run twice, sixteen hours
apart, on either side of the change. Paired: -1.1 ms TTFT, -0.9 ms decode,
-2.0 ms E2E, identical coverage, and no bucket beyond 5 ms except the 280-request
24k+ tail at 17 ms. That is the resolution floor for every number above.

**Still open here, resolved in §7.** One preemption remains in `lazy, =64`
(two before); §7 identifies the mechanism and the knob that bounds it.

## 6. The last milliseconds: what a short prompt still pays, and to whom

After §5, the only latency bucket where lazy trails eager is TTFT below
8000 tokens: +2 ms at 0--4k (at the drift floor) and +5 ms at 4--8k, while
the same requests' E2E already nets ahead (-1/-7 ms). Three extractions
from the §5 runs locate the cost exactly
(`agentic/attribute_short_ttft.py`).

**Part one is not the policy's.** Eager pays +7/+9 ms median TTFT against
`off` on the same buckets with an external hit rate of 0.003 -- that is
the MP connector's per-request lookup overhead, present before this PR.
Lazy's first octile, while L1 is still too cold to hit, pays the same
+7.5 ms. Everything the policy adds sits on top of a floor every
connector-bearing variant pays.

**Part two is retrieval on hits too small to pay for themselves.** The
per-request deltas are bimodal -- the fraction above 10 ms tracks the hit
fraction, near-zero against eager at step 0 (+1.2 ms median), 0.15 in the
cold first octile, 0.5--0.95 once L1 warms. The server log gives both
sides of the ledger:

| transfer | retrieval p50 | | prompt | hits <=2k tokens | median hit |
|---:|---:|---|---:|---:|---:|
| 0--2k tokens | 9.0 ms | | 0--4k | 91% | 1024 |
| 4--8k | 13.0 ms | | 4--8k | 60% | 1024 |
| 16k+ | 25.0 ms | | 16k+ | 43% | 8448 |

A retrieval has a ~9 ms floor before size matters, and what a short prompt
actually hits is almost always ~1024 tokens: all 74 trajectories share a
1242-token SWE-agent system prefix (chunk-aligned to 4 x 256). Saving a
thousand tokens of prefill while paying the floor plus a
WAITING_FOR_REMOTE_KVS scheduling round-trip is a net loss; the win mass
appears in the data exactly where the arithmetic says it must, at 8--16k
(paired TTFT p25 = -39.5 ms).

This also re-reads §4's gate-3 control. `min_prefix_tokens=12000` left the
short-prompt penalty unchanged, which looked like a refutation of
retrieval as the mechanism. It refuted a narrower claim: nothing short of
*its own session* was stored, but long sessions' stored prefixes contain
the shared 1242-token prefix, so short requests kept hitting it -- and
kept paying for a retrieval worth less than its floor.

**The consequence is a retrieve-side gate, not a store-side one.** The
policy's stores are fine; what does not pay is scheduling an external load
for a sub-break-even hit. A minimum-hit threshold in the connector's
lookup path (return no match below ~2k tokens; 91% of the 0--4k
retrievals move exactly that little) would remove the remaining gap for
every variant, eager included. That is common MP-connector code, a
separate change from this PR. Within this PR's scope the honest statement
is: the residual TTFT gap is the price of having a cache that actually
hits, eager avoids it only by hitting nothing, and E2E already nets ahead
of eager in every bucket.

## 7. The 10 GiB point retested, and the preemptions explained

Section 4 left two things standing against the policy: at L1 = 10 GiB
(3.7x oversubscription) lazy-d4 lost to eager by +10.9 ms paired E2E, and
the only configurations that ever preempted a vLLM request were
eviction-aware ones. The same four-variant panel as §5, at 10 GiB:

| variant | coverage | ext hit | cycles | vs off (TTFT/decode/E2E ms) | vs eager (E2E) | preempt |
|---|---:|---:|---:|---|---:|---:|
| off | 0.308 | 0.000 | 0 | -- | -46.0 | 0 |
| eager | 0.308 | 0.000 | 955 | +23.6 / +22.4 / +46.0 | -- | 0 |
| lazy | 0.429 | 0.121 | 298 | +13.3 / +3.6 / +16.8 | **-29.2** | 11 |
| lazy-d4 | 0.399 | 0.102 | 452 | +16.5 / +10.5 / +26.9 | -19.1 | 1 |

**The loss to eager is gone.** Lazy beats eager at every prompt length
here too (E2E -2 ms at 0--4k to -90 ms at 24k+), for the same reason as
§5: the read the old bound prepaid was largest exactly where the backlog
was largest. Against `off` the policy is still net positive cost at this
budget -- +16.8 ms E2E, decode nearly flat at +3.6 -- because a 0.121
external hit rate cannot pay back what §6 prices retrieval at. The working
range statement of §4 stands, but its floor moved: below the range the
policy loses least, rather than losing to eager. A same-night drift
control (`off` at 10 against `off` at 20, which no connector can tell
apart) pairs to -1.1/-1.9/-3.0 ms.

**The preemptions are the drain's transient pin, and the cap bounds
them.** Eleven preemptions at the default cap against one at `=4`, with
the read depth and decode now identical between the two -- the only
remaining per-step difference is how many operations one drain emits, so
the contrast is causal, not correlative. The mechanism, assembled from the
run logs (`agentic/preemption_census.py`):

- A pending operation's blocks sit *in* the free queue -- the policy
  watches them there; they cost nothing to hold. Emission is what pins
  them out of it for the duration of the D2H copy.
- Under budget pressure a drain emits whole-prefix operations in bursts
  (every one of the eleven events has 21k--116k tokens of stores within
  +/-3 s), and the burst fires exactly when allocation demand fires,
  because the demand is what pressed the free queue.
- The offered load alone never exceeds 38.2% of the pool in the sampled
  timeline -- `off` and `eager` peak there and preempt zero times -- while
  the lazy variant's sampled peaks reach 99.1%, 80.6%, 79.1%. The
  difference is the transient pin. Capping the drain at 4 cuts the peaks
  to 65.4% and the preemptions to one.
- vLLM preempts the newest request; each victim resumed and finished, the
  worst-observed cost +34 ms of TTFT, and its buffered operations were
  correctly dropped (`dropped_on_request_drop` stays closed in the
  ledger).

Big drain waves are necessary but not sufficient -- windows that large
occur in half the run without incident; the preemption needs the
collision with an equally bursty allocation demand, which is why the
count is 11 in 2212 requests and not hundreds. The knob trade this
restores is stated under Verdict: the default cap keeps the -29.2 ms E2E
win and accepts eleven ~30 ms victims; `=4` trades 10 ms of the win for
their absence.

## Verdict

1. **The policy's decisions are right in the agentic shape, at every prompt
   length.** It holds coverage where eager collapses -- 0.626 against 0.310
   at a 20 GiB budget with whole trajectories, 0.598 against 0.385 at a
   69.9 GiB working set with 12-step ones -- writes less, and cuts L1
   eviction cycles by more than half. Where the budget cannot hold the live
   set at all, eager's external hit rate is 0.000 and every byte it writes
   is wasted.
2. **Whether that converts into latency depends on prompt length, and the
   first cohort was too short to show it.** The policy costs a fixed
   ~21.7 ms per request and saves ~4.27 ms per 1000 prompt tokens, so it
   breaks even near 6000 tokens. The 12-step cohort's median continuation
   prompt was 5635 tokens and its median request therefore could not win;
   the untruncated one's is 11175, and the same policy cuts TTFT p90 by 18%,
   E2E p90 by 16%, and long-prompt TTFT by 28--29%.
3. **The policy has a working range, and outside it the failure is now
   graceful.** It needs a lower tier large enough to be worth deferring to:
   at a budget that nearly holds the live set the win narrows to the tail,
   and at 0.27x the live set a 0.121 external hit rate cannot pay back
   retrieval. But where §4's harness had it losing to eager at that floor,
   the decoupled read (§7) has it beating eager at every prompt length and
   every budget tested -- below the working range it loses least, against
   the one baseline nothing beats there (no connector at all).
4. **`max_drain_per_step` was priced for the drain, not for the read that
   sizing it implied -- and that was the whole cost.** One parameter set
   both the rate at which buffered KV is written and the depth of a
   free-queue read paid on every scheduler step, so raising it for the
   drain bought a per-step cost charged to every request. Reading only as
   deep as a drain's own emissions require (§5) removes it: paired against
   no connector at all, E2E goes from +32.4 ms to **-7.8 ms** at the
   default cap, decode from +39.2 ms to -1.9 ms, and the +73/+103 ms
   penalty on sub-8000-token prompts -- the reason the policy could look
   worse than eager on short requests -- falls to +7/+9 ms while the 24k+
   gain holds at -48 ms. The mean drain emits 0.04 operations; the old
   bound sized every one of them for 64.
5. **With that separated, the cap is a real trade with a right default.**
   `=4` and `=64` now read to the same depth and decode the same, so the
   knob no longer buys latency -- what it prices is the drain transient.
   At 20 GiB the two differ by 0.7 ms of E2E, inside the drift floor. At
   10 GiB the default cap wins 10 ms more E2E but lets a bursty drain pin
   enough blocks mid-copy to preempt 11 requests in 2212 (~30 ms each);
   `=4` reduces that to one at the cost of 186 more operations lost to
   eviction (§7). The default stays right -- the expected preemption cost
   is ~0.15 ms per request against a 10 ms win -- but an operator who
   cannot tolerate preemptions now knows which knob bounds them and what
   it costs.
6. **The short-prompt TTFT residual is the price of hitting, not a defect
   of deferral** (§6): a ~9 ms retrieval floor spent on the 1024-token
   shared prefix that is all a short prompt has to hit. It is removable
   for every variant at once by a retrieve-side minimum-hit gate in common
   connector code, outside this PR.

Both §4 loose ends are now closed: the 10 GiB loss to eager was the same
prepaid read as everything else and reverses under the decoupled bound,
and the eviction-aware-only preemptions are the drain's transient pin,
bounded by the cap (§7). Nothing here contradicts the reported hot/cold or
QASPER results.

## Guards and raw data

The 22 sweep runs, the 5 attribution runs and the 3 budget runs of
sections 1--3 pass every guard in `agentic/validate_agentic.py`: exact
request counts, zero failed steps, prompt-token counts identical to the
cohort's, a closing lazy ledger, no tracebacks, and zero vLLM preemptions.
Of section 4's 13 runs, 10 pass; the three that do not are named in "What
did not pass", and all three failures are preemption counts. Section 5's
four runs pass every guard except one preemption in `lazy, =64`; §7's four
pass except the preemptions the section itself is about (11 in `lazy`, 1
in `lazy-d4`, zero in `off` and `eager`). Schedule lag p90 was 0.0 ms in
every run reported here, so no run was reshaped by saturation.

Both cohorts (100+ MiB of trajectory text) stay on RAID; their identity,
per-session token profile and instance list are in
`results/agentic/cohort_manifest.json` and
`results/agentic_full/cohort_manifest.json`.

Reproduce sections 1--3 with `agentic/run_agentic_sweep.py` and
`agentic/run_attribution.py`, and sections 4--5 with
`agentic/run_full_replay.py`; the tables come from
`agentic/agentic_table.py` / `agentic/attribution_table.py`, and section 5's
from `agentic/analyze_panel.py` (paired per request, so it needs the runs to
share a request stream). Section 6's attribution is
`agentic/attribute_short_ttft.py` (run JSONs plus the policy run's server
log) and §7's event analysis is `agentic/preemption_census.py` (vllm log
plus server log). See
[`agentic/README.md`](agentic/README.md). Raw per-run JSON (every request
record, counter delta, L1 series and ledger) is under `results/agentic/`,
`results/agentic_full/` and `results/agentic_decoupled/`; the latter also
carries the rendered panels (`panel_l20.txt`, `panel_l10.txt`) and the
two-run drift control (`drift_control.txt`).
