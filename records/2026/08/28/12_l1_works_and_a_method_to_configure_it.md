# L1 works, and a method for choosing the parameters

Two rounds ran. The first produced the clearest L1 result of the project and
two operational failures. The second is running. Between them, most of the
session went into turning "configure the parameters until it looks reasonable"
into something with an inequality in it, because five consecutive attempts at
an operating point had each violated a constraint the previous one satisfied.

Three of my own conclusions are corrected here, two of them load-bearing.

## 1. The knee round: L1 96 G vs 384 G, matched

`N=20`, `--agentic-cache-warmup-duration 600`, `DUR=1800`, lazy on both arms,
only L1 differing. Working sets came out 4 % apart (497 vs 516 GiB) and
session counts 33 vs 35, so the two arms faced the same workload and the
comparison is controlled.

| | L1=96 G | L1=384 G | |
|---|---|---|---|
| rho = ws/(L0+L1) | **1.76** | **0.90** | crosses 1 |
| ext_hit mean | 0.29 % | **38.20 %** | 132x |
| retrieves | 28 | 296 | 10.6x |
| **L1 watermark events** | **104** | **14** | -87 % |
| tokens_stored | 16.03 M | 11.92 M | -26 % |
| **requests served** | **90** | **129** | **+43 %** |
| **TTFT p50** | **629.5 s** | **308.8 s** | **-51 %** |
| end-to-end p50 | 679.4 s | 346.3 s | -49 % |
| decode p50 | 5.0 tok/s | 12.6 tok/s | 2.5x |
| preempts | 9 | 4 | |
| GPU prefix hit | 14.5 % | 15.5 % | flat |

**Quadrupling L1 at identical load halved TTFT and raised throughput 43 %.**
The mechanism is legible in the watermark count: 104 crossings to 14, i.e.
from continuous eviction to a tier that holds its contents. GPU hit is flat
across the pair, so none of the gain came from the pool.

Both arms are ~117x past the production TTFT gate, so this is not a latency
claim. It is a clean demonstration that the tier does work once rho crosses 1.

## 2. Corrections

**(a) "TTFT is 95 % queue, so a cache policy can only touch the other 5 %."**
Wrong, and it was the basis for telling the user L1 could not help. From the
`/metrics` delta over the profiling window: queue 604.2 s, prefill 30.5 s,
TTFT 656.5 s. The identity holds, but `TTFT = queue + prefill` is
**accounting, not causation**. Reducing work per request raises X_max, and
near saturation the queue is superlinearly sensitive to service rate. The
measurement is the refutation: a 2.5 s change in the prefill term bought 43 %
throughput and a 320 s reduction in TTFT.

**(b) "L1 did not cure the collapse."** Said after comparing the 384 G arm to
`c2_20`, which differed in load *and* duration. Against the matched control it
plainly did help. The lesson is narrow and repeatable: only compare arms from
the same round.

**(c) "Think time is compressed 4.2x by the scenario's idle cap."** This drove
a proposal to run with `--unsafe-override`, which the user rejected; checking
it first showed the proposal was solving a non-problem. Measured directly --
per-conversation gaps from the export against the recorded gaps in the corpus:

| | p50 | p90 | mean |
|---|---|---|---|
| c2_14 replayed | 1.9 s | 39.8 s | 14.9 s |
| c2_20 replayed | 2.0 s | 57.6 s | 22.6 s |
| corpus recorded | 3.3 s | 59.1 s | **251.8 s** |

p50 and p90 replay to within 1.5-1.8x. **There is no 4.2x compression.** The
discrepancy is entirely in the mean, and the corpus's own p50/mean split
(3.3 s / 251.8 s) says why: the mean is carried by a tail of hours-long idle
periods, which a 900-1800 s window cannot contain. They are not compressed,
they are outside the window.

`system_idle_gap_cap_seconds` only shifts timers when **nothing** is in flight
(`agentic_replay.py`, and the cap is hard-locked in `validator.py:748` --
explicit disagreement is a violation, bypassable only with
`--unsafe-override`, which stamps `submission_valid=false`). At in-flight 3.60
the system is essentially never globally idle, so the cap never bit. And what
it prevents is the replay fast-forwarding dead air, which would inflate
throughput -- it protects the quantity we care about, in the direction that
would have flattered us.

Which also settles the duty cycle: the corpus's 5.27 % is a lifetime average
including nights; **our 24 % is the busy-period value, and busy-period is what
sizes a node.** 24 % is the right number.

## 3. Two operational failures

**`--benchmark-grace-period` doubles as the accelerated-warmup drain cap.**
`_wait_for_accelerated_warmup_handoff` (`timing/phase/runner.py:939-941`) uses
`self._config.grace_period_sec` as the timeout. With `GRACE=600` and cache
warmup on, the first 96 G arm cancelled 4 stragglers and aborted the whole run
(`Terminal warmup failure ... aborting run early`). Relaunched at `GRACE=1800`
and it completed. The 384 G arm drained inside 600 s, which is itself a data
point about the two configurations.

**Grace is a fixed wait, not a cap.** Both arms sat idle until the grace clock
expired even with `in_flight=0` and everything returned (`Phase profiling timed
out ... completed=129, cancelled=0, in_flight=0` at exactly T+600). `GRACE=1800`
therefore burns 30 minutes per arm. Back to 600 now that cache warmup is out.

**Cache warmup is not load-neutral, and I dropped it.** It couples offered load
to server throughput: the accelerated phase runs for a fixed *duration*, so a
faster server advances every lane further through its recorded timeline, and
profiling then resumes from a server-dependent trajectory state. The 384 G arm
issued 112 warmup requests, the two 96 G arms 88 and 86 -- 27 % more work in
the same 600 s, with 2 % run-to-run variation. The first-request spread
figures (247 s vs 3535 s) that I first cited as evidence are **not** evidence:
that statistic is max-minus-min over all trajectories including ones whose
next turn is hours out and never fires.

## 4. The offline reuse analysis

`requests` in the corpus is a **tree**: leaf requests carry
in/out/hash_ids/api_time, subagent nodes carry `agent_id`/`subagent_type` and
their own nested `requests`. Reading only the top level gives 58 495 requests;
the true count is **98 827** (which is what the original regex scanner
reported -- it was right and the json pass that "corrected" it was wrong).
`ttft` appears only on depth-0 requests, so the production TTFT reference is
unaffected.

`reuse_pass1.py` collapses 338 M block references to 98 827 rows, classifying
each request's blocks:

| depth | refs | new | hot | warm |
|---|---|---|---|---|
| 0 | 278.3 M | 1.2 % | 97.5 % | 1.3 % |
| 1 | 59.8 M | 3.9 % | 87.5 % | 8.6 % |
| all | **338.1 M** | **1.7 %** | **95.7 %** | **2.6 %** |

*hot* = also in that conversation's immediately preceding request; *warm* =
seen earlier in the trace but not in the last request; *new* = first sight.
Ceiling on hit rate is 98.3 %.

The trap here is reading "warm = 2.6 %" as L1's opportunity. It is not.
**GPU and L1 compete for the same 95.7 %**; which tier serves a hot reference
is purely a capacity question. A session's *peak context* is ~15 GiB but its
*total distinct content over its life* is 67-292 GiB (trace 4: 49 764 blocks x
6 MiB), because agents churn context. L1 lives between those two numbers.

`reuse_pass2.py` replays the corpus at conversation granularity -- within a
conversation every request re-reads its whole prefix, so all its resident
blocks share one recency and block-level LRU collapses to conversation-level,
turning 338 M references into 99 k events. Hit rate vs tier size:

| N | 24 G | 96 G | 187 G | 288 G | 384 G | 576 G | 1024 G | 2048 G |
|---|---|---|---|---|---|---|---|---|
| 16 | 68.5 | 91.2 | **97.8** | 98.1 | 98.2 | 98.3 | 98.4 | 98.4 |
| 32 | 54.9 | 78.1 | **96.1** | 97.9 | 98.1 | 98.3 | 98.5 | 98.5 |
| 64 | 38.4 | 59.7 | 87.1 | **95.2** | 96.7 | 97.6 | 98.1 | 98.4 |
| 128 | 25.8 | 42.5 | 68.5 | 83.0 | **89.4** | 94.9 | 97.2 | 97.9 |
| 256 | 23.2 | 34.3 | 54.4 | 67.4 | 76.0 | **85.7** | 94.3 | 97.3 |

Knee at roughly `6 GiB x N` -- 0.4x the per-session peak, because not every
session is simultaneously active. (First version of this had every trace
running to completion alone: the file is grouped by trace and I iterated it
unsorted, so the tier was never under pressure and every row read 97-98 %.
Sorting by `t` fixed it.)

The surface over-predicts against measurement (96 % vs the 4-15 % vLLM
reported at N=32) for one reason: **most of the pool is not cache, it is
in-flight KV.** `kv_mean` was 65-76 %, so of 187 GiB, 120-140 GiB was live
requests. Effective cache = tier - L*I. With that correction the surface
agrees with the measured 96 G -> 384 G jump in both direction and magnitude.

## 5. The configuration method

The simplification that makes it tractable: per-session working set `w` and
mean request size `I` are the same number (15.3 vs 15.1 GiB), because a
session's peak context *is* its largest request. Define `C = L0/w`, the pool's
capacity in average requests. Then both constraints are constraints on `N/C`:

```
latency        oversub = (N/C)*d < oversub_max
hierarchy+pressure     N/C > 1 + k          k = L1/L0
```

so a configuration exists only if

```
(1 + k) * d  <  oversub_max
```

**Feasibility is set by the duty cycle and nothing else** -- not L0, not L1,
not N. At the measured busy-period `d = 24 %` and my initial
`oversub_max = 0.5`, that caps `k` at **1.08**: the hierarchy is forced to be
inverted. That is why five hand-picked operating points in a row each broke a
different constraint.

`configure.py` closes the circularity (`d` needs `R`, `R` needs the hit rate,
the hit rate needs the cache left after in-flight, which needs `L = N*d`) by
damped fixed-point iteration, then screens against the constraints. Two things
it needed before it was honest:

- **A queue term.** `R = (OSL/decode + prefill)/(1-u)`, `u = L*I/L0`. Without
  it `R` never blows up, the collapsed branch does not exist in the model, and
  every configuration reports stable -- it happily returned a 300 s operating
  point as fine.
- **Performance gates.** The constraints were all capacity. Adding
  `TTFT <= 2.64 s` eliminated the configuration it had been recommending, and
  reporting end-to-end exposed the thing no configuration can fix: 43.6 s
  against production's 20.8 s, because `OSL/decode = 1370/57.6 = 23.8 s`
  versus production's 8.5 s. **TTFT is a reachable gate; end-to-end is not.**
  Record 8 concluded that by observation; this derives it.

Validation, on a path that used no latency measurement at all -- only model
constants, the reuse surface, and the fixed point: predicted in-flight at
`conc=14` is **3.14**, measured **3.60**. 15 % error.

## 6. Where it actually binds

`R` is ~90 % decode (`23.8 s` of `24.2 s` at a realistic hit rate). So an L1
tier cannot lower `R`, cannot lower `d`, and cannot raise the `k` it is
allowed -- an "A then B" plan where a small L1 bootstraps a larger one has no
torque. Lowering `d` needs faster decode (2.8x available if we matched
production) or a longer `T`, neither of which is a configuration knob.

What *is* soft is `oversub_max = 0.5`, which I picked rather than measured:

| oversub_max | allowed k at d=24 % |
|---|---|
| 0.5 (assumed) | 1.08 |
| 0.7 | 1.9 |
| **0.8** | **2.33** |

And the interval is unmeasured. `conc=14` sits at oversub 0.31 and passes;
`conc=20` sits at 1.02 and has collapsed; in-flight jumps 3.60 -> 12.74 for a
1.4x change in `conc`, which is a branch change rather than a slope. Nothing
between 0.31 and 1.02 has ever been run, and the reachable hierarchy ratio --
hence whether a normal L1 tier has any latency-feasible operating point on
this node -- rides entirely on that gap.

## 7. Running

```
ov16_s1  conc=16   |   ov18_s2  conc=18
both: L0 = 186.6 GiB (not overridden), L1 = 96 G, lazy,
      DUR=1800 GRACE=600, no cache warmup, no ramp, no scenario override
started 14:40
```

Nothing is configured differently from `c2_14`/`c2_20` except `conc`, so these
arms are directly comparable to the existing reference set -- unlike the last
several proposals, each of which would have reset it.

- `conc=18` still on the good branch (oversub ~0.4, gate passed, waiting < 1)
  -> the branch is wider than assumed; push toward oversub 0.8, `k=2.33`
  becomes reachable, and a normal hierarchy under pressure at a gate-passing
  load is available without breaking any lock.
- `conc=16` already collapsed -> the good branch ends at 14, `k <= 1.08` is a
  hardware-limited result for this node, and lazy's demonstrable claim narrows
  to write economy.

Both outcomes are results.

## 8. Tooling

```
reuse_pass1.py   corpus tree -> 98 827 rows of new/hot/warm per request
reuse_pass2.py   conversation-granularity LRU replay -> hit rate vs tier size
hit_surface.json cached g(N,S), 10 x 13 points
configure.py     fixed point in L, constraint screen, performance gates,
                 bistability probe, and the --concurrency that reproduces
                 each operating point under the harness's think time
mdelta.py        two /metrics scrapes -> queue/prefill/decode over a window
decode.py        per-user decode rate and TPOT, now on every arm
trace_walk.py    recursive corpus walk (the tree, not just the top level)
arm.sh           + CACHE_WARM, + ext_hit max/mean/last
ttft.py          REF lat_p50/p90 corrected to 8.34/32.41 (streaming only)
```

`/metrics` was never scraped before today and the 384 G arm's decomposition is
permanently lost with its server. `arm.sh` should scrape at profiling start and
before teardown and report the delta; not changed yet because an arm was running.

Repo clean at `3ec93178` before this record; nothing pushed.
