# Lanes, in-flight, and what L1 is actually for

Record 9 concluded that the experiment was over-determined: realistic latency
capped the session count, which capped the working set, which left no room for
a cache hierarchy. That conclusion was wrong, and the correction reframes the
whole project. It came from the user's objection: production latency is not set
by lane count, it is set by the pool and the in-flight set, and a request whose
context is not in GPU just pays an L1 transfer -- a p90 cost, not a wall.

## 0. What a lane is

The term is aiperf's own, from its trajectory log. Restating it because it
caused every measurement error in records 5-9.

**One lane = one recorded user session being replayed**, not a request slot.

```
TrajectorySource: built 10 trajectories from 393 traces
  lane=00  sample_time= 94%  root_next=115/130 ( 88% turns)  live=1   trace_id=002001296e8a...
  lane=06  sample_time= 86%  root_next=  0/54  (  0% turns)  live=11  trace_id=0470d446a451...
```

`root_next=115/130` is the turn it resumes at out of that session's 130;
`live=11` is how many streams that session has open at once (main agent plus
subagents). A lane alternates between a request in flight and the recorded
think-time gap, so:

| concept | meaning | our values |
|---|---|---|
| lane / `--concurrency` | sessions **open** | 10 / 14 / 20 / 32 / 64 |
| in-flight | API calls being served **right now** | 1.7 / 3.6 / 12.7 / 15.8 / 21.4 |
| duty cycle | fraction of a session's time in flight | corpus **5.27 %** |

`--concurrency` sounds like "concurrent requests" but under
`agentic_replay` it means "concurrent sessions". Record 5 calibrated it as if
it were the former.

## 1. The corrected causal chain

Two things I had collapsed into one:

- **in-flight demand > pool** causes queueing. Unbounded.
- **working set > pool** causes cache misses. Bounded, and it is what a cache
  tier exists to absorb.

Record 9 used the second as a latency constraint, which is equivalent to
denying L1 any reason to exist. What actually happens:

| arm | lanes | ws/pool | GPU hit | new tokens/req | prefill @13k | TTFT p50 |
|---|---|---|---|---|---|---|
| r0 | 10 | 0.76x | **60.4 %** | 65 k | 5.0 s | **1.1 s** |
| c2_14 | 14 | 1.32x | **81.0 %** | 34 k | 2.6 s | **1.3 s** |
| c2_20 | 20 | 1.89x | 31.7 -> 0 % | 112 k+ | 8.6 s+ | 46.0 s |
| cal_c32 | 32 | 2.29x | 4.3 % | 157 k | 12.1 s | 154.7 s |
| r3_lazy | 32 | 2.35x | 2.7 % | 167 k | 12.8 s | 255.3 s |

Working set crosses the pool, the GPU hit rate collapses, per-request prefill
work triples, service time triples, and the closed loop amplifies that into a
queue. **The collapse is the disease L1 is supposed to cure, not an artifact
to avoid.** Abandoning the 32-lane regime in record 8 was the wrong call for
the right reason.

L1 can cure it on paper: at 32 lanes the demand is ~48 k new tokens/s, which
as an L1 fetch is 4.7 GB/s against ~55 GB/s of PCIe Gen5, and a 100 k-token
copy measures 0.1-0.4 s against 7.7 s to recompute it. **~25x cheaper.** Our
L1 caught 0.4-7.4 % of it, so nothing was cured.

## 2. Lanes vs in-flight, in closed form

Each lane cycles: request resident for R seconds, then think for T seconds.
That is the interactive response time law:

```
X = N / (R + T)              throughput
L = X * R = N * R/(R+T)      in-flight
```

So **in-flight = lanes x duty cycle, and duty cycle = R/(R+T)**. T belongs to
the workload (the trace's think time); **R belongs to the server**. The duty
cycle is therefore not a workload constant.

Fitted from the exports, with N as sessions that actually opened:

| arm | N | X req/s | L | R | T | duty |
|---|---|---|---|---|---|---|
| r0 | 11 | 0.0953 | 1.66 | **17.4 s** | 98 s | **15.1 %** |
| c2_14 | 15 | 0.128 | 3.60 | 28.1 s | 89 s | 24.0 % |
| c2_20 | 20 | 0.0475 | 12.74 | 268 s | 153 s | 63.6 % |
| cal_c32 | 27 | 0.0449 | 15.78 | 351 s | 250 s | 58.5 % |
| r1_lazy | 28 | 0.0384 | 18.15 | **473 s** | 257 s | **64.8 %** |
| cal_c64 | 40 | 0.0432 | 21.42 | 495 s | 430 s | 53.5 % |

Three readings:

1. **T tracks the corpus think time, modulated by the scenario's global-idle
   compression.** 89-98 s at 11-15 sessions, 250-300 s at 28, 430 s at 40 --
   climbing toward the uncompressed ~222 s as sessions multiply and
   all-idle moments become rare. That is independent confirmation of the
   compression effect, which was previously only inferred.
2. **R explodes 17 -> 473 s while T merely doubles**, so the duty cycle goes
   15 % -> 65 %. Past the knee, in-flight is set by the server, not the
   workload.
3. **X falls 0.128 -> 0.038.** Throughput goes *down* as lanes go up. That is
   not saturation, which plateaus -- it is capacity loss, because each request
   now costs 3-5x the prefill work.

The two regimes and their slopes:

- **Unsaturated**: R ~ S0 fixed, `L = N * S0/(S0+T)` -- slope is the duty
  cycle, small.
- **Saturated**: X pinned at X_max, so `R = N/X_max - T` and
  `L = N - X_max*T` -- **slope 1**. Every extra lane is one more queued
  request.

Measured slope from 15 to 28 sessions: `(18.15-3.60)/(28-15) = 1.12`.
Textbook saturation.

## 3. Two knees, and the gap between them is L1's job

| knee | where | set by |
|---|---|---|
| **cache** | **N = 15-20 sessions** | working set crosses the pool (1.32x still fine at 81 % hit; 1.89x collapsed) |
| **capacity**, at 60-81 % hit | **N ~ 48 lanes** | `X_max*(S0+T) = 0.199 * (17+222)` |
| capacity, after the hit rate collapses to 3 % | N ~ 19 | `0.081 * 239` |

**The node has the raw prefill capacity for ~48 sessions but loses it at
~15-20, because the cache stops working first.** Closing that gap is exactly
what an L1 tier is for, and it sizes itself: reaching 48 sessions needs a
resident working set of `48 x 15.7 = 754 GiB`, minus the 186.6 GiB in GPU,
so **L1 ~= 570 GiB** -- 3x the pool, a normal hierarchy ratio, derived rather
than assumed.

Our sweep at 32 / 96 / 160 G covered 6 % / 17 % / 28 % of that. Three points
on the same side of the knee, with the tier smaller than the tier above it.
That is why `ext_hit` never exceeded 7.4 %.

**This also restates what lazy is worth.** Not a few percent of TTFT -- up to
**4x the session population at the same latency**, with the policy's role
being to not churn the 570 GiB away. Eager writes 1.8 TiB per 30 min into any
L1 it is given.

## 4. The knee, bracketed

`calib2`, 14 vs 20 lanes, L1=96 G, lazy, 900 s:

| | c2_14 | c2_20 |
|---|---|---|
| lanes / sessions | 14 / 15 | 20 / 20 |
| working set vs pool | **1.32x** | **1.89x** |
| GPU prefix hit | **81.0 %** | 31.7 % -> 0 % |
| TTFT p50 / p90 | **1.30 s / 4.93 s** | 45.97 s / 418.5 s |
| by-ISL shape | **monotone (prefill-bound)** | **NON-MONOTONE (queue-bound)** |
| in-flight / running / **waiting** | 3.60 / 3.15 / **0.16** | 12.74 / 5.19 / **7.33** |
| kv_mean | 47.8 % | 65.4 % |
| throughput | **7.7 req/min** (best of any arm) | 2.85 req/min |
| preempts | 0 | 2 |

**14 lanes is the honest operating point**: TTFT below production, zero queue,
the highest throughput any arm has produced, 81 % GPU hit. 20 lanes is over
the edge. So the knee sits between working set 1.32x and 1.89x the pool --
slightly later than the N=12 that record 9 computed from peak contexts, which
makes sense: a session in think-time does not need its whole peak resident.

## 5. Can working set and in-flight be set independently?

No, and the weka path has no pacing knob at all:

- `--replay-speedup`, `--open-loop-replay`, `--trace-session-sample-ratio`:
  honored only by the `baseten_trace` loader.
- `--num-sessions` maps to `expected_num_sessions`, "stop starting new
  sessions after this many" -- a cap, not a multiplier.
- `--use-think-time-only` is not a scale factor but an equivalent formula:
  default is `start-to-start gap - previous api_time`, the flag uses the
  recorded `think_time` field. Checked against the raw trace
  (`10.367 - 9.297 = 1.070 = think_time`) -- identical values.
- The only delay transform in the loader is `_delay_cap_tracker.clamp`, and
  the scenario forbids all three caps.

But in-flight should not be an input anyway. **`in-flight = arrival rate x
service time`**: arrival rate is the workload's (objective, from the trace),
service time is the server's -- and service time is precisely what L1 changes.
Fixing in-flight by hand would weld shut the link the experiment is trying to
measure. **As an outcome it is the evidence**: the duty cycle predicts
`L = N x 5.27 %`, and the gap between that and the measurement is the whole
miss-cost effect.

| N | predicted L | measured L | gap |
|---|---|---|---|
| 15 | 0.8 | 3.60 | 4.5x |
| 20 | 1.1 | 12.74 | 12x |
| 28 | 1.5 | 18.15 | 12x |
| 40 | 2.1 | 21.42 | 10x |

The one genuine working-set knob at fixed lanes is
`--trajectory-start-min-ratio` / `--max-ratio` (the scenario only sets
defaults 0.0/1.0). Resuming sessions later in their history gives larger
contexts -- though it also raises per-turn prefill, so it is not a clean
decoupling either.

## 6. Resource state

Nothing of ours is holding anything. **0 MiB of GPU** across all 8 cards;
`calib2` tore itself down cleanly -- no bound ports, no leftover
`lmcache`/`vllm`/`aiperf`. Our whole uid accounts for 6.1 GB of RSS across 95
processes, all of it VS Code server and Claude Code CLI sessions.

The 420 GB that has been squeezing node 1 all session is **not ours**:

```
pid=2647600  rss=201G  user=root
  /opt/venv/bin/python3 /opt/venv/bin/lmcache server --port 5555 --http-port 8080 --l1-size-gb 200.0 ...
pid=2650232  rss=201G  user=root   (identical)
```

Two root-owned `lmcache server` processes, 200 GB L1 each, ports 5555/8080,
from `/opt/venv`. Ours is `lmcache.v1.multiprocess.http_server` running as
`bo` on ports 272xx. Not killable or ours to kill; flagged for the user. If
they are stale, freeing them is what makes a 570 GiB L1 affordable.

Also standing: rui holds 38 GB on GPU0 and a 522 MiB context on every card.

## 7. Next

The cheap, sharp test, with a matched control already in hand:

```
20 lanes, L1 = 384 G, lazy, DUR=1800     against c2_20 (20 lanes, L1=96 G, collapsed)
```

Three independent predictions, all falsifiable:

1. `ext_hit` rises off the floor (was 0 % at the end of c2_20).
2. In-flight falls from 12.74 toward the duty-cycle value ~1.1-3.
3. TTFT p50 falls from 45.97 s toward production's 2.64 s, and the by-ISL
   shape returns to monotone.

All three together mean L1 pushed the cache knee past 20 lanes, which is the
first leg of the 4x-session-capacity claim. Then eager vs lazy at that L1 is
the real policy comparison, because retention is the thing that differs.

Repo clean at `220c8bcc` before this record; nothing pushed.
