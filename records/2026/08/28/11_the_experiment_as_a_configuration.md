# The experiment, written as a configuration

Records 5-10 were a sequence of corrections to one thing at a time. This one
fixes the whole thing at once: what the serving side is, what the workload's
truthfulness requires, and what is left over to sweep. The structure comes
from the user: decide a realistic serving configuration, pin the workload's
fidelity parameters, sweep only its intensity, and compare eager against lazy
at every point.

## 1. Production performance, re-measured properly

`traces.jsonl` was re-parsed with `json.loads` per line instead of the regex
scanner from record 8. The scanner was badly wrong: it reported 98 827
requests and found a `ttft` field on 393 of them, while `grep -c '"ttft":'`
counts **56 432**. The true totals are 58 495 requests, 56 432 of them
streaming (96.5 %).

| metric | p10 | p50 | p90 | p99 |
|---|---|---|---|---|
| TTFT | | **2.64 s** | **6.98 s** | 22.90 s |
| api_time | | **8.34 s** | **32.41 s** | 95.66 s |
| out tok/s per user | 67.3 | **161.7** | 310.4 | |
| TPOT | 3.2 ms | **6.2 ms** | 14.8 ms | |
| OSL | | 614 | 3 495 | 10 295 |

TTFT p50/p90/p99 come out identical to record 8, which is the cross-check that
those three numbers were right despite the broken scanner. **`api_time` was
not**: record 8's 6.65 / 26.32 mixed in non-streaming requests, and the
streaming-only reference is **8.34 / 32.41**. `ttft.py`'s `REF` is corrected.

By ISL bucket, TTFT stays nearly flat (1.80 / 2.16 / 2.43 / 3.12 s across a
20x range of prompt length) while the decode rate *rises* with context
(79.7 / 134.0 / 183.9 / 182.8 tok/s) -- the long-context turns are the ones
with long outputs, and production's prefix cache keeps their prefill off the
critical path.

## 2. The decode gap is ours to declare, not to fix

`decode.py` added: per-request `output_token_count / decode_duration` off the
aiperf export.

| arm | decode p50 | TPOT p50 | TTFT p50 |
|---|---|---|---|
| r0 (10 lanes) | 77.4 tok/s | 12.9 ms | 1.15 s |
| **c2_14** | **57.6 tok/s** | **17.4 ms** | **1.30 s** |
| c2_20 | 26.6 tok/s | 37.6 ms | 45.97 s |
| cal_c32 | 17.9 tok/s | 55.8 ms | 154.7 s |

We match or beat production on TTFT and lose 2.8x on tokens per second
(57.6 against 161.7). That gap is Qwen3-Coder-30B-A3B on two H200s against
whatever served the original traffic; no cache policy touches it. It is now
reported on every arm for one reason: so a decode regression is never read as
a load effect. **TTFT stays the only control target**, which is what record 8
argued and this measurement pins down.

`prefill_throughput_per_user` is worth watching alongside it -- 110 717 tok/s
at r0 and 3 977 at c2_20 -- because it is ISL/TTFT and therefore collapses the
moment queueing starts.

## 3. L0 is not a parameter

The GPU pool is HBM minus weights: 2 x H200 = 282 GB, ~61 GB of bf16 weights,
`gpu_util 0.90` -> **186.6 GiB / 2 038 512 tokens**. Three independent checks
say that is also the *right* size, so there is nothing to choose:

| check | requirement | actual |
|---|---|---|
| largest single request (truncation forbidden) | >= 70.2 GiB for 766 k tokens; 92 GiB to honour the declared 1 M | 186.6 |
| in-flight KV at the honest operating point | 3.6 x 165 k x 96 KiB = 55 GiB | `kv_mean=47.8 %` |
| must be **smaller** than the working set or L1 is pointless | 14 sessions x 15.7 = 220 GiB | 186.6, so 1.32x |

That third row retires record 9's proposal to shrink the pool. The workload
outgrows 186.6 GiB on its own at the operating point; there is no need to
manufacture the condition.

## 4. L1 sizes itself, twice, to the same answer

| derivation | result |
|---|---|
| capacity-knee arithmetic (record 10): 48 sessions x 15.7 GiB - 186.6 | 570 GiB |
| host DRAM per GPU: 2 TB / 8 GPUs x 2 GPUs for one arm | ~466 GiB |

Two unrelated arguments land within 20 % of each other, at 2.5-3x the pool --
a normal hierarchy ratio. **L1 = 384 GiB for the sweep, 512 GiB for one
confirmation pair.** The reason to run the sweep at 384 rather than 512 is
budget: 1 473 GB is available now, two parallel arms at 384 GiB cost 825 GB
and leave 650 GB of headroom, while 512 GiB each costs 1 100 GB and leaves
290 GB -- and paired eager/lazy arms on the same wall clock are worth more
than the last 25 % of tier capacity.

The 420 GB of root-owned `lmcache server` processes are still resident
(pids 2647600 / 2650232, 200 GB L1 each, from `/opt/venv`, ports 5555/8080).
Nothing of ours holds GPU memory; all eight cards read 549-553 MiB of other
users' contexts, except GPU0 at 40 GB (rui).

## 5. Which workload parameters are fidelity and which are intensity

### A. Locked by the scenario

`inferencex-agentx-mvp` rejects or auto-injects these; breaking one needs
`unsafe_override`, and a loader mismatch cannot be overridden at all.

| parameter | value | what it protects |
|---|---|---|
| timing mode | `agentic_replay` | trace replay at all |
| `--streaming` | forced on | TTFT/ITL exist |
| `ignore_eos` | injected | output length comes from the trace |
| `--ignore-trace-delays` | rejected | otherwise every turn dispatches back-to-back |
| `--synthesis-max-isl` | rejected | no input truncation |
| loader | weka trace family | the corpus |
| `--benchmark-duration` | >= 900 | statistics floor |
| `--trace-idle-gap-cap` | rejected | no per-trace timeline compression |
| `--inter-turn-delay-cap` | rejected | no per-turn cap |
| `--system-idle-gap-cap` | 10 s, pinned | only globally-idle replay time compresses |
| `--cache-bust` | `first-turn-prefix` | the first turn is genuinely cold |
| concurrency **sweep** | rejected under a scenario lock | one arm per lane count |

That last row is new and it matters for the harness: the rejection lives at
the envelope level (`AIPerfConfig._reject_scenario_with_sweep`), not in the
validator, so aiperf's own sweep machinery is unavailable here. `arm.sh` per
point is not a workaround, it is the only path.

### B. Free, but pinned for fidelity

| parameter | value | argument |
|---|---|---|
| `--public-dataset` | `semianalysis-cc-traces-weka-062126` | no `--num-dataset-entries`, no `--max-context-length` |
| `--random-seed` | 1234 | trajectory start positions are deterministic given it |
| `burst_phase_starts` | false (default) | its own docstring calls burst "a throughput-oriented run rather than a faithful arrival replay" |
| `--use-think-time-only` | unset | verified an equivalent formula, not a scale factor |
| `--grace-period` | 600 | drain, not load |
| `--agentic-cache-warmup-duration` | **600** | see C4 |

### C. Intensity

1. **`--concurrency N`** -- sessions open. The primary axis.
2. **`--trajectory-start-min-ratio` / `--max-ratio`** -- where in each
   session's history the lane resumes, so it sets per-session context size and
   therefore working set at fixed lanes. The scenario supplies defaults
   0.0/1.0 and honours explicit values; aiperf's own defaults are 0.25/0.75.
3. **`--prefill-concurrency K`** -- a hard cap on requests in the prefill
   stage. Acquired in the generic `CreditIssuer`
   (`aiperf/credit/issuer.py:334,392,519`) and configured for every phase in
   `timing/phase/runner.py:536`, so it applies to `agentic_replay` even though
   the strategy never mentions it. **This is the knob for holding the working
   set while moving in-flight**, which record 10 section 5 concluded did not
   exist. It does; the caveat is that throttling the client distorts the
   recorded arrival schedule, so it is a diagnostic rather than part of the
   main sweep.
4. **`--agentic-cache-warmup-duration S`** -- after the normal warmup drains,
   the live trajectories continue with zero idle delay and one-token outputs
   for S seconds, then profiling resumes from that trajectory state
   (`timing/strategies/agentic_replay.py:174-184,600-669`). It is a cache
   preload, and for a 384-512 GiB L1 it is close to mandatory: a run that
   starts from an empty tier spends most of its window filling it, which is
   exactly the confound that made `ext_hit` look like a floor in R1-R3.
5. `--benchmark-duration` / `sessions` -- statistics and a cap. Measured not
   to grow the session count (record 10).
6. `--concurrency-ramp` / `--prefill-ramp` -- transient shaping. Unused.

## 6. The configuration

```
serving, fixed for every arm:
  Qwen/Qwen3-Coder-30B-A3B-Instruct, TP=2, YaRN 1 M via --hf-overrides
  gpu_util 0.90, no --num-gpu-blocks-override
  L0 = 186.6 GiB / 2 038 512 tokens
  L1 = 384 GiB

workload fidelity, fixed:
  tables A and B above, plus --agentic-cache-warmup-duration 600

workload intensity, swept:
  CONC in {14, 20, 28, 40}    DUR=1800  GRACE=600  SEED=1234
  each point an eager/lazy pair, policy pinned to a slot within a point and
  swapped between points so slot equivalence stays checkable
```

14 is below the knee measured in record 10, 20 is where it collapses at
L1=96 G, and 28/40 test whether a real L1 moved the knee. About an hour per
point with both arms in parallel.

## 7. Next

The decision left open: run all four points, or spend one hour first on the
single sharp test from record 10 -- **20 lanes, L1 = 384 G against the
existing c2_20 at 96 G** -- whose three predictions (`ext_hit` off the floor,
in-flight down from 12.74, TTFT down from 45.97 s with a monotone by-ISL
shape) are all falsifiable. If enlarging L1 does nothing at 20 lanes the
premise of the whole sweep is wrong and the four points are not worth five
hours.

## 8. Tooling changes

- `trace_prod.py` -- the corpus reference, parsed with `json` instead of the
  regex scanner. Supersedes `trace_ttft.py` and `trace_tps.py`; the latter two
  are kept only as the record of how the under-count was found.
- `decode.py` -- per-user decode rate and TPOT, now on every arm in `arm.sh`.
- `ttft.py` -- `REF["lat_p50"/"lat_p90"]` corrected to 8.34 / 32.41.

Repo clean at `a8290714` before this record; nothing pushed.
