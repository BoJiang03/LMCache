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
token-prefix-stable under the serving chat template. The engine's own
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

These are single runs per point, not two-repetition results like §1, and
`throttled_drains` stayed at 0 in all three, so a budget of 4 was never the
binding constraint here. A cohort with larger per-step operations could
throttle at 4; the ledger's `throttled_drains` counter is the sensor.

## Verdict

1. **The policy's decisions are right in the agentic shape.** It holds
   coverage where eager collapses (0.598--0.657 versus 0.385 at a 69.9 GiB
   working set), writes less, and cuts L1 eviction cycles by more than half.
   At no pressure it writes nothing at all while serving the same hits.
2. **Its default per-step cost is real and workload-visible.** In a shape
   with a long-lived pending queue and low instantaneous concurrency, the
   bounded free-queue read doubles decode time, which swamps the retrieval
   win in E2E at every cohort size.
3. **That cost is a tuning artifact, not the price of deferring.** With
   `max_drain_per_step = 4` the buffering behaviour is unchanged, the cache
   benefit is retained or improved, and decode returns to the no-connector
   baseline.

The actionable finding for the PR is (3): `max_drain_per_step`'s default of
64 is priced for the drain, not for the read that sizing it implies, and an
agentic workload is where that shows. Nothing here contradicts the reported
hot/cold or QASPER results -- those run at higher instantaneous concurrency,
where the same per-step cost is amortized across a deeper batch.

## Guards and raw data

All 22 sweep runs, the 5 attribution runs and the 3 budget runs pass every
guard in `agentic/validate_agentic.py`: exact request counts, zero failed
steps, prompt-token counts identical to the cohort's, a closing lazy ledger,
no tracebacks, and **zero vLLM preemptions**. Schedule lag p90 was 0.0 ms in
every run, so no run was reshaped by saturation.

The cohort itself (100 MiB of trajectory text) stays on RAID; its identity,
per-session token profile and instance list are in
`results/agentic/cohort_manifest.json`.

Reproduce with `agentic/run_agentic_sweep.py`, `agentic/run_attribution.py`
and the tables in `agentic/agentic_table.py` / `agentic/attribution_table.py`;
see [`agentic/README.md`](agentic/README.md). Raw per-run JSON (every request
record, counter delta, L1 series and ledger) is under
`results/agentic/`.
