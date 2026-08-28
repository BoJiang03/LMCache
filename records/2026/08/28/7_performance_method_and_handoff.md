# How performance was measured, and where the session stands

Record 5 is the smoke and the calibration; record 6 is the sweep and its
numbers. This is the method behind those numbers, the one framing error worth
writing down, and the state to pick up from.

## 1. The framing error

Every round through R3 was first reported as `medD` in milliseconds -- -337,
-1422, -1375. Those numbers are correct and they are close to meaningless on
their own, because **TTFT p50 in this workload is 265-304 seconds**. At 32
lanes the scheduler runs ~8 and holds ~10 waiting, and a prompt is 107 k
tokens at p50 and 325 k at p90, so TTFT is mostly queue plus prefill. 1 400 ms
against 300 s is half a percent.

Reported as a ratio instead, the same data says something defensible: a
**2-5 % median TTFT improvement, on 69-73 % of matched turns, in all three
rounds**. The correction is not that the effect vanished -- it is that a
millisecond delta on a 300-second baseline needs its denominator printed next
to it or it invites the wrong conclusion.

The rule for every future round: **latency deltas go in the record as
percentages, with the baseline p50 beside them.**

## 2. The measurement set

Five things now get read off every arm. Three are new this session.

| script | reads | answers |
|---|---|---|
| `inflight.py` | `profile_export.jsonl` | client-side in-flight mean/peak, ISL distribution, oversubscription |
| `occupancy.py` | vLLM log + aiperf log | **engine-side** running/waiting/KV occupancy, profiling window only |
| `perf.py` | `profile_export.jsonl` | per-arm latency distributions and throughput; paired deltas |
| `early.py` | `profile_export.jsonl` | paired medD by time window (from the prior session) |
| `converge.py` | `server.log.gz` + `vllm.log.gz` | ledger and store/retrieve series over time (prior session) |

Three distinctions that turned out to matter:

**Client in-flight is not engine occupancy.** `inflight.py` counts requests
outstanding at the client, which includes those queued at the scheduler
holding no KV blocks. At 32 lanes it reads 15.78 against an engine running
7.3. The 0.7-0.8 oversubscription target is about pool occupancy, so it has to
come from the engine's own `GPU KV cache usage` line -- that is what
`occupancy.py` exists for.

**Warmup has to be excluded.** It runs 16-32 minutes at these lane counts and
would drag every mean down. `occupancy.py` finds the `PROFILING execute`
timestamp in the aiperf log and discards engine samples before it.

**Unpaired arm statistics are descriptive only.** The replay is closed-loop,
so a faster arm pulls in more work and the two arms do not serve identical
request sets. `perf.py` prints both views and labels which is which; the
controlled one is paired on `(conversation_id, turn_index)`.

## 3. Counting traps

- **At TP=2 the MP server logs each store and retrieve once per rank.** Raw
  token sums are double. `arm.sh` reports `tokens_stored_raw` and
  `tokens_stored = raw / TP`; the per-rank figure is the one that compares to
  served input tokens.
- **`Prefix cache hit rate` from the last stats line is usually 0**, because
  the engine emits a final line while draining. Take the max over the
  profiling window, or better, compute the mean over that window.
- **`preempt_events` is not lazy-only.** `_has_preemption_reqs` is called
  unconditionally in `build_connector_meta`
  (`lmcache/integration/vllm/lmcache_mp_connector.py:982`), so a 16-to-0 split
  between lazy and eager is a real difference in vLLM's behaviour, not a
  logging artifact. Both the `by preempted requests` and `by resumed
  requests` variants need counting.
- **Do not edit a running shell script.** Bash re-reads from a byte offset, so
  an arm that started before an `arm.sh` edit executes a mix of old and new.
  One calibration snapshot came back with `kv_mean=0.0%` for exactly this
  reason. Freeze the harness before a round.

## 4. The scenario's own constraints

- `inferencex-agentx-mvp` **refuses `--benchmark-duration < 900`**. Every arm
  is wall-time bounded and at least 900 s; `--request-count` is not usable.
- **Warmup wall time is set by the largest few requests, not the lane count.**
  967 s at 32 lanes, 1 918 s at 64 -- in both cases the phase sat with
  `in_flight=1` waiting on one straggler.
- Dataset load is ~4 min: 393 traces -> 9 843 conversations -> N trajectories,
  where N is the concurrency. Lanes are staggered across the trace timeline
  (first-request spread 102 607 s at 32 lanes), so how many are live at once
  is an outcome, not a setting.

## 5. Harness

`$SCRATCH/par/`, where `$SCRATCH` is this session's scratchpad. 58 MB
including per-arm artifacts.

```
env.sh      TP=2, YaRN 1M via --hf-overrides rope_parameters, slot->GPU pair
up.sh       MP server + vLLM for one slot; asserts the L1 target from the
            server's own log, and that lmcache resolves to $REPO
arm.sh      one arm end to end: down, up, aiperf, snapshot, errors, teardown
down.sh     slot-scoped teardown; handles GPUS as a comma-separated pair
calib.sh    two lane counts in one round
chain_r12.sh / chain_r3.sh / chain_r4.sh   the rounds as run
smoke_probe.sh / probe_eager.sh            the two gates
```

Slots: 1 -> GPU4+5, 2 -> GPU6+7, both NUMA node 1. Ports 2721x / 2722x.
GPU0-3 are node 0 and GPU0 belongs to another user.

## 6. Settled parameters

```
MODEL       Qwen/Qwen3-Coder-30B-A3B-Instruct
TP          2
max len     1048576   (YaRN factor 4 over native 262144)
pool        2 038 512 tokens / 186.6 GiB   (no --num-gpu-blocks-override)
CONC        32        (engine kv_mean 74.1%, inside the 0.7-0.8 band)
DUR         1800
GRACE       600
SEED        1234
scenario    inferencex-agentx-mvp, --public-dataset semianalysis-cc-traces-weka-062126
            no --max-context-length, no --num-dataset-entries
```

## 7. State

Rounds complete: R1 (32 G), R2 (96 G), R3 (160 G), all eager vs lazy paired on
the same node. R4 (off vs lazy at 96 G) started 08:52 and is running.

Repo clean at `6db2e6ce`, nothing pushed. Records 5 and 6 hold the results.

Open, in the order they matter:

1. **R4** -- the `off` floor, and the cross-round comparability check (R4 lazy
   s2 repeats R2's 96 G point on the same slot).
2. **Slot equivalence is still assumed, not measured under this
   configuration.** Every round puts eager/off on GPU4+5 and lazy on GPU6+7,
   resting on i60L's 92 ms spread -- measured on the old config. A
   lazy-vs-lazy round at 96 G would settle it and re-use the same cross-round
   point.
3. **The danger floor has never been on in this configuration.**
   `lazy_offload_danger_floor_max_blocks=0` in every round, and lazy is the
   only arm vLLM preempts (16 / 19 / 10 against eager's 0 / 1 / 0). The guard
   built for this has not been tested against the workload it was built for.
4. **Why eager retrieves nothing below 160 G** is explained by residency
   arithmetic (1.8 TiB written into a 32-96 G tier) but not directly
   observed -- the server does not log lookups, only hits. A lookup/miss
   counter would turn an inference into a measurement.
