# The corpus re-measured, and prefill as the bottleneck

Follows record 13. That record ended by saying the three requirements collide
and the only untested quantity was the congestion ceiling on `oversub`. Most of
that framing turned out to be built on two errors and one artefact. This record
re-measures the corpus from the raw trace file, corrects the identity, and
relocates the bottleneck to prefill throughput.

## 1. Decode is closed

Four arms, all negative. Record 13 section 6 carries the tables; the summary:

| arm | decode | verdict |
|---|---|---|
| baseline | 177.5 tok/s, TPOT 5.6 ms | above the corpus's own 161.7 p50 |
| ngram | 66-91 tok/s | 2x worse, acceptance length 1.97 |
| ngram_gpu | 59-83 tok/s | 2.3x worse, acceptance length 1.41 |
| fp8 KV | 167.6 tok/s | 5 % worse; pool doubled to 4 076 944 tok |
| MoE backend | n/a | TRTLLM refuses SM90; vLLM deprioritises FlashInfer on SM90 as slower than Triton |

A TPOT-vs-context sweep separated the fixed per-step cost from the KV read:

```
ctx tok        auto TPOT    fp8 TPOT
  2 533          4.28         4.51
 20 911          4.56         4.75
 33 645          4.78         5.11
 68 435          5.19         5.44
101 883          5.63         5.98
fit         4.247+0.0136/1k   4.469+0.0148/1k
100k split   fixed 75 % / KV 25 %   (both)
```

fp8's slope is not halved; it is slightly higher. The fp8 path saves no time on
the KV read. Per-GPU bandwidth at 100 k: KV bf16 4.92 GB in 1.38 ms = 3.56 TB/s
(74 % of peak, attention is fine); KV fp8 2.46 GB in 1.51 ms = 1.63 TB/s (34 %);
non-KV 3.0 GB in 4.25 ms = 0.71 TB/s (15 %). The 75 % fixed term is the MoE
expert GEMMs, and CUDA graphs are on (`enforce_eager=False`,
`cudagraph_mode=FULL_AND_PIECEWISE`, capture size 1 present), so it is kernel
execution at M=1, not launch overhead. Record 13's earlier attribution to launch
overhead is wrong.

Decode is not a lever. `177 tok/s` is what this machine does and it already
exceeds the reference.

## 2. The corpus, measured from the raw file

`traces.jsonl`, all 393 conversations / 58 495 requests, parsed directly rather
than through the harness.

```
in    p50 245,248   mean 304,446   p90 645,824   p99 895,808   max 989,824
out   p50     583   mean   1,334
turns/conv p50 67   mean 148.8   max 3052
api_time p50 8.34s  mean 14.59s
think gap p50 5.12s mean 405.80s  p90 140.23s
max concurrent requests within one conversation: p50 2, mean 2.1, max 6
```

Three findings.

**`in` is per-request, not cumulative.** Only 48 of 393 conversations have
monotonically non-decreasing `in`; 345 do not. The hypothesis that the corpus
sums a conversation's turns is dead.

**This is a cache-hit trace, not a compute trace.** TTFT by input bucket:
1.80 / 2.16 / 2.43 / 3.12 s across 0-50k / 50-150k / 150-400k / 400k-1M. A
400k-1M token prompt at 3.12 s implies 130-320 k tok/s of prefill, which no
hardware does. The production system was not prefilling these prompts; it was
serving them from a prefix cache. Our replay is cold and pays a price production
never paid. The 300 s vs 2.64 s TTFT gap is not a hardware gap.

**The working set is cluster-sized.** 393 conversations x 245 k tokens x 96 KiB
= 9.5 TB. Our L1 is 96-576 GiB, 1-6 % of it, and the measured `ext_hit` of
2-5 % is consistent with holding that fraction. Nothing was misconfigured; a
single node was replaying a cluster's traffic.

**The harness sends shorter prompts than the corpus contains.** Corpus `in` p50
is 245 248; our arms measure ISL p50 106 065, mean 184 157. A 2.3x gap. Not
investigated further; every number below that needs an ISL uses the harness
value, since that is what the server actually saw.

## 3. Two errors in record 13's identity

**`W` is not tied to in-flight.** Record 13 wrote requirement 2 as
`L1_tok < N x ISL_med` with `N` taken from the in-flight bound. The `N` that
sets the working set is the number of distinct conversations touched, which is
larger. Measured per arm:

| arm | conc | distinct conversations | working set |
|---|---|---|---|
| ov16_s1 | 16 | 17 | 3.30 M tok = 302 GiB |
| cal_c32_s1 | 32 | 27 | 4.68 M tok = 428 GiB |
| k20_384cw_s2 | 20 | 35 | 5.63 M tok = 516 GiB |

Collapsing the two `N`s is what made `L0` cancel and produced "the three are
mutually exclusive". With them separated the condition becomes
`d < oversub_max x ISL_conv/(K_MIN x ISL_mean)` = `oversub_max x 0.475`, so
`k=2` needs `oversub_max >= 0.48`.

That correction does not survive intact either: `W ~ conc x 175 k` because a
lane stays on one conversation for the whole arm (4 turns of a 148-turn
conversation), so `W` and in-flight both scale with `conc` and `conc` cancels
again. The number 0.48 stands; the reasoning behind it changed twice.

**`b` does not cancel.** Record 13 showed `k_max` invariant under bytes-per-token
because `b` appears on both sides. That holds only at fixed decode speed. Under
load the decode step is KV-bandwidth bound, so decode rate depends on `b`, so
`d` depends on `b`. The invariance argument was wrong; whether it matters in
practice is moot, since fp8 (section 1) delivers none of the theoretical gain.

## 4. The duty cycle, and what "cold" means

From the raw trace, per conversation, over the conversation's own span:

```
d_agg   = sum(api_time)/span                 = 0.0359   cold:in-flight = 26.9 : 1
d_union = fraction of wall time with >=1 request active
                                   p50 0.096  mean 0.168  = 9.4:1 to 4.9:1
```

`d_union` is the right one for KV capacity: parallel subagents share the
conversation prefix, so the KV object is one, not 2.1. With
`oversub = k x d/(1-d)`, `k=2` needs `oversub` between 0.21 and 0.41. The
already-proven-uncongested 0.31 sits inside that range.

The harness's effective `d` is 0.257 (c2_14: in-flight 3.60 at conc 14), 1.5-2.7x
the corpus value. Same GPU pressure, the real workload carries 28 sessions where
the compressed replay carries 14.

## 5. The think-gap distribution

The single most useful measurement of the day. All 54 951 gaps:

```
        band       count   share   cum     time share   cum
      0-5s        26979   49.1%   49.1%      0.32%     0.32%
      5-15s       14589   26.5%   75.6%      0.50%     0.82%
     15-30s        2515    4.6%   80.2%      0.24%     1.06%
     30-60s        2481    4.5%   84.7%      0.47%     1.53%
    60-120s        2183    4.0%   88.7%      0.83%     2.36%
   120-300s        3671    6.7%   95.4%      2.99%     5.35%
   300-600s         775    1.4%   96.8%      1.46%     6.80%
  600-1800s         944    1.7%   98.5%      4.41%    11.21%
    >1800s          814    1.5%  100.0%     88.79%   100.00%
```

With the L0 residency `H` as a free parameter:

```
     H     gaps < H (lazy skips)   gaps >= H share of idle time
    10s          71.9%                     99.29%
    30s          80.2%                     98.94%
    60s          84.7%                     98.47%
   120s          88.7%                     97.64%
   300s          95.4%                     94.65%
  1800s          98.5%                     88.79%
```

The two columns have very different sensitivity. Moving `H` from 10 s to 300 s
raises the skipped-write fraction 23.5 points while costing L1 only 4.6 points
of idle-time coverage, because 88.8 % of all idle time sits in the >1800 s band
that 1.5 % of gaps produce. **A long lazy horizon is close to free.** The current
`lazy_offload_horizon_steps=2.5` sits at the far left of this curve.

The 30-300 s band is 15.2 % of gaps and 4.29 % of idle time. It is the steepest
part of the curve and the only region where the policy decision is non-trivial:
below 30 s the block is certainly still in L0, above 300 s it is certainly gone.
An earlier draft of this analysis presented only a `<30s` / `>=300s` split and
dropped this band; that was a presentation error, and the corrected version
strengthens the case rather than weakening it.

This also explains every prior result at once. The scenario's global-idle cap
compresses long gaps and leaves short ones alone, so the population lazy skips
(80 %) survived and the store-side reduction was measurable (35-83 % across the
r-series), while the population L1 serves was compressed away and `ext_hit`
never rose above 5 %.

## 6. Forward derivation from L0

Rather than searching, derive the user count from hardware. L0 must hold the
in-flight requests plus the short-idle ones; L1 holds the long-idle ones.

```
occupancy = d + s        s = short-gap share of idle time x (1-d)
N         = L0_tok / (occupancy x ISL)
L1        = N x (1-occupancy) x ISL
```

```
d=0.168, ISL=184k harness
     H    short-idle    L0 occupancy   N     L1 wanted   k wanted   DRAM   k actual
   15s      0.68%         17.5%        63     881 GiB      4.7      no      2.51
   30s      0.88%         17.7%        63     869          4.7      no      2.51
   60s      1.27%         18.1%        61     846          4.5      no      2.51
  120s      1.96%         18.8%        59     808          4.3      no      2.51
```

With `d=0.096` the same table gives N = 94-107.

Two results. First, `H` barely moves `N` (63 to 59 across 15-120 s), because
short gaps are short: they are 76-89 % of turns but under 2 % of idle time.
Keeping them resident is nearly free in memory and buys most of the write
reduction. Second, the wanted L1 of 808-1618 GiB exceeds the per-TP2-slice DRAM
share of 469 GiB, so `k` lands at 2.51 by hardware and L1 holds only 29-58 % of
what wants to be in it. **k=2.51 is derived, not chosen, and L1 evicts
naturally.** All three requirements hold at this point without any compromise.

## 7. Prefill, not PCIe, is the bottleneck

```
L1 fetch 400 000 tok/s = 39.3 GB/s, 71 % of PCIe Gen5 x16 (confirmed gen 5, x16)
N=60, real think time: 60/(24+400) = 0.142 turns/s
  sustained fetch demand 26 060 tok/s = 2.56 GB/s = 4.7 % of PCIe
```

Peak and sustained differ 15x because 60 sessions each turn once per 424 s. PCIe
has 20x headroom and cannot be the constraint.

Prefill can:

```
                                   max miss rate     max N at 100 % miss
real think time (424 s/turn)
  prefill 13 000 tok/s                 49.9 %              29.9
  prefill 21 700 tok/s                 83.2 %              49.9
compressed replay (113 s/turn)
  prefill 13 000 tok/s                 13.3 %               8.0
  prefill 21 700 tok/s                 22.2 %              13.3
```

The last two rows are the cliff. With no working cache under the compressed
replay this machine supports 8-13 sessions; the measured cliff is at conc 15.
The whole causal chain closes:

```
idle compression -> 3.75x turn density -> 3.75x prefill demand
prefill capacity 13-22k tok/s -> 8-13 sessions without cache -> cliff at 15
```

A miss costs 19.7 s of prefill (later-turn `in` p50 256 064 at 13 k tok/s); an
L1 hit costs 0.46 s. L1 is therefore not only a capacity tier, it is the circuit
breaker that keeps `R` bounded and stops the feedback loop
(hits fall -> R inflates -> in-flight rises -> spare L0 falls -> hits fall).
Every arm to date ran with L1 too small or with L1's customers compressed away,
so the loop had nothing to break it.

The cache-bust the scenario forces is cheap: first-turn `in` p50 is 448 tokens,
so `FIRST_TURN_PREFIX` costs 0.03 s, not a full prefill.

## 8. The scenario locks twelve parameters, and self-releases the one that matters

`inferencex_agentx_mvp.py` sets:

```
timing_mode = AGENTIC_REPLAY          require_ignore_eos = True
require_streaming = True              forbid_ignore_trace_delays = True
forbid_input_truncation = True        require_loader = (...)
min_benchmark_duration_seconds = 900  default_trajectory_start_ratio 0.0/1.0
system_idle_gap_cap_seconds = 10.0    forbid_trace_idle_gap_cap = True
forbid_inter_turn_delay_cap = True    require_cache_bust = FIRST_TURN_PREFIX
```

Dropping `--scenario` to recover real think time does not work. `config.py:591`
states AGENTIC_REPLAY is set only by the scenario, so without it the run falls
back to closed-loop concurrency with no think time at all, and `ignore_eos` and
the cache-bust go with it. Three confounds, one of them the metric under test.

Overriding `--system-idle-gap-cap-seconds` explicitly is recorded as a
`ScenarioViolation` (`validator.py:762`) and invalidates the submission.

Neither is needed. The cap fires only when the replay is globally idle, so its
effect is a function of concurrency:

```
conc=14   P(all idle) = (1-0.096)^14  = 24.4 %   fires often, heavy compression
conc=60   P(all idle) = (1-0.096)^60  =  0.24 %  effectively never fires
```

Running at conc 60 with the scenario fully intact replays the trace at its
recorded spacing. `configure.py`'s note that the cap "releases above ~27" is the
same observation; it was read as a dead end and is in fact the way through.

## 9. ov15: the cliff is one concurrency step wide

conc=15, L1=96 GiB, 182 requests. Columns are avg/min/max/p99/p90/p50/std.

```
effective concurrency  avg 6.66  max 12      -> oversub 0.60
TTFT                   p50 4.69s  p90 154.2s  p99 192.0s  max 198.6s  min 0.24s
request latency        p50 31.1s  avg 87.8s
```

In context:

| conc | oversub | in-flight | TTFT p50 | TTFT p90 |
|---|---|---|---|---|
| 14 | 0.31 | 3.60 | 1.30 s | |
| 15 | 0.60 | 6.66 | 4.69 s | 154.2 s |
| 16 | 0.97 | 10.73 | 234.24 s | |

The cliff is not a line between 14 and 16, it is conc=15 itself: the median is
still serviceable while the tail has already gone. `oversub` 0.60 is exactly the
0.591 that section 4's identity requires for `k=2`, so "reaching a normal
hierarchy means standing on the edge of the cliff" is now measured, not derived.

The arm's aiperf restarted itself from warmup at 17:07:50 under the same
`arm.sh` process and overwrote `profile_export.jsonl`. The first pass had
already written its summary; it is saved as `ov15_s2_pass1_console.txt` and
`ov15_s2_pass1.json`. The second pass was killed as duplicate work.

## 10. Corrections to earlier records

- Record 13 section 6 attributed the 4.25 ms fixed decode cost to kernel launch
  overhead. CUDA graphs are on at batch 1; it is kernel execution at M=1.
- Record 13 section 2's `k_max` identity conflated in-flight sessions with
  working-set conversations. See section 3.
- Record 13 section 11's claim that `k_max` is invariant under bytes-per-token
  holds only at fixed decode rate. See section 3.
- The `<30s` / `>=300s` framing of the gap distribution omitted the 15.2 % of
  gaps between them, which is the decision-relevant band. See section 5.
- "fp8 will lower `d` through faster decode" was stated before measurement and
  is wrong; fp8 is 5 % slower at batch 1 and its slope is not reduced.
- An `n60_free` arm with `--scenario` removed was queued and cancelled before it
  ran, once section 8's reading of the timing-mode dependency was checked. No
  data from it exists.

## 11. Where this leaves the experiment

The configuration is derived, not searched:

```
N = 60 sessions      L0 = 186.6 GiB at util 0.90     L1 = 576 GiB (k = 3.09)
scenario intact, no cache warmup, DUR 1800
```

576 GiB rather than the 384 first queued: at 384 the combined residency is
33.8 of 60 conversations, a 44 % miss rate, and prefill utilisation 87 %, which
is the knee of the queue. 576 GiB puts residency at 45.3 of 60, miss 24 %, and
prefill utilisation 48 %.

The prediction, recorded before the result: TTFT p50 of a few seconds, p90 in
the tens of seconds, with a minority of long-context misses reaching minutes. If
p90 stays above 150 s the circuit-breaker reading is wrong and the problem is
L1's fill rate or hit logic rather than its capacity.

The arm was running when this record was written. Its warmup phase is slow
(`returned=3/65` at 180 s, in-flight 1), which is expected for a cold start
where every request is a full prefill, but if warmup consumes most of the window
the profiling phase will measure an unfilled L1 and read as a false negative.
`CACHE_WARM` is the remedy, at a small value: `k20_384cw` used 600 and that
pushed in-flight to 25.73 before L1 could fill, which is how a warmup overshoots
into the congested branch.
