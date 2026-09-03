# 2026-09-02 (8) — State of the investigation: the 9% is inside LMCache, and I have not found it

Read this one first. Records 1-7 are the chronological log and several of their
headline claims have since been falsified; each now carries a correction box.

## The question

Mentor handed over `vast__LMCache collab.pdf`: reproduce VAST's findings, and if
they hold, find out why LMCache underperforms. Finding (1) — GPU-only configs are
slower with LMCache than with vLLM alone — reproduces. Ratio on our H200s matches
VAST's MI355X number to within 0.6%.

User's objective for this session, verbatim: 定位为什么 lmcache 即使什么都不做，都有
10% 左右的开销 — and after the IP half was closed: 我要把剩下 9% 也解决.

**Not solved.** The search space is much smaller and the measurement is much
better, but the mechanism is not found.

## The one measurement that matters

Every arm runs `max_num_batched_tokens=8192`, so a forward step is a fixed 8192
tokens and vLLM's own in-engine `Avg prompt throughput` converts directly:

    ms/step = 1000 * 8192 / (steady-state tokens per second)

`scripts/engine_rate.py` does this. It reads the stat lines out of each arm's
`server.log`, splits them into blocks on 60 s gaps, and reports the p50 of the
blocks whose queue actually got deep.

Why it replaced end-to-end `vllm bench serve` numbers as the primary instrument.
The frozen output is `engine_rate_c1000.txt` next to this file: **23 valid blocks
across 5 sessions and both pool sizes, falling into exactly three levels with zero
overlap.**

    10 blocks @ 85.3 ms/step   no connector (1a x6, 1c x3) + null connector (1i)
     4 blocks @ 91.0 ms/step   LMCache MP  (1d x2, 1e x2)
     9 blocks @ 97.5 ms/step   LMCache IP  (1b x4, 1f x2, 1g x1, and 2 more 1b)
     1 block  @ 341.4          the discarded 1h py-spy run, kept as the outlier
                               that shows what a distorted arm looks like

Every block inside an arm agrees to 0.1 ms, across days and across pool sizes.
Pairing bench runs across sessions had already burned this investigation twice —
one arm moved 28% between sittings — and this does not.

| arm | pool | ms/step | p50 tok/s | Deferred |
|---|---|---|---|---|
| 1a no connector | 25,798,626 | **85.3** | 95,993-95,996 | 0 |
| 1c no connector | 13,724,416 | **85.3** | 95,997 | 0 |
| **1i null connector** | 13,724,416 | **85.3** | 95,993 | none |
| 1d LMCache MP | 25,798,626 | **91.0** | 89,989-89,991 | 0 |
| 1e LMCache MP | 13,724,416 | **91.0** | 89,988-89,990 | 0 |
| 1b LMCache IP | 13,724,416 | **97.5** | 83,993-83,996 | 531-926 |
| 1f IP + backoff patch | 13,724,416 | **97.5** | 83,991-83,994 | 415-599 |
| 1g IP + backoff = 0 | 13,724,416 | **97.5** | 83,993 | 575 |

## The decomposition this settles

| | ms/step | where a fix would go |
|---|---|---|
| vLLM's connector plumbing | **+0.0** (<= 1.5% end to end) | nowhere; not the problem |
| **LMCache, common to IP and MP** | **+5.7** | **LMCache** — this is the ~9%, the objective |
| LMCache, IP only | **+6.5** | LMCache — unexplained after 1f and 1g |

"LMCache costs ~10% while doing nothing" is a statement about **LMCache's own
per-step work**. It is not the cost of vLLM's connector interface, and it is not
the KV pool.

## Eliminated

Each with data, not argument. The first five cost no machine time at all — they
came out of logs already on disk.

| candidate | how it died |
|---|---|
| halved KV pool | no connector is 85.3 ms/step at **both** 25.8M and 13.7M; MP is 91.0 at both. The pool moves 1.88x and the per-step cost does not move at all |
| cudagraph mode downgrade | all six arms log `FULL_AND_PIECEWISE`; `requires_piecewise_for_cudagraph` only fires on `use_layerwise`, unset here |
| LMCache's store / D2H copy | 1b c=1000 warm window: TP0 did 1,423 stores totalling **4.5 s** of a 724 s run = 0.6% |
| prefix hash chain (O(N^2/chunk)) | LMCache times it itself inside `profile_process_tokens`; it is inside that same 4.4 s |
| idle time / stalls | duty cycle (mean/p90 rate) is 0.93-0.99 in every arm, MP highest. Each step is genuinely slower; the engine is not idling |
| scheduler config drift | `max_num_batched_tokens=8192` in every arm |
| KV cache layout / attention backend | MP returns None from `get_required_kvcache_layout` (with a comment saying why), IP does not override it; all arms pick FLASH_ATTN |
| per-layer `dispatch` hooks | MP's `dispatcher` is only built for `transfer_intermediate_tensors`, which we do not enable |
| prefix cache confound | every arm 0.0-0.4% hit rate, internal and external; nobody gets an advantage |
| async lookup backoff (`Deferred`) | 1f patched it, 1g removed it entirely — both 97.5 ms/step, identical to unpatched |
| vLLM's whole connector code path | 1i: a connector that does nothing costs **+0.0 ms/step** |

## Three mechanisms I proposed and the experiments killed all three

Worth writing down as a pattern, not just as outcomes.

1. **O(pending) scheduler stall.** `Deferred` peaked at 926, the backoff is 10 ms
   and held under the lock, so "up to 9.2 s of dead engine time in one pass".
   1f collapsed those sleeps to one per pass: **no change**.
2. **One sleep per scheduling pass.** The replacement, sized to +6.5 ms/step at
   10.25 steps/s. 1g set `lookup_backoff_time: 0.0` — no sleep at all:
   **no change**.
3. **`KVOutputAggregator`.** `vllm/v1/engine/core.py:158` installs it whenever
   `scheduler.connector is not None`, and it flips `execute_model` and
   `sample_tokens` from dequeuing one worker's output to dequeuing all eight —
   2 message-queue reads per step becomes 16, plus two straggler waits. Written
   into record 7 at 18:24, 17 minutes before 1i landed, precisely so it could be
   wrong on the record. 1i: **the gate is real and it is free**.

Each was read out of source, each had arithmetic that fit the measured gap, and
each was wrong. **Reading code and sizing a plausible mechanism has a bad track
record on this problem — 0 for 3. The next step has to be measurement, not a
fourth guess.**

## Deliverables state

- **PR: cleared.** `fix_async_lookup_backoff_stall_pr` deleted from the fork and
  locally. It existed to fix a performance problem; its commit is titled `[Perf]`;
  two independent measurements say it fixes no performance problem. Not
  repurposed into a correctness PR — that would ship a change whose real
  justification was never the one that motivated writing it. Content survives as
  `8ea23cd1` on `vast_repro_dev`.
- **venv: back to stock LMCache.** `b46f6a1c` reverts the patch; verified by
  importing and confirming the original sleep-under-lock is back. Every arm from
  here measures unmodified LMCache.
- **Records 1, 2, 3: corrected.** Each now opens with a box saying the halved pool
  is real but costs nothing, so no sentence in them may treat it as the cause.
  Record 2's `SupportsHMA` issue draft is still worth filing as a resource
  problem, but **must not claim a throughput regression**.
- Nothing pushed to the fork. Nothing filed. Nothing sent to VAST.

## Tools built today, both reusable

- `scripts/engine_rate.py` — the ms/step decomposition above. Use this, not paired
  bench runs, for anything comparing arms.
- `nullconn/null_connector.py` + `scripts/phase1i_null_connector.sh` — a
  do-nothing `KVConnectorBase_V1`. Holds every known confounder fixed by
  construction (no `SupportsHMA` so the pool matches; no piecewise so cudagraph
  matches; layout inherited so the backend matches; never returns None so it can
  never defer) and asserts all of them before spending 20 minutes.
- `scripts/analyze_pyspy.py` — written, smoke-tested, unused. Kept for
  single-process targets.

## Negative result on tooling: py-spy is unusable on a TP>1 vLLM

1h tried to profile MP with `py-spy record --subprocesses --idle -r 30` and ran at
23,998 tok/s = 341 ms/step against MP's true 91.0 — a **3.75x distortion**, 73% of
wall clock being the profiler. Its own pre-registered sanity gate caught it and
the run was stopped at 27 minutes instead of producing an uninterpretable file.

The cause is not ptrace permission. py-spy *can* attach without sudo here:
`ptrace_scope` is 1, which permits tracing descendants, and `record -- cmd` makes
py-spy the parent. The standing ask for `sudo sysctl -w kernel.yama.ptrace_scope=0`
is **withdrawn**. The cause is that with TP=8 the workers are lockstepped on NCCL
collectives, so ptrace-stopping any one rank stalls all eight. Lowering the rate
scales the distortion down but never removes the amplification. **Do not retry
py-spy on the workers.**

## Two candidate next steps — neither started, user's call

1. **Timing instrumentation inside LMCache** (preferred). Accumulate per-step wall
   clock in `build_connector_meta`, `start_load_kv`, `wait_for_save` and the
   per-layer hooks, run the MP arm, read off which hook owns the 5.7 ms. Requires
   an LMCache source change, but adds only counters — no logic touched. Covers the
   scheduler and the workers, and returns an attribution rather than a flame graph
   that still needs interpreting.
2. **vLLM's in-tree torch profiler** over a short window
   (`--profiler-config.profiler=torch` plus `/start_profile` and `/stop_profile`).
   No source change, in-process so no ptrace and no TP amplification, sees CUDA
   ops. Limitation: workers only, blind to the EngineCore scheduler.

A note for whoever picks this up: IP and MP do **not** share connector code — they
are separate implementations. So "common to IP and MP" is more likely something
structural both incur (both run LMCache inside all 8 worker processes and contend
for the same GIL) than a shared function. That is a hypothesis, and per the 0-for-3
record above it should be measured before it is believed.

## Still open, unrelated to the above

- `1a@200` and `1c@200` were never run, so the whole c=200 column is
  uninterpretable. 1d/1e at c=200 currently move in the **opposite** direction to
  c=1000 (1e/1d = 1.041x vs 0.994x) and nobody can say why without the baselines.
- Decode is entirely unmeasured: OSL=1 means every number here is prefill-only.
- Finding (2) from the PDF (IP vs MP in VAST's matrix) is parked by user decision.
- Ask VAST for the `GPU KV cache size: N tokens` line from each of their runs.
- `vast_repro_dev` is 13 ahead / 11 behind `origin/dev` and **not pushed**.
