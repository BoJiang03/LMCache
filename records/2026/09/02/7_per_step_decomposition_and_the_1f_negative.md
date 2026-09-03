# 2026-09-02 (7) — 1f is a negative result; the tax, measured in ms per forward step

## Headline

1. **The fix in record 6 buys nothing.** 1f (patched LMCache, otherwise byte-identical
   to 1b) measured 728.1 s against 1b's 727.7 s. The pre-registered band said
   "REFUTED if >= 715 s". Refuted. Record 6 now carries a correction box.

2. **The tax now has a unit that points at code.** Every arm runs
   `max_num_batched_tokens=8192`, so a forward step is a fixed 8192 tokens and
   vLLM's own in-engine `Avg prompt throughput` converts directly to ms/step.
   Across 14 independent blocks from 5 sessions and both pool sizes:

   | | ms/step | delta | blocks |
   |---|---|---|---|
   | no connector (1a, 1c) | **85.3** | — | 5 |
   | MP connector (1d, 1e) | **91.0** | **+5.7** | 4 |
   | IP connector (1b, 1f) | **97.5** | **+12.2** | 8 |

   Every block inside an arm agrees to 0.1 ms. **Pool size is irrelevant**: 1a at
   25,798,626 and 1c at 13,724,416 are both 85.3; 1d at 25.8M and 1e at 13.7M are
   both 91.0. So the decomposition is

       common connector tax  = +5.7 ms/step   <- the ~9%, the objective
       IP-only surcharge     = +6.5 ms/step

   and it reproduces the end-to-end ratios (1.067x, 1.143x) from a measurement
   taken inside the engine, not from pairing bench runs across sessions.

   Tool: `scripts/engine_rate.py`. This is the measurement to use from now on —
   it is immune to the cross-session drift that has bitten this investigation
   twice, because every arm's blocks are self-consistent to three digits.

---

## Why 1f measured zero

The patch's entire effect is to collapse multiple `lookup_cache` backoff sleeps
within one scheduling pass into one, and to move the sleep out from under
`self.lock`. It changed nothing, so **there was only ever about one
`lookup_cache` call per pass to collapse.**

The reason is the token budget. `max_num_batched_tokens` is 8192 and each waiting
request at ISL=60000 wants a full 8192-token chunk, so the scheduler's budget is
exhausted after roughly the first request and the loop ends. It never walks the
deferred queue. Record 6's "Deferred peaked at 926, so up to 9.2 s of dead engine
time in one pass" was an upper bound that is never reached, and every sentence
that reasoned from `Deferred` peak x 10 ms is void.

What survives: both defects are still real defects, and the regression tests
still bite (3 of 5 fail on the old code). What does not survive is the claim that
fixing them buys throughput. **The PR must not claim a performance win.**

What replaces it: **one 10 ms sleep per scheduling pass**, on the engine thread.
Sizing check — IP runs 10.2 steps/s, so if the sleep fires on about two passes in
three that is ~6.5 ms/step, which is exactly the measured IP-only surcharge.

---

## Eliminated today for the common +5.7 ms/step

All five from existing logs, no machine time:

| candidate | verdict | evidence |
|---|---|---|
| cudagraph mode downgrade | **dead** | all six arms log `FULL_AND_PIECEWISE`; `requires_piecewise_for_cudagraph` only fires on `use_layerwise`, which we do not set |
| LMCache's store / D2H copy | **dead** | 1b c=1000 warm window: TP0 did 1,423 stores totalling **4.5 s** of a 724 s run = 0.6% |
| prefix hash chain (O(N^2/chunk)) | **dead** | LMCache times it itself inside `profile_process_tokens`, which is part of that same 4.4 s |
| stalls / duty cycle | **dead** | duty (mean/p90 rate) is 0.93–0.99 in every arm, MP highest. Not idle time — each step is genuinely slower |
| scheduler config drift | **dead** | `max_num_batched_tokens=8192` in all arms |

The store path deserves a note even though it is not the mechanism: with
`local_cpu: false`, `cache_engine.store()` still runs the full prefix-hash chain
and still allocates and fills a host buffer before anything decides there is
nowhere to put it (`put_time` is 0.1 ms against a 14 ms `offload_time`). It is
wasted work — just not 9% of the run at this operating point.

---

## Unblocked: py-spy needs no sudo

`kernel.yama.ptrace_scope` is 1 on this box, which permits tracing **descendants**.
`py-spy record --subprocesses -- vllm serve ...` makes py-spy the parent, so the
EngineCore and the 8 TP workers are its descendants and are all sampled. Verified
before use on a throwaway spawn-based script: child frames captured, 0 errors, and
py-spy flushes its output file when the tree is torn down. The standing ask for
`sudo sysctl -w kernel.yama.ptrace_scope=0` is **withdrawn**.

---

## In flight (approved 17:07, queued behind 1f as `scripts/q_1g_1h.sh`)

**1g — `lookup_backoff_time: 0.0`, config only, no source change.**
`configs/lmcache_gpu_only_nobackoff.yaml` differs from 1b's config by exactly two
lines; the knob was verified to reach the client before launch. Tests whether the
IP-only +6.5 ms/step is the engine-thread sleep.
Pre-registered: confirmed at 680–700 s, refuted at >= 715 s.
If confirmed, the right fix is *never sleep on the engine thread*, not *throttle
the sleep* — and the PR gets rewritten around that.

**1h — py-spy profile of the MP arm at c=1000.** MP is the clean arm for the
common tax: `Deferred` is 0 in every block, so the async lookup client cannot
contaminate it, and MP shows +5.7 ms/step at both pool sizes. `--idle` is on
deliberately, because the 5.7 ms could be a block (CUDA sync, socket wait) rather
than CPU burn and a one-run budget cannot afford to miss half the hypothesis
space. The script gates on a sanity check: if the profiled run's own rate is not
~90,000 tok/s, py-spy distorted it and the profile is discarded.
Analysis tool is written and smoke-tested: `scripts/analyze_pyspy.py`.

Reading: connector frames at a few percent of a thread's wall clock => the cost is
Python-level and named, go read that call. Connector frames at ~0% => the cost is
native, and the follow-up is `--native` or a no-connector baseline to diff.

---

## State

| | |
|---|---|
| PR branch `fix_async_lookup_backoff_stall_pr` | pushed to fork, **not opened**, decision parked until 1g |
| venv LMCache | **PATCHED** (`8ea23cd1` on `vast_repro_dev`). Every run from here measures patched LMCache unless reverted |
| 1f | done, negative, `results/phase1/1f_ip_patched/` |
| 1g, 1h | queued, `logs/q_1g_1h.out` |
| `1a@200`, `1c@200` | still never run; the whole c=200 column stays uninterpretable |
| finding (2) (IP vs MP in VAST's matrix) | parked by user decision |
| records 1–3 | still lead with the falsified KV-pool mechanism, still need editing |

## Open

- The common +5.7 ms/step. Everything above is elimination; 1h is the first
  measurement aimed at it.
- `_cleanup_finished_aborted_lookups()` reads `reqs_status` unlocked — latent
  race, deliberately out of the PR, listed as a follow-up in its body.
- Decode is unmeasured (OSL=1 means prefill only).
- Ask VAST for the `GPU KV cache size: N tokens` line from each of their runs.
