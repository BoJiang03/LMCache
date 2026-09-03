# 2026-09-02 (7) — 1f is a negative result; the tax, measured in ms per forward step

## Headline

1. **The fix in record 6 buys nothing.** 1f (patched LMCache, otherwise
   byte-identical to 1b) measured cold 728.1 s vs 1b's 727.7 s, and warm
   718.1 s vs 1b's 724.1 s. The pre-registered band said "REFUTED if
   >= 715 s"; both passes are above it. Refuted. The warm pass's 0.8% is
   barely more than 1b's own 0.5% cold-to-warm spread. Record 6 now carries
   a correction box.

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

## 1g (done) and 1h (in flight) — approved 17:07, queued as `scripts/q_1g_1h.sh`

**1g — `lookup_backoff_time: 0.0`, config only, no source change. DONE, REFUTED.**

    arm                 backoff   ms/step   p50 tok/s   Deferred max
    1c no connector     --           85.3      95,997       0
    1e MP               --           91.0      89,990       0
    1b IP unpatched     10 ms        97.5      83,996     531
    1f IP patched       10 ms        97.5      83,994     599
    1g IP               0 ms         97.5      83,993     575

cold 726.3 s against a pre-registered refutation band of >= 715 s. **Removing the
sleep entirely changes nothing** — the three IP arms' steady-state rates differ by
3 tok/s. The arm is valid, not a misconfiguration: the knob was asserted to reach
the client before launch, and `Deferred` still peaks at 575, so the async lookup
path is demonstrably active.

**Conclusion: the async-lookup backoff is not the IP-only +6.5 ms/step.** Two
independent instruments — a source patch and a config knob to zero — both measure
nothing. Everything downstream of `Deferred` as a *throughput* explanation is dead,
including record 6's headline and the replacement hypothesis in this record's
first draft. `Deferred` being large is real, and it costs nothing.

**PR consequence.** `fix_async_lookup_backoff_stall_pr` now has no performance
justification at all. What is left is a genuine correctness fix (a sleep held
under the lock the response thread needs, and an O(pending) backoff) with tests
that bite, and it must be described that way and only that way.

**What the IP-only surcharge might be instead** — not yet tested, listed so the
next session does not re-derive it. IP runs a full LMCache engine *inside each of
the 8 TP worker processes*, where MP hands the work to a separate process:
`wait_for_save` walks every request in the connector metadata every step and calls
`lookup_unpin` per request; `build_connector_meta` builds a fresh O(tokens)
`slot_mapping` tensor per request per step; and all of it contends for the same
GIL as the worker's own Python. None of this is measured yet.

**1h — py-spy profile of the MP arm. ABANDONED: the instrument is unusable here.**

Its own pre-registered sanity gate caught it 27 minutes in. The profiled run was
sampling at **23,998 tok/s = 341.4 ms/step**, against MP's true 91.0 — a **3.75x
distortion**, i.e. 73% of the wall clock was py-spy's own overhead. The gate says
a profile taken under that must be discarded, so it was, and the run was stopped
at 18:16 rather than burning 30 more minutes producing an uninterpretable file.

**Why it is this expensive, and why lowering the rate does not fix it.** With
TP=8 the eight workers are lockstepped on NCCL collectives. py-spy PTRACE-stops a
process to walk its threads, so stopping *any one rank* stalls all eight. The cost
is not the sum of per-process sampling overhead, it is that overhead multiplied
through the collective. Dropping `-r` scales the distortion down proportionally
but never removes the amplification, and thinning the samples to ~1 Hz to get the
overhead into single digits leaves too few samples to resolve a 6% effect.

**Do not retry py-spy on a TP>1 vLLM.** The finding that py-spy *can* attach
without sudo (record above) still stands and is still useful for single-process
targets; it is the TP lockstep, not the ptrace permission, that makes it useless
for the workers.

`scripts/analyze_pyspy.py` is written and smoke-tested and kept for that case.
vLLM's in-tree torch profiler (`--profiler-config.profiler=torch` plus the
`/start_profile` and `/stop_profile` endpoints) is the instrument to reach for
instead if a profile is needed later: it is in-process, so no ptrace and no TP
amplification, and it can cover a short window rather than the whole run. Its
limitation is that it sees the workers only, not the EngineCore scheduler.

---

**1i — a connector that does nothing. Launched 18:18:56, approved 18:17.**

After 1h, stop trying to profile and bisect instead, using the instrument that
already works to 0.1 ms. `nullconn/null_connector.py` implements every abstract
method of `KVConnectorBase_V1` and does nothing in all of them. It lives in our
repo; no LMCache change and no vLLM change. vLLM still walks its entire connector
code path: `maybe_transfer_kv_layer` wraps all 36 attention layers,
`build_connector_meta` runs every scheduler step, the worker-side load/save hooks
fire every forward, the `KVOutputAggregator` is installed, and
`get_num_new_matched_tokens` is consulted for every waiting request.

Pre-registered:

| 1i lands at | conclusion |
|---|---|
| ~91 ms/step (686-700 s) | the +5.7 ms is **vLLM's own connector plumbing**; LMCache is not the thing to fix and this becomes an upstream vLLM finding |
| ~85 ms/step (620-635 s) | the plumbing is free; the +5.7 ms is **inside LMCache**, in code IP and MP share |
| 640-680 s | it splits, and the split is read off directly as (1i - 85.3) vs (91.0 - 1i) |
| > 700 s | the null connector is doing something unintended; discard |

**RESULT: 632.6 s cold, in-engine steady state 95,993 tok/s = 85.3 ms/step.**
Lands in the 620-635 s band: **the plumbing is free and the +5.7 ms is inside
LMCache.**

| arm | in-engine ms/step | p50 tok/s | cold dur |
|---|---|---|---|
| 1c no connector | 85.3 | 95,997 | 623.0 s |
| **1i null connector** | **85.3** | **95,993** | **632.6 s** |
| 1e MP | 91.0 | 89,990 | 686.0 s |
| 1b / 1g IP | 97.5 | 83,996 / 83,993 | 727.7 / 726.3 s |

Two numbers, and they disagree slightly, so both are reported. The in-engine
steady-state rates are identical to 4 tok/s, i.e. **+0.0 ms/step**. End to end 1i
is 1.5% slower than 1c (632.6 vs 623.0), which the steady-state p50 does not see —
that is startup and drain, not per-step cost. The honest statement is therefore:
**vLLM's entire connector code path costs at most ~1.5% end to end and is
indistinguishable from zero in steady state.**

All three assertions passed: 9 NullConnector instantiations (8 workers +
EngineCore, so both halves of the path are live), pool 13,724,416, and vLLM never
printed a `Deferred` field at all because this connector never returns None.

### The decomposition this settles

| | ms/step | where the fix would go |
|---|---|---|
| vLLM connector plumbing | **+0.0** (<= 1.5% end to end) | nowhere; not the problem |
| **LMCache, common to IP and MP** | **+5.7** | **LMCache** — this is the ~9% and the objective |
| LMCache, IP only | +6.5 | LMCache; still unexplained after 1f and 1g |

So "LMCache costs ~10% while doing nothing" is a statement about LMCache's own
per-step work, not about the cost of vLLM's connector interface. That is the
opposite of what the pre-registered mechanism predicted, and it puts the fix
squarely in LMCache.

Confounders held fixed by construction, each verified before launch: no
`SupportsHMA` so `--disable-hybrid-kv-cache-manager` is required and the pool is
13,724,416 like 1c/1e/1b/1g; `requires_piecewise_for_cudagraph()` False so the
cudagraph mode matches; `get_required_kvcache_layout()` inherited and None so the
attention backend matches; `get_num_new_matched_tokens` returns `(0, False)` and
never None, so this arm can never defer and `Deferred` must be 0. The script
asserts the pool, the attach line and the deferred count, and aborts rather than
producing a result that merely looks interpretable.

### A named mechanism, written down at 18:30 while 1i was still running

Not a post-hoc story: 1i launched 18:22 and lands ~18:44; this was read out of
vLLM's source in between, and 1i tests it directly.

`vllm/v1/engine/core.py:158`:

```python
if self.scheduler.connector is not None:
    self.model_executor.init_kv_output_aggregator(self.scheduler.connector)
```

The gate is **"is a connector attached"** and nothing else — not what it does, not
whether it stores anything, not which connector it is. Downstream, in
`multiproc_executor.py:359`:

```python
if kv_output_aggregator is not None:
    output_rank = None                      # collect from ALL ranks
else:
    output_rank = unique_reply_rank         # collect from ONE rank
...
if output_rank is not None:
    response_mqs = (response_mqs[output_rank],)
def get_response():
    for mq in response_mqs:                 # 8 dequeues instead of 1
        status, result = mq.dequeue(...)
```

and `kv_output_aggregator` is passed to **both** `execute_model` and
`sample_tokens` (`multiproc_executor.py:315` and `:327`). So attaching any
connector turns 2 message-queue dequeues per step into **16**, and makes the
engine wait on the slowest of 8 ranks twice per step instead of on one rank.

The aggregation itself is cheap — `KVOutputAggregator.aggregate` is a few set
operations over mostly-empty outputs — so the cost, if this is it, is the extra
collection and the straggler wait, not the merge.

Why this fits every constraint the measurement imposes:

| observation | fits? |
|---|---|
| flat per-step cost | yes, it is per collective RPC |
| identical for IP and MP (+5.7 both) | yes, gated only on `connector is not None` |
| independent of KV pool size | yes, nothing to do with the pool |
| present when the connector stores nothing | yes |
| invisible in LMCache's logs | yes, it is entirely vLLM-side |
| not idle time (duty cycle 0.93-0.99) | yes, it is real work plus a real wait |
| would scale with TP | yes — TP=8 here, so 8x the collection |

**REFUTED at 18:41.** 1i landed at 85.3 ms/step — identical to no connector at
all. The null connector trips exactly this gate (`connector is not None` ->
aggregator installed -> 16 dequeues per step instead of 2, two straggler waits
per step) and it costs nothing measurable. The mechanism above is wrong, and it
was written down 17 minutes before the result precisely so it could be wrong on
the record.

---

## State

| | |
|---|---|
| PR branch `fix_async_lookup_backoff_stall_pr` | **DELETED** from the fork and locally, 18:47. See below |
| venv LMCache | **STOCK again** — `8ea23cd1` reverted by `b46f6a1c`, verified by import. Every arm from here measures unmodified LMCache |
| 1f | done, negative, `results/phase1/1f_ip_patched/` |
| 1g | done 17:43, **refuted**, `results/phase1/1g_ip_nobackoff/` |
| 1h | running from 17:44, `logs/q_1g_1h.out` |
| `1a@200`, `1c@200` | still never run; the whole c=200 column stays uninterpretable |
| finding (2) (IP vs MP in VAST's matrix) | parked by user decision |
| records 1–3 | still lead with the falsified KV-pool mechanism, still need editing |

## The PR was cleared, not downgraded

User's instruction, 18:45: this PR exists to fix the performance problem; a
correctness change is not what it should carry, so empty it.

That is the right call and it is stricter than what I proposed. I had suggested
keeping the branch and rewriting it as a correctness-only PR. But the branch was
opened to fix a performance problem, its commit is literally titled
`[Perf] Stop the async lookup client from stalling vLLM's scheduler`, and two
independent measurements say it stops nothing. Repurposing it would have shipped
a change whose real justification was never the one that motivated writing it.

Done:
- `git push fork --delete fix_async_lookup_backoff_stall_pr` — gone from the fork
- worktree removed (verified clean first) and the local branch deleted
- the content is NOT lost: it survives as `8ea23cd1` on `vast_repro_dev`, which is
  exactly where `<line>_dev` is supposed to hold it
- `b46f6a1c` reverts `8ea23cd1` so the venv imports stock LMCache again; verified
  by importing and checking the original sleep-under-lock is back

The two defects it fixed are still real and still unfixed upstream: the backoff
sleep is held under the lock `process_responses_from_workers` needs, and the
backoff is O(pending) rather than O(1) per pass. They are worth a separate,
honestly-scoped correctness PR **if and when someone wants one** — with no
performance claim attached, since there is none. They are not part of this line
of work and should not ride along with it.

## Open

- The common +5.7 ms/step. Everything above is elimination; 1h is the first
  measurement aimed at it.
- The IP-only +6.5 ms/step is now **also** unexplained, since 1g killed the only
  hypothesis for it. Candidates in the 1g section above; none measured.
- `_cleanup_finished_aborted_lookups()` reads `reqs_status` unlocked — latent
  race, deliberately out of the PR, listed as a follow-up in its body.
- Decode is unmeasured (OSL=1 means prefill only).
- Ask VAST for the `GPU KV cache size: N tokens` line from each of their runs.
