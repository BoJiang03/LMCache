# Volume-neutrality controller and the acceptance rounds

Session continuation of record 10 (same conversation, post-compaction
segment). Code state: commit 42dc3c81 on lazy-offload-publish, 4 local
commits ahead of origin (230d15bc -> 7fe73fce -> 190d3c97 -> 42dc3c81),
NOT pushed -- everything stays local per user directive. Tree clean;
records/ is gitignored, so record edits never dirty it.

Detailed evidence, tables, and design rationale live in record 10's
appendices; this file is the conversation log for the segment.

## What happened, in order

1. **y60 verdict** (knob v1, EMA threshold): acceptance bar MET at 60G.
   Lazy+450 sumD +2.0/-4.2 vs eager-eager noise +6.4; preempts back to
   1/1. Wart: degrade_transitions=10/8 (EMA flapping on bursty
   eviction). Record 10 "y60 verdict".

2. **z90 verdict**: no regression at 90G. sumD -19.0/-17.7, same band
   as pure tail's historical wins (-18.5/-27.8). Also flapped
   (transitions=8). Record 10 "z90 verdict".

3. **User asked for the mechanism** (why lazy loses at 60G, how we fix
   it): answered in chat -- deferral's danger-triggered pins coincide
   with allocation bursts by construction; degradation removes the
   deferral when the retrieval dividend is gone.

4. **hot/cold rerun REGRESSED** (knob v1): at L1_GB=40 the knob
   degraded (self-fulfilling: short residence -> immediate emission ->
   +5G useless hot-set stores -> the churn that justified degrading)
   and destroyed the tail win (ext 0.5 -> 0.0, cold TTFT 299/639ms ->
   746ms = eager). Key discovery: NO fixed threshold separates
   hot/cold-40G (needs <60s) from agentx-60G (needs >320s). Also: the
   first launch died on system python (harness needs
   SMOKE_PYTHON/SMOKE_VLLM -> vllm-lazy venv). Record 10 "hot/cold
   rerun".

5. **User picked option (a)** -- fix the signal -- with the directive:
   design GENERALLY, not against our existing workloads.

6. **v2 controller designed and shipped** (commit 42dc3c81). Invariant:
   degrading may change the timing of stores, never their volume. The
   counterfactual volume is unmeasurable passively, so the controller
   runs bounded trials: churn (windowed residence < threshold) opens a
   45s trial of immediate emission; commit iff trial emitted-block rate
   <= 1.25x deferred baseline, else revert + 600s cooldown; committed
   degradation lifts on residence recovery (2x) or via a periodic 45s
   deferred probe (every 480s) showing filtering returned. Emission
   ledger is policy-local (no protocol change). Sliding windows replace
   all L1 EMAs (also kills the flapping wart). Constants are
   measurement properties, none workload-fit. 348 tests green, ruff
   clean, design docs rewritten. Record 10 "Signal v2".

## In flight at record time

- hot/cold v2 (GPU 7, pid 956614): expect trials=1, reverts=1,
  commits=0 per phase; one 45s degraded blip, cold-TTFT win mostly
  recovered.
- y2-60G (par slots 0/1/5/6, chain pid 957102): expect trials=1,
  commits=1, reverts=0; degraded coverage ~55% of window (vs y60's
  72%); question is whether parity holds at that coverage.
- Predictions recorded in record 10 before results, as usual.

## Standing constraints (unchanged)

- No PRs; all operations local; push only when told (fork
  BoJiang03/LMCache).
- Author/sign-off Bo Jiang <bo.jiang@temple.edu>, no Claude trailers.
- Shared box: writes in scratchpads; par harness in old-session
  scratchpad par/, hotcold harness in current-session scratchpad
  hotcold/ (now with SMOKE_DEGRADE_SECS knob; old results archived in
  logs/pre_degrade_baseline/ and logs/knob_v1/).
- Remaining after these rounds: 90G confirm with v2 (z-shape), load
  ramp, GSM8K coverage probe.
