# PR draft: lazy_offloading_policy_pr

Branch: `lazy_offloading_policy_pr` on BoJiang03/LMCache, base `LMCache/LMCache:dev`.
Five commits: policy core, connector wiring, MP pressure stats, tests, docs.

## Title

    [Core] Eviction-aware lazy offload policy for the MP connector

## Body (paste into the template)

**What this PR does / why we need it**:

Lazy offload today (`lmcache.mp.lazy_offload=true`) drains buffered stores
with a FIFO policy that triggers only when 100 finished requests are
buffered at once. Under a realistic serving workload that count is never
reached, so no store is ever submitted: in our agentic-corpus replay the
FIFO arm wrote zero chunks to L1 over a 30 minute run at every tested
concurrency.

This PR adds an EVICTION_AWARE drain policy and makes it the default for
lazy offload. It watches the GPU block pool's free queue and submits a
buffered store only when its blocks approach the eviction head, so KV that
is about to be lost is offloaded and KV that stays resident is not. A
deferral deadline, an adaptive danger floor, and per-step drain caps bound
the worst case. The scheduler-side pending store is reworked around an
explicit request state machine (`lazy_offload_state.py`) with a counter
ledger that closes exactly (admitted = emitted + pending + dropped).

Measured on a 4xH200, TP=4, fp8 KV, agentic-corpus replay (aiperf,
30 min + 10 min grace per arm, cold caches, identical seeds), against
eager (store at compute, the current default) at four concurrency levels:

| CONC | TTFT avg | output tok/s | requests completed |
|---|---|---|---|
| 32 | -15% | +2.5% | +6% |
| 40 | -4.5% | +3.1% | +0.4% |
| 48 | -21% | +3.8% | +4% |
| 72 | -16% | +9.9% | +13% |

ITL p99 improves 25-45%. The gap grows with load: lazy writes ~1/3 of
eager's L1 volume, and at saturation the saved store bandwidth keeps the
external hit rate up (43% vs 32%) where eager's collapses. GSM8K accuracy
through the lazy path is identical to the no-cache baseline (0.917, 120
questions, 94% of pass-2 prefill retrieved from L1).

**Special notes for your reviewers**:

- FIFO remains available unchanged via
  `lmcache.mp.lazy_offload_policy=FIFO`. Its threshold semantics are
  untouched; the observation above is documented in
  `docs/design/integration/vllm/lazy_offload.md` and left for a separate
  discussion.
- The MP management protocol gains L1 pressure counters
  (`l1_pressure_stats`) used by the policy's sizing sensors; the wire
  change is additive.
- Design docs: `lazy_offload.md` (updated),
  `lazy_offload_decision_model.md`,
  `lazy_offload_policy/eviction_aware.md` (new).

**If applicable**:

- [x] this PR contains user facing changes - docs added
- [x] this PR contains unit tests

## Push checklist (before Bo opens the PR)

- [x] gsm8k gate 1 on lazy-offload-dev code: off 0.917 / eager 0.917 /
      lazy 0.917, lazy ext 0.942, ledger closes 191 = 178 + 2 + 11.
- [ ] BLOCKED on .so decision: PR worktree needs lmcache_native built from
      upstream-HEAD csrc (new `is_kv_second_tuple` binding and
      `NL_X_TWO_X_NB_BS_NH_HS` enum member); the merge-base .so from the
      dev worktree cannot serve it, and the vllm-lazy venv's editable
      install additionally hijacks `lmcache.*` submodules to the dev
      worktree (pyguard sitecustomize needed for any engine run from the
      PR tree). Rebuilding lmcache .so is a standing red line; Bo decides.
- [ ] unit tests on PR tree (blocked on the same .so).
- [ ] gsm8k gate 2 on PR tree (blocked on the same .so).
- [ ] push both branches to BoJiang03/LMCache.
