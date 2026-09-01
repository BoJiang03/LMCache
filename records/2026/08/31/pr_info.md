# PR draft: lazy_offloading_policy_pr

Branch: `lazy_offloading_policy_pr` on BoJiang03/LMCache, base `LMCache/LMCache:dev`.
Three commits: policy core, connector wiring, docs. Slimmed 2026-08-31 (record 5): six unused policy mechanisms deleted, the L1 pressure-stats commit split out, all tests moved to dev. Refactored 2026-08-31 (record 6): the mode-branching pending-store facade replaced by the `OffloadPolicy` interface upstream already had.

## Title

    [Core] Eviction-aware lazy offload policy for the MP connector

## Body (paste into the template)

**What this PR does / why we need it**:

Lazy offload (`lmcache.mp.lazy_offload=true`) currently drains buffered stores with a count-triggered FIFO policy: a drain runs once `lmcache.mp.lazy_offload_threshold` finished requests (default 100) hold undrained stores at the same time, and each drain submits up to 10 of them. Buffered stores do not hold their blocks: under lazy offload `request_finished` returns False, so a finished request's GPU blocks return to the free queue and can be reallocated while its store waits. The drain revalidates each buffered chunk's admission-time block hash and drops the request's remaining chunks when they no longer match, the buffer-phase reallocation hazard already documented in `lazy_offload.md`. Under a steady serving workload the threshold and that race combine badly. In our agentic-corpus replay at CONC=40 the backlog needed about 11 minutes to reach 100, each drain then released 10 requests, and all 330 drained requests failed revalidation, so the 30 minute run stored nothing at all: final L1 usage 0 bytes, 0 lookup hits over 45M requested tokens. CONC=48 and 72 behave the same. Setting the threshold to 1 drains on the step after a request finishes and avoids the race, but that is close to storing at compute time and gives up most of what deferring is for.

This PR adds an EVICTION_AWARE drain policy and makes it the default for lazy offload. It watches the GPU block pool's free queue and submits a buffered store only when its blocks approach the eviction head, so KV that is about to be lost is offloaded and KV that stays resident is not. A deferral deadline and a per-step drain cap bound the worst case. The scheduler-side pending store is reworked around an explicit request state machine (`lazy_offload_state.py`) with a counter ledger that closes exactly (admitted = emitted + pending + dropped).

Results TL;DR, eviction-aware lazy vs the current eager default, same engine and cache config, only the policy differs:

- TTFT avg drops at every tested concurrency (-4.5% to -21%), and the gain grows with load: at CONC=72, e2e -14%, throughput +9.9%, and 13% more requests complete in the same 30 minutes.
- ITL and per-request decode speed improve at every level. One tail regression, detailed below: TTFT p99 at CONC=48.
- Why: lazy stores only what the GPU pool is about to evict, ~1/3 of eager's write volume into the same 250 GB L1, so L1 turns over ~3x slower -- effectively a larger cache. At saturation the effect is direct: eager's L1 lookup hit rate drops to 30% (entries evicted before their reuse arrives) while lazy holds 41%, and the served external share follows (32% vs 43%).
- GSM8K accuracy through the lazy path matches the no-cache baseline, and the store ledger closes exactly.

Benchmark setup: arcee-ai/Trinity-Large-Thinking-FP8-Block on 4x H200, TP=4, kv-cache-dtype fp8, max-model-len 262144, gpu-memory-utilization 0.90 (GPU KV cache 3.25M tokens, 26.5 GiB per rank); LMCache MP server with 250 GB CPU L1, chunk size 256, LRU. Workload: aiperf replay of a public agentic trace corpus (semianalysis-cc-traces-weka-062126-256k, inferencex-agentx-mvp scenario), 30 min benchmark plus 10 min grace per arm, fixed seed. Every arm starts cold (fresh MP server, fresh engine); the arms differ only in the lazy-offload connector keys. The lazy arm runs this PR's defaults plus lazy_offload_max_deferral_seconds=30 and lazy_offload_store_release=lru_tail. Baseline is the current default store-at-compute path ("eager" below). Driver script and raw per-arm artifacts (aiperf exports, metrics snapshots, store ledgers): [ab_chain.sh](https://github.com/BoJiang03/LMCache/blob/lazy_offloading_policy_dev/records/2026/08/31/artifacts/sweep/ab_chain.sh) and the surrounding [artifacts directory](https://github.com/BoJiang03/LMCache/tree/lazy_offloading_policy_dev/records/2026/08/31/artifacts). Values are eager -> lazy:

| CONC | TTFT avg (s) | e2e latency avg (s) | output tok/s | requests completed |
|---|---|---|---|---|
| 32 | 3.38 -> 2.88 (-15%) | 41.3 -> 38.6 (-6%) | 243 -> 249 (+2.5%) | 618 -> 656 (+6%) |
| 40 | 5.37 -> 5.13 (-4.5%) | 72.0 -> 71.8 (-0.4%) | 225 -> 232 (+3.1%) | 506 -> 508 (+0.4%) |
| 48 | 10.09 -> 8.02 (-21%) | 88.3 -> 81.5 (-8%) | 210 -> 218 (+3.8%) | 478 -> 497 (+4%) |
| 72 | 47.97 -> 40.47 (-16%) | 166.5 -> 142.4 (-14%) | 197 -> 216 (+9.9%) | 433 -> 488 (+13%) |

output tok/s is the aggregate across all in-flight requests. Per-request decode speed (output_token_throughput_per_user, avg) also improves: 29.2 -> 31.5 tok/s at CONC=32, 22.3 -> 22.7 at 40, 17.4 -> 19.0 at 48, 10.3 -> 11.7 at 72. ITL p99 improves at every level, most in the mid range (212 -> 160 ms at CONC=40, 313 -> 171 ms at 48). One regression to note: at CONC=48 lazy's TTFT p99 is worse (50.6 s vs 38.5 s) while avg/p50/p90 all improve; at 72 the p99 flips back in lazy's favor (105.4 s vs 159.6 s).

Direct policy-effect indicators, same runs (eager -> lazy; prefill shares over the profiling window):

| CONC | L1 chunks written | prefill from L1 | prefill from GPU cache | mean store deferral (drain steps) |
|---|---|---|---|---|
| 32 | 563k -> 396k (-30%) | 16.3% -> 14.8% | 54.5% -> 59.1% | 501 |
| 40 | 704k -> 400k (-43%) | 30.4% -> 34.3% | 32.3% -> 29.2% | 313 |
| 48 | 1,175k -> 370k (-69%) | 39.3% -> 42.0% | 20.8% -> 23.3% | 269 |
| 72 | 1,072k -> 435k (-59%) | 32.1% -> 43.1% | 4.6% -> 7.8% | 159 |

The policy stores 30-69% fewer chunks while the combined hit share (L1 + GPU cache) rises at every level. At low load the gain shows up as GPU-cache residence (deferred blocks stay hittable on the GPU); at saturation it shows up as L1 hit share: writing a fraction of the chunks into the same 250 GB slows L1 turnover, so entries survive until their reuse -- at CONC=72 eager's lookup hit rate is 30% (content evicted before reuse) against lazy's 41%, with near-identical rates at CONC=48 where L1 churn is not yet binding. Mean store deferral is how many drain steps a stored op waited between admission and submission; that wait is what converts into residence and saved bandwidth. FIFO arms were run at each concurrency as well; no store survived drain-time revalidation (see above), so they measure the no-offload path and are left out of the tables. As one reference point, at CONC=40 the FIFO arm reads as a no-L1 baseline: TTFT avg 9.24 s, 150.9 output tok/s, 381 requests completed, external hit share 0% -- which is also what either offloading policy is worth relative to no offload at all.

Correctness was checked end to end with GSM8K (20-shot, greedy, Qwen/Qwen3-8B, TP=4): 120 questions, two passes against one engine. Pass 1 runs cold; pass 2 re-sends the identical prompts after the GPU prefix cache has turned over (2048-block pool), so the prefill can only be served from L1. With lazy offload on, 93-94% of pass-2 prefill tokens came back from L1 (vLLM's own prefix-cache hit rate 0), and strict-match accuracy (0.908-0.917) stays inside the no-cache baseline's own cold/cached spread (0.900-0.925). The policy's store ledger closes exactly over the run (admitted = emitted + pending + dropped). Run on both the development branch and this branch. Harness: [repro/pr4499](https://github.com/BoJiang03/LMCache/tree/lazy-offload-policy-repro/repro/pr4499) (entry point accuracy.py; the directory is named after upstream #4499, whose lazy-offload results it was originally built to reproduce).

**Special notes for your reviewers**:

- On the diff size: +2.7k lines, of which 0.4k is docs. Production code is +2.3k in two independently reviewable commits: the policy core (1.6k, self-contained new files with no vLLM imports) and the connector wiring (0.7k). Reviewing commit by commit is recommended.
- FIFO remains available unchanged via `lmcache.mp.lazy_offload_policy=FIFO`. Its threshold semantics are untouched; the buffer-phase reallocation hazard behind the observation above is already documented in `docs/design/integration/vllm/lazy_offload.md`, and what to do about the default is left for a separate discussion.
- The policy ships three knobs: `lazy_offload_horizon_steps`, `lazy_offload_max_drain_per_step`, and `lazy_offload_max_deferral_seconds`, plus `lazy_offload_store_release` on the manager. Everything the development branch carried beyond those (a break-even prefix gate, an allocation-announcement path, content deduplication, a covered-prefix advance, an adaptive danger floor, and a per-step block-volume cap) was removed before this PR: measured over four 33-minute arms and 14,799 admissions, each of those either never fired or moved under 0.1% of the traffic, while the deferral deadline released 57-77% of all emissions.
- Both policies implement one `OffloadPolicy` protocol in `lazy_offload_policy/base.py`, restoring the abstract interface the package already had; `create_offload_policy()` selects between them. `LazyOffloadPendingStore` is removed: after the manager took over the lifecycle its remaining job was branching on the configured mode in every method.
- Design docs: `lazy_offload.md` (updated) and `lazy_offload_policy/eviction_aware.md` (new).

**If applicable**:

- [x] this PR contains user facing changes - docs added
- [ ] this PR contains unit tests

## Push checklist (before Bo opens the PR)

- [x] gsm8k gate 1 on lazy-offload-dev code: off 0.917 / eager 0.917 /
      lazy 0.917, lazy ext 0.942, ledger closes 191 = 178 + 2 + 11.
- [x] .so decision: Bo waived the red line for the PR worktree only.
      Built in-place with setup.py build_ext (4 extensions: cuda_ops,
      lmcache_native, lmcache_fs, lmcache_redis; upstream HEAD's CUDA
      profile no longer builds c_ops/native_storage_ops). The venv's
      editable-install hijack needed a sitecustomize at the repo root
      for engine runs (driver.py overwrites PYTHONPATH); removed after
      the gates.
- [x] unit tests on PR tree: 354 passed (lazy suites + cache_server +
      l1_pressure + l1_manager).
- [x] gsm8k gate 2 on PR tree: off 0.925/0.900, eager 0.908/0.925
      ext 0.961, lazy 0.917/0.908 ext 0.934; all within the off config's
      own cold/cached spread, apc 0, no evictions, ledger closes
      191 = 178 + 2 + 11. Engine verified running PR-tree code by
      log file:line fingerprints unique to the PR tree.
- [x] docs pass: eviction_aware.md 462 -> 306 lines, decision_model.md
      197 -> 159; contracts, config, and the ledger equation kept;
      measurement narratives cut. Docs commit amended; pre-commit clean after codespell+mypy fixes; head d45edab6.
- [x] pushed: `lazy_offloading_policy_pr` (head d45edab6; DCO sign-offs
      verified, pre-commit green locally) and
      `lazy_offloading_policy_dev` (updated to the session-log head) on
      BoJiang03/LMCache.

## After the slimming (record 5, 2026-08-31)

- [x] six dead mechanisms deleted, L1 pressure stats split out, docstrings
      brought below the norm of comparable repo modules (0.59 vs 0.64),
      all tests moved to dev. 9611 -> 2977 insertions, 5 commits -> 3.
- [x] ruff check clean, ruff format applied. mypy not run (not installed
      in any venv here; installing it would mutate a shared environment).
- [x] unit tests on the slimmed PR tree: 201 passed before the tests moved
      out; 116 passed after (the adapted pending-store suite plus
      cache_server, l1_manager, vllm_kv_cache_groups).
- [x] gsm8k gate 3 on the slimmed PR tree: off 0.908/0.908, eager
      0.908/0.917 ext 0.961, lazy 0.917/0.925 ext 0.961; apc 0 everywhere,
      l1 peak 0.75 under the 0.8 watermark, 0 evictions, all guards clean.
      Ledger closes: admitted 189 = emitted 178 + dropped_evicted 8 +
      pending 3.
- [ ] **open, needs Bo**: the CONC sweep in the body above was measured
      with `lazy_offload_danger_floor_max_blocks=8192`, a knob this PR no
      longer has. The floor raised 1-6 times in 35-58k drain steps per arm,
      so the numbers should be unchanged, but the body currently reports a
      config the PR cannot express. Either re-run one lazy arm (CONC=40,
      ~40 min, the e40 eager reference already exists) with the shipped
      defaults, or say so explicitly in the body.
- [x] all tests moved to dev per Bo. The one test file left in the PR,
      `tests/v1/test_lazy_offload_pending_store.py`, is the pre-existing
      upstream suite adapted to the interfaces this PR changes -- left
      untouched it would be red on merge. 217 lines against upstream's
      275, no new coverage.
- [ ] **flag for Bo before opening**: the PR adds ~1.3k lines of new
      production code with no new tests, and the "this PR contains unit
      tests" box is now unchecked. AGENTS.md and docs/coding_standards.md
      both ask for tests on new features, so a reviewer will raise it. The
      1,465 lines of new tests (which need porting to the refactored
      interface first, see record 6) are on
      `lazy_offloading_policy_dev` under
      `records/2026/08/31/artifacts/pr_slim/tests_moved_from_pr/slimmed/`
      and can go back in one commit.


## After the refactor (record 6, 2026-08-31)

- [x] `OffloadPolicy` protocol restored in `lazy_offload_policy/base.py`;
      `LazyOffloadPendingStore` (596 lines, one `if eviction else fifo`
      per method) deleted in favour of `create_offload_policy()`. Drain
      arguments bundled into `DrainSignals`; `observe_step` folded into
      `drain`; `AdmitResult` and `AddOutcome` deleted (the connector
      discarded the return value); several single-caller helpers and five
      pieces of dead API removed.
- [x] 2977 -> 2741 insertions. Lines of code, excluding docstrings,
      comments and blanks: 1255 -> 1089 (-13%).
- [x] ruff check + format clean, codespell clean, mypy 1.17.1 clean
      (run via uvx, no shared environment mutated).
- [x] unit tests: 253 passed.
- [x] gsm8k gate 4 on the refactored PR tree: off 0.900/0.908, eager
      0.908/0.908 ext 0.961, lazy 0.925/0.925 ext 0.935; apc 0 everywhere,
      l1 peak 0.73 under the 0.8 watermark, 0 evictions, all guards clean.
      Ledger closes: admitted 190 = emitted 177 + dropped_evicted 10 +
      pending 3. Lazy accuracy is the highest of the three arms; its pass-2
      external share sits between the two earlier gates (0.934 / 0.961), the
      run-to-run spread of this harness.
