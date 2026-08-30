# Session log: Trinity lands, and the ladder recalibrates the standard

Conversation record for 2026-08-30, continuing from record 11. Outcome: the
Trinity pick went from paper to a serving engine with a passing correctness
ladder in one afternoon. Steps 0-3 of the verification plan are done; the
CONC sweep is pending Bo's arm configuration. On the way, the acceptance
standard for L1 fidelity was recalibrated by a control experiment, which
also retro-softens the hybrid line's correctness verdict.

## 1. Two numbers before the launch

Bo asked what L0 lands at, and whether L0 should not simply be as large as
possible. Answers, now in `records/deployment_candidate.md` Part 4:

- L0 at the Trinity operating point is P(gap < 7 s) ~= 0.15-0.25, from the
  corpus gap CDF anchors (10.5 s -> 0.25, 12.6 s -> 0.333). Nothing throttles
  L0; its ceiling is set by the window, the window by the pool, the pool by
  the model size, which is the R4-compatible lever. Wanting L0 large again
  is wanting coder30 back: L0 0.8, W 272 s, L1/L0 0.05.
- The absolute L1 number implied by R2's ratio: L1 returns 20-60 percent of
  input tokens, target ~0.4, against 10.0 percent as the best figure any
  arm has ever measured. Same metric as `tokens_retrieved / isl_sum`.
- L1 size set to 250 GB: capacity must never be what caps realised L1/L0.
  Sized from T(r=3) ~= 60 s of offload stream at the no-dedup worst case.

## 2. SWA support: read the code, not the vibes

Bo asked whether LMCache supports SWA. It does, by design:

- `kv_cache_groups.py:76` reads vLLM's per-layer `SlidingWindowSpec` and
  tags kernel groups with `sw_size_tokens`; validation accepts SWA (only
  cross-attention and non-align mamba are rejected).
- `--separate-object-groups` on the MP server splits object groups by
  window size; retrieve loads only each SWA group's 16-chunk suffix
  (window 4096 / chunk 256). Added to the launch configuration.
- Two sub-risks named: (a) lazy stores snapshot block hashes at eviction
  time and SWA null blocks would reject the store
  (`lazy_offload_pending_store.py:321`, `REJECTED_UNHASHED_BLOCK`);
  (b) stored volume might be dense. (b) is now measured (section 5); (a)
  waits for the lazy arm of the sweep.

## 3. The launch, and the two environment traps

Bo approved the plan ("开始吧", then "ok" on GPU set 1-4 and the pull). The
2B probe server on GPU 7 was retired on his instruction first, including two
orphaned 45 GB EngineCores from earlier attempts.

Weights: 81 shards, 377 GiB, ~25 min to `/raid/data/hub`. transformers in
vllm-lazy knows `afmoe` natively, no trust-remote-code.

Bare start died twice, both environmental, both now in the candidate doc:

1. `Assertion failed: !cubin.empty() || isPathValid(path_)` in cudagraph
   capture: flashinfer's `fp8_blockscale_gemm_sm90` JIT-compiles via nvcc
   (`sh: nvcc: not found`). Fix: `/usr/local/cuda/bin` on PATH. Kernels
   cache under `~/.tensorrt_llm/cache`.
2. Triton launcher gcc failure: the doc's CPATH was incomplete. Both pydev
   include dirs are needed; the second resolves
   `x86_64-linux-gnu/python3.12/pyconfig.h`. Same as harness `env.sh:13`.

Third start clean. Measured: pool 26.73 GiB/GPU x 4 (3,275,586 tokens),
concurrency 12.50x at 262k (~31 contexts at ISL 107k, hybrid manager
windowing the SWA groups), smoke decode coherent. The predicted pool was
~106 GB; the measurement matched the prediction.

## 4. Registration and the storage volume answer

With the MP server (chunk 256, L1 250 GB, LRU, separate object groups) and
the connector, registration shows 4 kernel groups of 15 layers: three SWA
(`sw_size_tokens=4096`), one full attention. 512 B/token/layer/rank, so the
full-attention slice is 30,720 B/token across TP=4.

Stored bytes measured twice, and the first reading was misread: 3,460,300,800
bytes for a 28,155-token prompt is exactly 110 chunks x 256 x 122,880, i.e.
**all 60 layers stored dense** (8 objects per chunk). Not wrong -- retrieve
reads only the SWA suffix -- but the L1 stream is 4x the full-attention
figure, so the 250 GB sizing holds ~2.2M tokens and should be revisited
toward 500 GB for the sweep if eviction age drops below target.

## 5. The ladder, the failure, and the control that acquitted it

This 0.23 build has no `/reset_prefix_cache` route and does not report
`cached_tokens`. L0/L1 isolation is done the hard way: restart the engine
(L1 lives in the MP server). That is also the cross-instance reuse pattern a
real deployment cares about.

- Round 1: correctness2's prompt, cold engine, L1 read 872 chunks,
  byte-identical to the warm baseline.
- Ladder store phase: five 28k repeat-paragraph variants plus one 21k
  needle prompt, all stored cold with references saved.
- Check phase after restart: **5/5 byte-identical** on the paragraph
  prompts. The needle prompt FAILED the byte comparison -- the signature
  that killed the hybrid line.

Then the control that changed the interpretation. The divergence began at
generated token 0, but both texts recalled the needle and both were
coherent. A plain L0 partial-hit rerun on the same engine, LMCache not in
the loop, was measured at **0.3153 nats** max logprob delta with top-5 sets
reshuffled: fp8 KV plus recompute-tail kernel shapes make ~0.3 nats the
stack's own noise floor. Byte-exact long-form reproduction is not achievable
engine-to-itself.

The L1 legs measured against cold references: 0.2418 and 0.1637 nats, both
BELOW the L0 control, answer tokens (' BLUE-OR CH ID - 7 734') reproduced
exactly, flips only at near-tie punctuation/EOS after the answer.

Verdict: **the ladder passes** under the correct acceptance test, L1-path
perturbation <= L0-control perturbation. Corollary recorded in the candidate
doc: the hybrid line's 0.34 nats (record 11 section 3) is the same
magnitude as this stack noise, so that piece of its correctness verdict is
no longer evidence of corruption; the hybrid's hard blocker remains the
0.28 rank-4 layout bug.

## 6. State at close

- Engine (4th start) + MP server live on GPUs 1-4, ports 8973/8971/8972,
  L1 warm with the ladder prompts. GPU 0 belongs to other users; GPU 7 idle.
- `records/deployment_candidate.md` rewritten around Trinity and back-filled
  with every measured number; Parts 5, 6, 8 current as of this session.
- Probe scripts in the scratchpad: `run_once.py`, `ladder.py`,
  `logprob_probe.py` (the restart-based ladder), plus the older
  `correctness2.py` / `needle.py` which assume `/reset_prefix_cache`.
- Step 4, the CONC sweep (Running ~15/18/21, lazy arm, `rejected_unhashed`
  watch), is proposed and waiting on Bo's arm configuration.
