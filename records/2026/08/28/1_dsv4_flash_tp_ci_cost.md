# dsv4_flash_tp: CI cost and flakiness

Date: 2026-08-28
Worktree: `/home/bo/LMCache-worktrees/k3_mp_dsv4`
Base: `origin/dev` @ `0c2a6801`
Branches: `dsv4_ci_cost_pr` (code) @ `852cbc6c`, `dsv4_ci_cost_dev` (this record)

## Problem

The `dsv4_flash_tp` step landed in #4442 and turned out slow (40+ min) and
occasionally failing. A colleague reported that PRs wait on multiprocess CI
resources, some for 6h. (They named PRs 4522/4521; those numbers do not match
the complaint — 4522 is an MP-coordinator RFC issue, 4521 a closed
memory-allocator PR — so the queueing evidence is still second-hand. No
Buildkite access in this session, no build logs, no timing data.)

## What the step actually does

`.buildkite/k3_tests/multiprocess/scripts/run-dsv4-flash-tp.sh`, self-contained
(does not use `launch-processes.sh`): launch `lmcache server` with an L1-only
pool, build a fixed ~7.8k-word prompt, launch `vllm serve
deepseek-ai/DeepSeek-V4-Flash` at TP=4 with `fp8_ds_mla` / `deepseek_v4`
tokenizer / `--enforce-eager` / dev mode, send one greedy 128-token completion
(cold, populates LMCache through the slot-compression store path), sleep 20s
for the store to drain, `POST /reset_prefix_cache` (clears vLLM's APC only),
resend the identical request, then assert the two outputs are byte-identical
and that the LMCache retrieve count went up.

## Where the time goes

Never measured — this is the reason the timing instrumentation went in. Reading
the code, the ranking is:

1. `vllm_startup`: 160GB fp8 weights across 4 TP shards, plus a HF download on
   a cold `hf-cache` node. `VLLM_READY_TIMEOUT` is 2700s.
2. Per-job fixed cost shared by the whole MP suite, in
   `.buildkite/k3_harness/setup-env.sh`: `uv cache clean` and then a forced
   reinstall of vLLM nightly + torch cu130 (`--reinstall-package` on
   transformers/tokenizers/hf-hub/safetensors/vllm), then LMCache editable
   from source. Deliberate — it works around a base-image `GenerationConfig`
   ImportError — so left alone.
3. DeepGEMM JIT: every kernel is compiled on first use with nvcc. Estimated a
   few dozen kernels for this model, 10-30s each, so roughly 2-10 min. Not
   measured.
4. The SM120 DeepGEMM build (1-2 min), now removed.
5. `sleep 10` + `sleep 20`: 30s, ignored.

## The real problem is not the 40 minutes

`dsv4_flash_tp` is the only 4-GPU step in the whole repo. Everything else,
across all nine pipelines, takes one or two GPUs:

```
dsv4_flash_tp        4 GPU   90m  retry
kimi_linear_tp       2 GPU   60m  retry
hma_lm_eval_gemma4   1 GPU   60m  retry
rest of MP         1-2 GPU   30m
comprehensive      1-2 GPU   30m x10
correctness/integration/blend/sglang/unit  1 GPU  15-60m
```

`kimi_linear_tp` also runs 60 min on a 48B model and nobody complains, because
2 GPUs coexist with other jobs on a node. So the harm is node occupancy, not
duration: on a 4-GPU node the step holds the entire node for the length of the
build, up to 3x that under `x-flaky-retry` (limit 2 on both `-1` and `"*"`),
while other builds' 1-GPU steps queue.

It was also placed first in the pipeline on purpose ("so the heaviest job gets
scheduled before the 2- and 1-GPU jobs"), which maximises the effect.

## How this CI skips work (surveyed)

Three layers, none of them per-test capability checks:

1. Pipeline level, Buildkite web trigger filter (`BK_WEB_SETUP.md`):
   integration and correctness run on every PR; multiprocess needs label `mp`
   or `full`, or a `dev` push; amd/xpu/comprehensive similar. So the MP suite
   does *not* run on every PR — an earlier claim of mine that was wrong.
2. Build level, `common_scripts/upload-pipeline.sh` + `path-filter.sh`:
   docs-only changes annotate and `exit 0` without uploading any test step.
   `.buildkite/**` always runs; `force-ci` label bypasses; scheduled builds are
   never skipped.
3. Step/group level, `if: build.env(...)` in `pipeline.yml`:
   `VERIFY_AND_PIN_VLLM` (pin canary mode) and `NEED_UPLOAD` (nightly baseline
   upload) are the only two.

No GPU test uses an env gate for cost. Cost is controlled entirely by PR labels
at the pipeline level, and inside a suite everything runs. Temporarily
disabling a step has been done by commenting out its YAML.

## DeepGEMM: the SM120 workaround is dead

vLLM #52035 (`025d56a11`, 2026-08-12) repointed the pin back to
`deepseek-ai/DeepGEMM` nv_dev tip `8b1392b9` ("Pinned to the tip of the nv_dev
branch (SM120 support)"). Verified:

- `8b1392b9`'s `csrc/apis/layout.hpp` has the `arch_major == 12` branches
  (lines 49/57/112/116/132), so the old `DG_HOST_UNREACHABLE("Unknown SF
  transformation")` fallthrough is unreachable again. `csrc/jit_kernels/
  heuristics/sm120.hpp` exists.
- CI's pin (`buildkite_latest_tested_vllm/latest_tested_vllm.txt`) is
  `0.28.1rc1.dev43+g6f7df92a8`, dated 2026-08-28, which is ahead 612 / behind 0
  of `025d56a11`, and its `deepgemm.cmake` carries `8b1392b9`.

Also: the script's short-circuit `python3 -c "import deep_gemm"` could never
succeed, because the wheel ships it as `vllm.third_party.deep_gemm`. So on
SM120 the clone + `bdist_wheel` ran on every single run.

## DeepGEMM JIT cache: safe to share, but scope it

`vllm/utils/deep_gemm.py:250` sets `DG_JIT_CACHE_DIR` to
`VLLM_CACHE_ROOT/deep_gemm` only when unset, so an explicit value wins. The pod
mounted only `hf-cache` and `dshm`, so the cache was empty every run.

Cache key, from `csrc/jit/compiler.hpp:101`:

```
hash(name $$ signature $$ flags $$ code)
  signature = "NVCC<major>.<minor>"                    (or the NVRTC version)
  flags     = -std=c++20 ... -I<install path> <arch>
              arch = -gencode=arch=compute_120a,code=sm_120a  on SM120
                     --gpu-architecture=sm_90a               otherwise
  code      = full kernel source
```

Arch, CUDA version, DeepGEMM version and install path are all in the key, so
reuse cannot serve a wrong cubin. Writers compile into a temp dir and `rename`
into place (`compiler.hpp:111-142`, with a comment about losing the rename race
being fine), so concurrent builds on one node are safe.

First cut mounted all of `/root/.cache/vllm`. Narrowed on review: that root
also holds `torch_compile_cache/` and `gpu_p2p_access_cache_for_*.json`, and
while `--enforce-eager` means this step writes neither today, the blast radius
is unnecessary. Final form sets `DG_JIT_CACHE_DIR=/root/.cache/deep_gemm` on
the step and mounts only that.

## Changes (branch `dsv4_ci_cost_pr`, commit `852cbc6c`)

```
 .buildkite/k3_tests/multiprocess/BK_WEB_SETUP.md   |  29 +++++
 .buildkite/k3_tests/multiprocess/pipeline.yml      |  56 ++++++---
 .../multiprocess/scripts/run-dsv4-flash-tp.sh      | 134 +++++++--------------
```

Three substantive edits in `pipeline.yml`:

- 4-GPU group condition:
  `build.env("VERIFY_AND_PIN_VLLM") !~ /true/ && (build.source == "schedule" || build.pull_request.labels includes "dsv4" || build.env("RUN_DSV4_TEST") == "true")`
- `env: DG_JIT_CACHE_DIR: /root/.cache/deep_gemm` on the step
- `deepgemm-jit` volume: hostPath `/data/deep_gemm_jit` -> `/root/.cache/deep_gemm`

In the script: `provision_deepgemm_sm120()` and its two ref variables deleted
along with the 38-line pin-history comment (the `VLLM_USE_DEEP_GEMM=0` warning
was kept); `begin_phase`/`end_phase`/`print_timing_summary` added with a `trap
... EXIT` so a failed or killed run still reports, and eight phase marks
(`lmcache_server`, `prompt_build`, `vllm_startup`, `cold_run`, `store_drain`,
`apc_reset`, `retrieve_run`, `verify`).

No test logic touched: prompt, `MAX_TOKENS=128`, `STORE_DRAIN_SECONDS=20`, both
requests, the byte comparison and the non-vacuous retrieve assertion are
unchanged. `MAX_TOKENS` and the two sleeps were considered and left alone — 30s
of noise inside a 40-minute step.

`build.source` and `build.pull_request.labels` were both checked against the
Buildkite conditionals documentation; `schedule` is a documented value of
`build.source`.

## Design decision: why not gate on paths (yet)

The gate as written gives no pre-merge coverage for an LMCache change that
breaks DSv4 (`lmcache/integration/vllm/kv_cache_group_edits.py`,
`lmcache/v1/kv_layer_groups.py`, `lmcache/v1/platform/*/cache_context.py`, the
MP connector). The nightly catches it post-merge, within 24h. A label does not
help, because the author of a `kv_layer_groups.py` change has no reason to
think of DSv4.

If pre-merge coverage is wanted, the trigger has to be path-based **at upload
time**, not inside the step. An in-step early exit would still request 4 GPUs,
so on a saturated fleet the job waits for a free node before it can print
"skip", and the PR's build stays red-until-finished for that whole wait — which
is exactly the symptom being fixed. `upload-pipeline.sh` already runs
`path-filter.sh` and already computes changed files, so the decision belongs
there: split the 4-GPU group into its own pipeline fragment and upload it only
when the relevant paths changed. Cost: one line in `buildkite-pipeline.yml`,
which means re-pasting it in the Buildkite web UI.

Deferred until after a real run confirms the step still passes without the
SM120 build.

## Not verified

- Nothing ran on a real 4-GPU SM120 node. Local box is 8xH200 (SM90) and GPUs
  4-7 were busy; the user asked for no GPU experiments. `bash -n` passed, the
  timing helpers were exercised standalone (normal, empty, failure paths), the
  YAML parses and the gated group's fields were inspected.
- Whether `/data/deep_gemm_jit` is writable on the agent nodes. Same hostPath
  pattern as `/data/huggingface`, and `DirectoryOrCreate` runs as root, so
  probably fine. The tell is the second run's `vllm_startup` + `cold_run`
  timings: no drop means the cache is not landing.
- Whether the JIT estimate (2-10 min) is right at all.

## Needs a human with Buildkite access

1. Create a Scheduled Build on the multiprocess pipeline: branch `dev`, daily
   (`0 4 * * *` suggested, offset from comprehensive's 2am), no env vars. Until
   this exists there is no nightly coverage — only the label and manual paths.
   It must be a *new* schedule: the pin canary is also a scheduled build and
   the `VERIFY_AND_PIN_VLLM` clause deliberately excludes this group from it.
2. Route nightly failures somewhere a person reads.
3. Confirm `/data/deep_gemm_jit` writability with whoever runs the k8s fleet.

Suggested first action before any of that: one manual build with
`RUN_DSV4_TEST=true` to confirm the step still passes and to read the phase
timings.

## Side decisions

- Branch naming convention stated by the user: `<line>_pr` carries only what
  goes into the PR, `<line>_dev` carries everything else including `records/`.
  Written to `~/.claude/CLAUDE.md` (created; there is no global memory store,
  only per-project ones) and folded into the project memory entry
  `push-to-fork-user-opens-pr`.
- `.git/hooks/pre-push` allowlist extended to `*_dev` so the convention works
  without per-line edits.
