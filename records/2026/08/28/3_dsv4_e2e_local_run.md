# dsv4_flash_tp: local 4-GPU end-to-end run, and the L1 rollback

Date: 2026-08-28 (session 3, continues `2_dsv4_test_restructure.md`)
Worktree: `/home/bo/LMCache-worktrees/k3_mp_dsv4`
Branches: `dsv4_ci_cost_pr` @ `c3454869`, `dsv4_ci_cost_dev` (this record)

## What this session settled

The 4-GPU end-to-end test **passes on this box**, in 375s, and the two premises
that `852cbc6c` rests on both held: the wheel's bundled DeepGEMM is enough (no
local build, nothing installed), and the DeepGEMM JIT cache really does carry a
run when it is warm. The 1-GPU test from session 2 was dropped on the user's
instruction; correctness stays with the 4-GPU test.

## The L1 rollback

Session 2 added a 1-GPU bitwise round-trip test (`a41726b8`, +362 lines in
`tests/v1/multiprocess/test_cache_server.py`, plus a `## Test coverage` section
in the hybrid-KV design doc and cross-reference comments in the pipeline, the
BK setup doc and the script header). The user ruled it out: correctness is the
4-GPU end-to-end test's job.

`dsv4_ci_cost_pr` was moved back to `852cbc6c` with `git update-ref` plus
`git checkout 852cbc6c -- <files>` (`git reset --hard` was refused by the
permission classifier). `tests/v1/multiprocess/test_cache_server.py` and
`docs/design/integration/vllm/hybrid-kv-cache-groups.md` are byte-identical to
`origin/dev` again. `a41726b8` stays reachable via reflog.

The PR branch is now exactly one commit ahead of `origin/dev` (`852cbc6c`,
ahead 1 / behind 0, fast-forwardable), which is what it should be.

A process note worth keeping: the user's question in session 2 was "is there a
way to verify correctness without launching 4 GPUs?" -- a question. It was
answered by building the thing, which is a scope error, and the cost of the
error was the rebuild plus a session of review. The rebuild itself was
approved; adding the test was not.

Also checked, since the user believed they had said so earlier: the full
session transcript has no message asking to drop L1 before this one, and
`test_cache_server.py` was never removed by anyone -- it is an existing file
with nine pre-existing tests. Stating that plainly was the right move; the
disagreement was about a fact, and the fact was checkable.

## Getting the 4-GPU test to run on this box

The script is self-contained (its own `lmcache server` + `vllm serve`, no
`launch-processes.sh`, no k8s), so it runs locally with four env overrides:

```bash
export PATH=<worktree>/.venv/bin:/usr/local/cuda/bin:$PATH
export CUDA_VISIBLE_DEVICES=1,2,3,6
export HF_HUB_CACHE=/raid/data/hub
export HF_HUB_OFFLINE=1
bash .buildkite/k3_tests/multiprocess/scripts/run-dsv4-flash-tp.sh
```

Preconditions checked before launching, all of which held:

| precondition | state |
|---|---|
| 4 idle GPUs | 1/2/3/6 idle; 0 partly used, 4/5 held at 132GB by a colleague |
| model weights | fully cached, `/raid/data/hub`, 149G, two complete snapshots (47 shards each), no `.incomplete` |
| vLLM supports the model | `DeepseekV4ForCausalLM` -> `vllm.models.deepseek_v4`, importable; `fp8_ds_mla` in `config/cache.py`; `deepseek_v4` in the tokenizer-mode list |
| ports | 8000 and 6555 free |
| host RAM for the 40GB L1 pool | 1357GB available |

`HF_HUB_OFFLINE=1` matters twice: it stops any download, and it makes hub
resolve the local snapshot path directly. It logs one
`Ignoring corrupted tree cache file ... Permission denied` for
`/raid/data/hub/.../trees/*.json` -- hub trying to write its tree cache onto a
directory we only have read access to. Harmless, and it is the desired
outcome: the run touched nothing under `/raid`.

## Result

```
=== Phase timing (total 375s) ===
  lmcache_server             10s
  prompt_build                0s
  vllm_startup              301s     80% of the run
  cold_run                   23s
  store_drain                20s
  apc_reset                   0s
  retrieve_run               21s
  verify                      0s

PASS: vLLM-run and LMCache-retrieve outputs are identical.
  outputs identical; LMCache served 4 retrieves.
```

Non-vacuity, from the LMCache server log:

- cold run stored `8192` + `768` = 8960 tokens, i.e. 35 chunks of 256, so
  several slot-compressed blocks per group were written.
- retrieve run logged four `Retrieved 8960 tokens` lines, 0.036-0.051s each --
  one per TP rank. `retrieves_before=0, retrieves_after=4`.
- both generated texts are 628 chars and `cmp` clean.

The model-side flags all engaged, from the vLLM log:
`Detected quantization_config.scale_fmt=ue8m0; enabling UE8M0 for DeepGEMM`,
`Using fp8_ds_mla data type to store kv cache`,
`Using FP8 indexer cache for Lightning Indexer`,
`Resolved architecture: DeepseekV4ForCausalLM`. `--max-model-len auto` resolved
to 1048576 and the GPU KV cache came out at 7,265,314 tokens.

Nothing was installed. That is the direct check on the SM120 deletion in
`852cbc6c`: the wheel's bundled DeepGEMM served the whole run.

## The DeepGEMM JIT cache carried this run entirely

`~/.cache/vllm/deep_gemm/cache` holds 324 kernel directories, all dated
2026-08-05 (this worktree's earlier DSv4 work). `find -newermt 2026-08-28` over
the whole cache returns **zero** files, so today's run compiled nothing: a 100%
warm hit.

This is the best evidence yet for the `DG_JIT_CACHE_DIR` + `/data/deep_gemm_jit`
hostPath change, and it also fixes the number session 1 could only guess at.
The 301s `vllm_startup` measured here is the floor -- weights from local disk,
zero JIT. CI's first run pays the download plus 324 nvcc compiles on top of
that floor, which is where the 40+ minutes goes and what the mount removes from
every subsequent run.

## Half the startup was FlashInfer nvcc, not model work

The 301s startup was opened up from the vLLM log. Weight loading is 63s and
process/NCCL init another 45s, but the two long silent stretches -- 77s for
"profile" and 102s for "warmup model" -- turned out not to be forward passes.
Timestamps under `~/.cache/flashinfer` place two nvcc builds exactly inside
them: the `sampling` module, 13:56:40 to 13:57:42 (62s), and
`fp8_blockscale_gemm_90`, 13:57:59 to 13:59:14 (75s). 137s of the 301s, with
the DeepGEMM cache fully warm. The actual dummy forwards were ~15s and ~18s.

So `DG_JIT_CACHE_DIR` fixed one JIT framework and left another one, larger,
untouched.

### The A/B

The first run left those two modules in `~/.cache/flashinfer`, so re-running
the script unchanged is the warm-cache arm:

| phase | cold flashinfer | warm flashinfer |
|---|---|---|
| lmcache_server | 10s | 10s |
| vllm_startup | 301s | **103s** |
| cold_run | 23s | 20s |
| store_drain | 20s | 20s |
| retrieve_run | 21s | 20s |
| **total** | **375s** | **173s** |

Both runs PASS, both `cmp`-clean on 628-char outputs, both `retrieves 0 -> 4`.
The warm run wrote **zero** files to `~/.cache/flashinfer`, `~/.cache/vllm`,
`~/.triton` and `~/.tilelang`, so nothing compiled anywhere.

Attributing the 198s, honestly, needs two buckets and only one of them
transfers to CI:

- **`init engine` 186.4s -> 24.0s (162s).** This is the FlashInfer cache. 137s
  of it is directly accounted for by the two `.so` timestamps above; the rest
  is the ninja/filelock/probe work around them.
- **weight loading 59.9s -> 22.1s (38s).** This is the OS page cache -- 149GB
  of weights still resident from a run 35 minutes earlier on a 1.3TB box. It is
  not a property of the change and will not reliably repeat in CI.

So the CI-transferable prediction is ~150s off the step per run, not ~200s.

### The change

`pipeline.yml` mounts `/data/flashinfer_jit` at `/root/.cache/flashinfer` on
the dsv4 step, alongside the DeepGEMM mount. No env var: FlashInfer resolves
`$FLASHINFER_WORKSPACE_BASE/.cache/flashinfer/<version>/<arch>/cached_ops` and
the base defaults to `$HOME`, so mounting the directory is the whole change.
Locally that path is `0.6.15.post1/90a`, which is the same key-on-a-new-pin
property the DeepGEMM cache argument rests on.

One difference from DeepGEMM worth stating in review: FlashInfer builds in
place under `cached_ops/<name>/` and serialises concurrent builders on a
per-module `filelock`, rather than compiling into a temp dir and renaming. Two
consequences on a shared hostPath: concurrent builds on one node wait instead
of each compiling (cheaper than recompiling, but it is serialisation), and a
pod killed mid-link can leave a partial build for ninja to redo on the next
run.

Still on the table and not done here: `~/.triton` and `~/.tilelang`. The
`jit_monitor` lines show five kernels (CuTeDSL, TileLang, Triton) compiling at
the *first request*, i.e. inside `cold_run`, not startup. Both caches were warm
on this box so the 20s cold_run does not show what CI pays for them.

## The cold run: 788s, and where it goes

A third run with every JIT cache cold. Nothing was deleted -- the five cache
roots were redirected at an empty scratch dir instead, which is both
non-destructive on a shared box and a closer match to CI's empty pod:

```
DG_JIT_CACHE_DIR  FLASHINFER_WORKSPACE_BASE  TRITON_CACHE_DIR
TILELANG_CACHE_DIR  VLLM_CACHE_ROOT   ->  <scratch>/coldcache/*
```

`vllm_startup: 788s`. The run then died in `cold_run` and never reached a PASS
(see below), but the startup number is the one that was wanted and it is clean.

Three points now:

| DeepGEMM | FlashInfer | Triton/TileLang | vllm_startup |
|---|---|---|---|
| warm | warm | warm | 103s |
| warm | cold | warm | 301s |
| cold | cold | cold | 788s |

The DeepGEMM share is directly measured rather than inferred -- its warmup
prints a progress bar: `1281/1281 [06:40]` cold, `[00:00, 8629it/s]` and
`[00:00, 15183it/s]` on the two warm runs. So **DeepGEMM ~400s, FlashInfer
~198s, Triton + TileLang + the rest ~87s** (that last one is a residual, not a
measurement, and absorbs any other cold-start variance).

Two things I had said earlier and got wrong, corrected here:

- "DeepGEMM first compile is ~487s" -- no. 788 - 301 = 487s also contains
  Triton and TileLang being cold. The progress bar splits it: 400s.
- "CuTeDSL/TileLang/Triton compile at the first request" -- that was read off
  `jit_monitor`, which only reports a handful of kernels. Bucketing the cold
  cache's file mtimes by the phase boundary shows the bulk lands in startup:

| cache | files written in startup | after cold_run began |
|---|---|---|
| deep_gemm | 240 | 6 |
| flashinfer | 22 | 0 |
| triton | 665 | 40 |
| tilelang | 45 | 15 |
| VLLM_CACHE_ROOT | 1 | 0 |

All four are now mounted in `pipeline.yml`. The whole set of caches the cold
run filled is 35MB.

### Why the cold run has no PASS

It reached `cold_run`, stored 8192 + 768 tokens successfully, and then at
15:50:50 the `lmcache server` was SIGKILLed and vLLM's EngineCore died with it
(ranks 2/3 reporting TCPStore broken pipe); the completion request returned
HTTP 500. Not a defect in anything under test: the `vllm-lazy` work line
restarted five seconds later, at 15:50:55, and its startup sweep takes out
processes matching `lmcache server`. Worth remembering when running anything
here alongside that line. `cold_run: 54s` and the 852s total from that run are
both meaningless.

## What this run does *not* prove

1. **vLLM version.** Local is `0.26.1rc1.dev306+gcb8104839`; the CI pin
   (`buildkite_latest_tested_vllm/latest_tested_vllm.txt`) is
   `0.28.1rc1.dev43+g6f7df92a8`. Two minor versions apart, and the MP pipeline
   force-reinstalls vLLM nightly on every job, so CI runs against something
   newer still.
2. **Arch.** This box is H200 (SM90). The SM120 deletion is exactly the change
   whose risk lives on SM120, and SM90 cannot exercise it. What SM90 shows is
   the weaker statement: on a supported arch, the bundled DeepGEMM needs no
   local build.
3. **Cold-start cost.** No download (weights pre-cached) and no JIT (cache
   warm), so 375s says nothing about CI's first run. It says "with everything
   warm, load + two requests is about six minutes".

So the manual `RUN_DSV4_TEST=true` build before merge is still worth doing --
now not to answer "does the test still pass" but to answer "does it pass on
SM120 against the pinned nightly, and does `/data/deep_gemm_jit` actually
land". The tell for the mounts is the second build's `vllm_startup`: with both
JIT caches persisting it should land near the 103s measured here, and anything
near 300s means the hostPaths are not sticking.

## Cleanup

Killed the two PIDs from each run's `/tmp/lmcache_mp_pids_local_dsv4_*` (the
`lmcache server` needed `-9` the first time), then confirmed GPUs 1/2/3/6 back
to ~6MiB. Both times the script's EXIT trap printed the timing summary but did
not reap the servers -- that is a property of invoking the script directly, not
a CI bug: `run-single-test.sh:93` sets `trap cleanup.sh EXIT` above it, and the
script's own header says it leaves teardown to the dispatcher's PID_FILE.

One thing that is a real CI cost, found when a cold-run attempt lost a GPU to
another process mid-load: `wait_for_server` (`common_scripts/helpers.sh:155`)
only polls `curl /v1/models`. It never checks whether the vLLM PID is still
alive, so a crashed startup is indistinguishable from a slow one and the step
sits out the whole `VLLM_READY_TIMEOUT` -- 2700s, holding four GPUs -- before
reporting failure. Not touched here; it is shared helper code and a separate
change.
Left alone: the `vllm-lazy` venv processes on other GPUs and the `/opt/venv` /
`/workspace` servers, which belong to other work lines and other people.

## Branch state

- `dsv4_ci_cost_pr` @ `c3454869` (`852cbc6c` before the four JIT-cache
  mounts and the DCO sign-off were folded in) -- `[CI][MP] Take dsv4_flash_tp off the
  default multiprocess run`. One commit, ahead 1 / behind 0 of `origin/dev`.
  Pushed to `fork` earlier and unchanged since.
- `dsv4_ci_cost_dev` -- `852cbc6c` + this session's records commit. Pushed with
  `--force-with-lease` after the rollback (`1ac6a1cd` -> `17ffc10e`), and
  updated again by this record.

A PR title/body draft was handed over in chat and is not repeated here.

## Still open

1. One manual Buildkite build with `RUN_DSV4_TEST=true`, for the two questions
   in "What this run does not prove". Doing it via the `dsv4` label needs
   `mp`/`full` **and** `dsv4` on the PR (the `mp`/`full` filter gates the
   pipeline upload; `dsv4` gates the 4-GPU group), plus GitHub triage
   permission to self-label. The env-var path needs neither.
2. A human with Buildkite access to create the nightly Scheduled Build (branch
   `dev`, `0 4 * * *`, no env vars; must be a *new* schedule, since the
   `VERIFY_AND_PIN_VLLM` clause deliberately keeps this group out of the pin
   canary), and to route nightly failures somewhere a person reads.
3. Confirm one hostPath, `/data/jit_cache`, is writable on the agent nodes
   (it started as four separate dirs; the user picked a single mount with a
   subdirectory per framework, which turns four questions for the fleet owner
   into one),
   and that the mounts actually take effect. None of the four has ever run in
   CI -- this branch has never been built. The evidence they will work is
   layered, and only part of it is measured: the three env vars are known to be
   honoured (the cold run wrote into redirected dirs), `DG_JIT_CACHE_DIR` is
   known not to be overridden (`vllm/utils/deep_gemm.py:250` guards with
   `if not os.environ.get`), and `/data/<x>` + `DirectoryOrCreate` is an
   established pattern here (`/data/huggingface` is used seven times in this
   pipeline and across eight others). What is *not* verified is that pattern on
   the 4-GPU node specifically. That worry did not survive checking: all 17
   steps in this pipeline share one queue (`k8s`) with no `nodeSelector`,
   `tolerations` or `affinity`, so there is no separate 4-GPU pool -- the
   scheduler just picks a node with four free GPUs, and every step mounts
   `/data/huggingface`. What remains is only that `/data/jit_cache` is a new
   directory, created by kubelet through `DirectoryOrCreate` exactly as the
   existing two were. A dead mount would be silent: it just recompiles. The user is asking the CI owner for a real build to confirm
   the caching. `FLASHINFER_WORKSPACE_BASE=/root` was added so that no cache
   depends on the image's `HOME`, and the script now reports each cache's
   resolved path, existence, writability and entry count before and after the
   run (`new_this_run=N` on the after pass), so two consecutive builds answer
   the question from their own logs. The report never fails the step -- on a
   first build every cache legitimately reads `entries=0`. Tested against the
   real caches and against each failure mode: unset env var -> `<unset>`,
   missing mount -> `exists=False`, files written between the two passes ->
   `new_this_run=2`, and a FlashInfer base it cannot use -> `<unresolved: ...>`
   rather than a crash (flashinfer does `os.makedirs` at import time).
4. Two consecutive builds, to read the JIT cache report. Neither path to
   running the 4-GPU group is available to the PR author: the `dsv4` label does
   not exist in the repo (only `full` and `mp` do) and applying labels needs
   triage permission, which a fork-based author lacks -- the `full` label on
   #4804 was applied by a maintainer. What *is* available is the branch itself:
   Buildkite uploads `pipeline.yml` from the PR branch, so temporarily dropping
   the group's `if:` to just the `VERIFY_AND_PIN_VLLM` clause runs it, with the
   commit reverted before merge. The second build has to wait for the first to
   finish -- the pipeline cancels running branch builds -- and hostPath is per
   node, so a second build scheduled elsewhere reads cold legitimately
   (`exists=True writable=True entries=0`, as against `exists=False` for a
   broken mount).
5. A cold run that actually reaches PASS. The one measured here was killed in
   `cold_run` by the `vllm-lazy` line's restart sweep, so the cold total is
   still unknown; only `vllm_startup` survived.
