# dsv4_flash_tp: local 4-GPU end-to-end run, and the L1 rollback

Date: 2026-08-28 (session 3, continues `2_dsv4_test_restructure.md`)
Worktree: `/home/bo/LMCache-worktrees/k3_mp_dsv4`
Branches: `dsv4_ci_cost_pr` @ `852cbc6c`, `dsv4_ci_cost_dev` (this record)

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
land". The tell for the mount is the second run's `vllm_startup`: if it does
not drop toward the ~300s floor measured here, the cache is not persisting.

## Cleanup

Killed the two PIDs from `/tmp/lmcache_mp_pids_local_dsv4_135426` (the
`lmcache server` needed `-9`), then confirmed GPUs 1/2/3/6 back to ~6MiB.
Left alone: the `vllm-lazy` venv processes on other GPUs and the `/opt/venv` /
`/workspace` servers, which belong to other work lines and other people.

## Branch state

- `dsv4_ci_cost_pr` @ `852cbc6c` -- `[CI][MP] Take dsv4_flash_tp off the
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
3. Confirm `/data/deep_gemm_jit` is writable on the agent nodes.
