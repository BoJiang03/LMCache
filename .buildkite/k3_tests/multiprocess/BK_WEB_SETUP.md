# Buildkite Web UI Setup: Multiprocess Tests

**Steps editor**: paste contents of `buildkite-pipeline.yml` (fill in `HF_TOKEN`).

**GitHub trigger settings**:
- Filter: `build.pull_request.labels includes "mp" || build.pull_request.labels includes "full" || build.branch == 'dev'`
- Rebuild on PR label change: Yes
- Skip queued / cancel running branch builds: Yes

Heavy test (2 GPUs, Docker-in-Docker, ~45 min) — run on `"mp"`/`"full"` label or dev push, not every PR.

**`dsv4_flash_tp` (DeepSeek-V4-Flash, 4 GPUs, 40+ min)** is the only 4-GPU step
here, so any build that includes it holds a whole 4-GPU node and pushes every
other build's 1- and 2-GPU steps behind it. It is off by default and runs when
any of these holds:

- the build is **scheduled** (see below) — the nightly coverage
- the PR carries the **`dsv4`** label, alongside the `mp`/`full` label that
  gates this pipeline — for a change to the hybrid-KV-group / slot-compression
  path
- the build was started with **`RUN_DSV4_TEST=true`** in "New Build" env

> Builds whose only changes are docs/`*.md`/`LICENSE`/`.github/**` auto-pass
> via the [path filter](../README.md#path-based-skip-auto-pass-on-docs-only-changes).
> Changes under `.buildkite/` always run. Add `force-ci` label to the PR to
> bypass.

## Nightly Scheduled Build (dsv4_flash_tp)

`dsv4_flash_tp` runs on any scheduled build of this pipeline, so it needs one
to exist. Create a **Scheduled Build**:

- **Schedule**: daily, offset from the comprehensive pipeline's 2am upload
  (e.g. `0 4 * * *` — 4am UTC)
- **Branch**: `dev`
- **Extra Environment Variables**: none needed

The whole multiprocess suite runs in that build, with `dsv4_flash_tp` added.
Scheduled builds are never path-skipped, so it runs even on a day when `dev`
saw only docs commits.

The vLLM pin canary (`VERIFY_AND_PIN_VLLM=true`) is also a scheduled build, and
`dsv4_flash_tp` stays out of it: that build answers "does this vLLM candidate
work", on one 2-GPU smoke step.
