# PR 4499 lazy-offload reproduction package

This package is intentionally kept on a non-merge branch. It reproduces the
hardware results reported by PR 4499 without adding one-off workload code to
the LMCache source tree.

## Source identity

- Production code under test: `8e4e851f91316bb7994be3d096966f0d1ef0b52b`
- The reproduction commit is printed in the PR description and should be
  checked out by hash, not by the mutable branch name.
- `results/` contains the raw JSON retained from the reported H200 runs.

The reproduction commit changes no file under `lmcache/` relative to the code
SHA above. `run_apc_backfill_ab.sh` checks that invariant before running.

## Environment

Reported TP=1 runs used one NVIDIA H200; tensor-parallel runs used the
corresponding number of H200 GPUs. The software environment was vLLM 0.23.0,
PyTorch 2.11.0+cu130, CUDA 13.0, and Python 3.12.13. See
`results/environment.json`.

Use the Python environment in which the PR source and vLLM are installed:

```bash
export SMOKE_PYTHON="$(command -v python3)"
export SMOKE_VLLM="$(command -v vllm)"
export SMOKE_GPU=0
python repro/pr4499/capture_environment.py
```

The scripts start and stop both the LMCache MP HTTP server and `vllm serve`.
They use ports 26555, 28085, and 28100 by default; override them with
`SMOKE_MP_PORT`, `SMOKE_HTTP_PORT`, and `SMOKE_VLLM_PORT` when necessary.
Models are downloaded through the normal Hugging Face/vLLM path. Set
`HF_HUB_CACHE` to a non-home model volume when appropriate.

For tensor parallel runs, list the visible GPUs and set the matching TP size:

```bash
export SMOKE_GPU=0,1
export SMOKE_TP=2
```

Both workload JSON formats retain `tensor_parallel_size`; TP=1 remains the
default.

## 1. Primary hot/cold workload

This is the performance claim in the PR. It runs eager and eviction-aware lazy
offload against the same code and workload. The exact preset uses Qwen3-8B,
a 20 GiB GPU KV pool, 40 GiB of L1, 14 warmup requests, and 120 measured
requests.

```bash
export SMOKE_MODEL=Qwen/Qwen3-8B
export SMOKE_HORIZON=2.5
export REPETITIONS=3
export L1_GB=40
./repro/pr4499/run_hot_cold.sh
```

Each run writes JSON under `repro/pr4499/logs/` and checks the actual connector
mode, request count, metrics, counter ledger, warnings, and tracebacks. On the
reported H200, eager took approximately 41--43 seconds with 14--15 L1 eviction
cycles; eviction-aware took approximately 27--31 seconds with 3--6 cycles.

For a shorter functional run, set `REPETITIONS=1`. It verifies behavior but is
not enough to support a stable timing comparison. The same script supports
multi-GPU runs through `SMOKE_GPU` and `SMOKE_TP`; see the retained
[`TP2.md`](TP2.md) and [`TP4.md`](TP4.md) validations.

## 2. GSM8K retrieval correctness

The included GSM8K train/test JSONL files make prompt generation deterministic.
The exact preset runs 120 questions twice (cold and cached), concurrency four,
using Qwen3-8B and a 68 GiB L1.

```bash
export SMOKE_MODEL=Qwen/Qwen3-8B
export REPETITIONS=3
export QUESTIONS=120
export CONCURRENCY=4
export L1_GB=68
./repro/pr4499/run_gsm8k.sh
```

For a smoke run, use `QUESTIONS=20 REPETITIONS=1`; do not compare its accuracy
or timing directly with the full table in the PR.

## 3. Eager APC-backfill isolated A/B

This script compares two temporary worktrees built from the same production
SHA. The baseline differs by one visible source patch: eager mode does not
record an APC hit when LMCache misses. The script prints that diff before
starting either engine.

```bash
export SMOKE_MODEL=Qwen/Qwen3-0.6B
./repro/pr4499/run_apc_backfill_ab.sh
```

The sequence is:

1. populate both vLLM APC and LMCache;
2. clear LMCache while leaving vLLM alive;
3. replay the prompt from APC;
4. displace it from GPU with four distinct prompts;
5. request it again and measure retrieve versus recompute.

Five reported repetitions rebuilt 10 L1 objects and retrieved 2560 tokens with
the fix. The one-line baseline rebuilt nothing and retrieved nothing. Median
third-request latency was 375 ms versus 506 ms (25.9% lower, 1.35x speedup).
All outputs matched and both variants had zero warnings and tracebacks.

## 4. QASPER long-context working-set sweep

[`QASPER_WORKING_SET.md`](QASPER_WORKING_SET.md) reports a real 8K--16K-token
paper-QA workload at TP=4 while sweeping the distinct KV working set from 23 to
67 GiB against a 40 GiB L1. Every point has two repetitions with reversed
policy order. The retained per-request CSV and counter JSON are under
`results/qasper_working_set/`; the 31 MiB source dataset remains outside Git.

The sweep shows the operating envelope rather than a universal speedup. At a
34.6 GiB unique working set, eviction-aware preserved 0.461 aggregate coverage
versus 0.001 for eager and reduced returning-session E2E p50 by 28% in both
orders. Benefits declined as the working set exceeded L1, and a 23 GiB set that
fit comfortably showed no E2E p50 improvement.

## 5. Agentic session replay

[`AGENTIC_WORKLOAD.md`](AGENTIC_WORKLOAD.md) replays real SWE-agent
trajectories as growing-prefix agent sessions at TP=4, sweeping the cohort
from 8 to 48 concurrent sessions (10.8 to 69.9 GiB of distinct KV) against
the same 20 GiB GPU pool and 40 GiB L1 as the other workloads. Two
repetitions per point with reversed policy order, plus a decision-loop
attribution set and a drain-budget follow-up.

```bash
source repro/pr4499/agentic/env.sh   # see agentic/README.md for the variables
AGENTIC_SWEEP_SIZES=8,16,24,32,48 AGENTIC_SWEEP_REP=0 \
python repro/pr4499/agentic/run_agentic_sweep.py
```

It reproduces the capacity result in a third workload shape -- coverage
0.598--0.657 versus eager's 0.385 at the largest working set, with 16 L1
eviction cycles against 39 -- and isolates a per-scheduler-step cost that
scales with the pending queue and is removed by lowering
`lazy_offload_max_drain_per_step`. Raw per-run JSON is under
[`results/agentic/`](results/agentic/).

## Reading results

Treat a run as invalid if any of these guards fail:

- the requested mode is absent from the vLLM log;
- request counts differ from the preset;
- a traceback is present;
- the lazy counter ledger does not close;
- the hot/cold workload does not create the documented cache pressure;
- APC-backfill outputs differ across the three passes.

Absolute timings depend on GPU, model cache state, and server load. The useful
review signal is the same-machine A/B together with the non-vacuity guards and
raw JSON, not one isolated wall-clock number.
