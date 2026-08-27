# Lazy offload smoke run, native transfer path

First clean end-to-end run of the branch. Every earlier attempt died in the MP
server's transfer path (see record 3): a stale `lmcache/cuda_ops.so` collided
with the rebuilt `lmcache_native.so` on the pybind type `PageBufferShapeDesc`,
`ensure_native()` swallowed the resulting `ImportError` as "extension not
found", and the torch fallback issued one unsplit `cudaMemcpy` across two
separately registered 64 MiB pin chunks. Rebuilding `cuda_ops` against HEAD
removes the fallback, so this is the first run where the C++ path is actually
under test.

## Configuration

| | |
|---|---|
| model | Qwen/Qwen3-Coder-30B-A3B-Instruct, TP=1, GPU 2 |
| GPU KV pool | 16384 blocks x 16 tok = 262144 tok (24 GiB) |
| max model len | 131072 |
| L1 | 200 GB pinned host |
| connector | LMCacheMPConnector, lazy_offload=true, EVICTION_AWARE |
| policy | `horizon_steps=2.5, min_prefix_tokens=0, max_drain_per_step=64` |

Fallback warning count: 0 in the MP server log, 0 in the vLLM log. Both
processes bind `lmcache.cuda_ops` natively.

## Run 1: AgentX replay (the workload that previously killed the server)

`aiperf profile --scenario inferencex-agentx-mvp --public-dataset
semianalysis-cc-traces-weka-062126 --max-context-length 100000
--num-dataset-entries 24 --concurrency 8 --benchmark-duration 180
--use-think-time-only`

62 requests, 187.91 s, exit 0. **The MP server and vLLM were both still alive
at the end** -- the point of the run. 24 of 223 scanned traces survived the
100k context filter; largest observed peak context in the dataset is 996,579
tokens.

`--use-think-time-only` replays the traces' own think time, so it is not a load
generator: the profiling phase reported a first-request spread of 109,319 s and
produced only 51 store batches in 188 s. It exercises the shapes AgentX
actually produces, not throughput.

## Run 2: concurrent probe (transfer path under real pressure)

40 requests x 48,003 prompt tokens at concurrency 8, against the same live
server. This is the load run 1 does not supply, and it is 8x the concurrency of
the sequential verification in record 3.

| | |
|---|---|
| completed | 40/40, 0 failed |
| wall | 96.2 s |
| latency | p50 19.2 s, p99 19.9 s, max 21.8 s |
| store batches added | 42 (51 -> 93) |

## Errors

Zero, across both runs, in both logs:

| pattern | MP server | vLLM |
|---|---|---|
| `cudaMemcpy failed` | 0 | -- |
| `cudaErrorInvalidValue` | 0 | 0 |
| `AcceleratorError` | 0 | 0 |
| `Traceback` | 0 | 0 |
| `index_select` | 0 | -- |
| `compiled extension not found` | 0 | 0 |

The vLLM log does contain 41 `ERROR` lines, all `BadRequestError` inside
13:09:20-13:09:37. Those are self-inflicted: the first version of the
concurrent probe passed its prompt size as a word count, and each word in the
generator (`tok` + 5 digits) is exactly 6 Qwen3 tokens, so 48000 "tokens" was
288k tokens and every request was refused. The probe now sizes by tokens and
prints the conversion. Zero `ERROR` lines from 13:10 onward, which is the
entire window of the successful probe.

Worth noting the 400 body reports `at least 131065 input tokens` -- a lower
bound clamped at the context limit, not the true count. Reading it as the true
count is what produced the 2.73 tok/word figure that briefly contradicted the
measured 6.0.

## Lazy policy ledger

Cumulative over both runs, from the last periodic ledger line:

```
admitted=533 emitted=477 dropped_evicted=33 rejected_short_prefix=0
rejected_unhashed=0 rejected_prefix_broken=0 dropped_on_request_drop=9
dropped_failed_store=0 dropped_id_reuse=0 deduplicated=1 throttled_drains=0
drain_steps=24065 free_queue_blocks_read=1201457 requests_validated=819
blocks_validated=1692820 pending=14 held=0
```

The ledger closes: `533 == 14 pending + 0 held + 477 emitted + 42 dropped`.

Note the equation covers the `dropped_*` counters **only**. `deduplicated` and
the `rejected_*` counters return from `admit()` before `admitted += 1`
(`eviction_aware.py:710`), so they are admission-time refusals, not outcomes of
an op that was ever buffered. Summing them in as well makes the ledger appear
to over-count by exactly `deduplicated + sum(rejected_*)`; the docstring's
phrase "every drop counter" is what it says, and does not extend to refusals.

Sensors:

| sensor | value | reading |
|---|---|---|
| `dropped_evicted / admitted` | 6.2% | gate-1 drop rate |
| `throttled_drains` | 0 | `max_drain_per_step=64` is not the constraint |
| `free_queue_blocks_read / drain_steps` | 49.9 | mean free-queue depth walked per step |
| `blocks_validated / drain_steps` | 70.3 | mean block-hash comparisons per step |
| `dropped_failed_store` | 0 | consistent with zero transfer errors |

L1 ended at 6523 objects / 152.9 GiB of 200 GiB (76.4%).

## What this run does not cover

`min_prefix_tokens=0`, so **gate 3 never held anything**: `held=0` and
`rejected_short_prefix=0` for the whole run. That is the code path
924e2c1c ("Move gate 3 (economy) from emission to admission") changed, and it
is the newest commit on the branch. The gate-3 admission path therefore has
unit coverage only, and needs a run with `min_prefix_tokens` set above zero
before the branch can claim it is exercised end to end.

`log_final_stats()` did not run: `down_safe.sh` uses `SIGKILL`, which skips the
shutdown hook, so the `Lazy offload final counters` line never appears. The
periodic ledger is the only record after a `-9` teardown.

## Environment

Built out of tree (`--build-lib`/`--build-temp` into the session scratchpad);
only `lmcache/cuda_ops.so` was swapped into the repo, originals in
`scratchpad/so_bak/`. `*.so` is gitignored, so the tree stays clean.

Teardown via `down_safe.sh` (pidfile pid-tree plus a GPU sweep, both gated on
`uid == $(id -u)`, `GPUS=2` only). GPU 2 returned to 4 MiB; `rui`'s process on
GPU 0, `root`'s 89.8 GB on GPU 4, and my own unrelated multi_modal bisect on
GPU 1/3 were all left running. The old `down.sh` pattern-kills every
`vllm serve` on the box and must not be used here.

## Artifacts

- `artifacts/probe_conc.py.txt` -- the concurrent probe
- `artifacts/check.sh.txt` -- post-run assertions incl. the ledger equation
- aiperf export: `smoke/artifacts/lazy_smoke/profile_export_aiperf.{csv,json}`
