# fs_native read depth: benchmark scripts and raw results

Scripts and logs behind the `read_io_depth` / `read_max_bytes_in_flight`
change to the native filesystem connector (LMCache PR #5078). Everything here
talks to `LMCacheFSClient` directly: no vLLM, no GPU. The end-to-end numbers
quoted in the PR came from a separate vLLM harness that is not included; its
configuration and per-point results are in `results/e2e_chunk256_summary.txt`
and described at the end of this file.

## Running

Build the checkout under test in place (`python setup.py build_ext --inplace`)
and point the scripts at it:

```
export TREE=/path/to/LMCache          # checkout under test, built in place
export WORK=/mnt/nvme/l2bench         # directory ON THE STORAGE UNDER TEST
export PY=python3                     # interpreter with torch and lmcache deps
bash bench_objsize.sh                 # the object-size sweep, ~40 min
```

`WORK` receives the corpus (tens of GiB) and the logs. Scripts that read
`/proc/diskstats` take the device name through `--dev` (default `md1`); on
tmpfs the device columns read zero and are ignored. `exp_move_check.sh` takes
two checkouts, `NEW_TREE` and `OLD_TREE`.

Every script prints one `RESULT` or `pass N:` line per arm. `summarise.py`
turns an object-size sweep log into a per-rung table with a verdict.

## Method

Three things made the numbers reproducible; the scripts encode all of them.

1. **A discarded warmup.** The first arm after a large corpus write reads
   low while the drives settle. Three data sets were thrown away during this
   work because the first arm was 8%, 26% and 40% under its own repeat.
2. **Interleaved arms, two rounds.** `L P L P`, never `L L P P`, so drift
   moves both arms together. Each arm is scored on both rounds and the
   within-arm spread is reported as drift.
3. **A verdict, not a number.** `summarise.py` calls a rung `solid` only when
   the effect exceeds twice the larger drift, `weak` between one and two, and
   `NO CLAIM` below. Several rungs in the results are `NO CLAIM` and are
   reported as ties.

Each `exp_*.sh` header states its prediction and its falsifier before the
run. Three of the four experiments designed to show `num_workers` had side
effects were falsified by their own falsifiers and are kept here as such
(`exp_write_quick.sh`, `exp_mixed.sh`, `exp_probe_quick.sh`).

## Scripts

| script | what it measures |
|---|---|
| `benchB.py` | read throughput of a corpus through `LMCacheFSClient`, legacy or pooled, with `/proc/diskstats` brackets |
| `benchN.py` | `benchB.py` plus read IO count, queue depth and service time per request |
| `bench_probe.py` | a saturating read stream plus a single-object probe, reporting stream GB/s and probe latency p50/p95 |
| `bench_mixed.py` | concurrent loads and stores; store latency under read load |
| `summarise.py` | per-rung table with drift and verdict for an object-size sweep log |
| `bench_objsize.sh`, `_h2.sh`, `_h3.sh` | the 8-rung object-size sweep, default `num_workers=4` against the pool, on an array, a single NVMe (guarded, 8 GiB) and tmpfs |
| `bench_objsize_3arm.sh`, `bench_retune.sh`, `bench_l2_transfer.sh` | earlier forms of the same sweep, kept for provenance |
| `exp_ceiling.sh` | the depth-ceiling comparison: `num_workers` 4 and 256 against the byte budget at 1.5 MiB and 192 MiB objects |
| `exp_numworkers.sh`, `exp_default_cell.sh`, `exp_smallbatch.sh` | pool against a tuned `num_workers` at 6 MiB objects, several batch sizes |
| `exp_barrier.sh` | is the per-group barrier the limit at small batches (workers varied, depth fixed) |
| `exp_budget_sweep.sh`, `exp_h2_budget.sh` | budget sweeps on the array and the single NVMe |
| `exp_probe.sh`, `exp_probe_quick.sh` | Little's law check: latency of a single read under each arm |
| `exp_mixed.sh`, `exp_write_quick.sh` | write-path collateral of a large `num_workers` |
| `exp_move_check.sh` | pool in `ConnectorBase` against pool in `FSConnector`, and the depth-based tile split for a lone large batch |
| `exp_192_w64.sh` | 192 MiB objects, `num_workers` 64 and 4 against the pool, controlled |
| `exp_h3_w64.sh` | tmpfs, `num_workers` 64 and 4 against the pool, controlled |

## Hardware

* **Array:** 8x Micron 7450 NVMe in RAID0 (`md1`), fio O_DIRECT ceiling
  ~53.5 GB/s. All "array" numbers below.
* **Single NVMe:** one drive, ~6 to 7 GB/s.
* **tmpfs:** `/dev/shm`, reads are memcpy.

## Results

All GB/s are application bytes over wall time, three passes over the corpus,
O_DIRECT except on tmpfs. "default" is `num_workers=4`, `read_io_depth=0`.
"pool" is `num_workers=4`, `read_io_depth=64`, budget 1536 MiB unless stated.

### Object-size sweep, default against pool (`bench_objsize*.sh`, `results/H*_objsize.log`)

| object (chunk_size) | array default | array pool | ratio | single NVMe default | single NVMe pool | ratio | tmpfs default | tmpfs pool | ratio |
|---|---|---|---|---|---|---|---|---|---|
| 1.5 MiB (64) | 10.88 | 48.05 | 4.41x | 6.00 | 5.15 | 0.86x | 15.05 | 87.28 | 5.80x |
| 3 MiB (128) | 19.12 | 48.41 | 2.53x | 6.47 | 6.41 | 0.99x | 16.00 | 70.95 | 4.44x |
| 6 MiB (256) | 30.23 | 50.56 | 1.67x | 6.61 | 6.08 | 0.92x | 15.13 | 73.31 | 4.84x |
| 12 MiB (512) | 42.87 | 42.95 | tie | 6.96 | 6.06 | 0.87x | 16.79 | 69.99 | 4.17x |
| 24 MiB (1024) | 50.46 | 51.69 | 1.02x | 6.95 | 6.29 | 0.90x | 14.09 | 57.78 | 4.10x |
| 48 MiB (2048) | 53.03 | 52.64 | tie | 6.96 | 6.37 | 0.92x | 16.52 | 43.92 | 2.66x |
| 96 MiB (4096) | 54.51 | 53.37 | 0.98x (weak) | 6.90 | 6.43 | 0.93x | 14.38 | 26.39 | 1.84x |
| 192 MiB (8192) | 54.52 | 53.38 | 0.98x | 6.30 | 6.29 | tie | 14.66 | 13.87 | tie |

The single NVMe is saturated by four workers at every size, and the pool
costs it 7 to 14% in dispatch overhead. That device is the case the design
doc says to pin a budget for; the default is not sized for it.

### Depth ceiling: no `num_workers` value covers both ends (`exp_ceiling.sh`, `results/ceiling.log`)

`read_io_depth=256` so the budget is what binds. Batch 512.

| reads in flight bounded by | 1.5 MiB | 192 MiB | worst |
|---|---|---|---|
| `num_workers=4` | 11.02 (78% below best) | 53.94 | 78% |
| `num_workers=256` | 49.27 | 50.41 (6.5% below) | 6.5% |
| budget 1536 MiB | 50.97 | 53.49 (0.8% below) | 0.8% |

### Against a tuned `num_workers` at 6 MiB (`exp_barrier.sh`, `exp_move_check.sh`)

Batch 16, 16 batches outstanding. `num_workers=64` and the pool both reach
the array ceiling when the pool has 8 or more workers (54.03 against 54.11).
With the default four workers the pool reads between 47 and 53 on the same
binary depending on the evening's per-read latency, because each worker
drains a 16-object group before dispatching the next; `num_workers=64`
streams continuously and read 53.3 to 54.1 every time. This is the
small-batch cost recorded in the design doc.

### 192 MiB objects, controlled (`exp_192_w64.sh`, `results/w64_192.log`)

| arm | round 1 | round 2 | probe p95 |
|---|---|---|---|
| `num_workers=64` | 48.93 | 49.46 | 182 / 145 ms |
| pool | 52.45 | 52.43 | 25.5 ms |
| `num_workers=4` | 53.82 | 53.67 | 146 ms |

### tmpfs, controlled (`exp_h3_w64.sh`, `results/h3_w64.log`)

| object | `num_workers=4` | `num_workers=64` | pool |
|---|---|---|---|
| 6 MiB | 15.0 / 15.5 | 81.3 / 47.3 | 86.9 / 86.1 |
| 1.5 MiB | 17.9 / 15.9 | 69.6 / 69.7 | 71.7 / 72.1 |

The second `num_workers=64` round at 6 MiB fell by 42%; that arm is void and
the pool's steady 86 is not claimed as a win over it. At 1.5 MiB the 3% lead
is within twice the drift: a tie.

### Pool placement and the tile split (`exp_move_check.sh`, `results/move_check.log`)

| arm | round 1 | round 2 |
|---|---|---|
| pool in `FSConnector`, batch 16 | 47.44 | 47.26 |
| pool in `ConnectorBase`, batch 16 | 47.42 | 47.57 |
| pool in `FSConnector`, one 480-object batch | 50.54 | 50.55 |
| pool in `ConnectorBase` (one tile), same | 41.12 | 40.44 |
| `num_workers=64`, batch 16 | 53.29 | 53.75 |

### End to end (not reproducible from this directory)

vLLM 0.11-series with the LMCache multiprocess connector, gpt-oss-20b, TP 2
on two H200s, LMCache MP server with L1 300 GB and the `fs_native` L2 on the
array, `chunk_size` 256 (6 MiB objects), store policy `skip_l1`. 100
concurrent requests of 122,880 tokens each. Round 1 prefills and stores
562.5 GiB to L2 and is discarded; the prefix cache is reset; round 2 serves
the same requests from L2 and is measured. Arms alternate and the first arm
is repeated as a drift control. A gate reads the connector's own log line to
confirm each arm received its `num_workers`, `read_io_depth` and effective
budget before the round is accepted.

| arm | warm round | tok/s | mean TTFT |
|---|---|---|---|
| `num_workers=4` (default) | 24.05 s | 510,964 | 13.0 s |
| `num_workers=64` | 15.56 s | 789,894 | 9.34 s |
| pool, `read_io_depth=64`, budget 1536 MiB | 14.49 s | 848,037 | 8.71 s |
| `num_workers=64`, repeated | 15.60 s | 787,828 | 9.31 s |

Store rounds were 278.2 s and 278.5 s for `num_workers=64` and the pool.
