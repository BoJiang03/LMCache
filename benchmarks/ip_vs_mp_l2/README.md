# LMCache in-process vs multiprocess, with an L2 disk tier

Does it cost throughput to move LMCache out of the vLLM process and into a
separate server? This benchmark answers that with a **fair pair** — two arms
that differ only in the connector — and then measures what the multiprocess
arm can do when nothing is held back for the sake of comparability.

| arm | connector | L2 | hybrid KV cache manager |
|---|---|---|---|
| `ip_l2` | `LMCacheConnectorV1`, in-process | `fs` remote connector | disabled |
| `mp_l2_fs` | `LMCacheMPConnector`, separate server | `fs` L2 adapter | disabled |
| `mp_ceiling` | `LMCacheMPConnector`, separate server | `fs_native` (C++) L2 adapter | **on**, plus `--separate-object-groups` |

`ip_l2` vs `mp_l2_fs` is the comparison. `mp_ceiling` is **not** an MP-vs-IP
number and must never be read as one.

## What makes the pair fair

Both arms get the same L2 mechanism, the same DRAM, the same vLLM command line
apart from `--kv-transfer-config`, the same workload and the same protocol.

- **Same L2 approach.** Both use the `fs` filesystem backend: aiofiles,
  `O_DIRECT` on read and write, one `.data` file per key, `.tmp` + atomic
  rename. They are two independent implementations of it — the in-process
  `FSConnector` and the server-side `fs` L2 adapter share no code — but the
  mechanism, the flags and the array are identical. The C++ `fs_native`
  adapter is deliberately *not* used in the pair: it is registered only as an
  L2-adapter type, so the in-process path cannot select it, and using it on one
  side would compare storage implementations rather than connectors.
- **Same DRAM, and DRAM is not a cache tier on either side.** `L1_GB` is split
  across ranks for the IP arm (`max_local_cpu_size` is per rank) and passed
  whole to the server as `--l1-size-gb`. IP sets `local_cpu: false`, which
  makes its CPU pool a staging arena rather than a lookup tier; MP sets
  `--l2-store-policy skip_l1`, which drains L1 after each L2 store. So the
  measured round reads from disk on both sides.
- **Hybrid KV cache manager disabled on both.** vLLM keeps it enabled only for
  connectors that declare `SupportsHMA`. `LMCacheMPConnector` does;
  `LMCacheConnectorV1` does not, so vLLM turns it off by itself for the IP arm.
  The pair passes `--disable-hybrid-kv-cache-manager` explicitly on both so the
  asymmetry is visible rather than silent — and so the only difference between
  the two command lines is `--kv-transfer-config`.

## Why two rounds

The measured round must exercise the L2 → GPU load path, so it must not be
servable from vLLM's own GPU prefix cache. With a single round it is: a 120k × c
corpus fits inside the GPU KV pool at the lower concurrencies, so the KV is
still resident on the second pass and the connector is never asked for
anything. **That failure is silent** — it looks like a fast warm pass. Per
point:

1. `POST /reset_prefix_cache`, verified
2. round 1 — discarded, this is what populates L2
3. drain: wait until the L2 directory stops growing
4. `POST /reset_prefix_cache` again, verified
5. round 2 — **measured**

## The two acceptance gates

Both are checked by `run_point.sh`; both must hold or the numbers mean nothing.

**G1 — the reset must actually have happened.** The endpoint returns HTTP 200
unconditionally, and `block_pool.reset_prefix_cache()` refuses (with only a
`logger.warning`) while any block is still referenced. The real signal is the
server log: `Successfully reset prefix cache` versus `Failed to reset prefix
cache because some blocks (N) are not freed yet`. The script counts those lines
before and after each POST and retries with backoff, up to six times, then
fails the point. A silently-failed reset is indistinguishable from a good run
in the throughput numbers, which is why this is not assumed.

**G2 — round 2 must have been served by the connector.** vLLM's own counters
`vllm:external_prefix_cache_queries` / `_hits` are sampled either side of round
2 and differenced. A zero-hit point is labelled (`ZERO_HIT`) and kept, not
dropped: a zero-hit arm is a finding, not a run error.

## Prerequisites

**LMCache.** Run this branch, or any tree containing
`lmcache/v1/hash_seed.py`. Without that fix `LMCacheConnectorV1` at `TP > 1`
is a **write-only cache**: the rolling prefix hash is seeded from vLLM's
`NONE_HASH`, which is `os.urandom(32)` when `PYTHONHASHSEED` is unset, so the
scheduler and each worker derive keys under a different seed and the names
never coincide. Measured on 8×H200 / gpt-oss-120b / c=100 × 120k tokens:
**0 hits out of 12,000,000 queried tokens**, every object present on disk, and
round 2 rewriting the 787.7 GiB it had just written — 74,308 tok/s against
232,758 with the fix, i.e. **3.1× slower**, with no round-1-to-round-2 speedup
at all. If the IP arm reports `ZERO_HIT`, this is the first thing to check.

**vLLM.** Tested against 0.22.1 (torch 2.11.0+cu130, python 3.12). Needs
`SupportsHMA`, `--disable-hybrid-kv-cache-manager` and the
`/reset_prefix_cache` dev endpoint.

**`fs_native`** — for `mp_ceiling` only — needs the C++ extension built:
`pip install -e .`.

**RAM.** `L1_GB` + 250 GB, checked before launch. The default `L1_GB=1200`
therefore wants ~1.45 TB. Lower it to fit your box, but **lower it for both
arms** — that is what `L1_GB` does — or the comparison is void.

**Disk.** Each point writes ~8.5 GB per request and is wiped before the next,
so the requirement is set by the largest point, not the sum:
c=600 × 120k tokens ≈ 5.0 TB. `run_point.sh` gates on `9 × c + 300` GB free and
skips a point it cannot fit.

**File descriptors.** One `.data` file per key per rank; c=600 keeps ~8400
chunks in flight. `env.sh` raises `ulimit -n`; the default 1024 is not enough.

**GPUs.** 8 for the defaults (`TP=8`). Set `GPU_WAIT_MIN` to wait for a busy
shared machine rather than failing.

## Running it

```bash
cd benchmarks/ip_vs_mp_l2
export MODEL=/path/to/gpt-oss-120b L2_DIR=/mnt/nvme/lmcache_l2 L2_DEV=nvme0n1
export OUT=$PWD/out

./run_ladder.sh                                    # 3 arms x 6 concurrencies
ARMS="ip_l2 mp_l2_fs" CONC="100 200" ./run_ladder.sh   # just the fair pair

./collect.py "$OUT" > ladder.csv                   # CSV on stdout, summary on stderr
```

`L2_DEV` is the block device behind `L2_DIR`; it is read from
`/sys/block/$L2_DEV/stat` to account the bytes each measured round moves. Byte
accounting is skipped if it is unset, and it is the check that says whether both
arms are simply sitting at the device ceiling — leave it set if you can.

Every knob is an environment variable with a default; see `env.sh`.

## What we measured

8×H200, TP=8, `gpt-oss-120b`, 120k-token prompts, `L2_DIR` on a local NVMe RAID
(measured 12.9 GB/s `O_DIRECT` read), `L1_GB=1200`, one repeat per point.

| c | `ip_l2` | `mp_l2_fs` | IP / MP |
|---|---|---|---|
| 100 | 232,758 | 228,018 | 1.021 |
| 200 | 255,617 | 231,049 | 1.106 |
| 300 | 230,228 | 235,873 | 0.976 |
| 400 | 246,849 | 237,055 | 1.041 |
| 500 | 247,427 | 237,607 | 1.041 |
| 600 | 245,374 | 249,938 | 0.982 |
| **mean** | **243,042** | **236,590** | **1.028** |

**The two connectors perform the same.** The ratio straddles 1.0, the sign
flips point to point, and every difference is smaller than a single arm's own
spread across the ladder (IP 11.0%, MP 9.6%) — so ~10% is this experiment's
resolution at one repeat per point, and nothing here is outside it.

Three controls say both arms really did the same work: hit rate 95.57% at all
twelve points (identical to four decimal places), L2 bytes read equal to within
0.3 GiB, and TTFT tracking per point.

The hit rate is 95.57% *by construction*, not by luck: with `chunk_size: 8192`
and `save_unfull_chunk: false`, a 120,000-token prompt stores 14 full chunks =
114,688 tokens and never stores the 5,312-token remainder. 114,688 / 120,000 =
0.955733, and the counters land on exactly `c × 114,688` hits out of
`c × 120,000` queries at every point.

`mp_ceiling` runs **+9.4%** above the fair MP arm on average (+16.8% at c=100
falling to +4.5% at c=600), on 1.87× the GPU KV pool (25.73M vs 13.72M tokens)
and half the L2 store traffic (400 vs 788 GiB). Roughly 60% of that is the C++
`fs_native` client — worth +10.0% on its own at c=100 with the hybrid manager
off on both sides — and the rest the hybrid manager. Both are out of the
in-process connector's reach as shipped.

## Reading the output

Per point, under `$OUT/c<N>/<arm>/`:

| file | what |
|---|---|
| `c<N>_warm.json` | the measured round's `vllm bench serve` result |
| `c<N>_prefill.json` | round 1, discarded — kept for reference |
| `server.log` | vLLM, including the reset lines G1 checks and the pool size |
| `lmcache_server.log` | the MP server (MP arms only) |
| `external_counters.txt` | G2: round-2 hit/query deltas |
| `disk.txt` | bytes and rate over the measured round |
| `pool.txt` | GPU KV cache size and max concurrency, as vLLM reported them |
| `cmdline.txt` | the exact `vllm serve` line used |
| `ZERO_HIT` / `FAILED` | present when a gate did not hold |
