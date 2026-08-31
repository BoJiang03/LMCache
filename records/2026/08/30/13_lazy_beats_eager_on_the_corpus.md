# Session log: lazy beats eager on the corpus, and the gap grows with load

Conversation record for 2026-08-30 evening, continuing from record 12.
Bo asked for one lazy trial ("先跑lazy一次试试"), then delegated the rest
("你自己推进实验设计,目的是展示lazy相比于eager的各方面优势。特别是
ttft, throught 以及 e2e") and left for a few hours. Outcome: a shakedown
plus two clean A/B pairs, lazy ahead on every headline metric, with the
one early regression (TTFT p99) inverting at saturation.

## 1. Design

One arm = fresh MP server (empty L1: 250 GB, chunk 256, LRU, separate
object groups) + fresh engine (empty GPU pool) + aiperf replay of the
coder30 corpus recipe from record 2026/08/29/5 appendix:
`inferencex-agentx-mvp` on `semianalysis-cc-traces-weka-062126-256k`,
seed 1234, 1800 s benchmark, 600 s grace. Engine: Trinity TP=4 on GPUs
1-4, fp8 KV, util 0.90, max len 262144. The ONLY difference between arms
is the lazy connector block (EVICTION_AWARE, horizon 2.5, lru_tail,
danger floor 8192, deferral 30 s -- the same record's recipe, all keys
verified present on this branch). Arms serialized by `ab_chain.sh` in the
session scratchpad `sweep/` dir: kill engine, fresh MP server, fresh
engine, sampler (15 s cadence on both `/metrics`), aiperf, final prom
snapshots, teardown with GPU-free check.

Shakedown first: `lazy1` (lazy, CONC=48, L1 pre-warmed with record 12's
ladder prompts) ran clean end to end. Running landed 19-26 against the
15-21 target band, waiting 0-6; TTFT p50 4.5 s; the lazy ledger showed
`rejected_unhashed=57` of 3597 admissions. That settled the operating
points: a healthy-region pair at CONC=48 and a saturation pair at
CONC=72.

## 2. CONC=48: lazy wins, one tail regression

Both arms cold. Eager / lazy / delta:

| TTFT avg | 10,091 / 8,017 ms | -20.6% |
| TTFT p50 | 5,407 / 4,194 ms | -22.4% |
| TTFT p90 | 26,268 / 20,740 ms | -21.0% |
| TTFT p99 | 38,470 / 50,591 ms | +31.5%, the one loss |
| e2e avg | 88.3 / 81.5 s | -7.7% |
| e2e p50 | 43.2 / 36.5 s | -15.5% |
| out tok/s | 210.5 / 218.4 | +3.8% |
| ITL p99 | 313 / 171 ms | -45.5% |
| requests | 478 / 497 | +4.0% |

Sources over the profiling window (by_source token deltas): eager
external 39.3% / L0 20.8% / compute 39.9%; lazy 42.0% / 23.3% / 34.8%.
L1 writes: eager 1,175,091 chunks, lazy 369,638 -- lazy writes 31% of
eager's volume and still lands MORE external hit tokens.

## 3. CONC=72: saturation, and the gap widens

Running p50 25 / max 34, waiting to 21 -- queue-dominated for both arms.
Eager / lazy / delta:

| TTFT avg | 47,970 / 40,475 ms | -15.6% |
| TTFT p99 | 159,637 / 105,426 ms | -34.0%, the 48-loss inverts |
| e2e avg | 166.5 / 142.4 s | -14.5% |
| e2e p90 | 339.7 / 275.8 s | -18.8% |
| out tok/s | 196.6 / 216.0 | +9.9% |
| requests | 433 / 488 | +12.7% |

Sources: eager external 32.1% / L0 4.6% / compute 63.3%; lazy 43.1% /
7.8% / 49.1%. The compute-share gap grows from 5.1 points at 48 to 14.2
at 72. L1 write:read for eager is 2.1:1 (1,071,707 written, 504,073
read); for lazy it is 1:1.8 (434,979 written, 766,889 read).

## 4. The mechanism, in the numbers

Eager stores every prefill chunk immediately, and on this model storage
is dense (122,880 B/token, record 12), so the store stream is 3x the
useful volume. Under pressure that stream competes with the loads: eager's
external share COLLAPSES exactly when the pool churns fastest (39.3% ->
32.1% going from 48 to 72), while lazy's holds (42.0% -> 43.1%). Lazy's
deferral drops what gets evicted before it is read
(`dropped_evicted`~830-1060 per arm) -- and the write:read ratios above
show most of eager's extra writes were never read anyway. The bandwidth
saved shows up as decode smoothness (ITL p99 -45% at 48) and as more
prompt tokens processed per window (+4.8% at 48, +14.4% at 72).

Sub-risk (a) from record 12 is answered along the way:
`rejected_unhashed` = 57/63/39 per lazy arm (1.1-1.8% of admissions),
without `VLLM_PREFIX_CACHE_RETENTION_INTERVAL`. Minor, not a blocker.
`rejected_prefix_broken` is larger (1194/1804) and unexamined.

## 5. Caveats and open items

- Single rep per arm; no variance estimate. The 48-pair deltas are large
  enough (20%+) that a rep would likely not flip signs, but it is unrun.
- TTFT p99 at 48 (+31% for lazy) is unattributed; candidates are
  deferral-drain bursts at admission or a hit eager's denser L1 served
  that lazy's dropped_evicted lost. It disappears at 72.
- The A/B measures the loaded workload, not the pure-decode tpot gauge;
  Part 4's predicted table (49-62 tok/s at B=15-20) is untouched by this
  data. Measured tok/s/user avg was 17-19 at CONC=48 with prefill
  interleaved.
- L1 never filled (0.69-0.78 of 250 GB at arm end), so eviction age and
  the 500 GB question stay open for longer runs.
- One harness note: `local a=$1 b=${a...}` under `set -u` expands before
  assignment; split `local` lines.

## 6. State at close

- All four arms + shakedown archived in the session scratchpad `sweep/`
  dir: per-arm server/mp/client logs, 15 s samples, final prom snapshots,
  aiperf exports, `ab_analysis.md` with the full tables.
- Engines stopped, GPUs 1-4 free; the l72 MP server left running (warm
  L1), ports 8971/8972.
- `records/deployment_candidate.md` Part 6 item 2 and Part 8 updated:
  sub-risks both measured, step 4 done in redirected (A/B) form, R2
  absolute target met at 0.42, R1 pure-decode check still open.
