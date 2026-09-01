# Lazy vs eager A/B on Trinity (2026-08-30 afternoon)

Design: one arm = fresh MP server (empty L1, 250 GB, chunk 256, LRU,
separate object groups) + fresh engine (empty GPU pool) + aiperf
inferencex-agentx-mvp on semianalysis-cc-traces-weka-062126-256k,
seed 1234, benchmark 1800 s, grace 600 s. Only difference between arms:
the lazy offload connector keys (record 2026/08/29/5 appendix recipe,
EVICTION_AWARE, horizon 2.5, lru_tail, deferral 30 s). Engine: Trinity
TP=4 GPUs 1-4, fp8 KV, util 0.90, max len 262144.

Shakedown before the A/B: lazy1 (lazy, CONC=48, L1 pre-warmed with ladder
prompts) ran clean; Running landed 19-26, waiting 0-6.

## CONC=48 pair (both cold)

| metric | e48 eager | l48 lazy | lazy delta |
|---|---|---|---|
| TTFT avg (ms) | 10,091 | 8,017 | -20.6% |
| TTFT p50 | 5,407 | 4,194 | -22.4% |
| TTFT p90 | 26,268 | 20,740 | -21.0% |
| TTFT p99 | 38,470 | 50,591 | +31.5% (worse) |
| e2e latency avg (ms) | 88,309 | 81,490 | -7.7% |
| e2e p50 | 43,184 | 36,488 | -15.5% |
| e2e p90 | 162,355 | 165,211 | +1.8% |
| e2e p99 | 970,053 | 908,475 | -6.3% |
| output tok/s (total) | 210.5 | 218.4 | +3.8% |
| tok/s/user avg | 17.4 | 19.1 | +9.5% |
| ITL avg (ms) | 82.4 | 72.1 | -12.5% |
| ITL p90 | 137.5 | 119.2 | -13.3% |
| ITL p99 | 313.3 | 170.6 | -45.5% |
| requests completed | 478 | 497 | +4.0% |

Prompt token sources over the profiling window (samples delta, _total):

| source | e48 | l48 |
|---|---|---|
| external_kv_transfer | 20,837,360 (39.3%) | 23,331,584 (42.0%) |
| local_cache_hit | 11,057,360 (20.8%) | 12,921,072 (23.3%) |
| local_compute | 21,151,083 (39.9%) | 19,317,589 (34.8%) |
| total | 53,045,803 | 55,570,245 |

L1 traffic (MP counters, whole arm): writes e48 1,175,091 chunks vs l48
369,638 chunks (lazy writes 31% of eager's volume); reads e48 2,061,377 vs
l48 1,070,902. L1 usage at end: 0.73 vs 0.69 of 250 GB. Corrected
08-31 (record 4 section 3): that end value is the LRU watermark, not
headroom. Usage rises to ~0.80 by t+250-400 s in every arm and then
sawtooths between 0.65 and 0.84 for the rest of the run, so both arms
evict continuously from about minute 5. The store is dense at 122,880
B/token, so 250 GiB holds only ~1.7M tokens -- half the GPU pool's
3,250,930 -- and entries live 276-336 s (lazy) or 164 s (eager). The
500 GB question is not open, it is the binding constraint.

Lazy ledger (l48): admitted=3523 emitted=2687 dropped_evicted=831
(6,394,880 tokens) rejected_prefix_broken=1194 rejected_unhashed=63
danger_floor_raises~small. rejected_unhashed is 1.8% of admissions:
sub-risk (a) (lazy x SWA unhashed blocks) is real but minor at this
operating point, without VLLM_PREFIX_CACHE_RETENTION_INTERVAL set.

## Reading

- Lazy wins TTFT (avg/p50/p90 all ~-21%), e2e (avg/p50/p99), throughput
  (+3.8% total, +9.5% per user), and especially ITL tail (p99 -45%).
- Mechanism consistent with the design: eager stores every chunk of every
  prefill immediately (dense 122,880 B/token on this model), a store
  stream that competes with loads and decode; lazy defers stores to
  eviction time and ends up writing 3.2x less while losing nothing that
  was actually reused inside the window -- its external hit share is
  HIGHER (42.0% vs 39.3%), because the store bandwidth saved goes to
  loads and the GPU pool retains more (L0 23.3% vs 20.8%).
- R2's absolute L1 number: tokens_retrieved/isl_sum = 0.42 on the lazy
  arm, on target (~0.4, band 0.2-0.6).
- The one regression is TTFT p99 (+31%). Not yet attributed; candidates:
  deferral-drain bursts at admission (danger floor), or a hit that
  eager's denser L1 served and lazy's dropped_evicted lost.
- Single rep per arm; no variance estimate yet.

## CONC=72 pair (both cold; saturation region)

Load profile: e72 Running p50 25 / max 34, Waiting up to 21. TTFT is
queue-dominated for both arms.

| metric | e72 eager | l72 lazy | lazy delta |
|---|---|---|---|
| TTFT avg (ms) | 47,970 | 40,475 | -15.6% |
| TTFT p50 | 43,172 | 36,093 | -16.4% |
| TTFT p90 | 118,454 | 95,561 | -19.3% |
| TTFT p99 | 159,637 | 105,426 | -34.0% |
| e2e avg (ms) | 166,478 | 142,405 | -14.5% |
| e2e p50 | 98,435 | 85,990 | -12.6% |
| e2e p90 | 339,706 | 275,785 | -18.8% |
| e2e p99 | 1,278,416 | 1,138,749 | -10.9% |
| output tok/s (total) | 196.6 | 216.0 | +9.9% |
| tok/s/user avg | 10.3 | 11.7 | +13.4% |
| ITL avg (ms) | 123.8 | 102.5 | -17.2% |
| requests completed | 433 | 488 | +12.7% |

Prompt token sources over the profiling window:

| source | e72 | l72 |
|---|---|---|
| external_kv_transfer | 13,948,240 (32.1%) | 21,479,904 (43.1%) |
| local_cache_hit | 2,012,176 (4.6%) | 3,870,112 (7.8%) |
| local_compute | 27,554,060 (63.3%) | 24,447,434 (49.1%) |
| total | 43,514,476 | 49,797,450 |

L1 traffic: e72 wrote 1,071,707 chunks and read 504,073 (write-heavy,
2.1 writes per read); l72 wrote 434,979 and read 766,889 (1.8 reads per
write). Ledger (l72): admitted=3636 dropped_evicted=1062
rejected_prefix_broken=1804 rejected_unhashed=39 (1.1%).

The gap widens at saturation, as the contention mechanism predicts:

- The TTFT p99 regression seen at 48 inverts: -34% at 72. Every latency
  percentile now favors lazy.
- Eager's external hit share collapses under pressure (39.3% -> 32.1%)
  while lazy holds (42.0% -> 43.1%). At saturation the GPU pool churns
  fastest exactly when eager's store stream is largest; the stores steal
  the bandwidth the loads need, so hits get slower to materialize and
  the scheduler falls back to compute. Lazy's deferral drops the stores
  that would never be read (dropped_evicted) and keeps the wire clear
  for reads.
- Throughput gap grows from +3.8% to +9.9%, completed requests from
  +4.0% to +12.7%, compute share gap from 5.1 to 14.2 points.

Artifacts: e48/l48/e72/l72_{server,mp,client}.log, _samples.log,
_mp_final.prom, _vllm_final.prom, _artifacts/ (aiperf exports), all in
this directory. lazy1_* is the shakedown.

## CONC=32 pair (low load; L0-dominated)

e32 / l32: TTFT avg 3,380 / 2,876 ms (-14.9%), p90 10,235 / 8,272
(-19.2%); e2e avg 41,276 / 38,648 (-6.4%); thpt 242.6 / 248.7 (+2.5%);
ITL avg 45.6 / 43.0; requests 618 / 656 (+6.1%). Sources (whole arm):
e32 ext 16.3% / L0 54.5% / compute 29.1%; l32 ext 14.8% / L0 59.1% /
compute 26.2%. At low load the GPU pool retains most reuse (window
theory: low f -> large W -> high L0) and L1 matters less; margins narrow
but every sign still favors lazy. Ledger note: l32 rejected_unhashed=279
of 3702 (7.5%) -- the rate RISES as load falls, because blocks live
longer and stores meet out-of-window SWA null blocks more often.

## CONC=40 pair (the TTFT ~5 s operating point)

First e40 was ruined by a duplicate chain instance (quarantined in
bad_run1/, flock added); numbers below are the clean rerun.

e40 / l40: TTFT avg 5,368 / 5,126 ms (-4.5%), p50 2,613 / 2,495; e2e avg
72,033 / 71,769 (-0.4%), p90 156,191 / 147,389 (-5.6%); thpt 224.9 /
231.8 (+3.1%); ITL avg 67.1 / 62.8 (-6.5%), ITL p99 212 / 160 (-24.7%);
requests 506 / 508. Sources: e40 ext 30.4% / L0 32.3% / compute 37.3%;
l40 ext 34.3% / L0 29.2% / compute 36.5%. Ledger (l40): admitted=3938
dropped_evicted=931 rejected_unhashed=98 (2.5%).

CONC=40 is eager's TTFT ~5 s operating point (Bo's target). The eager
TTFT curve over CONC: 32 -> 3.4 s, 40 -> 5.4 s, 48 -> 10.1 s, 72 -> 48 s;
queueing takes off between 40 and 48.

## FIFO result: the default never offloads (f40 measured)

f40 (lazy_offload=true, policy=FIFO, all defaults): external_kv_transfer
= 0 for the whole arm; l1_write_chunks_total never incremented. Cause
(corrected 08-31 from the engine log, see record 3): fifo.py:80 drains
only when 100 finished-but-undrained requests coexist. That set never
shrinks on its own, so it does reach 100, at 02:20:45, ~11 min into the
load, and then sawtooths at 10 requests per drain. By then the blocks
are gone: lazy offload's request_finished returns False, the finished
requests' blocks return to the free queue and are reused, and the
manager's hash revalidation (lazy_offload_manager.py:571) drops every
drained request. 330 drained, 330 dropped, 0 stores. The arm
is effectively a no-L1 baseline: TTFT avg 9,240 ms (vs eager 5,368 at
the same CONC), thpt 150.9 (vs 224.9), ITL avg 113.8, requests 381,
sources 0% ext / 24.0% L0 / 76.0% compute. Per Bo: the policy is not
ours to fix; f48/f72 run with the same defaults and stand as the
no-offload reference curve. (The prior line's own harness set
lazy_offload_threshold=1 for its FIFO arms -- driver.py:1171 -- which is
how FIFO was ever made to drain at all.)

## FIFO predictions (registered 02:1x, before f40 lands)

Mechanism: FIFO defers stores (shares the deferral TTFT win) but drains
blind -- backlog > 100 ops, drain 10/step, no eviction awareness, no
skip-never-reused. Expect between eager and EA, converging to eager as
load rises: f40 TTFT ~5.2s / thpt ~228 / ext 31-33%; f48 TTFT 8.5-9.5s /
thpt 212-215 / ext ~40%; f72 TTFT 44-48s / thpt 200-205 / ext 33-36%,
ITL near eager. Ledger: dropped_evicted and rejected_prefix_broken well
above EA. Tail branch: if admit rate exceeds drain rate at 72 the
backlog grows unboundedly, stores arrive very late, external collapses
below eager; diagnose via pending depth and dropped counters.
