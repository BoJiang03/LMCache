# 4 — The benchmark ran on 0.4% of the corpus, and 2.9x oversubscribed

Continues records 1–3. The user rejected record 2's `agentx-std` spec on a
point of principle: "我说的真实的 workload 意思是与我们怎么 serve 无关…
一个 node/一个 vllm 实例服务的 workload 难道不是客观存在的吗？" Following
that through invalidated more than the spec. **Every round from i60A to
i60N was measured on 42 of the corpus's 393 sessions, carrying 0.4% of its
token volume, on a GPU pool oversubscribed 2.9x.**

No code changed this session. Tree clean at `f080487b`.

## 1. The conflation, and the flag that mattered

`agentx-std` mixed workload with serving config, and put `L1_GB=120` in the
workload column having chosen it because the hit rate looked good there —
record 2's own circularity, one level up. The split is: the corpus, the
arrival process, the model class and context distribution are the workload;
model, TP, block size, pool size, L1 size, policy are ours.

Applying it exposed the real defect. **`--max-context-length` is a
trace-level filter, not a truncation** (`aiperf/dataset/loader/selection.py`:
"FILTER: drop a candidate whose peak context exceeds `max_context_length`",
then "CAP: keep the first N eligible"). It discards a whole session if its
*peak* context ever exceeds the limit.

| `--max-context-length` | traces kept | share | token volume kept |
|---|---|---|---|
| **100,000 (every round to date)** | **42 / 393** | **10.7%** | **0.4%** |
| 131,072 | 82 | 20.9% | 1.0% |
| 262,144 | 220 | 56.0% | 6.4% |
| 400,000 | 292 | 74.3% | 16.2% |
| none | 393 | 100% | 100% |

Confirmed directly in i60N's own log: `TrajectorySource: built 32
trajectories from 42 traces`. `--num-dataset-entries 256` never bound —
only 42 sessions were ever eligible. The survivors are the *smallest*
sessions, i.e. precisely the ones where an offload tier matters least.

Root cause was `MAX_MODEL_LEN=131072` in the harness: vLLM was capped at
128k, so the filter had to be set below it.

## 2. The corpus, measured

393 real Claude Code sessions, 98,827 leaf requests (subagent records nest
their own `requests`), 7 models. A node serves one; `claude-opus-4-8` is
62,108 requests / 16.7 B input tokens / 314 sessions.

Per-request context (`in + out`), opus only:
p50 **201,985**, p90 624,530, p99 893,140, max 996,579.
Per-trace *peak* context: p25 142,160, p50 226,315, p75 412,037, p90 692,567.

Sessions are a developer's whole day: span p50 10,121 s, p90 168,526 s;
p50 inter-request gap 9.2 s; turns per session p50 86 (72 for opus alone).
`hash_id_scope` is `"local"` on all 393 — block ids mean something only
within one session, so **the corpus cannot measure cross-user sharing at
all**.

## 3. The preemption anomaly is oversubscription, not policy

Records 1–2 left preemptions unexplained (122–145 on lazy arms against
eager's 4). Measured mean in-flight requests directly from
`profile_export.jsonl` (`request_start_ns` / `request_end_ns`), which no
round had ever done:

| arm | `--concurrency` | mean in-flight | peak | mean latency |
|---|---|---|---|---|
| i60M eager | 32 | 15.8 | 27 | 79.0 s |
| i60N 30 G | 32 | 16.0 | 27 | 80.7 s |
| i60N 120 G | 32 | 13.8 | 23 | 51.3 s |
| i60N 240 G | 32 | 13.5 | 22 | 45.9 s |

14 in-flight x 54 k mean ISL = **756 k tokens demanded against a 262 k-token
pool: 2.9x oversubscribed**. That is the whole preemption story. Record 2
§2's "prefill pressure, which collapses when the cache hits" was directionally
right and mechanically wrong.

`--concurrency` is **lanes, not in-flight requests**. Lane duty cycle was
~45% here. Every past statement of the form "at concurrency 32" described
13.5–16 concurrent requests.

## 4. What 6 in-flight means in users

The corpus records the original system's `api_time` (p50 6.9 s, p90 27.0 s)
and `ttft` (p50 2.4 s) — the production cluster was ~10x faster per request
than our node. Session duty cycle (busy / open) from those: pooled **4.1%**,
median session 10.4%.

So 8 in-flight requests corresponds to ~196 open sessions at the original
system's speed, ~50–80 at ours. **A single-digit in-flight count is tens to
hundreds of concurrent Claude Code users, not a handful.**

## 5. Official aiperf tooling that already existed

Surveyed after the user pointed out that hand-built generators are not
authoritative. All of this shipped in aiperf 0.12.0 and was unused:

- **`aiperf synthesize agentic-code`** — parameterised agentic-coding
  workload generator with a bundled canonical manifest
  (`aiperf/dataset/agentic_code_gen/datasets/1k_sessions_200k_ctx/`, README
  gives the exact command), a target-vs-realised error report
  (`comparison.txt`), and an explicit L1/L1.5/L2/L3 prefix-sharing model.
- **`aiperf analyze-trace`** — ISL/OSL distributions and theoretical hit
  rates for a mooncake trace.
- **`simulation.html`** (emitted by synthesize) — an interactive LRU node
  simulation with Concurrency / Prefill TPS / Decode TPS / Total KV Cache
  (GB) / KV Bytes-per-Token as inputs, plotting hit rate, memory pressure,
  evictions by layer, and a session Gantt.
- **`aiperf validate --target mooncake-trace`**, and the `trace_replay.yaml`
  / `fixed_schedule.yaml` config templates for open-loop timestamp replay.

A `benchmarks/node_workload/` tool (session superposition + LRU stack
distance) was written and then **deleted** this session: superseded by the
above, and not authoritative. Reconstructible in ~30 min from this record
if the open-loop path is ever wanted; aiperf supplies the mechanism
(`--fixed-schedule` + mooncake) but no timestamped node stream for this
corpus.

## 6. Synthesized vs real, and the one assumption that dominates

`synthesize agentic-code` at the canonical manifest (1131 sessions,
seed 42) against the real opus traffic:

| | synth p50 | real p50 | synth p90 | real p90 |
|---|---|---|---|---|
| prompt tokens sent | 98,049 | **200,448** | 169,951 | **622,784** |
| turns per session | 11 | **72** | 23 | 547 |
| inter-turn gap | 2 s | **12 s** | 17 s | 95 s |
| session span | 56 s | **11,162 s** | 190 s | 169,764 s |

Shape right, scale deliberately small and fast. The session-span gap is
disqualifying for this line: an offload tier exists to hold KV across time,
and synth sessions do not live long enough to need one.

Counting over the accumulated prefixes actually sent, **42.7% of everything
the server is asked to prefill in the synthetic workload is cross-session
shared** (L1 global 32.4%, L1.5 group 10.3%). The real corpus cannot confirm
this: the weka loader honours `hash_id_scope: "local"` strictly, reseeding
the prompt RNG per trace (`weka_trace.py:2191`), so blocks never collide
across sessions. **agentx therefore models zero cross-user sharing and
understates hit rate; synthesize asserts 42.7% and probably overstates it.**
Claude Code's system prompt and tool definitions are in fact identical
across users, so the truth is between. Neither is calibrated on this axis.

Decision: **agentx (real corpus) is primary; synthesize is a sensitivity
check on the cross-user-sharing axis only.**

## 7. Model choice for the full context distribution

The box is **8x H200, 143,771 MiB each** — the 24 GiB pool was always a
harness choice, never a hardware limit. GPU0–3 on NUMA node 0, GPU4–7 on
node 1, all NV18. GPU0/1 carry a foreign job. Host RAM 2015 GB
(node 0 1031 GB, node 1 1032 GB).

Cached candidates, KV bytes/token computed from `config.json`:

| model | KV B/tok | max_pos | verdict |
|---|---|---|---|
| **Qwen3-Coder-30B-A3B** | 98,304 | 262,144 | YaRN to 1M, architecture unchanged |
| Llama-4-Scout-17B-16E FP8 | 196,608 | 10,485,760 | 2x KV, chunked attention -> hybrid allocator |
| DeepSeek-V4-Flash | ~9.4 k (est.) | 1,048,576 | see below |
| gpt-oss-120b | 73,728 | 131,072 | shorter than now |
| Qwen3.5-122B / Qwen3.8-Flash-Next | 98,304 | 262,144 | no gain |
| Devstral-2-123B | 360,448 | 262,144 | 3.7x KV |
| Qwen3-Coder-480B-A35B FP8 | 253,952 | 262,144 | 2.6x KV |

**DeepSeek-V4-Flash** is the only true 1M-native option and the support
chain exists: `vllm-main` (0.28.1rc1.dev) registers `DeepseekV4ForCausalLM`;
LMCache's [kv_cache_group_edits.py:108](../../../../lmcache/integration/vllm/kv_cache_group_edits.py)
explicitly does **not** reject its `compress_ratio > 1` slot packing (served
by `lmcache.v1.kv_layer_groups`); and the lazy policy is already group-aware
([eviction_aware.py:995](../../../../lmcache/integration/vllm/lazy_offload_policy/eviction_aware.py)
iterates per-group `tokens_per_block` and takes the cross-group minimum).

Rejected anyway, on four counts, none of which is "unsupported":
its KV is ~10x smaller so the GPU pool holds 10x more tokens and **the
pressure we are studying disappears**; it needs vllm-main while the whole
lazy branch is built on vllm-lazy 0.23.0; 149 GB of weights forces TP>=2
(realistically 4), destroying the same-NUMA-node multi-arm design; and its
256/64/8/4 slot layout with per-layer 4x/128x compression has never been
exercised by our 225 tests.

It is the right answer to "can this corpus be served at 1M context" and the
wrong platform for "is lazy offload worth it".

## 8. The settled configuration

KV pool, H200 140.4 GiB/card, weights 56.9 GiB bf16, util 0.90, ~7 GiB
overhead:

| TP | KV pool | pool tokens | max request (996 k) uses |
|---|---|---|---|
| 1 (all rounds to date) | 62.5 GiB | 0.68 M | 146% — does not fit |
| **2** | **181.8 GiB** | **1.99 M** | **50.2%** |
| 4 | 420.5 GiB | 4.59 M | 21.7% |

At TP=2 with YaRN, **100% of requests are servable**; at 262 k without YaRN,
58.6%.

In-flight target, against real context sizes:

| in-flight | x p50 202 k | x mean 270 k | prefix cache left |
|---|---|---|---|
| 5 | 50.9% | 68.1% | 58–89 GiB |
| **6** | **61.0%** | **81.7%** | **33–71 GiB** |
| 7 | 71.2% | 95.3% | 8–52 GiB |

7 squeezes the prefix cache to nothing; 5 is not enough pressure. **6 is the
operating point**, and the pressure now comes from the workload rather than
from an artificially shrunk pool.

Serving (per arm):

```
vllm serve Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --tensor-parallel-size 2 --max-model-len 1048576 \
  --rope-scaling '{"rope_type":"yarn","factor":4.0,"original_max_position_embeddings":262144}' \
  --gpu-memory-utilization 0.90 --block-size 16
```

No `--num-gpu-blocks-override`. vLLM 0.23.0 derives
`max_model_len = original_max_position_embeddings * factor` for
`rope_type: yarn` (`vllm/config/model.py:2185`), so the mechanism is
supported; Qwen's own recommendation for this model was not verifiable from
the local cache, and static YaRN costs short-sequence quality, which does
not affect the cache/latency behaviour being measured.

Load:

```
aiperf profile --model agentx --url http://127.0.0.1:$PORT \
  --endpoint-type chat --streaming \
  --tokenizer Qwen/Qwen3-Coder-30B-A3B-Instruct \
  --scenario inferencex-agentx-mvp \
  --public-dataset semianalysis-cc-traces-weka-062126 \
  --concurrency 10 --benchmark-duration 1800 --benchmark-grace-period 600 \
  --random-seed 1234 --output-artifact-dir out/artifacts
```

`--max-context-length` and `--num-dataset-entries` removed. Grace raised
180 -> 600 s (a p90 request is 625 k tokens, ~30 s of prefill alone).
`--concurrency 10` is a starting point; recalibrate once as
`lanes = 10 * 6 / measured_in_flight`.

Placement, under the user's constraint of **at most 4 GPUs and modest host
memory**: two arms on NUMA node 1, **GPU4+5 and GPU6+7**. Node 0 is
excluded (foreign job on GPU0/1).

L1 size is a swept variable, not a setting. Its reference point has changed:
in every past round the GPU pool was 24 GiB and L1 (60–240 G) was always far
larger, so the sweep measured total cache volume. At TP=2 the pool is
182 GiB and in-flight requests hold ~81% of it, leaving **33–71 GiB of GPU
prefix cache**. L1 crossing that boundary is the event worth resolving, so
the points bracket it:

| round | arms (paired, same node) | L1 per arm | node 1 memory |
|---|---|---|---|
| R0 | smoke, 1 arm | 96 G | 96 GB |
| R1 | eager vs lazy | 32 G — below the GPU prefix cache | 64 GB |
| R2 | eager vs lazy | 96 G — 1.5–3x it | 192 GB |
| R3 | eager vs lazy | 256 G — 4–8x it | 512 GB |
| R4 | off vs lazy | best of R1–R3 | ≤ 512 GB |

Both arms of a round carry the same L1 (eager also stores into L1), so round
memory is 2 x L1; the ceiling is 512 GB of node 1's 1032 GB. Pairing eager
against lazy *within* a round keeps the policy comparison on the tight
same-node footing (i60L: 92 ms spread between identical arms); the L1 trend
is read across rounds, which is the coarser question. Repeat one L1 point in
two rounds as the cross-round comparability check.

Report every round: mean/peak in-flight and `in-flight * mean ISL / pool`
(target 0.7–0.8); `tokens_stored / served input tokens`; retrieves,
`tokens_retrieved`, `l1_gib`; `preempt_events` (should now be near zero);
medD at 600 s / 900 s / full; per-arm request count.

The in-flight meter, which no round has had and every round now needs:

```python
ev, busy = [], 0.0
for line in open(f"{arm}/artifacts/profile_export.jsonl"):
    md = json.loads(line)["metadata"]
    s, e = md.get("request_start_ns"), md.get("request_end_ns")
    if not s or not e:
        continue
    busy += (e - s) / 1e9
    ev += [(s, 1), (e, -1)]
ev.sort()
span = (ev[-1][0] - ev[0][0]) / 1e9
cur = peak = 0
for _, d in ev:
    cur += d
    peak = max(peak, cur)
print(busy / span, peak)          # mean in-flight, peak in-flight
```

## 9. What this invalidates

Every scored round, i60A through i60N. They ran on 42 small sessions
(0.4% of token volume) at 2.9x oversubscription with a 24 GiB pool and a
128 k context cap. Records 1–2's ceiling tables, the reuse-distance CDF in
record 2 §1 (built from ISL as a proxy, on the filtered subset), and the
`agentx-std` spec are all superseded. What survives is methodological: the
NUMA placement rule, the paired-arm design, `converge.py` / `early.py`, and
the three counters added in `0436a131` / `0b75fa3b`.

Blocking item before any round: **TP=2 has never been run with LMCache and
this lazy offload path.** Smoke test first — bring up, confirm store and
retrieve, run ~20 requests, check the counters and that the ledger balances.
