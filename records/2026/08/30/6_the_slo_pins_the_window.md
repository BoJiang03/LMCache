# The SLO pins the window, and three requirements do not fit

This record covers the session after record 5. No experiment was launched. Every
number below comes either from artifacts already on disk or from arithmetic on
them. Four proposals were made and withdrawn in the course of it; section 7
lists them, because the pattern matters more than any one of them.

## 1. The requirement that started it

Two requirements were set: per request decode must reach about 50 tok/s, and
L1 must carry a larger share of reuse. They are not both reachable here. The
argument is one line long once the tpot law is measured.

## 2. The tpot law, from twenty arms already on disk

Every prior arm's artifacts were re-parsed. Same model, TP=2, fp8 KV.

| arm | in-flight | ISL mean | tpot | per-req tok/s |
|---|---|---|---|---|
| r0_lazy | 1.66 | 164,990 | 12.9 ms | 77.4 |
| n14L576 | 2.64 | 192,407 | 16.0 ms | 62.3 |
| n14L256 | 3.06 | 204,392 | 17.5 ms | 57.1 |
| c2_14_s1 | 3.60 | 178,238 | 17.4 ms | 57.6 |
| ov18_s2 | 12.00 | 193,900 | 31.9 ms | 31.4 |
| l64r1 | 25.05 | 103,274 | 61.0 ms | 16.4 |
| f8k256c72 | 30.22 | 107,443 | 81.6 ms | 12.3 |
| f8k256c84 | 39.38 | 97,187 | 94.7 ms | 10.6 |
| l72b64L512d0cw | 100.59 | 89,060 | 147.7 ms | 6.8 |

Least squares over twenty such points:

```
tpot_ms = 8.03 + 16.5692 * (in-flight * ISL / 1e6)      mean |residual| 20%
```

The slope converts to 2.97 GB/ms of KV read across two GPUs, about 31 percent
of H200 peak. Record 3 section 9 and record 5 section 2 both recorded the
isolated harness reading KV at 2.5 to 8 percent of peak and flagged it as
blocking. The live fit says 31 percent. **The anomaly is the isolated harness,
not the machine.** That open item narrows to "why is bs.sh slow", and it no
longer qualifies anything measured live.

At ISL 107k the law gives 50 tok/s at in-flight 6.8, aggregate 338 tok/s.

## 3. 50 tok/s is reachable, and the hardware is not the problem

The four low in-flight arms above deliver 57 to 77 tok/s, at ISL 165k to 204k,
which is heavier than the CONC=64 pair's 107k. The pair sits at 15.9 tok/s
because it holds 25 requests in flight, not because the box is slow.

OpenRouter lists this exact model at Alibaba Cloud 53 tok/s, NovitaAI 47,
SiliconFlow 21. Their figure is unnormalised live traffic, and OpenRouter
documents no normalisation, so it is dominated by short prompts. Our own
isolated numbers on the same model are 73.9 tok/s at 8k and 28.1 at 32k, so at
short context we are already above the market.

The binding quantity is in-flight KV bytes, not request count. At tpot 20 ms on
2xH200 the budget is 33.1 GiB. What that buys:

| context | concurrent requests, all at 50 tok/s |
|---|---|
| 2k | 361 |
| 8k | 90 |
| 32k | 23 |
| 107k | 6.8 |

So "many requests" and "50 tok/s each" coexist freely at short context. That is
the whole of the OpenRouter number.

For long context providers add GPUs. The budget scales with GPU count:

| | 107k concurrent at 50 tok/s | aggregate | per GPU |
|---|---|---|---|
| TP=2 | 6.8 | 338 tok/s | 169 |
| TP=4 | 13.5 | 675 | 169 |
| TP=8 | 27.0 | 1350 | 169 |
| TP=16 | 54.0 | 2701 | 169 |

**Per GPU long context decode throughput is a hardware constant.** Adding GPUs
makes nothing faster; it only decides how many users divide a fixed per GPU
throughput at a given latency. Our measured CONC=64 point is 199 tok/s per GPU,
slightly above the law because the fixed term amortises better at high batch.

15.9 tok/s per request is therefore not a deficiency. It is two GPUs carrying 25
users of 107k each. A provider would put those 25 users on eight to sixteen GPUs.

## 4. The theorem

L1's share of reuse is `P(gap > window)` with `window = T(1-f)/f`, `T = OSL x
tpot`, `f = in-flight KV / pool` (record 2 section 4).

A per token latency SLO fixes tpot. Fixing tpot fixes the in-flight KV budget,
because that budget is `(tpot - fixed term) x bandwidth`. It also fixes T. So
with the pool held constant, f is constant, the window is constant, and L1's
share is constant. Concurrency and context length can be traded against each
other freely inside the budget and nothing moves.

At 50 tok/s, 2xH200, 187 GiB pool: f = 0.19, window 84 s, L1 share 14.7%.

**The two requirements are over-determined, not merely hard.** Any two of the
three are already in hand: 50 tok/s with long context is the low in-flight arms;
long context with L1 relevance is the CONC=64 pair; 50 tok/s with L1 relevance
has no solution on this hardware.

## 5. More GPUs make it worse

Weights are sharded once while HBM scales linearly, so the pool grows faster
than the SLO pinned numerator.

| TP | pool | in-flight KV | f | window | L1 share | users at 107k |
|---|---|---|---|---|---|---|
| 2 | 187 GB | 35.5 GB | 0.190 | 84 s | 16.3% | 7.2 |
| 4 | 435 GB | 71.0 GB | 0.163 | 101 s | 13.8% | 14.5 |
| 8 | 930 GB | 142.0 GB | 0.153 | 109 s | 12.8% | 29.0 |
| 16 | 1921 GB | 284.0 GB | 0.148 | 114 s | 12.3% | 58.0 |

Chat had claimed TP leaves f unchanged. It does not. It is monotonically
against us, because 61 GB of weights is a quarter of the TP=2 pool and three
percent of the TP=16 pool.

## 6. The model search, and why it closes

Solving the identity for the model rather than the load gives

```
f = (tpot x BW - c*P) / (HBM x util - P)      P = weight bytes resident
                                              c = fraction of weights read per step
df/dP > 0  iff  c < tpot x BW / (HBM x util) = 0.234
```

so f rises with weight size only for models sparse enough that a step reads less
than 23 percent of them. KV bytes per token cancels out entirely, which kills
the idea of looking for a model with a big KV.

What is on the box:

| model | ckpt | KV/token fp8 | TP=2 pool | 107k contexts |
|---|---|---|---|---|
| Qwen3-Coder-30B-A3B (incumbent) | 61 GB bf16 | 49,152 B | 187 GiB | 38 |
| Qwen3.5-122B-A10B-FP8 | 118 GiB | 12,288 B | 129 GiB | 105 |
| Qwen3-Coder-480B-A35B-FP8 | 449 GiB | 126,976 B | does not fit | TP=4 gives 3.7 |

Filling the GPUs with a larger model makes it worse. The 122B holds more weight
but its KV per token is four times smaller, so the pool measured in contexts
goes from 38 up to 105. Current model design cuts KV per token while growing
parameters, and both effects push f down.

### DeepSeek-V4-Flash: right profile, wrong silicon

284B total, 13B active, 256 routed experts at top-6, one shared, MLA,
`index_topk: 512` sparse attention, 1M positions. Already on `/raid`, 148.7 GiB.
vLLM 0.23.0 ships a dedicated `vllm/models/deepseek_v4` package.

Its dtype, read from the safetensors headers rather than from `config.json`:

| dtype | tensors | size | share | example |
|---|---|---|---|---|
| I8 (packed FP4) | 33,792 | 132.00 GiB | 88.8% | `layers.0.ffn.experts.0.w1.weight [2048, 2048]` |
| F8_E8M0 (ue8m0 scales) | 34,167 | 8.25 GiB | 5.6% | `layers.0.attn.wkv.scale [4, 32]` |
| F8_E4M3 (attention) | 375 | 5.61 GiB | 3.8% | `layers.0.attn.wkv.weight [512, 4096]` |
| BF16 (embedding) | 433 | 2.64 GiB | 1.8% | `embed.weight [129280, 4096]` |

The I8 is packed four bit: `w1` should hold moe_intermediate 2048 x hidden 4096
= 8.39M parameters but the tensor is [2048, 2048] = 4.19M I8 elements, exactly
half. `config.json` saying `quant_method: fp8` describes the attention path
only. vLLM agrees: `_DEEPSEEK_V4_EXPERT_DTYPES = ("fp4", "fp8")`, default fp4,
and `is_scale_e8m0` is true exactly when the experts are fp4.

And that dtype ends it. `vllm/models/deepseek_v4/nvidia/model.py:285`:

```python
if torch.cuda.get_device_capability(device)[0] != 10:
    raise NotImplementedError("DeepGEMM MegaMoE requires SM100 GPUs.")
```

SM100 is Blackwell. All eight cards here report compute capability 9.0. The FP4
expert path does not exist on Hopper. The only escape vLLM names is
`expert_dtype="fp8"`, the Flash-Base checkpoint, whose experts would be 277 GB,
which does not fit on two cards and on four gives a larger pool and therefore a
worse f.

Quantisation format is now locked to hardware generation. A checkpoint is no
longer portable across GPU families, which did not use to be true.

### One hypothesis killed cheaply

`f8k256c72fic` ran `MOE_BACKEND=flashinfer_cutlass` against `f8k256c72` on auto
(Triton), everything else equal: tpot 83.4 vs 81.6 ms, in-flight 30.16 vs 30.22,
ISL within 0.3 percent. **A production grade MoE kernel changes nothing.** The
isolated harness anomaly is not kernel selection.

## 7. Four withdrawn proposals, and the shape of the mistake

1. **Lower CONC.** Raises per request tok/s and shrinks T, but also shrinks f,
   and f wins: window goes 48.5 s to 102.8 s and L1 falls from 20.7% to 13.6%.
2. **`NO_SCENARIO=1`.** Proposed from the comment in `arm.sh` claiming the
   scenario compresses think time and removes the long gap tail. Record
   2026/08/29/4 section 6 had already measured that claim false and withdrawn the
   proposal: `agentic_replay.py:433` fires only when the whole benchmark is idle
   and then shifts all pending timers equally, preserving relative spacing, and
   aiperf prints `jumps=0 skipped=0` for three of four arms at CONC 14 to 24. I
   read the harness source and repeated a claim that a record had already killed.
3. **Shrink the KV pool.** Correct arithmetic, wrong engineering. Real
   deployments do not run `gpu_memory_utilization` down; they take all the KV
   they can get. Proposed three times in different clothes.
4. **Swap the model.** Closed by section 6.

The common fault is not that any one was wrong. It is that an over-determined
system was repeatedly treated as an under-explored one. Once section 4's
identity was written down, in record 2, the answer was available. Four rounds
were spent looking for a way around an equality.

## 8. The corpus over-weights subagents by an order of magnitude

Raised in review: the trace's inter-turn gaps look too short for agentic work.
Measured from `reuse.tsv`, which carries `depth`:

| depth | turns | % turns | blocks | % blocks | conversations |
|---|---|---|---|---|---|
| 0, main agent | 56,798 | 57.5% | 278.3M | 82.3% | 393 |
| 1, subagent | 42,029 | 42.5% | 59.8M | 17.7% | 1,697 |

4.3 subagents per main session. Against the user's own 135 Claude Code sessions
on this machine, 2026-08-05 to 08-30:

| | measured local usage | corpus | ratio |
|---|---|---|---|
| subagents per session | 0.25 (34 `Agent` calls in 135 sessions) | 4.3 | 17x |
| subagent share of turns | 3.0% (2,799 of 91,925) | 42.5% | 14x |
| sessions with any subagent | 25% | n/a | |

Tool histogram for those sessions: Bash 24,192, Edit 1,999, Read 1,104, Write
442, `Agent` 34. **The corpus over-weights subagent fan-out by roughly an order
of magnitude and a half.** Caveat: one user, one workload, infrastructure
debugging. It is not proof about the wider population, but 17x is large enough
to record as a known corpus bias.

Block weighted reuse gaps, split by depth:

| subset | reused blocks | p10 | p25 | p50 | p75 | p90 | p95 |
|---|---|---|---|---|---|---|---|
| all | 331.6M | 6.5s | 10.4s | 18.1s | 42.6s | 132.9s | 239.9s |
| depth 0 only | 274.9M | 8.2s | 12.3s | 21.0s | 50.6s | 166.1s | 303.4s |
| depth 1 only | 56.7M | 3.9s | 5.9s | 8.7s | 14.6s | 30.6s | 60.1s |

L1 share by window:

| window | all | main agent only |
|---|---|---|
| 18 s | 50.3% | 56.7% |
| 48 s | 22.7% | 26.1% |
| **84 s (the SLO pinned one)** | **14.7%** | **17.1%** |
| 109 s | 11.8% | 13.8% |

**Stripping subagents entirely is worth 2.4 points.** A 17x error in the corpus
moves the conclusion by 2.4 percentage points, because subagents carry only
17.7 percent of reused blocks. The result is insensitive to the bias.

Note also that the main agent median gap is 21.0 s. Even with no subagents at
all, most turns in a Claude Code session are the model's own tool calls, not a
human typing. The human paced gaps are the p90 and p95 tail at 166 s and 303 s.

Switching to `semianalysis_cc_traces_weka_no_subagents` was considered and
rejected. It buys 2.4 points and costs the ability to say the corpus was not
chosen for its answer.

## 9. What the harness structurally cannot measure

The single replica residency frame is the wrong place to look for a shared
cache tier's value, and this is a property of the harness, not of L1.

**Cross replica portability.** A production fleet runs many replicas. A
session's next turn lands wherever the router sends it. GPU local prefix cache
does not follow the session; a shared L1 or L2 does. This value is entirely
independent of f, and a single replica benchmark measures none of it. It is
probably the primary production reason for a shared tier.

This belongs in the PR as a stated limitation, not as an omission.

## 10. What to claim

Decouple the two requirements, because they are independent quantities.

- **50 tok/s** is reachable and is the only number directly comparable with the
  hosted APIs. It needs in-flight near 7 at ISL 107k. The CONC that produces
  that is not known: in-flight over CONC measures 0.19 to 0.26 in the low load
  arms and 0.39 at CONC=64, and the middle is unmeasured.
- **L1's read share** is a function of the SLO and the hardware, pinned near 15
  percent, and is not ours to move. Do not claim it.
- **The write path** is where lazy offload acts and is independent of f. Every
  block is stored whether or not it is ever read, so the write path is fully
  loaded at any f. Store traffic falls only about a quarter going from CONC=64
  to in-flight 7, from roughly 42k to 32k new prefix tokens per second.

So: report stores down 86.8 percent, stored/isl down 30.7 points, L1 watermark
31 to 19, tpot down 7.0 percent, waiting down 40 percent, and state
cross-replica as unmeasured.

## 11. Open

- The isolated harness runs at 2.5 to 8 percent of KV peak where the live fit
  gives 31 percent. Narrowed by section 2 from blocking to a `bs.sh` bug. It
  still means no isolated number may be read as a live prediction, which
  includes all of record 5.
- in-flight versus CONC between 14 and 64 is unmeasured, so the CONC that
  delivers in-flight 7 is unknown.
- Whether lazy's write path advantage survives at in-flight 7. Untested, and it
  is the one thing that would invalidate section 10.
- Cross replica reuse, unmeasurable on this harness at any configuration.
- L0 intersect L1 point in time overlap, still unmeasured.
- `phase.py` compares wall clock times as strings, so an arm crossing midnight
  gets an empty window.
- `bs.sh` truncates `off_vllm.log` per server start, which lost the k=1
  acceptance in record 5.
