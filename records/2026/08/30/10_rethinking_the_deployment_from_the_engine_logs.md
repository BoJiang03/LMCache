# Session log: rethinking the deployment, and the probe that followed

Conversation record for 2026-08-30. Technical findings live in records 8 and 9
and in `records/deployment_candidate.md`; this is the arc of the session and
what each of Bo's interventions changed.

## 1. What was asked

> 你基于目前的硬件。思考怎么部署个 满足要求的模型。模型你选较新的就行。
> [...] 从新思考，不要遵循之前ctx的惯例

with R2 gaining a numeric band (`l1 / l0 in [1,3]`) and R7 added
(`ttft < 10s 最好 <5s`). Both are new requirements, recorded into
`records/deployment_requirements.md`.

The instruction to rethink rather than inherit was the right call and it paid
off immediately: the previous sessions had been inferring quantities the engine
had been logging all along.

## 2. The instrument that was there the whole time

Every arm archived its engine log as `<arm>/vllm.log.gz`. One line per ten
seconds carries f, per request decode speed, and both hit rates:

```
Avg generation throughput: Y tokens/s, Running: R reqs, Waiting: Q reqs,
GPU KV cache usage: U%, Prefix cache hit rate: H%, External prefix cache hit rate: E%
```

`GPU KV cache usage` is f exactly, because vLLM keeps completed-but-cached
blocks in the free queue, so the gauge counts only blocks pinned by running
requests. `Y/R` is per request decode speed. `E x (1-H) / H` is R2's ratio.

Reading it directly gave coder30's R1 operating point without any fitting:
50 tok/s is Running 10 at f = 0.345.

This corrected the tpot law of record 6 (fitted from client-side Little's law
in-flight) and the window identity of record 7, whose `ISL/A = 59` factor came
from the corpus's idealised new-block fraction rather than from what an engine
actually allocates, which is `1 - L0`.

## 3. Where the requirements collide, and what moves it

R1 fixes tpot, which caps bytes read per step. R4 and R5 fix the pool at
whatever the model leaves. So

```
f_max = tpot b / (HBM u - overhead) = 20 x 2.4 / 130.67 = 0.367
```

and any model small next to HBM sits at f ~= 0.3 at 50 tok/s regardless of GPU
count. coder30 measures 0.305 to 0.33 at TP=2, 4 and 8. At f = 0.31 the window
is 272 s and L1/L0 is 0.05, twenty times short.

The lever is the model, and it is R4-compatible: the pool shrinks because the
model is large, not because a knob was turned. Wanting f = 0.6 means about
65 GB of weights per GPU with `c(B) <~ 0.2`, hence `E/k >= 32`.

Sweeping the local catalogue on that basis picked Qwen3.5-397B-A17B-FP8 at
TP=4: 512 experts top-10, E/k = 51.2, the highest on the box.

## 4. Four interventions, three of which corrected me

**"btw, 没必要局限于本地的llm，如果有需要，可以上hf找。"** Checked and reported
back that it does not change the answer: the local `/raid/data/hub` tree
already carries GLM-5.1, GLM-5.3-Flash, Qwen3.5-122B, Qwen3.8-Flash-Next,
MiniMax-M2.7, Devstral-2-123B, Qwen3-Coder-480B and DeepSeek-V4-Flash. The one
model with the ideal shape, DeepSeek-V4-Flash, is blocked by the cards, not by
availability.

**"你说不定可以创建一个新的 uv venv来升级新的vllm"** and then
**"我之前要你换vllm就是为了去支持 混合模型"**. I had argued against switching on
comparability grounds, which missed the point entirely. The newer vLLM is not a
convenience here:

| | `attn_type == "hybrid"` |
|---|---|
| vllm-lazy 0.23.0, `config/model.py:1852` | False, "still experimental" |
| vllm-main 0.28.1rc1, `config/model.py:2128` | True, default on |

The pick is a hybrid model. Without the newer vLLM it has no L0 and no L1
except through an override vLLM itself calls experimental. Dropping the
comparability argument was correct.

**"能不能换个结构的模型？我怕lmcache不支持"** A reasonable worry that the code
answers: LMCache's hybrid support is deliberate, including in the lazy offload
path (`lazy_offload_policy/eviction_aware.py:280`). But chasing it down found
the real hazard, which is not hybrid support but silent fallback: vLLM's shim
prefers LMCache's external connector and falls back to its own builtin on any
ImportError, and only the external one declares `SupportsHMA`. vLLM disables
the hybrid KV cache manager when the connector does not declare it, without an
error.

**"store的cache size对吗？"** The sharpest question of the session and the
answer was no. vLLM forces the attention block size up to match the mamba page
(2096 on 0.28.1rc1, 2112 on 0.23.0) and LMCache requires
`chunk % block == 0`. The default 256 fails, and no power of two can ever work,
because 2096 = 2^4 x 131 and 2112 = 2^6 x 33.

## 5. Corrections I had to make in-session

- Misread my own arms table and briefly claimed the CONC=64 arms hit 50 tok/s.
  They are 16.4 tok/s; the tok/s and tpot columns were swapped in my reading.
  The previous session's "four arms at 57 to 77 tok/s at in-flight 1.7 to 3.6"
  was right.
- Said the block size was 2112 on the run that mattered. It is 2112 on
  vllm-lazy 0.23.0 and 2096 on vllm-main 0.28.1rc1.
- Predicted the scheduler-side `ValueError` at `lmcache_mp_connector.py:450`
  would fire. What fired was a worker-side `assert` at
  `vllm_multi_process_adapter.py:616`, same rule, different site.
- Set `LMCACHE_CHUNK_SIZE` on the vLLM process. Inert: the chunk size is read
  from the MP server over the message queue, so it is the server's
  `--chunk-size` flag.

## 6. Three failures, none of them the model

1. Triton's launcher build shells out to gcc and there is no `python3.12-dev`
   on this box, so `Python.h` is missing. `env.sh:40` already carries the
   `CPATH` workaround; launching without sourcing it loses it.
2. The vllm-main venv lacks `sortedcontainers` and
   `opentelemetry.exporter.prometheus`. Fixed without touching shared state: the
   first into a scratchpad directory on `PYTHONPATH`, and the MP server run on
   the vllm-lazy interpreter, which has the deps. Separate processes over ZMQ,
   same worktree code on both sides.
3. Self-inflicted. A `p3.sh` still inside its 120 s MP health-wait loop passed
   its check when a later launch brought a server up on the same port, and
   `exec`ed a second `vllm serve` onto the same four cards. Two EngineCores,
   two workers claiming rank 3, CUDA OOM at 139 of 139.8 GiB. Check for a live
   launcher before relaunching, not only for a live server.

## 7. Where it stands

Measured: weights 98.5 GB/GPU (394 GB, 150 s), pool 115.7 GB / 3,568,733
tokens / 32,434 B per token, forced block 2096, hybrid prefix caching on by
default, connector resolving to the `SupportsHMA` class. With `--chunk-size
2096` the engine gets past the assert and reaches CUDA graph capture.

Computed but not measured: the whole operating point. B = 17 gives 53.6 tok/s
at an L1/L0 ceiling of 2.09, which satisfies R1 and R2 together and is the
first configuration in this line of work that does. Both constants behind it
were calibrated on coder30: `b = 2.4 GB/ms/GPU` on 4 KV heads at head_dim 128,
and `c(B)` at E/k = 16 against this model's 51.2.

Not measured at all: no request has been served. No observed tok/s, TTFT, hit
rates, Stored or Retrieved.

Asked directly whether there is a conforming configuration yet, the answer was
no. There is a candidate that computes as conforming and is confirmed to start.

## 8. Open

1. Stored and Retrieved at block 2096. The server is coming up now.
2. The 68 MB storage granule that a 2096-token chunk implies at 32,434 B per
   token, two orders of magnitude coarser than the 256 every arm on disk ran.
   The assert passing does not mean L1 behaves.
3. A CONC sweep to replace every computed number. Design not yet discussed.
4. `assert` used for runtime validation at
   `integration/vllm/vllm_multi_process_adapter.py:616` and :1153, which
   `docs/coding_standards.md` forbids. Under `python -O` it vanishes and the
   failure mode becomes misaligned block-id slicing instead of a dead engine.
   Separate fix, worth a PR of its own.
