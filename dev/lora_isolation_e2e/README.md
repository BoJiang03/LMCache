# LoRA KV isolation e2e (MP mode)

Checks that LMCache MP mode does not serve one LoRA adapter's KV to another
adapter or to the base model. Dev-only; not part of the PR.

## What it does

vLLM (Qwen2.5-1.5B, two random LoRA adapters on q/k/v/o) with
`LMCacheMPConnector`, L1 only. One 4096-token prompt is sent as
`lora-a`, `base`, `lora-b`, `lora-a` again. Before each request vLLM's own
prefix cache is reset, so every hit comes from LMCache. Per request it
records `vllm:external_prefix_cache_hits_total` (LMCache hit tokens) and
whether the output equals a reference vLLM without LMCache
(`VLLM_BATCH_INVARIANT=1` on both).

Hit counts are the decisive signal. The random-token prompt gives
degenerate outputs, so a wrong KV can still produce a matching output.

Lazy groups use the FIFO policy with threshold 1, and `client.py` sends a
short unrelated request plus 3 s after every request, so the deferred store
lands before the next lookup. A later hit on the same prefix (g4 `A-again`,
g3 `base`) proves the store happened.

## Groups

| group | code | lazy |
|---|---|---|
| g1-prefix | #5843 head, no fix | off |
| g2-fix | this branch | off |
| g3-utilsonly-lazy | connector tag fix only, lazy coalesce not fixed | on |
| g4-fix-lazy | this branch | on |

## Run

Needs one free GPU (about 0.6 of an H200), about 30 min including startup.
Paths and ports are in `env.sh` and can be overridden from the environment.

```bash
cd dev/lora_isolation_e2e
# 1. native extensions in place (CUDA 13 toolchain; the system nvcc is 11.5)
(cd ../.. && CUDA_HOME=/usr/local/cuda-13.0 PATH=/usr/local/cuda-13.0/bin:$PATH \
  SETUPTOOLS_SCM_PRETEND_VERSION_FOR_LMCACHE=0.0.0 MAX_JOBS=48 TORCH_CUDA_ARCH_LIST=9.0 \
  $PY setup.py build_ext --inplace)
# 2. adapters and control trees
$PY make_adapters.py adapters /raid/models/Qwen2.5-1.5B-Instruct
bash make_controls.sh
# 3. reference server in one shell, groups in another
bash baseline.sh &
bash run_all.sh
```

Per-group logs land in `out/<group>/` (`tree.txt` shows which tree was
imported, then `lmcache.log`, `vllm.log`, `client.log`, `results.json`).

## Result (2026-10-29, H200 GPU 1)

LMCache hit tokens per request (`*` = output differs from reference):

| group | A | base | B | A-again |
|---|---|---|---|---|
| g1-prefix | 0 | 4095 * | 4095 | 4095 |
| g2-fix | 0 | 0 | 0 | 4095 |
| g3-utilsonly-lazy | 0 | 4095 * | 0 | 0 |
| g4-fix-lazy | 0 | 0 | 0 | 4095 |

- g1: base and lora-b read lora-a's KV; base output is corrupted.
- g2: isolated, and lora-a still hits its own KV.
- g3: the coalesced lazy store dropped the tag, so lora-a's KV landed in
  the untagged (base) namespace: base hits it, lora-a itself misses.
- g4: isolated under lazy offload too.

Raw rows are in `results/`.
