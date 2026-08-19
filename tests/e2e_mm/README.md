# Multimodal Model Support Acceptance Suite (`tests/e2e_mm`)

This suite defines what it means for LMCache to **support** a multimodal
model. A model is declared supported only when it passes the full matrix
below on every claimed deployment path. The suite launches real vLLM engines
and requires a GPU; it is strictly opt-in.

## How to run

```bash
export LMCACHE_MM_E2E=1
cd tests/e2e_mm && pytest .            # run from THIS directory (see below)
```

Environment knobs:

| Variable | Default | Meaning |
|---|---|---|
| `LMCACHE_MM_E2E` | unset | Must be `1` for any test to run (opt-in guard). |
| `LMCACHE_MM_E2E_MODELS` | `qwen2.5-vl-3b` | Comma-separated model keys from `specs.py`. |
| `LMCACHE_MM_E2E_PRESSURE_N` | `64` | Number of distinct images in the collision pressure test (T0.2). Nightly runs should raise this to `1000`. |

Run from inside `tests/e2e_mm` so its local `pytest.ini` anchors the rootdir
and the global `tests/conftest.py` (autouse mocks and allocator patches that
interfere with a real engine) is not loaded. The suite forces
`VLLM_ENABLE_V1_MULTIPROCESSING=0` so the LMCache stats singleton is
readable in-process. The conftest pins `import lmcache` to THIS repo and
fails loudly if a stray editable install resolves it elsewhere — without
that guard the suite can silently certify a different source tree.

## Verdict method

- **Baseline comparison**: every request is first run on a plain vLLM engine
  (no LMCache, no prefix caching, greedy decoding) in a subprocess; the
  LMCache engine must reproduce those outputs token-for-token.
- **Semantic probe fallback**: test images are synthetic solid-color images
  and prompts ask for the dominant color in one word. If an exact-match
  comparison fails but the probe answer is still correct, the case passes
  with a warning (GPU nondeterminism); if the probe answer is wrong, the
  case fails hard — that is a cross-image contamination, no appeal.
- **Hit-counter assertions**: every step also asserts LMCache lookup hit
  counts. A false hit is "a hit where a miss was expected" and is caught by
  counters even when the output happens to survive.

## Acceptance matrix

### T0 — Correctness (any failure = NOT supported)

| # | Test | Assertion |
|---|---|---|
| T0.1 | No cross-image hits | After caching image A with prompt P, image B (same resolution, identical placeholder tokens) with the same P must produce B's baseline output. |
| T0.2 | Collision pressure | N distinct same-shape images, two passes: pass 1 hit counts stay flat (any excess hit = false hit; regression for issue #3301); pass 2 answers equal pass 1. |
| T0.3 | Hit equivalence | The same request twice: the second run must hit and produce the identical output. |
| T0.4 | Chunk-boundary phases | T0.1/T0.3 hold when the image placeholder span crosses chunk boundaries at varying phases (text prefix padded 0..chunk_size-1 words, chunk_size=16). |
| T0.5 | Mixed traffic | Interleaved text-only and multimodal requests do not contaminate each other. |

### T1 — Effectiveness (the cache must actually work)

| # | Test | Assertion |
|---|---|---|
| T1.1 | Reuse depth | The repeat request in T0.3 hits at least prompt length minus a small tail. |
| T1.2 | Prefix reuse | Same image + a different follow-up question partially hits (shared system+image prefix). |
| T1.3 | Non-degenerate | Multimodal requests record nonzero hits — implementations that pass T0 by bypassing the cache for MM requests fail here. |

### T2 — Scenario coverage (as applicable per model)

| # | Test | Assertion |
|---|---|---|
| T2.1 | Multi-image order | A request with images (A, B) and one with (B, A) do not cross-hit and each answers correctly. |
| T2.2 | Partial sharing | Request [A] then request [A, C]: shared prefix hits, C computed correctly. |
| T2.3 | Other modalities | For models with video/audio, T0.1/T0.3 rerun per modality (not yet implemented). |

### T0.6 — Benchmark score parity (`benchmark_parity.py`)

The synthetic matrix proves cache-key isolation; this tier proves the cache
HIT path does not degrade real model quality. It scores the full **MME**
benchmark (2374 yes/no questions, standard Perception/Cognition scoring)
three ways: plain-vLLM baseline, LMCache cold pass (miss path), and an
identical second pass where every prompt's KV is restored from LMCache (hit
path). Pass criteria: answer flips ≤ 0.5% and |total score delta| ≤ 10
points (of 2800) on BOTH comparisons (pass1 vs baseline, pass2 vs pass1),
and pass2 lookup hit ratio ≥ 0.8. The pass1-vs-baseline gate matters:
cross-image contamination poisons the cache on the cold pass and then
replays deterministically, so the pass2-vs-pass1 comparison alone cannot
detect it.

```bash
cd tests/e2e_mm && CUDA_VISIBLE_DEVICES=0 python benchmark_parity.py
```

Long-running (three full benchmark passes); intended for nightly/release
validation rather than PR CI. Certification for the "supported" level
requires one recorded parity run per model.

### T3 — Deployment paths

The same T0+T1 set must pass per path. Currently implemented: in-process
`LMCacheConnectorV1`. Planned: `LMCacheMPConnector`, CPU offload round-trip,
remote backend cross-instance.

### Special-architecture add-ons

- **DeepStack** (Qwen3-VL / Qwen3.5): hit-path output must equal baseline
  despite multi-layer visual injection outside paged KV.
- **Gemma 3**: bidirectional image attention vs chunk-boundary phases
  (strengthened T0.4).
- **Phi-4-multimodal**: different LoRA on identical tokens must not share
  cache entries.

## Support levels

- ✅ **Supported**: T0 + T1 + applicable T2 + claimed T3 paths all green.
- ⚠️ **Safe but not accelerated**: T0 green with an explicit MM bypass
  (T1.3 waived and documented).
- ❌ **Not supported**: any T0 failure.

## Adding a model

Add one `ModelSpec` entry in `specs.py` (HF id, modalities, smallest
variant, optional `extra_suites` flags) and run the suite with
`LMCACHE_MM_E2E_MODELS=<key>`. No new test code is needed for standard
placeholder-injection architectures.
