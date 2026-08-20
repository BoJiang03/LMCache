# Multimodal Model Support Acceptance Suite (`tests/e2e_mm`)

This suite defines what it means for LMCache to **support** a multimodal
model. The claim is operationalized by `certify.py`: one command per model
that runs every layer and emits a machine-readable certificate whose verdict
(`SUPPORTED`) — never an individual green test — is the artifact behind the
support claim. The suite launches real vLLM engines and requires a GPU; it
is strictly opt-in.

## Why passing implies support (completeness argument)

LMCache's multimodal job decomposes into: compute cache keys that carry
image identity, look up and retrieve KV without corrupting it, store every
missed token exactly once, and survive real scheduling (batching, split
prefills, eviction). The suite is built by enumerating the ways each duty
can fail and pointing at least one detector at every mode:

| Failure mode | Symptom | Detector |
|---|---|---|
| Keys blind or partially blind to image identity (e.g. issue #3301 16-bit truncation) | cross-image KV serving | T0.1 / T0.2 / T0.4 counter margins + semantic probes; MME pass1-vs-baseline gate |
| Unstable keys (same image, different keys across requests) | no reuse, duplicate stores | T0.3 / T1.1 full-hit floor; T0.7 replay-stores-nothing bound |
| Placeholder-span misalignment vs chunk boundaries | either of the above at specific alignments | T0.4 (all 16 phase offsets) |
| KV corrupted on the hit path (retrieve bugs) | wrong output despite correct keys | T0.3 hit-equals-miss output; MME pass2-vs-pass1 gate |
| MM requests silently bypass the cache | "safe" but useless | T1.1 / T1.3 nonzero-hit floors |
| Store drops KV (missed tokens never stored / never resident) | wasted recompute, cold cache | T0.7 conservation: store intent vs resident keys/bytes |
| Store and lookup disagree on keys | stored but never hittable | T1.1 full hit + T0.7 replay re-store bound |
| Scheduler step splits an image span (chunked prefill) | store-side truncated-span bugs | T0.9 (dedicated small-budget engine) |
| Concurrent store/lookup races in a batch | contamination or loss under load | T0.8 duplicate-in-batch; MME (2374 batched requests) |
| Eviction misbehavior (false hit after evict, unbounded growth, corrupt recompute) | wrong output or OOM at capacity | T0.10 (dedicated tiny-capacity engine) |
| Preemption recompute corrupts state (stale keys, wrong restored tokens, poisoned cache) | wrong output after a preemption round-trip | T0.11 (dedicated tiny-block-pool engine, preemption PROVEN via the vLLM counter) |
| A different deployment path skips or breaks the MM key handling (e.g. MP connector) | per-path cross-image serving | T3 mp_connector scenario (real MP cache server) incl. its own negative control |
| Modality-specific ingestion misses identity (video frames, temporal merge) | cross-video KV serving | T2.3 per declared modality |
| Quality drift only visible on real data (resolutions, aspect ratios, numerics) | statistical score loss | T0.6 MME three-way parity |
| The detectors themselves are broken | false green on everything above | negative control: induced identity blindness MUST trip the counter check (run on BOTH deployment paths) |

Layering: the synthetic tests are deterministic, minute-scale, and
localizing — a single false hit trips a counter invariant. MME parity is the
certification layer for anything only statistically visible. The negative
control makes a green run self-evidencing: the suite proves its own tripwire
fires before it certifies anything.

**Replay oracle** (`check_replay_text`): a hit-path replay is compared to
its own miss-path output, but the miss pass (KV computed) and the hit pass
(KV loaded) are different numeric regimes, so byte equality is expected, not
required. A divergent replay still passes — with a warning — when the
extracted final answer matches (`ModelSpec.answer_extract_pattern`, e.g.
GLM's boxed answers) or the semantic probe passes; it fails hard otherwise.
Verbose answer styles give regime noise many tokens to flip phrasing, while
real KV corruption or contamination flips the answer itself. Models with no
pattern and no probe keep strict byte equality (Qwen answers are 1–8 direct
tokens; equality holds there in practice).

**What a pass does NOT claim** (recorded verbatim in every certificate):
only the deployment paths listed in the certificate scope are certified —
currently the in-process `LMCacheConnectorV1` and the `LMCacheMPConnector`
+ MP cache server pair, each on a single GPU (TP=1). TP>1, remote/disk
backends, the audio modality (no audio model registered), and
allocator-level buffer accounting are outside the claim until their tests
exist; on the MP path the chunk-boundary phases and collision pressure
tiers run only in-process (cache keys are computed identically on both
paths, so the keyspace properties are transport-independent).

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
| T0.7 | Storage conservation | On the T0.2 traffic: every token the lookup missed is store-requested and lands as resident chunk keys in the local CPU backend (deficit = silently dropped KV); the full-hit replay stores ~nothing new, never loses resident keys, and resident bytes track keys (growth without keys = a leak). |
| T0.8 | Concurrent batch | One batch containing duplicate image requests plus mixed traffic: every entry's output verified, and the entry cached during the batch must fully hit afterwards. |
| T0.9 | Chunked prefill | Dedicated engine with `max_num_batched_tokens` far below the prompt length, pad phases sweeping the step boundary across the image span: miss/full-hit/isolation invariants and store conservation all hold when stores end mid-image (`test_isolated_paths.py` / `isolated_cases.py`). Outputs are checked against a plain-vLLM baseline computed under the SAME scheduling config — small models misname colors behind long pads even without LMCache, so a bare probe would misattribute model weakness to the cache. |
| T0.10 | Capacity eviction | Dedicated engine with a ~50 MB cache overflowed several times by distinct images: no false hits ever, resident bytes stay under the cap, and evicted requests recompute to exactly their first-pass output. |
| T0.11 | Preemption recompute | Dedicated engine with a tiny GPU block pool and forced-length decodes so the scheduler MUST preempt (proven via `vllm:num_preemptions`; zero preemptions fails as vacuous): every batch output verifies against the config-matched baseline, and every request afterwards fully hits and verifies again — the preemption round-trip neither corrupts KV nor poisons the cache. (Byte-equality between the concurrent batch and the solo replay is not asserted: the ignore_eos garbage tail amplifies cross-regime kernel numerics; contamination still fails hard via the probe.) |
| — | Detector negative control | With MM identity substitution deliberately disabled for a fresh salt, the T0.1-style counter check MUST trip (the second image must falsely hit). A failure here invalidates every green counter assertion. Run on the in-process path (main suite) AND inside the T3 MP scenario. |

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
| T2.3 | Other modalities | For models whose spec declares `video`: synthetic solid-color MP4s rerun T0.1/T0.3/T1 on the video ingestion path (multi-frame decode, temporal merge). Deselected — not skipped — at collection for models without the modality, so certification stays exactly as wide as the spec. Audio: no audio model registered yet. |

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

The same T0+T1 core must pass per path. Implemented:

- **In-process `LMCacheConnectorV1`** — the full matrix above.
- **`LMCacheMPConnector` + MP cache server** (`isolated_cases.py
  mp_connector`): a real `lmcache.v1.multiprocess.http_server` subprocess,
  the engine driven through this repo's connector via
  `kv_connector_module_path`. Reruns T0.1/T0.3/T0.5/T0.8, T1.1–T1.3,
  T2.1/T2.2, store conservation against the server's resident-object API
  (`/status`), and its own detector negative control. Lookup/hit counters
  come from wrapping the scheduler adapter's lookup submit/check calls;
  store intent from the worker adapter's batched store submissions.
  T0.4 phases and T0.2 pressure stay in-process: both paths compute cache
  keys with the same `apply_mm_hashes_to_token_ids`, so keyspace properties
  are transport-independent.

Planned: CPU offload round-trip, remote backend cross-instance, TP>1.

### Special-architecture add-ons

- **DeepStack** (Qwen3-VL / Qwen3.5): hit-path output must equal baseline
  despite multi-layer visual injection outside paged KV.
- **Gemma 3**: bidirectional image attention vs chunk-boundary phases
  (strengthened T0.4).
- **Phi-4-multimodal**: different LoRA on identical tokens must not share
  cache entries.

## Certification (`certify.py`)

One command produces the support verdict and its evidence:

```bash
cd tests/e2e_mm
python certify.py qwen2.5-vl-3b --run-parity            # full certification
python certify.py qwen2.5-vl-3b --parity-report mme_full.json  # reuse a run
python certify.py qwen2.5-vl-3b                         # suite only
```

It runs the entire synthetic suite (including the isolated scenarios and
the negative control), combines it with an MME parity result, and writes
`certificate_<model>.json` recording the verdict, the exact commit, the
certified scope (deployment paths, modalities, backend), all measurements,
and the known-not-covered list. Exit codes: 0 `SUPPORTED`,
2 `PROVISIONAL` (suite green, parity not provided), 1 `NOT_SUPPORTED`.
A skipped or empty suite can never certify (skips are counted as failure).

## Support levels

- ✅ **Supported**: certificate verdict `SUPPORTED` — synthetic suite green
  AND MME parity gate passed, for the scope named in the certificate.
- ⚠️ **Safe but not accelerated**: T0 green with an explicit MM bypass
  (T1.3 waived and documented).
- ❌ **Not supported**: any T0 failure or a failed parity gate.

## Adding a model

Add one `ModelSpec` entry in `specs.py` (HF id, modalities, smallest
variant, optional `extra_suites` flags) and run the suite with
`LMCACHE_MM_E2E_MODELS=<key>`. No new test code is needed for standard
placeholder-injection architectures.

Per-model answer-style adaptations are declared on the spec, never coded
into the tests:

- `chat_template_kwargs` — e.g. `{"enable_thinking": False}` for
  hybrid-thinking models (GLM-4.6V). The suite's oracles read the generated
  text directly, so a reasoning preamble must be disabled or budgeted for.
- `min_decode_tokens` — floor for every request's `max_tokens` (suite,
  baselines, MME parity). Needed when the model answers after a preamble
  even with thinking off (GLM's `<|begin_of_box|>`-boxed answers land
  within ~64 tokens).
- `mme_mm_processor_kwargs` — model-specific per-image token cap for the
  MME parity engines (Qwen: `max_pixels`; GLM: `size.longest_edge` in
  total pixels). MME photos are arbitrarily large; without a cap a single
  image can exceed the 8192-token parity context.
- `mme_max_tokens` — decode budget for the MME parity runs only. Real MME
  photos draw much longer reasoning than the suite's synthetic images
  (GLM needs 256 there vs 64 in the suite); too small a budget truncates
  answers and fails the parse-rate gate.

The MME parity gate additionally enforces a baseline answer parse-rate
(`MIN_PARSE_RATIO`): if a model's yes/no verdict does not land inside the
decode budget, all three passes parse to '' and the flip/score comparisons
would pass vacuously — the parse-rate guard turns that into a hard FAIL
instead.
