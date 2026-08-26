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
| A cold cache server plus a fresh engine breaks the MM key handling | cross-image serving on a path nothing else exercises | T3 mp_connector scenario (its own server and engine) incl. its own negative control |
| Modality-specific ingestion misses identity (video frames, temporal merge) | cross-video KV serving | T2.3 per declared modality |
| One modality's identity is dropped while another's covers for it, or items are keyed as a set rather than a sequence | cross-modal false hit invisible to every single-modality test | T2.5 (image held constant while the clip swaps; then the same two items reversed) |
| Quality drift only visible on real data (resolutions, aspect ratios, numerics) | statistical score loss | T0.6 MME three-way parity |
| The detectors themselves are broken | false green on everything above | negative control: induced identity blindness MUST trip the counter check (run in the session suite AND in the T3 scenario) |
| A "hit" is reported but the retrieve path never ran (vLLM's own prefix cache served it) | green suite proving nothing about the load path | hit-provenance oracle on every measured step (see below) |

Layering: the synthetic tests are deterministic, minute-scale, and
localizing — a single false hit trips a counter invariant. MME parity is the
certification layer for anything only statistically visible. The negative
control makes a green run self-evidencing: the suite proves its own tripwire
fires before it certifies anything.

**Hit-provenance oracle** (`MMHarness._check_hit_provenance`): every
measured step cross-checks LMCache's counters against vLLM's own per-prefill
accounting (`PrefillStats`, split into local-prefix-cache and
external-connector tokens). LMCache's counters report what the cache HELD
for a prompt, not what the engine loaded from it, and the two diverge
whenever vLLM prefix caching is on: measured on Qwen3.5-2B (2026-08-21), a
repeated prompt is served entirely out of vLLM's GPU cache (544 local / 0
external) while the connector still reports a 544-token hit. Without this
check every hit-count assertion in the hybrid suite would pass with the
retrieve path never running. Two rules per step: vLLM's own cache must not
have served a single-request replay, and the hits LMCache reported must
have actually been loaded (one token of slack per request — vLLM always
recomputes a prompt's final token). The preemption scenario opts out of the
second rule via `harness.unloaded_hits_allowed()`, since LMCache
deliberately declines to reload a preempted request.

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
only the deployment path listed in the certificate scope is certified —
`LMCacheMPConnector` + MP cache server on a single GPU (TP=1). The
in-process `LMCacheConnectorV1` path is NOT covered: the suite drives the
multi-process deployment only and no longer contains an in-process harness
(removed 2026-08-26; branch `archive/e2e_mm-inprocess-and-mp` and git
history carry it). TP>1, remote/disk backends, and allocator-level buffer
accounting are outside the claim until their tests exist.

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
`VLLM_ENABLE_V1_MULTIPROCESSING=0` so the connector adapters it wraps for
its lookup/store counters run in the test process. The conftest pins `import lmcache` to THIS repo and
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
| T0.7 | Storage conservation | On the T0.2 traffic: every token the lookup missed is store-requested and lands as resident objects in the MP server's L1 pool (deficit = silently dropped KV); the full-hit replay stores ~nothing new, never loses resident keys, and resident bytes track keys (growth without keys = a leak). |
| T0.8 | Concurrent batch | One batch containing duplicate image requests plus mixed traffic: every entry's output verified, and the entry cached during the batch must fully hit afterwards. |
| T0.9 | Chunked prefill | Dedicated engine with `max_num_batched_tokens` far below the prompt length, pad phases sweeping the step boundary across the image span: miss/full-hit/isolation invariants and store conservation all hold when stores end mid-image (`test_isolated_paths.py` / `isolated_cases.py`). Outputs are checked against a plain-vLLM baseline computed under the SAME scheduling config — small models misname colors behind long pads even without LMCache, so a bare probe would misattribute model weakness to the cache. |
| T0.10 | Capacity eviction | Dedicated engine with a ~50 MB cache overflowed several times by distinct images: no false hits ever, resident bytes stay under the cap, and evicted requests recompute to exactly their first-pass output. |
| T0.11 | Preemption recompute | Dedicated engine with a tiny GPU block pool and forced-length decodes so the scheduler MUST preempt (proven via `vllm:num_preemptions`; zero preemptions fails as vacuous): every batch output verifies against the config-matched baseline, and every request afterwards fully hits and verifies again — the preemption round-trip neither corrupts KV nor poisons the cache. (Byte-equality between the concurrent batch and the solo replay is not asserted: the ignore_eos garbage tail amplifies cross-regime kernel numerics; contamination still fails hard via the probe.) |
| — | Detector negative control | With MM identity substitution deliberately disabled for a fresh salt, the T0.1-style counter check MUST trip (the second image must falsely hit). A failure here invalidates every green counter assertion. Run in the session suite AND inside the T3 scenario. |

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
| T2.3 | Other modalities — video | For models whose spec declares `video`: synthetic solid-color MP4s rerun T0.1/T0.3/T1 on the video ingestion path (multi-frame decode, temporal merge). Deselected — not skipped — at collection for models without the modality, so certification stays exactly as wide as the spec. |
| T2.4 | Other modalities — audio | For models whose spec declares `audio`: synthetic clips rerun T0.1/T0.3/T1 on the audio ingestion path (own processor, resampler and encoder). Paired with its **own** negative control: the image control proves nothing about audio, because if audio items never reached vLLM's `mm_features` the isolation assertion would pass trivially and look green. Same collection-time deselection policy. |
| T2.5 | Cross-modal — image + audio | For models declaring **both** modalities (all `requires_modality` markers must hold, not just the closest one). One prompt carries an image and a clip, and two swaps are compared against it. Holding the **image constant** while swapping only the clip makes the audio hash the sole separator — everywhere else in the suite audio isolation is confounded with a differing text or image prefix. Reversing the **order** of the same two items changes no content at all, so only position can separate them; a key over the multiset of items would collide here and nowhere else. Measured caveat: both orders answer identically (`red, tone` either way, 5/5 combos), so the order half is a counter-only assertion. |

### T0.6 — Benchmark score parity (`benchmark_parity.py`)

The synthetic matrix proves cache-key isolation; this tier proves the cache
HIT path does not degrade real model quality. It scores a benchmark three
ways: plain-vLLM baseline, LMCache cold pass (miss path), and an identical
second pass where every prompt's KV is restored from LMCache (hit path).
Which benchmark comes from `ModelSpec.parity_benchmark`:

| `--benchmark` | Data | Scoring | Score delta budget |
| --- | --- | --- | --- |
| `mme` (default) | lmms-lab/MME, 2374 yes/no questions over images | standard Perception/Cognition | ≤ 10 of 2800 |
| `mmau` | TwinkStart/MMAU test-mini, 1000 four-way questions over audio | mean of per-task (sound/speech/music) accuracy | ≤ 1.0 of 100 |

An image benchmark cannot measure an audio model's quality, so a
`Benchmark` subclass supplies the four things that differ — items,
conversations, answer parsing, scoring — and everything else (the three
passes, the counters, the hit-coverage arithmetic, the gate) is shared. The
per-task breakdown is load-bearing for MMAU: accuracy ranges from 59 to 71
across its three tasks, so an aggregate-only score would average away a
regression confined to one of them.

Pass criteria: answer flips ≤ 0.5% and |total score delta| within the
benchmark's budget on BOTH comparisons (pass1 vs baseline, pass2 vs pass1),
plus a non-vacuity gate on the hit path. The pass1-vs-baseline gate
matters: cross-image contamination poisons the cache on the cold pass and
then replays deterministically, so the pass2-vs-pass1 comparison alone
cannot detect it.

The non-vacuity gate is granularity-dependent. At the 16-token chunk size
it is the raw pass-2 lookup hit ratio (≥ 0.8). A hybrid model caches at
vLLM's unified block size (e.g. 544 tokens), where an 800-token MME prompt
caps out at a 0.68 raw ratio however perfect the hit path is, so those runs
are gated on **coverage** instead: pass 2 must find ≥ 95% of the tokens
pass 1 actually stored. Coverage is granularity-free and strictly sharper;
the report records which criterion applied and the certificate refuses a
parity report produced on a path the model is not certified on.

```bash
cd tests/e2e_mm && CUDA_VISIBLE_DEVICES=0 python benchmark_parity.py
```

The LMCache passes run against a cache server the script starts.
`--hybrid-block-tokens N` chunks that server at the unified block size and
gives each KV cache group its own objects, applies the mandatory
hybrid engine settings to the baseline engine too, and empties vLLM's own
prefix cache between passes — `align` mode forces vLLM prefix caching on,
which would otherwise serve pass 2 out of GPU memory and leave LMCache
unasked. `certify.py` passes it from the spec.

Long-running (three full benchmark passes); intended for nightly/release
validation rather than PR CI. Certification for the "supported" level
requires one recorded parity run per model.

### T3 — Cold-start replay on a dedicated engine and server

The suite drives **one** deployment: `LMCacheMPConnector` + an
`lmcache.v1.multiprocess.http_server` subprocess, with the engine pointed
at this repo's connector via `kv_connector_module_path`. Lookup/hit
counters come from wrapping the scheduler adapter's lookup submit/check
calls; store intent from the worker adapter's batched store submissions;
residency from the server's `/status` API.

The in-process `LMCacheConnectorV1` path was removed on 2026-08-26 (branch
`archive/e2e_mm-inprocess-and-mp` keeps it). Two independent reasons: vLLM
offers its hybrid KV cache manager solely to connectors advertising
`SupportsHMA`, which that connector does not — so it fails engine init
outright on any multi-KV-group model ("Hybrid KV cache manager is disabled
but failed to convert the KV cache specs to one unified type") — and on
vLLM ≥ 0.26 its fused KV layout is corrupted (LMCache #4463 / #4467, both
of which state the MP connector is unaffected). Certifying two paths only
diluted every verdict.

T3 (`isolated_cases.py mp_connector`) survives that removal as what it now
is: the T0+T1 core replayed against a **freshly started** server whose L1
has seen nothing else, on an engine of its own built at the isolated GPU
fraction. It reruns T0.1/T0.3/T0.5/T0.8, T1.1–T1.3, T2.1/T2.2, store
conservation against `/status`, and its own detector negative control.
T0.4 phases and T0.2 pressure run once, in the session suite: they are
keyspace properties of `apply_mm_hashes_to_token_ids` and do not change
with cache state.

Planned: CPU offload round-trip, remote backend cross-instance, TP>1.

**Multi-KV-group models come in two kinds**, which is why `hybrid_family`
exists alongside `hybrid_block_tokens`.

Which isolated scenarios a model runs is decided in one place,
`isolated_routing.isolated_scenarios`, which both the parametrization and
the certificate read — they are the same statement seen from two sides, and
they drifted apart once when only one of them was updated. `mp_connector`
and `capacity_eviction` apply to every model; eviction's cap becomes
per-model wherever one cache object is too big for the shared default
(`ModelSpec.eviction_capacity_gb`, needed by the 27B recurrent-state
hybrids whose state page is ~154 MB). `chunked_prefill` is excluded for
models whose scheduling forbids a sub-prompt token budget.
`preemption` needs a measured `ModelSpec.preemption_gpu_blocks` on a
hybrid, and is unavailable to the `RECURRENT_STATE` family at any pool
size — `certify._PREEMPTION_NOT_COVERED` carries the measurements.

The two kinds differ in what else the engine needs:

- `RECURRENT_STATE` (Mamba/Gated-DeltaNet: Qwen3.5/3.6/3.8) keeps
  per-sequence state pages instead of per-token KV, so `align` mode and its
  two companions are mandatory, and a hit *restores* a lossy state summary
  rather than reproducing KV bit-for-bit.
- `SLIDING_WINDOW` (Gemma 4) is ordinary paged KV throughout; its groups
  differ only in window and block size, so it needs none of the align
  settings and runs with prefix caching off like a single-group model.

Gemma 4 also shows why the chunk size is not simply the block size vLLM
reports. Its 512-wide full-attention layers page at block 16 while its
256-wide sliding layers page at block 32 (vLLM equalizes page size by
varying the block size), and LMCache requires the chunk to be a multiple of
every paged group's block — so the chunk is 32 and
`cache_config.block_size` is 16. `_validate_block_size` checks that rule
rather than equality. Only 24 of its 42 layers have their own KV at all:
`num_kv_shared_layers=18` makes the rest reuse another layer's, so per-token
cost is measured, not derived from layer count.

**A small chunk size starves the server's heartbeat, which is why the MP
servers run several CPU workers.** The server registers `PING` in the same
NORMAL thread pool as `LOOKUP`, and that pool defaults to one worker, so a
liveness probe queues behind the workload it is monitoring. A GDN hybrid
hides this (one 784-token chunk per prompt, so lookups are rare); Gemma 4's
32-token chunk issues ~25 lookups per prompt and the ping never lands --
measured 2026-08-21, the connector declared the server dead after five 60s
intervals while the server log showed retrieves completing in 4ms, and the
run then died in the fatal-load-error path below. `MP_SERVER_CPU_WORKERS`
keeps the probe answerable. That a health check shares a queue with the
data plane is an upstream defect, not something the suite can fix.

**A failed KV load is fatal on a hybrid, so the suite must not manufacture
one.** When the connector reports load errors, vLLM rewinds the affected
requests in `_update_requests_with_invalid_blocks`, which unpacks a single
KV cache group (`(req_block_ids,) = ...get_block_ids(req_id)`, carrying an
explicit "TODO: add support for hybrid memory allocator") and so raises
`ValueError: too many values to unpack` on a multi-group model. The
connector only reports load errors while it believes the server is dead,
and it believes that after a single heartbeat ping missing its 10s window
-- which happens on this host because a real MP retrieve is reported back
0.3-20s after submission though the server transfers it in ~3ms. The suite
therefore runs every MP engine with a 60s heartbeat window and a matching
server reap timeout (`MP_HEARTBEAT_INTERVAL_S`), so the pressure cases
measure cache behaviour rather than that latency. The latency and the
unrecoverable-load-error pair are recorded findings, not fixed ones; the
certificate excludes degraded-mode recovery on hybrids explicitly.

### Special-architecture add-ons

Declared per model via `ModelSpec.extra_suites`; add-on tests carry
`@pytest.mark.requires_extra_suite("<name>")` and are deselected — not
skipped — for models without the flag (same policy as `requires_modality`).
A declared suite the repo cannot currently run stays declared: `certify`
turns it into an exclusion, so the gap is stated on every certificate
rather than quietly disappearing with the test file.

- **DeepStack** (`"deepstack"` — Qwen3-VL family): **declared but NOT
  RUNNABLE; every certificate for such a model carries it as an
  exclusion** (`certify.DEEPSTACK_NOT_COVERED`). The vision tower's
  multiscale features are injected into the first LLM layers through a
  per-step side buffer OUTSIDE the paged KV, and the risky path is a hit
  boundary INSIDE an image span, where vLLM resumes prefill mid-span and
  must scatter the payload at the right offsets. TD.1–TD.4 produced such
  boundaries surgically (evict a stored request's tail chunks, replay) and
  compared the KV the resume re-stored against the KV the full prefill
  stored. That oracle had to be KV-level, because outputs are MEASURED
  blind here: fully disabling the injection on Qwen3-VL-2B changes no
  output byte on the synthetic probes, while per-chunk KV divergence
  separates cleanly (recompute noise rel-Frobenius 0.02–0.04 vs 0.55–0.70
  with the payload zeroed; measured 2026-08-21). Reading stored KV back
  required the in-process `LocalCPUBackend`; the MP cache server has no
  equivalent (its object listing covers L2 only, and its checksum API
  hashes GPU blocks, which a recompute never reproduces bit-exactly), so
  the suite was removed with the in-process path rather than replaced by a
  check known to be blind. Restoring it needs a server-side read-back or
  KV-distance API first. The earlier green result stands as a measurement,
  not as ongoing coverage: the payload's effect is baked into stored KV,
  so skipped prefixes need no side buffer; only the resumed span needs
  (and gets) reinjection.
- **Gemma 3/4**: bidirectional image attention vs chunk-boundary phases
  (strengthened T0.4); vLLM issue #40106 makes this a live concern.
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
the negative control), combines it with a benchmark parity result (MME for
image/video models, MMAU for audio), and writes `certificate_<model>.json`
recording the verdict, the tested tree, the certified scope (deployment
paths, modalities, backend), all measurements, and the known-not-covered
list. Exit codes: 0 `SUPPORTED`, 2 `PROVISIONAL` (suite green, parity not
provided), 1 `NOT_SUPPORTED`. A skipped or empty suite can never certify
(skips are counted as failure).

`schema_version` is **5** since the in-process path was dropped. Every
certificate at 4 or below was produced by the two-path suite: its `scope`
block claims a deployment the suite no longer drives, and its
`known_not_covered` list predates the in-process and deepstack
exclusions. Those models need re-certifying, not re-labelling — no
recorded certificate carries a claim this suite would issue today.

Three fields exist because a certificate that overstates itself is worse
than no certificate, and each began as a real defect in a published one:

- **`tested_tree`** — HEAD is read at launch *and* again at write time,
  with a dirty-tree check on both. `commit` names the tree under test only
  when `tested_tree.stable` is true; committing (or editing) mid-run makes
  it false and prints a warning. The first Qwen3-Omni certificate recorded
  a commit that was not the tree it had measured.
- **the audio exclusion** — emitted only for a model whose spec does not
  declare `audio`. It used to be unconditional, which made Qwen3-Omni's
  certificate list audio in `scope.modalities` and disclaim it in
  `known_not_covered`, in the same document.
- **`gate.pass2_hit_coverage`** — `null`, never `0.0`, when the report has
  no denominator for it (one recorded before the coverage fields existed,
  or one from the removed in-process path, which had no per-request lookup
  lengths); a literal `0.0` there read as "the cache achieved nothing" for
  runs whose raw hit ratio was 1.0. Fine-granularity runs gate on
  `raw_hit_ratio`, so no verdict ever depended on the bad number — only
  its readers did.

The scenario-shaped entries in `known_not_covered` are derived from
`isolated_scenarios(spec)`, the same predicate the pytest parametrization
uses, so a scenario that starts or stops running for a model cannot leave a
stale claim behind. Restating them by hand is how Gemma 4's certificate came
to omit two scenarios it had actually passed.

## Support levels

- ✅ **Supported**: certificate verdict `SUPPORTED` — synthetic suite green
  AND the parity gate passed, for the scope named in the certificate.
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
- `mme_max_flip_fraction` — per-model override of the parity gate's
  answer-flip budget (default 0.5%, calibrated on short-answer models).
  Running with vs. without the LMCache connector are two different — each
  fully deterministic — numeric regimes, and a long chain-of-thought
  amplifies the regime difference into ~1% of borderline answers flipping
  or repetition-looping past the decode budget. Measured for GLM
  (2026-08-20): flip counts and flipped-question sets reproduce exactly
  across runs, baseline reruns are byte-identical (0 self-flips), and
  per-question inspection shows no corruption signature; KV corruption is
  still caught by the replay, hit-ratio, and score-delta oracles.
- `mme_max_local_cpu_gb` — LMCache local-CPU capacity for the MME parity
  run (0 = the 40 GB default, which holds the full benchmark's KV for
  GQA-2 models at 28–36 KB/token). A wider-KV model (InternVL3.5-2B's
  Qwen3 backbone is GQA-8, 112 KB/token) overflows the default: the
  pass-2 replay revisits requests in store order, the LRU scan evicts
  every entry before its revisit, and the hit-ratio gate fails at ~0
  with zero flips (pure recompute, not corruption). Size it to hold the
  whole benchmark: questions × prompt tokens × KV bytes per token.

- `hybrid_block_tokens` — vLLM's unified block size for a Mamba/GDN hybrid
  (Qwen3.5-2B: 544; vLLM prints it at startup), 0 for every other model.
  Setting it chunks the cache server at that block size, gives each KV
  cache group its own cache objects, and becomes the granularity every
  hit-count tolerance is derived from (`harness.chunk`). The harness
  validates it against the live engine (a multiple of every paged group's
  block size), so a stale value fails loudly instead of making assertions
  trivially true. Must be set together with `hybrid_family`, which decides
  whether the align settings (`mamba_cache_mode=align`, vLLM prefix caching
  on, `max_num_batched_tokens` >= one block) are forced onto every engine;
  `ModelSpec.__post_init__` rejects half a statement. Because a cacheable unit is
  now hundreds of tokens — larger than a 196-token image span — the prompts
  are padded to whole blocks around each image (`HYBRID_PRE_PAD_BLOCKS` /
  `HYBRID_POST_PAD_BLOCKS` and the mid-pad in `conftest.py`); without the
  post-pad the block-granularity hit counts of two *different* images are
  identical and the suite's primary cross-image detector goes blind.
- `hybrid_object_groups` — cache objects stored per block, i.e. the number
  of KV cache groups the server keeps separate under
  `--separate-object-groups` (2: full-attention KV plus recurrent state, or
  full-attention plus sliding-window). The storage-conservation bounds
  multiply by it.
- `hf_overrides` — config repairs applied identically to every engine for
  the model. Gemma 4 needs it because transformers 5.15 folded the
  per-layer attention dims into `per_layer_config` and stopped exposing the
  flat `global_head_dim` / `num_global_key_value_heads` names vLLM still
  reads with a defaulting `getattr`, so the full-attention layers would be
  built at the sliding geometry. It must be identical across the test
  engine, the baseline and the parity engines: it changes the model's
  geometry, so a baseline without it is not a comparison.

The MME parity gate additionally enforces a baseline answer parse-rate
(`MIN_PARSE_RATIO`): if a model's yes/no verdict does not land inside the
decode budget, all three passes parse to '' and the flip/score comparisons
would pass vacuously — the parse-rate guard turns that into a hard FAIL
instead.
