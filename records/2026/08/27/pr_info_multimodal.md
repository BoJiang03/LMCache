# PR info: full-entropy multimodal cache keys

Branch `multi_modal_pr`, base `09bc14c0`. Draft body below.

---

## [Bugfix][vLLM] Full-entropy mm_hash substitution for multimodal cache keys

### Problem

The multimodal placeholder substitution collapsed each image's identifier to
16 bits (`hex_hash_to_int16`) and filled the entire placeholder span with that
one repeated value. Two different images therefore shared every KV cache key
with ~50% probability at around 300 distinct same-shape images (birthday bound
over 65536 buckets), and the second image was served the first image's KV.

Reproduced end to end on Qwen2.5-VL-3B: 6 of 800 distinct images cross-hit,
with visible answer flips. Issue #3301.

### Fix

`mm_hash_to_token_values()` derives a distinct 31-bit value per placeholder
position from the full identifier (SHA-256 in counter mode). A chunk
overlapping k placeholder tokens now carries 31*k bits of item identity, the
offset within the item is encoded per position (mirroring vLLM's
`(mm_hash, offset)` block-hash extra keys), and prefixes stay stable under
save-path truncation. Values stay below 2^31 for signed-int32 safety.

No interface change. Both substitution paths are fixed at once, and every MP
connector operation is keyed on the MM-adjusted token ids rather than the raw
prompt ids.

### Changes

384 lines: key derivation in `lmcache/integration/vllm/utils.py`, connector
wiring for the three supported vLLM versions, 181 lines of unit tests, and a
design doc at `docs/design/integration/vllm/multimodal_cache_keying.md`.

### Validation

15 models, each run as a two-pass MME (or MMAU) parity benchmark on the MP
deployment path: pass 1 populates the cache, pass 2 reads it back. Only the
answer flip counts are reported here; whether they are acceptable is a
judgment call, so the raw numbers are given in both directions.

| Model | Benchmark | Hit rate | p1 vs plain vLLM | p2 vs p1 | wrong / right | yes->no / no->yes |
|---|---|---|---|---|---|---|
| Qwen/Qwen2.5-VL-3B-Instruct | MME 2374 | 0.984 | 0 | 24 | 13 / 11 | 11 / 13 |
| zai-org/GLM-4.6V-Flash | MME 2374 | 0.976 | 0 | 18 | 11 / 7 | 11 / 7 |
| Qwen/Qwen2-VL-2B-Instruct | MME 2374 | 0.984 | 0 | 18 | 10 / 8 | 10 / 8 |
| moonshotai/Kimi-VL-A3B-Instruct | MME 2374 | 0.989 | 0 | 13 | 8 / 5 | 7 / 6 |
| deepseek-ai/DeepSeek-OCR | MME 2374 | 0.989 | 0 | 11 | 8 / 3 | 5 / 6 |
| microsoft/Phi-4-multimodal-instruct | MME 2374 | 0.990 | 0 | 10 | 2 / 8 | 4 / 6 |
| mistralai/Mistral-Small-3.1-24B-Instruct-2503 | MME 2374 | 0.996 | 5 | 8 | 4 / 4 | 5 / 3 |
| Qwen/Qwen3-VL-2B-Instruct | MME 2374 | 0.983 | 0 | 8 | 2 / 6 | 3 / 5 |
| allenai/Molmo2-4B | MME 2374 | 0.992 | 0 | 7 | 2 / 5 | 5 / 2 |
| OpenGVLab/InternVL3_5-2B-HF | MME 2374 | 0.983 | 1 | 6 | 4 / 2 | 1 / 5 |
| Qwen/Qwen3.8-27B | MME 2374 | 1.059* | 0 | 4 | 4 / 0 | 4 / 0 |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | MMAU 1000 | 0.975 | 0 | 2 | 1 / 1 | n/a (4-way) |
| Qwen/Qwen3.6-27B | MME 2374 | 1.059* | 0 | 2 | 1 / 1 | 1 / 1 |
| google/gemma-4-E4B-it | MME 2374 | 1.008* | 0 | 1 | 0 / 1 | 1 / 0 |
| google/gemma-3-4b-it | MME 2374 | 0.965 | 0 | 0 | 0 / 0 | 0 / 0 |

`*` marks the three hybrid-attention models plus Gemma-4-E4B, where the gate
reads cache coverage rather than the raw lookup hit ratio; coverage can exceed
1.0 because vLLM's own prefix cache serves part of the request.

Reading the columns:

- `p1 vs plain vLLM` is a control. Pass 1 only writes, so nothing LMCache does
  can feed back into the computation, and the column should be 0. Note that 0
  flips is not byte-identical output: the baseline runs in a separate process,
  so request batching differs and raw text differs in tens of places per model
  without moving any verdict. Mistral's 5 and InternVL's 1 are that same
  run-to-run nondeterminism crossing a parse boundary, not a cache effect.
- `p2 vs p1` is the number that matters: same engine, same questions, answers
  computed from loaded KV instead of recomputed KV.
- `wrong / right` splits pass-2 flips by whether the new answer is wrong or
  right against MME ground truth. `yes->no / no->yes` splits the same flips by
  direction on MME's binary questions.

Across the 14 MME models: 130 flips out of 33236 questions (0.39%), worst
model 24 of 2374 (1.0%), 69 wrong against 61 right, 68 yes->no against
62 no->yes. Both splits are close to even, which is what an unbiased
perturbation looks like rather than a systematic corruption. Qwen3.8-27B is
the one row that is one-sided in both (4 flips, all yes->no, all wrong); at
that count it is not separable from chance.

Two models are not in the table. Qwen3.5-2B is still on an older certificate
schema and has no direction data. The other 15 all certify SUPPORTED at
schema 8 against this branch.

### Tests

The acceptance harness used to produce the table above is large (about 9500
lines: model catalog, MME/MMAU runners, parity benchmark, certificate
generator) and is deliberately kept out of this PR. It lives on the
`multi_modal_repro` branch. This PR carries only the unit tests for the key
derivation and the connector keying.

### Defects found while validating this change, fixed separately

Three unrelated LMCache bugs surfaced while running the parity benchmark. None
of them are caused by this change and none of their code is in this PR. Each is
a single commit on its own branch, based on `09bc14c0`.

**1. Torch-fallback `lmcache_memcpy_async` is not stream-ordered.**
Branch `fix_memcpy_stream_order`. Silent data corruption. The pointer-mode
fallback issued a synchronous `cudaMemcpy`, which runs on the legacy default
stream and is unordered against PyTorch's non-blocking streams. The MP server
enqueues the paged-KV gather kernel on the cache context's stream and then
copies the staging slot to the pinned host object; when the gather was still
queued, the copy read the slot's previous contents and committed another
chunk's KV under this chunk's key. The retrieve path has the mirror hazard.
Only affects deployments where the compiled `lmcache.cuda_ops` extension fails
to load; the C++ path already uses `cudaMemcpyAsync` on the current stream.
Fixed by mirroring the native implementation: `cudaMemcpyAsync` on the current
torch stream, split at `cudaHostRegister` boundaries, falling back to draining
the stream if the async symbol is missing.

**2. Failed MP retrieves are reported as successes.**
Branch `fix_mp_load_error`. Silent data corruption. The MP worker adapter's
drain loop in `get_finished` logged a failed retrieve and moved on, but still
listed the request in `finished_retrieves` without flagging its blocks in
`error_block_ids`. vLLM saw a clean completed load, kept the tokens it had
already counted as computed, never recomputed them, and the model read
whatever those blocks happened to hold. The server side already reports the
failure; only the last hop discarded it, and the unhealthy-drain branch of the
same method had always handled it correctly. Fixed by extracting the drain into
`_collect_finished_retrieves()` so both callers share it, and flagging the
block ids there. Failed retrieves are still reported as finished, because an
unreported async load hangs the request in `WAITING_FOR_REMOTE_KVS` instead of
recomputing.

**3. An expired read lock is reported as a write collision.**
Branch `fix_l1_read_lock_reason`. Misleading diagnostics only, no data effect.
`read_prefetched_results` labelled `unsafe_read`'s `KEY_NOT_READABLE` as
`reason="write_locked"`, but `unsafe_read` never consults the write lock. The
real condition is an expired read lock: `reserve_read` stamps expiry at lookup
time and only `lock()` refreshes it, so any lookup-to-transfer gap longer than
`read_ttl_seconds` silently unlocks the entry. The label is the only signal
separating the two causes and it pointed at a concurrent writer that did not
exist, which cost real debugging time on a 7699-failure incident. Renamed to
`read_lock_expired` on the l1_retrieve path only, and added one aggregate log
line naming the count and the configured TTL.
