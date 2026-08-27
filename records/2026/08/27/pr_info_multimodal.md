# PR info: multimodal cache keys

Branch `multi_modal_pr`. Draft body below.

---

## [Bugfix][vLLM] Different images can share the same multimodal cache keys

### Problem

The multimodal placeholder substitution reduced each image's identifier to
16 bits and repeated that one value across the whole placeholder span. Once a
run has seen a few hundred same-shape images, two different images share every
cache key with roughly even odds, and the second one is served the first one's
KV. Issue #3301.

Reproduced on Qwen2.5-VL-3B: 6 of 800 distinct images cross-hit, with wrong
answers.

### Fix

`mm_hash_to_token_values()` gives every placeholder position its own 31-bit
value, derived from the whole identifier with SHA-256 in counter mode. A chunk
covering k placeholder tokens then carries 31*k bits of image identity, and the
position inside the item is part of the value, which is what vLLM already does
with its `(mm_hash, offset)` extra keys. Prefixes stay stable when the save path
truncates.

No interface change, and both substitution paths are fixed. 384 lines: the key
derivation in `utils.py`, connector wiring for three vLLM versions, unit tests,
and a design doc.

### Validation

15 models, each run twice on the MP path: the first pass fills the cache, the
second reads it back. The table counts answers that changed between the two
passes.

| Model | Benchmark | Hit rate | pass 1 vs plain vLLM | pass 2 vs pass 1 | became wrong / right | yes->no / no->yes |
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
| Qwen/Qwen3.8-27B | MME 2374 | 1.059 | 0 | 4 | 4 / 0 | 4 / 0 |
| Qwen/Qwen3-Omni-30B-A3B-Instruct | MMAU 1000 | 0.975 | 0 | 2 | 1 / 1 | n/a |
| Qwen/Qwen3.6-27B | MME 2374 | 1.059 | 0 | 2 | 1 / 1 | 1 / 1 |
| google/gemma-4-E4B-it | MME 2374 | 1.008 | 0 | 1 | 0 / 1 | 1 / 0 |
| google/gemma-3-4b-it | MME 2374 | 0.965 | 0 | 0 | 0 / 0 | 0 / 0 |

A hit rate above 1.0 means vLLM's own prefix cache served part of the request.

`pass 1 vs plain vLLM` is a control: the first pass only writes, so nothing
LMCache does can change what is computed, and the column should be 0. It is not
a byte-identical check. The baseline runs in its own process, so raw text
differs in a few dozen places per model without moving any answer; Mistral's 5
and InternVL's 1 are that same nondeterminism crossing a parse boundary.

Totals across the 14 MME models: 130 changed answers out of 33236 questions
(0.39%), worst model 24 of 2374 (1.0%), 69 became wrong against 61 became
right, 68 went yes->no against 62 no->yes. Both splits are close to even, which
is what an unbiased perturbation looks like rather than a systematic
corruption.

The harness that produced this table is about 9500 lines and is deliberately
kept out of this PR. It is on the `multi_modal_repro` branch. This PR carries
only the unit tests for the key derivation and the connector keying.

### Unrelated bugs found while validating

None of these are caused by this change and none of their code is in this PR.
Each is one commit on its own branch.

**Torch-fallback `lmcache_memcpy_async` is not stream-ordered.** Branch
`fix_memcpy_stream_order`. Silent data corruption. The pointer-mode fallback
issued a synchronous `cudaMemcpy`, which runs on the legacy default stream and
is unordered against PyTorch's non-blocking streams. The MP server queues the
paged-KV gather kernel on the cache context's stream and then copies the
staging slot to the pinned host object; when the gather was still queued, the
copy read the slot's previous contents and stored another chunk's KV under this
chunk's key. The retrieve path has the mirror hazard. Only affects deployments
where the compiled `lmcache.cuda_ops` extension fails to load. Fixed by
mirroring the native path: `cudaMemcpyAsync` on the current torch stream, split
at `cudaHostRegister` boundaries.

**An expired read lock is reported as a write collision.** Branch
`fix_l1_read_lock_reason`. Diagnostics only, no data effect.
`read_prefetched_results` labelled `unsafe_read`'s `KEY_NOT_READABLE` as
`reason="write_locked"`, but `unsafe_read` never consults the write lock. The
real condition is an expired read lock: `reserve_read` stamps expiry at lookup
time and only `lock()` refreshes it, so any lookup-to-transfer gap longer than
`read_ttl_seconds` silently unlocks the entry. The label is the only signal
separating the two causes and it pointed at a writer that did not exist, which
cost real debugging time on a 7699-failure incident. Renamed to
`read_lock_expired` on the l1_retrieve path, plus one aggregate log line naming
the count and the configured TTL.

**Failed MP retrieves reported as successes.** Branch `fix_mp_load_error`. The
data-loss part of this was fixed independently on dev by #4709 while this work
was in progress, so what is left on the branch is the deduplication of the two
drain loops that #4709 left duplicated, plus a warning at registration when the
model has more than one KV cache group, where vLLM cannot recompute a trimmed
prefix and aborts the engine on a load error instead.
