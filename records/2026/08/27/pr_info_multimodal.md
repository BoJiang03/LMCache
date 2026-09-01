# PR info: multimodal cache keys

Branch `multi_modal_pr`. Everything below the rule is the paste-ready body,
one paragraph per line because GitHub renders a single newline as a line
break. Do not re-wrap it.

Title: `[Bugfix][vLLM] Different images can share the same multimodal cache keys`

---

**What this PR does / why we need it**:

### Problem

LMCache derives cache keys from token ids alone, and vLLM emits the same placeholder token id for every multimodal item, so the connectors substitute an item's placeholder span with values derived from its `mm_hash` before hashing. `hex_hash_to_int16()` reduced that hash to 16 bits and wrote the one value across the whole span. The identifier is a blake3 hexdigest by default, so this keeps only its last four hex characters: a colliding identifier is constructed, not searched for. Two images that collide share every cache key their spans touch, and the second is served the first one's KV. Issue #3301, which the stale bot closed in August without a fix.

### Fix

`mm_hash_to_token_values()` gives every placeholder position its own 31-bit value, expanded from the full identifier with SHA-256 in counter mode. Widening the reduction alone would not have been enough: one value repeated across a span carries that value's entropy however long the span is, so a chunk overlapping k placeholder tokens now carries 31*k bits instead of 31. Values are position-dependent, mirroring the `(mm_hash, offset)` extra keys vLLM already appends to its own block hashes, and prefixes stay stable so a span the save path truncates hashes consistently with the full one. 31 bits keeps them positive in a signed int32, the narrowest type token ids pass through downstream. Collision resistance is now bounded by LMCache's 64-bit chunk hash, the same bound text keying has.

No interface change, both substitution paths are fixed, 398 lines including unit tests and a design doc.

### Validation

[`repro/mm_hash_collision_repro.py`](https://github.com/BoJiang03/LMCache/blob/multi_modal_repro/repro/mm_hash_collision_repro.py) runs the MP path end to end: a real cache server subprocess, a vLLM engine on `LMCacheMPConnector`, 800 distinct 448x448 images that differ only in a corner pattern, six dominant colours between them. It records the identifiers the connector actually sees, finds a pair that collides under the old 16-bit reduction, and asks for each image's colour.

On dev at `09bc14c0` it exits 1: images 280 and 513 have identifiers `2a7c4bc5...0590` and `cf0b1a39...0590`, both truncating to `0x0590`, and the yellow one is answered `Purple`. With this change it exits 0 on the same pair, each image answered from its own KV.

The full acceptance suite is about 9500 lines and stays on [`multi_modal_repro`](https://github.com/BoJiang03/LMCache/tree/multi_modal_repro). This PR carries only the unit tests for the key derivation and the connector keying.

**Special notes for your reviewers**:

Multimodal entries cached before this change miss afterwards rather than collide.

`apply_mm_hashes_to_token_ids` must be given the full prompt or a prefix of it, since placeholder offsets are absolute and a suffix would substitute the wrong positions with no error raised. All five call sites satisfy this and the docstring now says so.

A chunk overlapping only one or two placeholder tokens, at a span boundary, carries 31 or 62 bits rather than 64, and damage there would be confined to that one chunk since its neighbour overlaps many more and diverges.

Routing `mm_hash` through `TokenDatabase._hash_tokens(extra_keys=...)` is the vLLM-aligned design and stays a TODO in the design doc. It needs the lookup protocol, the MP connector metadata and the SDK to carry per-chunk extra keys.

### Unrelated bugs found while validating

Neither is caused by this change and neither one's code is in this PR. Each is one commit on its own branch.

**Torch-fallback `lmcache_memcpy_async` is not stream-ordered.** Branch [`fix_memcpy_stream_order`](https://github.com/BoJiang03/LMCache/tree/fix_memcpy_stream_order). Silent data corruption. The pointer-mode fallback issued a synchronous `cudaMemcpy`, which runs on the legacy default stream and is unordered against PyTorch's non-blocking streams, so the copy of a staging slot could read the slot's previous contents while the gather kernel was still queued, storing another chunk's KV under this chunk's key. The retrieve path has the mirror hazard. Only affects deployments where the compiled `lmcache.cuda_ops` extension fails to load. Fixed by mirroring the native path: `cudaMemcpyAsync` on the current torch stream, split at `cudaHostRegister` boundaries.

**An expired read lock is reported as a write collision.** Branch [`fix_l1_read_lock_reason`](https://github.com/BoJiang03/LMCache/tree/fix_l1_read_lock_reason). Diagnostics only. `read_prefetched_results` labelled `unsafe_read`'s `KEY_NOT_READABLE` as `reason="write_locked"`, but `unsafe_read` never consults the write lock. The real condition is an expired read lock: `reserve_read` stamps expiry at lookup time and only `lock()` refreshes it, so a lookup-to-transfer gap longer than `read_ttl_seconds` silently unlocks the entry. That label is the only signal separating the two causes and it pointed at a writer that did not exist. Renamed to `read_lock_expired` on the l1_retrieve path, plus one aggregate log line naming the count and the configured TTL.

**If applicable**:

- [x] this PR contains user facing changes - docs added
- [x] this PR contains unit tests
