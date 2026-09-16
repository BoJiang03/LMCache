# Landing the branches, and measuring what the diff cannot lose

2026-09-16, continuing from `3_the_control_that_told_us_which_share_was_ours.md`.

That record closed the measurement. This one is all mechanics: get the work onto
`dev`, get the PR onto the current upstream, squash it, and then answer Bo's
question "can you make it smaller" with numbers instead of an opinion.

Nothing measured here changes any claim in the previous three records.

## Branch state

`dev` was fast-forwarded from `c40b7807` to `f556df90`, the 12 commits carrying
the implementation experiment, the benchmark harness and the six records. Loss
free (`c40b7807` is an ancestor); `git branch -f dev c40b7807` undoes it.
`delta_token_ids` still points at the same commit, so both names name it.

The PR branch was rebased from `c40b7807` onto `upstream/dev` at `1b7dff2c`,
seven commits newer, and squashed to a single commit. Backup tag
`backup/pr-before-rebase-20260916` holds the old two-commit tip `5d96a232`.
Nothing pushed.

## The conflict git did not report

git reported exactly one conflict, an import block in
`modules/blend/store.py`. The rebase then produced a branch whose gRPC
transport could not start.

Upstream `#4953` added a gRPC request transport whose codec maps a
`msgspec.Struct` onto a protobuf message **by field name**
(`grpc_impl/proto_codec.py:325-331`). `IpcCacheServerKey.token_ids` in
`protos/common.proto` no longer had a field to bind to, so compiling the codec
raises `TypeError: IPCCacheServerKey.token_bytes does not match
lmcache.mp.IpcCacheServerKey.token_ids`. Three edits:

- `common.proto` field 4 `repeated int64 token_ids` becomes `bytes token_bytes`,
  reusing the number, which is consistent with a change that already declares a
  wire break.
- `_proto_gen/common_pb2.pyi` regenerated. `grpc_tools` is a declared build
  dependency (`requirements/proto.txt`) but is not in the venv, so the generator
  ran from a throwaway venv in the job scratch directory. The `_pb2.py` files
  are gitignored and rebuilt at install; only the `.pyi` is checked in.
- `test_grpc_transport.py` constructed a key with `token_ids=(1, 2, 3)`.

The lesson is narrow and worth keeping: **a rename that a structural codec
resolves by name is invisible to a three-way merge.** Both sides edited files
git merged cleanly; the break lives in the relationship between a Python field
and a `.proto` field, which no textual merge inspects. Grep for the old name
across schema files, not just Python, after any rebase that renames a wire field.

## The reduction, and what it cost to find out

Bo asked twice to make the diff smaller. Start `+774/-203`, end `+715/-200`,
41 files either way.

What came out, in decreasing order of how defensible the cut was:

1. **PR2 infrastructure that had leaked in.** `maybe_submit_lookup_request` had
   been changed from `-> None` to `-> int` returning the token count, with a
   seven-line docstring about taking the maximum over calls. Zero callers use
   it; it exists for the delta PR. `LMCacheMPRequestTracker.packed_token_slice`
   and `_pack_through` likewise: the only caller was `packed_token_ids(0, n)`.
   Arbitrary-range slicing is a delta requirement. About 40 lines.
2. **Docstrings restating the PR description.** The `IPCCacheServerKey` wire
   break note, `LoadStoreOp.token_bytes`, the `token_codec` module header, two
   wire-compat test docstrings. Kept the sentence a reader cannot derive from
   the code, dropped the expansion.
3. **Gratuitous churn.** Three `tuple(...)` -> `list(...)` edits in tests;
   `from_token_ids` takes a `Sequence`, so those lines never needed to change.
4. **One duplicated test.** `TestSessionHashesMatchTokenListPath::
   test_whole_sequence` asserts what `test_session.py::
   test_session_matches_standalone_hasher` already asserts once `set_tokens`
   takes bytes.

## The cut that was wrong

I deleted `IPCCacheServerKey.num_tokens` as unused. It has five callers in
`modules/blend/lookup.py` and `modules/lookup.py`. The grep that "proved" it
unused was `grep ... | head -8`, and the first eight hits were all unrelated
`args.num_tokens` / `config.num_tokens`. Six tests caught it; it was restored.

`head` on a search whose whole purpose is to establish absence is not a filter,
it is a way to manufacture the answer you were hoping for. Absence needs the
full result set, or a count.

## Three refactors that were measured and rejected

This is the part worth keeping. Each looked like the big lever and each is
quantified, so none of them needs to be reconsidered from scratch.

**Factor the shared rolling-hash loop.** `compute_chunk_hashes` and
`compute_packed_chunk_hashes` run the same loop (hash from token 0 because each
chunk's hash depends on all previous, return only chunks from `start`, drop a
trailing partial chunk). Extracting it gives a helper needing its own
Args/Returns block, about 18 doc lines plus 10 code, and two wrappers at 7 each:
**48 added against the 35 the duplicate costs today**, plus a lambda per chunk.
The shared abstraction is more expensive than the duplication it removes.

**Split the PR along the process boundary.** The change has a clean seam: the
`LoadStoreOp` half (scheduler to worker, through vLLM's shm broadcast) and the
`IPCCacheServerKey` half (worker to server, over ZMQ). Measured, the
`LoadStoreOp` half is **89 insertions across 8 files**, about 14% of the diff.
There is no small PR hiding in there, only a second review cycle and a worker
that would have to pack what the scheduler already could have.

**Merge `token_codec.py` into `token_hasher.py`.** Saves a module header, about
12 lines. But `custom_types.py` needs only the codec, and `token_hasher`
imports numpy, blake3 and vLLM's `kv_cache_utils`. The separate module is what
keeps the key type cheap to import.

## What the remaining 715 lines are

Code 475, docstrings and comments 154, blank 86.

The three largest code blocks are `test_packed_token_ids.py` at ~105,
`token_hasher.py` at ~60 and `token_codec.py` at ~45. About 100 more lines are
mechanical renames across 17 test files, which cannot be avoided: every
production caller of `Session.set_tokens` passes `key.token_bytes`
(`blend/store.py:99`, `blend/lookup.py:376`, `engine_context.py:299`,
`modules/lookup.py:264`), so the parameter is `bytes` and every test that sets
tokens has to pack.

The 18 peripheral construction sites cost one or two lines each because they
route through `IPCCacheServerKey.from_token_ids`, which is cheaper than an
import plus a keyword change and also drops a redundant `tuple()` copy at each.

## Two things found but not changed

Both unpack a whole sequence to use a slice of it, and both are the shape the
code already had (`list(key.token_ids)[a:b]`), so neither is a regression this
change introduced:

- `modules/blend/store.py:106`, CacheBlend path, off by default.
- `lmcache_driven_transfer.py::_publish_token_bindings`, gated on
  `event_bus.has_subscribers(MP_TOKENS)`, so no subscriber means no cost.

Both become O(chunk) by slicing the buffer before unpacking. That adds lines
rather than removing them and neither sits on a path these runs measured, so it
belongs in a follow-up.

One production call site did get fixed, because it was a wart this change
introduced rather than inherited: `blend/lookup.py:666` read
`compute_chunk_hashes(unpack_token_ids(key.token_bytes))`, unpacking 200k ints
so the hasher could pack them again, 300 lines below a call that already used
`compute_packed_chunk_hashes`. **That branch (`segmented` with
`retained_chunks`) has no test coverage anywhere in `tests/`.** What supports
the edit is the equivalence itself, which is pinned for whole sequences,
sub-ranges and a non-blake3 algorithm in `test_packed_token_ids.py`, and the
fact that both methods return `list[bytes]`. The property is tested; this call
site is not.

Ten `token_codec` imports were also inserted out of alphabetical order. ruff
does not catch it: this repo enables E, F, B and SLF only, no isort rules.

## State

- `dev` and `delta_token_ids` at `f556df90`. Not pushed.
- `delta_token_ids_pr` at `8a3a338c`, one commit on `upstream/dev` `1b7dff2c`,
  `+715/-200` over 41 files. Not pushed.
- Tests: `tests/v1/multiprocess` 806 passed, 1 skipped, 0 failed; the touched
  files outside it 232 passed, 0 failed. ruff check and format clean. mypy not
  run, it is not installed in the venv.
- PR description at `$CLAUDE_JOB_DIR/tmp/PR1_description.md`, 208 lines, not in
  the repo. Updated this session for the single commit, the new test counts and
  the gRPC proto change.
- Still open: the benchmark harness
  (`benchmarks/microbenchmark/packed_token_ids_benchmark.py`, `benchmarks/e2e/`)
  is on `dev` but not on the PR branch, so the performance numbers in the commit
  message are not reproducible from the PR alone. Bo has not ruled on moving it.
