"""Chunk-hash equivalence: whole-prefix stores vs delta stores.

The fix stops STORE from shipping the request's growing prompt prefix and
sends only ``[start, end)`` with ``token_offset=start``. The server's session
must derive byte-identical chunk hashes either way, or every stored object
lands under a different key and the cache silently stops hitting.
"""
import random, sys
from lmcache.v1.multiprocess.session import Session
from lmcache.v1.multiprocess.token_hasher import TokenHasher
from lmcache.v1.multiprocess.custom_types import SessionTokenGapError

CHUNK = 64
rng = random.Random(20260903)
hasher = TokenHasher(chunk_size=CHUNK, hash_algorithm="blake3")
N_CHUNKS = 12
full = [rng.randrange(100000) for _ in range(CHUNK * N_CHUNKS + 17)]  # ragged tail

def store_ranges(first_start):
    """[start, end) pairs the scheduler would emit, chunk-aligned, contiguous."""
    out, s = [], first_start
    while s + CHUNK <= CHUNK * N_CHUNKS:
        n = rng.choice([1, 1, 2, 3])
        e = min(s + n * CHUNK, CHUNK * N_CHUNKS)
        out.append((s, e)); s = e
    return out

fails = 0
def check(name, a, b):
    global fails
    ok = a == b
    print(f"{'PASS' if ok else 'FAIL'}  {name}  ({len(a)} chunk hashes)")
    if not ok:
        fails += 1
        for i, (x, y) in enumerate(zip(a, b)):
            if x != y:
                print(f"   first divergence at chunk {i}: {x!r} != {y!r}"); break

# ---- A: stores only, no preceding lookup -------------------------------
for label, first_start in (("no lookup, from 0", 0),):
    ranges = store_ranges(first_start)
    old_s = Session(request_id="old", hasher=hasher)
    new_s = Session(request_id="new", hasher=hasher)
    old_h, new_h = [], []
    for s, e in ranges:
        old_s.set_tokens(list(full[:e]))                 # what the code did before
        old_h += list(old_s.get_hashes(s, e))
        new_s.extend_tokens(list(full[s:e]), s)          # what it does now
        new_h += list(new_s.get_hashes(s, e))
    check(f"stores tile the prefix ({label})", old_h, new_h)

# ---- B: LOOKUP first (full prompt), then deltas from the hit point ------
for hit_chunks in (0, 1, 5):
    ranges = store_ranges(hit_chunks * CHUNK)
    old_s = Session(request_id="old", hasher=hasher)
    new_s = Session(request_id="new", hasher=hasher)
    # LOOKUP: whole prompt, offset 0, both paths identical
    old_s.set_tokens(list(full)); old_lookup = list(old_s.get_hashes(0))
    new_s.extend_tokens(list(full), 0); new_lookup = list(new_s.get_hashes(0))
    check(f"lookup hashes (hit_chunks={hit_chunks})", old_lookup, new_lookup)
    old_h, new_h = [], []
    for s, e in ranges:
        old_s.set_tokens(list(full[:e]))
        old_h += list(old_s.get_hashes(s, e))
        new_s.extend_tokens(list(full[s:e]), s)
        new_h += list(new_s.get_hashes(s, e))
    check(f"stores after lookup (hit_chunks={hit_chunks})", old_h, new_h)
    # the deltas must not have corrupted the chain: re-hashing the whole
    # prompt on a fresh session must reproduce what the spliced one holds
    ref = Session(request_id="ref", hasher=hasher)
    ref.set_tokens(list(full))
    check(f"chain intact vs fresh full-prompt session (hit_chunks={hit_chunks})",
          list(ref.get_hashes(0)), list(new_s.chunk_hashes))

# ---- C: resend of the same range is idempotent -------------------------
s2 = Session(request_id="idem", hasher=hasher)
s2.extend_tokens(list(full[:2 * CHUNK]), 0)
h1 = list(s2.get_hashes(0, 2 * CHUNK))
s2.extend_tokens(list(full[CHUNK:2 * CHUNK]), CHUNK)   # duplicate delta
h2 = list(s2.get_hashes(CHUNK, 2 * CHUNK))
check("duplicate delta is idempotent", h1[1:], h2)

# ---- D: a gap raises, and leaves the session untouched ------------------
s3 = Session(request_id="gap", hasher=hasher)
s3.extend_tokens(list(full[:CHUNK]), 0)
before = list(s3.token_ids)
try:
    s3.extend_tokens(list(full[3 * CHUNK:4 * CHUNK]), 3 * CHUNK)
    print("FAIL  gap did not raise"); fails += 1
except SessionTokenGapError as e:
    print(f"PASS  gap raises SessionTokenGapError")
    if list(s3.token_ids) != before:
        print("FAIL  session mutated despite the gap"); fails += 1
    else:
        print("PASS  session left untouched")

# ---- E: 8 TP ranks submit the same ranges to one shared session ---------
# This is what the first GPU run actually hit: every rank sends the same
# [start, end) deltas asynchronously against one session keyed by request_id.
# A straggler's low-offset delta must not truncate the sequence out from under
# a rank that is further along.
print()
def interleave(n_ranks, ranges, seed):
    """Per-rank order preserved, ranks interleaved arbitrarily."""
    r = random.Random(seed)
    pending = {k: list(ranges) for k in range(n_ranks)}
    out = []
    while any(pending.values()):
        k = r.choice([k for k, v in pending.items() if v])
        out.append((k, pending[k].pop(0)))
    return out

ref = Session(request_id="ref2", hasher=hasher)
ref.set_tokens(list(full))
ref_hashes = list(ref.get_hashes(0))

for seed in (1, 2, 3, 7, 11):
    ranges = [(i * CHUNK, (i + 1) * CHUNK) for i in range(N_CHUNKS)]
    sess = Session(request_id="tp8", hasher=hasher)
    # LOOKUP lands first with the whole prompt, as it does in production.
    sess.set_tokens(list(full))
    gaps, wrong = 0, 0
    for _rank, (s, e) in interleave(8, ranges, seed):
        try:
            sess.extend_tokens(list(full[s:e]), s)
            got = list(sess.get_hashes(s, e))
        except SessionTokenGapError:
            gaps += 1
            continue
        if got != ref_hashes[s // CHUNK : e // CHUNK]:
            wrong += 1
    ok = gaps == 0 and wrong == 0
    print(f"{'PASS' if ok else 'FAIL'}  8-rank interleave seed={seed}: "
          f"gaps={gaps} wrong_hashes={wrong}")
    if not ok:
        fails += 1

# and again with no preceding lookup (stores alone must still tile)
for seed in (1, 5):
    ranges = [(i * CHUNK, (i + 1) * CHUNK) for i in range(N_CHUNKS)]
    sess = Session(request_id="tp8b", hasher=hasher)
    gaps, wrong = 0, 0
    for _rank, (s, e) in interleave(8, ranges, seed):
        try:
            sess.extend_tokens(list(full[s:e]), s)
            got = list(sess.get_hashes(s, e))
        except SessionTokenGapError:
            gaps += 1
            continue
        if got != ref_hashes[s // CHUNK : e // CHUNK]:
            wrong += 1
    ok = gaps == 0 and wrong == 0
    print(f"{'PASS' if ok else 'FAIL'}  8-rank interleave, no lookup, seed={seed}: "
          f"gaps={gaps} wrong_hashes={wrong}")
    if not ok:
        fails += 1

print("\n" + ("ALL PASS" if fails == 0 else f"{fails} FAILURE(S)"))
sys.exit(1 if fails else 0)
