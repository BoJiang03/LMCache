"""Scratch check of _PendingOperations invariants after the refcount removal.

Not part of the PR. Drives the queue through shared-block admissions,
evictions, emissions and drops, and asserts the reverse index and the
counter ledger stay exact.
"""
import sys
sys.path.insert(0, "/home/bo/LMCache-worktrees/lazy_offloading_policy_pr")
from types import SimpleNamespace

from lmcache.integration.vllm.lazy_offload_policy.base import DrainSignals
from lmcache.integration.vllm.lazy_offload_policy.eviction_aware import (
    EvictionAwareStoreQueue,
    LazyOffloadPolicyConfig,
)


class FakePool:
    def __init__(self, n):
        self.hashes = {b: f"h{b}" for b in range(n)}
        self.free = set(range(n))
        self.order = list(range(n))

    def free_queue_block_ids(self):
        for b in self.order:
            if b in self.free:
                yield b

    def is_free(self, b):
        return b in self.free

    def block_hash(self, b):
        return self.hashes.get(b)


def meta(rid, blocks, start, end):
    return SimpleNamespace(
        request_id=rid,
        op=SimpleNamespace(flat_block_ids=list(blocks), start=start, end=end),
    )


def index_matches(q):
    """The reverse index must equal what the pending lists imply."""
    ops = q._pending_ops
    expected = {}
    for rid, oplist in ops._by_request.items():
        for op in oplist:
            for b in op.block_hashes:
                expected.setdefault(b, set()).add(rid)
    assert ops._requests_by_block == expected, (
        f"index drift\n  got      {ops._requests_by_block}\n  expected {expected}"
    )


def ledger_closes(q):
    c = q.stats()
    lhs = c.admitted
    rhs = (
        q.num_pending_ops()
        + c.emitted
        + c.dropped_evicted
        + c.dropped_on_request_drop
        + c.dropped_failed_store
        + c.dropped_id_reuse
    )
    assert lhs == rhs, f"ledger {lhs} != {rhs} ({c})"


def signals(alloc=0, nxt=0, ids=(), blocked=()):
    return DrainSignals(alloc, nxt, set(ids), set(), set(blocked))


def hashes(pool, blocks):
    return {b: pool.hashes[b] for b in blocks}


# 1. shared blocks within one request (chunk boundary), partial emission
pool = FakePool(40)
q = EvictionAwareStoreQueue(LazyOffloadPolicyConfig(horizon_steps=2.0), pool)
q.add(meta("A", [0, 1], 0, 512), hashes(pool, [0, 1]), 0)
q.add(meta("A", [1, 2], 512, 1024), hashes(pool, [1, 2]), 0)   # shares block 1
q.add(meta("B", [1, 3], 0, 512), hashes(pool, [1, 3]), 0)      # shares block 1 too
index_matches(q); ledger_closes(q)
assert q._pending_ops._requests_by_block[1] == {"A", "B"}, "shared block lost a request"

# idle step: nothing due
r = q.drain(signals())
assert not r.items, r
index_matches(q); ledger_closes(q)

# a drop of A alone must leave block 1 indexed for B (the shared-block case
# the removed refcount used to guard)
assert q.drop_request("A") == 2
index_matches(q); ledger_closes(q)
assert q._pending_ops._requests_by_block[1] == {"B"}, q._pending_ops._requests_by_block

# pressure: block 1 is in the window, so B is due; prefix closure emits it
r = q.drain(signals(alloc=4, nxt=4, ids=[0]))
assert [i.request_id for i in r.items] == ["B"], r.items
index_matches(q); ledger_closes(q)
assert q._pending_ops._requests_by_block == {}, q._pending_ops._requests_by_block

# 2. an evicted block drops the covering op and every later op of its request
q.add(meta("C", [10, 11], 0, 512), hashes(pool, [10, 11]), 0)
q.add(meta("C", [11, 12], 512, 1024), hashes(pool, [11, 12]), 0)
index_matches(q); ledger_closes(q)
pool.hashes[10] = "recycled"
r = q.drain(signals(alloc=4, nxt=4, ids=[10]))
assert q.stats().dropped_evicted == 2, q.stats()
index_matches(q); ledger_closes(q)
assert q.num_pending_ops() == 0, q.num_pending_ops()
assert q._pending_ops._requests_by_block == {}, q._pending_ops._requests_by_block

# 3. drop_request / discard_for_reuse / mark_store_failed clear the index
pool2 = FakePool(40)
q2 = EvictionAwareStoreQueue(LazyOffloadPolicyConfig(), pool2)
for rid in ("X", "Y"):
    q2.add(meta(rid, [5, 6], 0, 512), hashes(pool2, [5, 6]), 0)
    q2.add(meta(rid, [6, 7], 512, 1024), hashes(pool2, [6, 7]), 0)
index_matches(q2); ledger_closes(q2)
assert q2.drop_request("X") == 2
index_matches(q2); ledger_closes(q2)
assert q2._pending_ops._requests_by_block[6] == {"Y"}
assert q2.mark_store_failed("Y") == 2
index_matches(q2); ledger_closes(q2)
assert q2._pending_ops._requests_by_block == {}

# 4. admission order survives a request emptying and coming back
q3 = EvictionAwareStoreQueue(LazyOffloadPolicyConfig(), FakePool(40))
for rid in ("r1", "r2", "r3"):
    q3.add(meta(rid, [1], 0, 512), {1: "h1"}, 0)
assert q3._pending_ops.admission_order() == {"r1": 0, "r2": 1, "r3": 2}
q3.drop_request("r1")
q3.add(meta("r1", [1], 0, 512), {1: "h1"}, 0)
assert q3._pending_ops.admission_order() == {"r2": 0, "r3": 1, "r1": 2}, \
    q3._pending_ops.admission_order()

# 5. deferral deadline still releases without any window pressure
q4 = EvictionAwareStoreQueue(
    LazyOffloadPolicyConfig(max_deferral_seconds=0.001), FakePool(40)
)
q4.add(meta("D", [9], 0, 512), {9: "h9"}, 0)
import time as _t
_t.sleep(0.02)
r = q4.drain(signals())
assert [i.request_id for i in r.items] == ["D"], r.items
assert q4.stats().emitted_overdue == 1, q4.stats()
ledger_closes(q4)

# 6. window widening: emitting pins blocks, which moves later blocks toward
# the head and brings fresh requests into view in a second discover round.
# The admission-order snapshot must cover those late candidates too.
pool5 = FakePool(64)
q5 = EvictionAwareStoreQueue(LazyOffloadPolicyConfig(horizon_steps=1.0), pool5)
for i in range(12):
    rid = f"w{i}"
    blocks = [2 * i, 2 * i + 1]
    q5.add(meta(rid, blocks, 0, 512), hashes(pool5, blocks), 0)
index_matches(q5); ledger_closes(q5)
r = q5.drain(signals(alloc=2, nxt=2, ids=[0]))
emitted_ids = [i.request_id for i in r.items]
assert len(emitted_ids) > 1, f"window never widened: {emitted_ids}"
assert emitted_ids == sorted(emitted_ids, key=lambda x: int(x[1:])), emitted_ids
index_matches(q5); ledger_closes(q5)

print("index + ledger invariants hold across all six scenarios")
