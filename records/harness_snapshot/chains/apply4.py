import pathlib
ROOT = pathlib.Path("/home/bo/LMCache-worktrees/vast_repro")
def patch(rel, old, new, count=1):
    p = ROOT / rel; s = p.read_text(); n = s.count(old)
    assert n == count, f"{rel}: expected {count}, found {n}\n---\n{old[:300]}"
    p.write_text(s.replace(old, new)); print(f"OK  {rel}")

F = "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py"

patch(F,
"""from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
)""",
"""from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    KVCache,
    SessionTokenGapError,
)""")

# Only the STORE call site (line ~1097) gets the guard; the RETRIEVE one below
# is reached only by whole-prefix keys, which cannot gap.
patch(F,
"""        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
            key, list(range(num_object_groups))
        )
        num_chunks = len(obj_keys_per_obj_group[0])

        # CPU-synchronous sentinel: a GPU store is about to be enqueued.""",
"""        num_object_groups = cache_context.kv_layer_groups_manager.num_object_groups
        try:
            obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
                key, list(range(num_object_groups))
            )
        except SessionTokenGapError:
            # A store carries only its own token range; the chunks before it
            # live in the session's rolling hash state. If that state is gone
            # the chain cannot be rebuilt from this request alone. Answer
            # terminally rather than raising: an unanswered handler leaves the
            # client's future pending forever, whereas a False here just makes
            # the later retrieve miss and the engine recompute.
            logger.warning(
                "Skipping STORE for request %s [%d, %d): %s",
                key.request_id,
                key.start,
                key.end,
                "session token state is gone, cannot continue the hash chain",
            )
            return b"", False
        num_chunks = len(obj_keys_per_obj_group[0])

        # CPU-synchronous sentinel: a GPU store is about to be enqueued.""")
