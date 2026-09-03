import pathlib
ROOT = pathlib.Path("/home/bo/LMCache-worktrees/vast_repro")
F = ROOT / "lmcache/v1/multiprocess/modules/lmcache_driven_transfer.py"
lines = F.read_text().splitlines(keepends=True)
# 1-indexed 1097..1099 is the STORE resolve_obj_keys call
old = "".join(lines[1097:1100])
assert old == ("        obj_keys_per_obj_group = self._ctx.resolve_obj_keys(\n"
               "            key, list(range(num_object_groups))\n"
               "        )\n"), repr(old)
new = '''        try:
            obj_keys_per_obj_group = self._ctx.resolve_obj_keys(
                key, list(range(num_object_groups))
            )
        except SessionTokenGapError as e:
            # A store carries only its own token range; the chunks before it
            # live in the session's rolling hash state. If that state is gone
            # the chain cannot be rebuilt from this request alone. Answer
            # terminally rather than raising: an unanswered handler leaves the
            # client's future pending forever, whereas False here only makes
            # the later retrieve miss and the engine recompute.
            logger.warning("Skipping STORE for request %s: %s", key.request_id, e)
            return b"", False
'''
lines[1097:1100] = [new]
F.write_text("".join(lines))
print("OK  store site guarded")
