import pathlib
ROOT = pathlib.Path("/home/bo/LMCache-worktrees/vast_repro")
def patch(rel, old, new, count=1):
    p = ROOT / rel; s = p.read_text(); n = s.count(old)
    assert n == count, f"{rel}: expected {count}, found {n}\n---\n{old[:300]}"
    p.write_text(s.replace(old, new)); print(f"OK  {rel}")

F = "lmcache/v1/multiprocess/modules/blend.py"

patch(F,
'''            session = self._ctx.session_manager.get_or_create(key.request_id)
            # Request-end cleanup may have deleted the session; get_or_create
            # then returns a fresh one whose hash chain is garbage. Re-set
            # tokens: idempotent if the session survived, corrective if not.
            session.set_tokens(list(key.token_ids))''',
'''            session = self._ctx.session_manager.get_or_create(key.request_id)
            # Request-end cleanup may have deleted the session; get_or_create
            # then returns a fresh one whose hash chain is garbage. Splicing
            # the key's tokens back in is idempotent if the session survived,
            # and corrective if not -- unless the key is a delta that does not
            # reach back to what is held, which raises and lands in the
            # ``except`` below (fingerprinting is best-effort by design).
            session.extend_tokens(list(key.token_ids), key.token_offset)''')

patch(F,
'''            tokens_in_range = list(key.token_ids)[key.start : key.end]''',
'''            tokens_in_range = list(key.token_ids)[
                key.start - key.token_offset : key.end - key.token_offset
            ]''')
