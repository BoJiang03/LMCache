import pathlib
ROOT = pathlib.Path("/home/bo/LMCache-worktrees/vast_repro")
def patch(rel, old, new, count=1):
    p = ROOT / rel; s = p.read_text(); n = s.count(old)
    assert n == count, f"{rel}: expected {count}, found {n}\n---\n{old[:300]}"
    p.write_text(s.replace(old, new)); print(f"OK  {rel}")

patch("lmcache/v1/multiprocess/engine_context.py",
'''        Raises:
            ValueError: If ``key.worker_id`` is ``None``.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.set_tokens(list(key.token_ids))
        if session.lookup_ipc_key is None:
            session.lookup_ipc_key = key.no_worker_id_version()
''',
'''        Raises:
            ValueError: If ``key.worker_id`` is ``None``.
            SessionTokenGapError: If the key carries a token delta that does
                not join onto the session's tokens. Store callers must turn
                this into a terminal failure response, never let it escape as
                an unhandled handler exception.
        """
        session = self.session_manager.get_or_create(key.request_id)
        session.extend_tokens(list(key.token_ids), key.token_offset)
        if session.lookup_ipc_key is None and key.token_offset == 0:
            # A delta key holds only part of the prefix, so it cannot stand in
            # for a lookup key: ``Session.prepare_failed_retrieve_release``
            # proves range ownership by comparing ``token_ids`` against this
            # one. Leaving it unset costs a lock release that falls back to the
            # TTL; setting it from a delta would silently reject every real
            # retrieve.
            session.lookup_ipc_key = key.no_worker_id_version()
''')
