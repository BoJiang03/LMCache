import pathlib
ROOT = pathlib.Path("/home/bo/LMCache-worktrees/vast_repro")
def patch(rel, old, new, count=1):
    p = ROOT / rel; s = p.read_text(); n = s.count(old)
    assert n == count, f"{rel}: expected {count}, found {n}\n---\n{old[:300]}"
    p.write_text(s.replace(old, new)); print(f"OK  {rel}")

patch("lmcache/v1/multiprocess/session.py",
"""from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey""",
"""from lmcache.v1.multiprocess.custom_types import (
    IPCCacheServerKey,
    SessionTokenGapError,
)""")

patch("lmcache/v1/multiprocess/session.py",
'''    def set_tokens(self, full_token_ids: list[int]) -> None:
        """Update the token sequence (idempotent, replaces not extends).

        Args:
            full_token_ids: Complete token sequence.
        """
        with self._lock:
            self.token_ids = full_token_ids
''',
'''    def set_tokens(self, full_token_ids: list[int]) -> None:
        """Update the token sequence (idempotent, replaces not extends).

        Args:
            full_token_ids: Complete token sequence.
        """
        with self._lock:
            self.token_ids = full_token_ids

    def extend_tokens(self, token_ids: list[int], token_offset: int) -> None:
        """Splice a token range in at ``token_offset``.

        ``token_offset == 0`` is exactly ``set_tokens``: the caller sent the
        whole prefix and it replaces whatever is here. A non-zero offset is a
        delta -- the caller sent only ``[token_offset, token_offset + len)``
        because the chunks before it are already folded into the rolling hash
        state (``num_chunks_processed`` / ``last_prefix_hash``), which this
        never touches.

        Splicing rather than appending keeps the call idempotent: a resend of
        the same range overwrites itself instead of duplicating. The tokens
        below ``token_offset`` are left in place, so ``_compute_hash`` can
        still walk any chunk it has not processed yet.

        Args:
            token_ids: The tokens covering ``[token_offset, token_offset + len)``.
            token_offset: Absolute position of ``token_ids[0]``.

        Raises:
            SessionTokenGapError: If ``token_offset`` is past the end of the
                tokens held, which would leave a hole the hash chain cannot
                cross. The session is left untouched.
        """
        if token_offset == 0:
            with self._lock:
                self.token_ids = token_ids
            return

        with self._lock:
            held = len(self.token_ids)
            if token_offset > held:
                raise SessionTokenGapError(
                    f"token_offset ({token_offset}) is past the {held} token(s) "
                    f"held by session {self.request_id!r}; the session was most "
                    "likely swept or recreated mid-request, so the hash chain "
                    "cannot be continued from this delta"
                )
            self.token_ids[token_offset:] = token_ids
''')
