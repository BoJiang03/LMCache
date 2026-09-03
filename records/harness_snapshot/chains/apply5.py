import pathlib
ROOT = pathlib.Path("/home/bo/LMCache-worktrees/vast_repro")
def patch(rel, old, new, count=1):
    p = ROOT / rel; s = p.read_text(); n = s.count(old)
    assert n == count, f"{rel}: expected {count}, found {n}\n---\n{old[:300]}"
    p.write_text(s.replace(old, new)); print(f"OK  {rel}")

F = "lmcache/integration/vllm/vllm_multi_process_adapter.py"

# ---- LoadStoreOp gains the offset
patch(F,
'''    skip_first_n_tokens: int = 0
    """Number of tokens to skip writing at the beginning of the retrieve
    range. Used to avoid overwriting APC-shared GPU blocks during retrieve."""
''',
'''    skip_first_n_tokens: int = 0
    """Number of tokens to skip writing at the beginning of the retrieve
    range. Used to avoid overwriting APC-shared GPU blocks during retrieve."""

    token_offset: int = 0
    """Absolute position of ``token_ids[0]``.

    0 means ``token_ids`` is the request's whole prefix and ``start``/``end``
    index into it -- what retrieve ops send. A store op instead carries only
    ``[start, end)`` with ``token_offset == start``: the server's session
    already holds the rolling hash of everything before it, so resending the
    prefix every step only costs a list copy on the scheduler, a serialize per
    step into the connector metadata that vLLM broadcasts to every worker, and
    a tuple build plus msgpack encode on each of them. ``start``/``end`` stay
    absolute in both cases."""
''')

# ---- _create_key forwards it
patch(F,
'''        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs.
            start: Start token index.
            end: End token index.
            request_id: The request ID.
            cache_salt: Per-user isolation salt.

        Returns:
            IPCCacheServerKey: The constructed key.
        """
        return IPCCacheServerKey(
            num_kv_readers=self.parallel_strategy.num_kv_readers,
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
''',
'''        token_ids: list[int],
        start: int,
        end: int,
        request_id: str,
        cache_salt: str = "",
        token_offset: int = 0,
    ) -> IPCCacheServerKey:
        """Convert token IDs to an IPC cache engine key.

        Args:
            token_ids: The token IDs covering
                ``[token_offset, token_offset + len(token_ids))``.
            start: Start token index (absolute).
            end: End token index (absolute).
            request_id: The request ID.
            cache_salt: Per-user isolation salt.
            token_offset: Absolute position of ``token_ids[0]``. 0 (the
                default) means ``token_ids`` is the whole prefix.

        Returns:
            IPCCacheServerKey: The constructed key.
        """
        return IPCCacheServerKey(
            num_kv_readers=self.parallel_strategy.num_kv_readers,
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            token_ids=tuple(token_ids),
            start=start,
            end=end,
            request_id=request_id,
            cache_salt=cache_salt,
            token_offset=token_offset,
        )
''')

# ---- the store submit path passes the op's offset through
patch(F,
'''        assert op.token_ids is not None
        key = self._create_key(
            op.token_ids,
            op.start,
            op.end,
            request_id=request_id,
            cache_salt=cache_salt,
        )
        if self.transfer_ctx is None:
            raise RuntimeError(
                "Transfer context is not initialized. "
                "Call register_kv_caches() before submitting store requests."
            )''',
'''        assert op.token_ids is not None
        key = self._create_key(
            op.token_ids,
            op.start,
            op.end,
            request_id=request_id,
            cache_salt=cache_salt,
            token_offset=op.token_offset,
        )
        if self.transfer_ctx is None:
            raise RuntimeError(
                "Transfer context is not initialized. "
                "Call register_kv_caches() before submitting store requests."
            )''')
