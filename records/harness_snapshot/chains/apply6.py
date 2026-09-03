import pathlib
ROOT = pathlib.Path("/home/bo/LMCache-worktrees/vast_repro")
def patch(rel, old, new, count=1):
    p = ROOT / rel; s = p.read_text(); n = s.count(old)
    assert n == count, f"{rel}: expected {count}, found {n}\n---\n{old[:300]}"
    p.write_text(s.replace(old, new)); print(f"OK  {rel}")

F = "lmcache/integration/vllm/lmcache_mp_metadata.py"

patch(F,
'''    def get_token_ids(self) -> list[int]:
        """Return the token ids to use for LMCache key derivation."""
        if not self.mm_adjusted_prompt_ids:
            return list(self.all_token_ids)
        num_prompt_tokens = len(self.mm_adjusted_prompt_ids)
        return self.mm_adjusted_prompt_ids + list(
            self.all_token_ids[num_prompt_tokens:]
        )
''',
'''    def get_token_ids(self) -> list[int]:
        """Return the token ids to use for LMCache key derivation."""
        if not self.mm_adjusted_prompt_ids:
            return list(self.all_token_ids)
        num_prompt_tokens = len(self.mm_adjusted_prompt_ids)
        return self.mm_adjusted_prompt_ids + list(
            self.all_token_ids[num_prompt_tokens:]
        )

    def get_token_ids_range(self, start: int, end: int) -> list[int]:
        """Return ``get_token_ids()[start:end]`` without materializing the rest.

        A store fires on roughly every step of a long prefill, so building the
        whole prefix just to slice one chunk out of it is an O(prompt) copy per
        step that then gets serialized into the connector metadata and shipped
        to every worker. This copies only the range asked for.

        Args:
            start: Start token index (absolute).
            end: End token index (absolute), exclusive.

        Returns:
            The token ids in ``[start, end)``.
        """
        if not self.mm_adjusted_prompt_ids:
            return list(self.all_token_ids[start:end])
        num_prompt_tokens = len(self.mm_adjusted_prompt_ids)
        if start >= num_prompt_tokens:
            return list(self.all_token_ids[start:end])
        if end <= num_prompt_tokens:
            return self.mm_adjusted_prompt_ids[start:end]
        return self.mm_adjusted_prompt_ids[start:num_prompt_tokens] + list(
            self.all_token_ids[num_prompt_tokens:end]
        )
''')

patch(F,
'''            token_ids = tracker.get_token_ids()
            op = LoadStoreOp(
                token_ids=token_ids,
                block_ids=block_ids,
                start=start_token_idx,
                end=end_token_idx,
            )
''',
'''            # Ship only the range being stored. The server keeps a per-request
            # session whose rolling chunk hash already covers everything before
            # ``start_token_idx`` (Session._compute_hash), so the prefix is dead
            # weight -- and it is not cheap weight: it is copied here, serialized
            # into the connector metadata vLLM broadcasts every step, decoded by
            # every worker, then tuple-built and msgpack-encoded once per rank.
            token_ids = tracker.get_token_ids_range(start_token_idx, end_token_idx)
            op = LoadStoreOp(
                token_ids=token_ids,
                block_ids=block_ids,
                start=start_token_idx,
                end=end_token_idx,
                token_offset=start_token_idx,
            )
''')
