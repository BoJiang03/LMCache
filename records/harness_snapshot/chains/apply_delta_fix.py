import io, sys, pathlib
ROOT = pathlib.Path("/home/bo/LMCache-worktrees/vast_repro")

def patch(rel, old, new, count=1):
    p = ROOT / rel
    s = p.read_text()
    n = s.count(old)
    assert n == count, f"{rel}: expected {count} occurrence(s), found {n}\n---\n{old[:200]}"
    p.write_text(s.replace(old, new))
    print(f"OK  {rel}")

# ---------------------------------------------------------------- 1. key type
patch("lmcache/v1/multiprocess/custom_types.py",
"""    # Number of workers that retrieve this key's object; the server reserves
    # that many read locks (see ``require_num_kv_readers``). 0 = not sent;
    # lookups reject it.
    num_kv_readers: int = field(default=0, compare=False)
""",
"""    # Number of workers that retrieve this key's object; the server reserves
    # that many read locks (see ``require_num_kv_readers``). 0 = not sent;
    # lookups reject it.
    num_kv_readers: int = field(default=0, compare=False)

    # Absolute position of ``token_ids[0]`` in the request's token sequence.
    #
    # 0 (the default, and what LOOKUP/RETRIEVE always send) means ``token_ids``
    # is the request's whole prefix and ``start``/``end`` index directly into
    # it. STORE instead sends only ``[start, end)`` with ``token_offset=start``,
    # because the server's ``Session`` already caches the rolling chunk hashes
    # of everything before it (see ``Session._compute_hash``) and so never needs
    # the prefix resent. ``start``/``end`` stay absolute either way.
    #
    # msgspec encodes dataclasses as maps keyed by field name, so an old payload
    # without this field decodes on new code as 0 -- the whole-prefix meaning.
    token_offset: int = 0
""")

patch("lmcache/v1/multiprocess/custom_types.py",
"""            token_ids=self.token_ids,
            start=self.start,
            end=self.end,
            request_id=self.request_id,
            cache_salt=self.cache_salt,
        )
""",
"""            token_ids=self.token_ids,
            start=self.start,
            end=self.end,
            request_id=self.request_id,
            cache_salt=self.cache_salt,
            token_offset=self.token_offset,
        )
""")

patch("lmcache/v1/multiprocess/custom_types.py",
'''"""


@dataclass(order=True, frozen=True)
class IPCCacheServerKey:''',
'''"""


class SessionTokenGapError(ValueError):
    """A key's token delta does not join onto the session's token sequence.

    Raised when ``token_offset`` is past the end of what the session already
    holds, which leaves a hole the server cannot hash across. Callers on the
    store path must convert this into a terminal failure response (the engine
    then recomputes); it must never escape as an unhandled handler exception,
    because the MQ loop answers those with silence and the client's future
    would never complete.
    """


@dataclass(order=True, frozen=True)
class IPCCacheServerKey:''')
