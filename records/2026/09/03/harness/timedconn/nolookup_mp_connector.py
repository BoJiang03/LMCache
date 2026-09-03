"""TimedMPConnector with the LOOKUP removed entirely, and nothing else.

THE QUESTION
------------
The twin of the no-store arm.  LMCache does two jobs; this one switches off the
other.  Together they partition the +5.7 ms/step, and if neither moves it, the
tax is the mere presence of LMCache's processes and threads.

NOT THE SAME EXPERIMENT AS 1l.  1l kept the LOOKUP -- same request, same server,
same work -- and only changed who waits for the answer.  That bought nothing.
This arm does not send the LOOKUP at all: no message, no server-side chunk
locking, no messaging thread wakeup, no reply.  1l tested "is the scheduler's
waiting on the critical path"; this tests "does the lookup round trip cost
anything anywhere".  They can disagree, and if they do that is the finding.

WHAT IS CHANGED, EXACTLY
------------------------
One instance attribute on the scheduler adapter:

    scheduler_adapter.maybe_submit_lookup_request -> no-op

Nothing else is touched, and nothing else needs to be.  With no submit, the
request id never enters `_pending_lookups`, so the very next line of the stock
hook -- `check_lookup_result` -- takes its own early-return branch:

    if request_id not in self._pending_lookups:
        return self._finished_lookup_results.get(request_id, 0)   # 0

and `get_num_new_matched_tokens` returns (0, False).  That is the identical code
path a real cache miss takes, reached without any server contact.

WHY THIS LOSES NOTHING OBSERVABLE
---------------------------------
Every prompt in the cold pass is 60,000 unique random tokens, so every stock
lookup already returns 0.  This arm returns the same 0 for the same requests --
it removes the round trip that was going to say "miss" and says "miss" locally.
The store path is untouched, so the cache is still populated exactly as before.

FAILURE MODES, PRE-REGISTERED
-----------------------------
    ms/step falls to the no-connector baseline -> the tax is the lookup path,
        and since 1l showed the scheduler's WAIT is free, the cost must be in
        the server or the messaging threads, not in the blocking.
    ms/step unchanged                          -> lookup is free end to end;
        with the no-store arm this leaves only LMCache's idle footprint.
    lookups_skipped == 0 in the log            -> patch did not take; VOID.
"""

import os

from timedconn.timed_mp_connector import TimedMPConnector, logger

SKIPPED = [0]


def _install_no_lookup(conn) -> bool:
    adapter = getattr(conn, "scheduler_adapter", None)
    if adapter is None:
        logger.warning("NOLOOKUP: no scheduler_adapter; nothing patched.")
        return False

    def no_lookup(request_id, token_ids, cache_salt=""):
        SKIPPED[0] += 1
        if SKIPPED[0] % 2000 == 0:
            logger.info("NOLOOKUP pid=%d lookups_skipped=%d", os.getpid(), SKIPPED[0])
        return

    adapter.maybe_submit_lookup_request = no_lookup
    return True


class NoLookupMPConnector(TimedMPConnector):
    """LMCacheMPConnector, timed, with the LOOKUP round trip never sent."""

    def __init__(self, vllm_config, role, kv_cache_config=None):
        super().__init__(vllm_config, role, kv_cache_config)
        from timedconn.timed_mp_connector import _ROLE

        patched = _install_no_lookup(self) if _ROLE[0] == "SCHEDULER" else False
        logger.info(
            "NoLookupMPConnector attached pid=%d role=%s no_lookup=%s",
            os.getpid(), _ROLE[0], patched,
        )
