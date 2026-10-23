# SPDX-License-Identifier: Apache-2.0
"""Measure how long a lazily stored chunk's L1 copy overlaps vLLM's L0.

A lazy store lands shortly before vLLM may recycle the chunk's GPU blocks.
Until it does, the L1 copy duplicates L0. :class:`StoreLeadMeter` times that
overlap per stored chunk from the scheduler process, without hooks into vLLM:
it reads each tracked block's ``ref_cnt`` and ``block_hash`` at every
:meth:`StoreLeadMeter.poll`, so every event is known only to the poll
interval (one scheduler step).

One chunk goes through::

    stored ──(in use)──> freed ──(in the free queue)──> recycled | re-hit
       └──── inuse ────┘     └──────── free_recycled / free_rehit ────────┘

- ``inuse``: the store landed while vLLM still referenced the block; the
  time until vLLM released it.
- ``free_recycled``: the time from ``max(stored, freed)`` until the block's
  hash left vLLM's prefix cache, i.e. vLLM evicted or reused the block.
- ``free_rehit``: the same until a request referenced the block again,
  which ends the free-queue overlap with the block back in use.

A chunk is tracked by its last block of one full-attention group: vLLM frees
a request's blocks in reverse, so that block is recycled first and ends the
chunk's servable prefix, and a hit touching it touched the whole chunk.

Each distribution is a histogram over :data:`LEAD_BUCKET_EDGES_S` (log2
seconds), counted in chunks and in tokens, which are proportional to the
chunk's bytes. One INFO line ``Lazy offload lead: ...`` carries cumulative
counts, at most every :data:`LOG_INTERVAL_S`.
"""

# Standard
from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import Literal, Protocol
import math

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Upper edges of the histogram buckets, in seconds; the last bucket has none.
LEAD_BUCKET_EDGES_S = tuple(2.0**exponent for exponent in range(-3, 9))
NUM_LEAD_BUCKETS = len(LEAD_BUCKET_EDGES_S) + 1
# Minimum seconds between two lead lines, as for the counter ledger.
LOG_INTERVAL_S = 5.0

LeadDistribution = Literal["inuse", "free_recycled", "free_rehit"]
LEAD_DISTRIBUTIONS: tuple[LeadDistribution, ...] = (
    "inuse",
    "free_recycled",
    "free_rehit",
)


def lead_bucket(seconds: float) -> int:
    """Return the histogram bucket a duration falls into.

    Args:
        seconds: A non-negative duration.

    Returns:
        The index of the first edge of :data:`LEAD_BUCKET_EDGES_S` above
        ``seconds``, or ``len(LEAD_BUCKET_EDGES_S)`` past the last one.
    """
    if seconds < LEAD_BUCKET_EDGES_S[0]:
        return 0
    # frexp puts seconds in [2**(e-1), 2**e): bucket e+3 has upper edge 2**e.
    return min(math.frexp(seconds)[1] + 3, NUM_LEAD_BUCKETS - 1)


class TrackedBlock(Protocol):
    """The part of vLLM's ``KVCacheBlock`` the meter reads."""

    @property
    def block_id(self) -> int:
        """The block's index in the pool."""

    @property
    def ref_cnt(self) -> int:
        """References to the block; 0 while it sits in the free queue."""

    @property
    def block_hash(self) -> object:
        """Its prefix-cache hash, reset when vLLM recycles the block."""


@dataclass
class LeadHistogram:
    """One distribution, in chunks and in tokens per bucket.

    Attributes:
        chunks: Chunks per bucket of :data:`LEAD_BUCKET_EDGES_S`.
        tokens: Their tokens per bucket.
    """

    chunks: list[int] = field(default_factory=lambda: [0] * NUM_LEAD_BUCKETS)
    tokens: list[int] = field(default_factory=lambda: [0] * NUM_LEAD_BUCKETS)

    def add(self, seconds: float, tokens: int) -> None:
        """Count one chunk.

        Args:
            seconds: Its duration.
            tokens: Its tokens.
        """
        bucket = lead_bucket(seconds)
        self.chunks[bucket] += 1
        self.tokens[bucket] += tokens


@dataclass
class LeadSnapshot:
    """Cumulative counts since the meter was built.

    Attributes:
        histograms: Per distribution, its histogram.
        counts: ``stored``, ``stored_inuse``, ``after_recycle`` (stores that
            completed once the block was already recycled, a near-drop),
            ``replaced`` (a new store of a still tracked block, which ends
            the earlier one's tracking uncounted), each also as
            ``<name>_tokens``; and the gauges ``alive_inuse`` and
            ``alive_free`` with their ``_tokens``: chunks still tracked.
    """

    histograms: dict[LeadDistribution, LeadHistogram]
    counts: dict[str, int]


class _Overlap:
    """One tracked chunk: its block, its stored hash and when events happened."""

    __slots__ = ("block", "block_hash", "tokens", "stored_at", "freed_at")

    def __init__(
        self,
        block: TrackedBlock,
        block_hash: object,
        tokens: int,
        stored_at: float,
        freed_at: float | None,
    ) -> None:
        self.block = block
        self.block_hash = block_hash
        self.tokens = tokens
        self.stored_at = stored_at
        self.freed_at = freed_at


class StoreLeadMeter:
    """Time each stored chunk's overlap with vLLM's prefix cache.

    Bounded by the number of GPU blocks: one entry per tracked block, gone
    once its overlap ends. Scheduler thread only.
    """

    def __init__(self, pin_counts: Mapping[int, int]) -> None:
        """Create an empty meter.

        Args:
            pin_counts: Per block, the references the caller itself holds on
                it, which its ``ref_cnt`` counts too and which do not make it
                in use by vLLM. Read live, never written.
        """
        self._pin_counts = pin_counts
        self._live: dict[int, _Overlap] = {}
        self._histograms: dict[LeadDistribution, LeadHistogram] = {
            name: LeadHistogram() for name in LEAD_DISTRIBUTIONS
        }
        # Per scalar count, the name of its token count.
        self._token_names = {
            name: f"{name}_tokens"
            for name in ("stored", "stored_inuse", "after_recycle", "replaced")
        }
        self._counts: dict[str, int] = dict.fromkeys(
            (key for pair in self._token_names.items() for key in pair), 0
        )
        self._changed = False
        self._last_log_time = -math.inf

    def on_stored(
        self, now: float, block: TrackedBlock, block_hash: object, tokens: int
    ) -> None:
        """Start tracking a chunk whose L1 store just completed.

        Args:
            now: The completion time, in ``time.monotonic()`` seconds.
            block: The chunk's tracked block.
            block_hash: The hash the block had when the chunk was buffered.
            tokens: The chunk's tokens.
        """
        self._changed = True
        self._bump("stored", tokens)
        if block.block_hash != block_hash:
            self._bump("after_recycle", tokens)
            return
        if block.ref_cnt > self._pin_counts.get(block.block_id, 0):
            self._bump("stored_inuse", tokens)
            freed_at = None
        else:
            freed_at = now
        replaced = self._live.get(block.block_id)
        if replaced is not None:
            self._bump("replaced", replaced.tokens)
        self._live[block.block_id] = _Overlap(block, block_hash, tokens, now, freed_at)

    def poll(self, now: float) -> None:
        """Observe the tracked blocks once and close the overlaps that ended.

        A block found freed and recycled since the last poll counts as freed
        now, with an empty free-queue overlap.

        Args:
            now: The poll time, in ``time.monotonic()`` seconds.
        """
        pins = self._pin_counts
        ended: list[int] = []
        for block_id, overlap in self._live.items():
            block = overlap.block
            if block.block_hash != overlap.block_hash:
                if overlap.freed_at is None:
                    self._histograms["inuse"].add(
                        now - overlap.stored_at, overlap.tokens
                    )
                    overlap.freed_at = now
                self._histograms["free_recycled"].add(
                    now - overlap.freed_at, overlap.tokens
                )
                ended.append(block_id)
            elif block.ref_cnt > pins.get(block_id, 0):
                if overlap.freed_at is not None:
                    self._histograms["free_rehit"].add(
                        now - overlap.freed_at, overlap.tokens
                    )
                    ended.append(block_id)
            elif overlap.freed_at is None:
                self._histograms["inuse"].add(now - overlap.stored_at, overlap.tokens)
                overlap.freed_at = now
                self._changed = True
        if ended:
            self._changed = True
            for block_id in ended:
                del self._live[block_id]

    def snapshot(self) -> LeadSnapshot:
        """Return a copy of the cumulative counts and the live gauges.

        Returns:
            The counts as of the last :meth:`poll` or :meth:`on_stored`.
        """
        counts = dict(self._counts)
        for state in ("alive_inuse", "alive_free"):
            counts[state] = counts[f"{state}_tokens"] = 0
        for overlap in self._live.values():
            state = "alive_inuse" if overlap.freed_at is None else "alive_free"
            counts[state] += 1
            counts[f"{state}_tokens"] += overlap.tokens
        return LeadSnapshot(
            histograms={
                name: LeadHistogram(list(hist.chunks), list(hist.tokens))
                for name, hist in self._histograms.items()
            },
            counts=counts,
        )

    def maybe_log(self, now: float) -> None:
        """Log the lead line if anything changed and the throttle allows.

        Args:
            now: The current ``time.monotonic()`` seconds.
        """
        if not self._changed or now - self._last_log_time < LOG_INTERVAL_S:
            return
        self.log()
        self._last_log_time = now

    def log(self) -> None:
        """Log the lead line now, as at shutdown."""
        logger.info("Lazy offload lead: %s", self.render())
        self._changed = False

    def render(self) -> str:
        """Render the cumulative counts as one greppable ``key=value`` body.

        Scalar counts first, then per distribution ``<name>=c0,...,c12`` and
        ``<name>_tokens=...``, one value per bucket of
        :data:`LEAD_BUCKET_EDGES_S`.

        Returns:
            The line body.
        """
        snapshot = self.snapshot()
        scalars = " ".join(f"{name}={value}" for name, value in snapshot.counts.items())
        histograms = " ".join(
            f"{name}={','.join(map(str, hist.chunks))} "
            f"{name}_tokens={','.join(map(str, hist.tokens))}"
            for name, hist in snapshot.histograms.items()
        )
        return f"{scalars} {histograms}"

    def _bump(self, name: str, tokens: int) -> None:
        """Count one chunk under ``name`` and its tokens under ``name_tokens``.

        Args:
            name: A scalar count of :attr:`LeadSnapshot.counts`.
            tokens: The chunk's tokens.
        """
        self._counts[name] += 1
        self._counts[self._token_names[name]] += tokens
