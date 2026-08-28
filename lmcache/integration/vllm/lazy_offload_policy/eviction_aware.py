# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project
"""Eviction-aware lazy offload policy (gates 1 and 3 of the store decision).

Implements the drain policy described in
``docs/design/integration/vllm/lazy_offload_decision_model.md``: store
operations are buffered instead of submitted eagerly, and are released only
when the GPU blocks holding their data are about to be evicted (gate 1,
"replace prediction with timing"). A chain whose covered prefix is too short
for the store to beat recomputation is held outside the pending machine --
where it costs nothing per step -- until its request grows past the
break-even length, and dropped if it never does (gate 3, static break-even
threshold at admission).

This module is pure policy: it never touches vLLM at runtime (vLLM types
appear only in annotations) and performs no I/O, so it is unit-testable
without a GPU or a vLLM installation. The connector owns execution: taking
block-hash snapshots at admission, calling :meth:`EvictionAwareStoreQueue.
observe_step` / :meth:`EvictionAwareStoreQueue.collect_due` once per
scheduler step, pinning (``touch``) the blocks of emitted operations, and
submitting them to the worker.
"""

# Standard
from collections import deque
from dataclasses import dataclass, field, replace
from typing import TYPE_CHECKING, Iterable, Iterator, Protocol
import enum
import math

# First Party
from lmcache.utils import init_logger

if TYPE_CHECKING:
    # Third Party
    from vllm.v1.core.block_pool import BlockPool
    from vllm.v1.core.kv_cache_utils import BlockHashWithGroupId

    # First Party
    from lmcache.integration.vllm.lmcache_mp_metadata import (
        LMCacheMPRequestMetadata,
    )

logger = init_logger(__name__)

# Smoothing factor for the per-step block-consumption EMA. Not a config knob:
# the horizon (in steps) is the tunable quantity; the EMA only smooths noise.
_EMA_ALPHA = 0.3

# Adaptive-degradation controller constants. None of these encodes a
# workload: each is a property of the measurement itself, documented with
# the physical quantity that sets it.

# Sliding window for the eviction byte rate. Eviction arrives in watermark
# bursts tens of seconds apart, so the window must span at least two burst
# cycles or the rate reads zero between bursts and the residence estimate
# flaps across any threshold.
_PRESSURE_WINDOW_SECS = 120.0

# Minimum history span before the eviction rate -- and hence residence --
# counts as known at all. Half the window: one burst cycle.
_PRESSURE_MIN_SPAN_SECS = 60.0

# Length of one trial (immediate emission measured against the deferred
# baseline) and one probe (the reverse). Long enough to span several store
# cycles, short enough that a wrong regime during the measurement is cheap.
_TRIAL_SECS = 45.0

# Emission-rate ratio treated as "volume unchanged". Covers the sampling
# noise of comparing two short windows of a bursty emission process; a
# genuine volume effect (deferral filtering stores out) shows up as a
# multiple, not a percentage.
_NEUTRALITY_FACTOR = 1.25

# Fraction of the windowed admissions that deferral may lose to eviction
# before the loss counts as material and opens a trial on its own. The same
# tolerance the neutrality check uses: losing a quarter of the intake is a
# volume effect, not sampling noise. Measured over the trailing trial-length
# window -- the controller attacks on the timescale it verifies at.
_MATERIAL_LOSS_SHARE = _NEUTRALITY_FACTOR - 1.0

# While degraded, how often to re-measure the deferred counterfactual.
# With _TRIAL_SECS this sets the probe duty cycle (~9%): the price of
# noticing that deferral has become worthwhile again.
_PROBE_INTERVAL_SECS = 480.0

# Minimum spacing between probes when recovered residence arms one early.
# A failed probe is evidence about the current regime; asking again faster
# than a few trial lengths would re-measure the same conditions.
_PROBE_RETRY_MIN_SECS = 4 * _TRIAL_SECS

# Cap on the exponential probe backoff, in probe intervals. Every probe
# re-defers for a trial length, so a workload that keeps failing them pays
# that exposure forever at a fixed duty cycle. Backing the interval off per
# consecutive failure bounds the lifetime exposure while keeping recovery
# reachable if the workload changes.
_PROBE_BACKOFF_MAX = 8

# After a reverted trial or a probe recovery, how long before churn may
# start another trial. Bounds the degraded-blip duty cycle on a workload
# whose deferral filters volume (~7%).
_REVERT_COOLDOWN_SECS = 600.0

# Residence must recover to this multiple of the threshold (hysteresis)
# before an early recovery probe is armed. Recovery itself still goes
# through the probe's volume verdict: the windowed estimate reads infinite
# residence whenever bursts space out past the window, so acting on it
# directly would expose the deferred backlog to the next burst.
_RESIDENCE_RECOVERY_FACTOR = 2.0

# Adaptive danger floor constants (opt-in via
# LazyOffloadPolicyConfig.danger_floor_max_blocks). Like the degradation
# constants above, each is a property of the measurement, not a workload
# tunable.

# Per-drain decay of a raised floor, applied only after the hold below has
# expired. A drain runs once per scheduler step (tens of milliseconds), so
# 0.999 halves the floor in ~700 steps once decay starts.
_DANGER_FLOOR_DECAY = 0.999

# How long a raised floor holds flat before decay may start, in multiples
# of the measured interval between raises. Bursts recur on a cadence; a
# floor that decays between two bursts loses the leading edge of every
# burst and re-learns the same size forever (measured on i60F: 58 raises,
# 272 operations lost to exactly this cycle). Holding for two measured
# gaps means a standing cadence keeps the floor up, while a workload that
# genuinely quiets down waits two of its own burst intervals and then
# decays as before.
_DANGER_FLOOR_HOLD_GAPS = 2.0

# Hold length while no raise interval has been measured yet (a single
# raise so far), and the lower bound afterwards, in drains. Sized to span
# at least one burst inter-arrival: bursts arrive tens of seconds apart
# and a drain runs every few tens of milliseconds.
_DANGER_FLOOR_MIN_HOLD_DRAINS = 2048

# A raise at least multiplies the standing requirement by this factor, so
# consecutive losses escalate geometrically even when the peak-allocation
# sample undershoots the burst that caused them.
_DANGER_FLOOR_GROWTH = 2.0

# Steps of gross allocation retained for the peak sample a raise covers.
# The loss a drain discovers happened within the last few steps, so the
# burst that caused it is still in a window this size.
_RECENT_ALLOC_STEPS = 8


class _DegradeRegime(enum.Enum):
    """Where the adaptive-degradation controller currently stands."""

    #: Defer stores; watch the churn gate for a reason to run a trial.
    NORMAL = "normal"
    #: Emit immediately for a bounded window, measuring the volume effect.
    TRIAL = "trial"
    #: Committed immediate emission; recovery only through a probe.
    DEGRADED = "degraded"
    #: Defer for a bounded window, measuring whether filtering returned.
    PROBE = "probe"


class BlockPoolReader(Protocol):
    """Read-only view over the GPU block pool required by the policy.

    The production implementation is :class:`GPUBlockPoolView`; tests provide
    a fake. Both must be side-effect free: the policy never mutates pool
    state.
    """

    def free_queue_block_ids(self) -> Iterator[int]:
        """Iterate the free queue from the eviction head, lazily.

        The policy consumes only as many blocks as the step's decisions
        actually compare against, so the iterator must be lazy: this runs
        once per scheduler step on the critical path while the queue holds
        every free block in the pool, and materialising it would be
        O(free blocks) -- tens of thousands on a pool sized to fill the GPU.

        Returns:
            Block ids in eviction order, the next victim first. The position
            of a block in this sequence is its rank; blocks the caller never
            reaches, and blocks that are not in the free queue at all, mean
            the same thing to the policy -- not at risk this step.
        """
        ...

    def is_free(self, block_id: int) -> bool:
        """Whether the block currently sits in the free queue.

        Answers in O(1) what walking to the block's rank would answer in
        O(rank): a block the policy is about to pin leaves the queue and
        shifts every block behind it toward the head, and that shift has to
        be counted whether or not the block is inside the window the step
        happened to read.

        Args:
            block_id: The GPU block id to inspect.

        Returns:
            True when the block is evictable (in the free queue), False when
            it is referenced by a live request or otherwise out of the queue.
        """
        ...

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the current prefix-cache hash of a GPU block.

        Args:
            block_id: The GPU block id to inspect.

        Returns:
            The block's current hash, or None if the block holds no cached
            (full, hashed) content -- e.g. it was evicted and reallocated,
            or never completed.
        """
        ...


class GPUBlockPoolView:
    """Production :class:`BlockPoolReader` over a bound vLLM ``BlockPool``.

    All accesses are read-only. ``free_queue_block_ids`` is a generator, so
    it costs O(the depth the caller consumes) rather than O(the free queue),
    which matters because it runs on the scheduler thread once per step: a
    pool sized to fill the GPU keeps tens of thousands of blocks free.
    """

    def __init__(self, block_pool: "BlockPool") -> None:
        """Wrap a vLLM block pool obtained via ``bind_gpu_block_pool``.

        Args:
            block_pool: The scheduler's GPU block pool.
        """
        self._block_pool = block_pool

    def free_queue_block_ids(self) -> Iterator[int]:
        """Walk the free queue's links from the eviction head, lazily.

        Yields ids instead of calling ``get_all_free_blocks()``, which
        materialises the whole queue into a list (its own docstring in vLLM
        says it is mainly for testing), and instead of snapshotting a fixed
        depth, which would charge the step for ranks it never compares.

        Yields:
            Block ids in eviction order, the next victim first.

        Raises:
            RuntimeError: If the free list's fake head has no successor,
                i.e. the queue is not in the shape vLLM maintains.
        """
        block = self._block_pool.free_block_queue.fake_free_list_head.next_free_block
        if block is None:
            raise RuntimeError("free_block_queue.fake_free_list_head has no successor")
        # The fake tail is the one node with no successor, so this stops
        # before reaching it.
        while block.next_free_block is not None:
            yield block.block_id
            block = block.next_free_block

    def is_free(self, block_id: int) -> bool:
        """Whether the block is an eviction candidate.

        vLLM keeps exactly the blocks with no live reference in the free
        queue (``BlockPool.touch`` removes a block from it on the same
        condition), so the reference count answers queue membership without
        walking the list. The null block of a hybrid-attention model is
        excluded: it is popped out of the queue at construction and its
        count is not maintained.

        Args:
            block_id: The GPU block id to inspect.
        """
        block = self._block_pool.blocks[block_id]
        return block.ref_cnt == 0 and not block.is_null

    def block_hash(self, block_id: int) -> "BlockHashWithGroupId | None":
        """Return the current hash of the block, or None if uncached.

        Args:
            block_id: The GPU block id to inspect.
        """
        return self._block_pool.blocks[block_id].block_hash

    def num_free_blocks(self) -> int:
        """Return the number of blocks currently in the free queue."""
        return self._block_pool.get_num_free_blocks()


class _FreeQueueWindow:
    """The head of the free queue, materialised only as deep as it is used.

    A drain compares ranks against the danger depth, extended by the blocks
    the drain itself pins out of the queue. How deep that reaches is not
    known before the drain runs, and the number it *could* reach --
    ``max_drain_per_step`` operations at the largest pending size -- is the
    wrong thing to pay for: it is the per-step cost of a burst that almost
    never happens, charged to every step. This window starts at the danger
    depth and is extended block by block as emissions actually widen the
    threshold, so a step reads the ranks it compares and no more.
    """

    def __init__(self, block_ids: Iterator[int]) -> None:
        """Open an empty window over a lazy free-queue walk.

        Args:
            block_ids: Free-queue block ids in eviction order, consumed on
                demand. An empty iterator opens a window that never grows.
        """
        self._block_ids = block_ids
        self._exhausted = False
        self.ranks: dict[int, int] = {}

    def depth(self) -> int:
        """Return how many blocks the window currently holds."""
        return len(self.ranks)

    def extend_to(self, depth: int) -> dict[int, int]:
        """Walk the queue until the window holds ``depth`` blocks.

        Args:
            depth: Target depth, counted from the eviction head. A target at
                or below the current depth reads nothing.

        Returns:
            The entries this call revealed, block id -> rank, in ascending
            rank order. Empty when the window already reached the target or
            the queue ended first.
        """
        revealed: dict[int, int] = {}
        while not self._exhausted and len(self.ranks) < depth:
            block_id = next(self._block_ids, None)
            if block_id is None:
                self._exhausted = True
                break
            rank = len(self.ranks)
            self.ranks[block_id] = rank
            revealed[block_id] = rank
        return revealed


class AdmitResult(enum.Enum):
    """Outcome of admitting a store operation into the lazy queue.

    The connector maps each outcome to an action:

    - ADMITTED: nothing to do now; the operation is in the policy's custody.
      It is emitted later -- or dropped by a gate: an operation below the
      gate-3 break-even prefix length is held until its request grows past
      the threshold, and dropped if the request finishes below it.
    - REJECTED_UNHASHED_BLOCK: a covered block has no hash, so eviction of
      that block could not be detected later (a reallocated block would also
      read None, masking the loss). The connector must skip the store and
      warn. Because the caller's tracker has already advanced past the
      skipped range, the request's later chunks are unreachable and will be
      rejected as prefix-broken. With plain prefix caching, chunk-aligned
      ranges never cover unhashed blocks; hybrid-attention models (sliding
      window, mamba) can place hash-less null blocks in block tables.
    - REJECTED_PREFIX_BROKEN: an earlier chunk of this request was already
      dropped, so this chunk would be unreachable on retrieval. The
      connector must skip the store entirely.
    - DEDUPLICATED: identical content (same salt, range, and block-hash
      chain) is already buffered under another request. Nothing to do: the
      content will be stored -- or dropped -- with the operation that
      buffered it, and this operation must not defer its own request's
      session teardown.
    """

    ADMITTED = enum.auto()
    REJECTED_UNHASHED_BLOCK = enum.auto()
    REJECTED_PREFIX_BROKEN = enum.auto()
    DEDUPLICATED = enum.auto()


@dataclass
class PendingStoreOp:
    """A deferred store operation with the state needed to validate it.

    Attributes:
        request_id: The vLLM request this operation belongs to.
        store_metadata: The ready-to-send store metadata produced by
            ``LMCacheMPRequestMetadata.GetStoreMetadata``; opaque to the
            policy.
        block_hashes: Hash of every GPU block covering the operation's token
            range, snapshotted at admission. All values are non-None
            (enforced by admission); a later mismatch against the pool means
            the block was evicted or reallocated.
        prefix_start_tokens: Token index of the start of this operation's
            range. Used to detect holes in a request's pending list: after
            a deduplicated chunk, the next operation does not start where
            the previous pending one ended, and an emitted batch must never
            span such a hole (it is coalesced into one contiguous store).
        prefix_end_tokens: Token index one past the end of this operation's
            range, i.e. the request-prefix length covered once this
            operation and all earlier ones are stored.
        epoch: Store epoch that produced this operation.
        cache_salt: The request's cache salt, part of the operation's
            content identity for deduplication (two requests with the same
            block hashes but different salts store under different keys).
    """

    request_id: str
    store_metadata: "LMCacheMPRequestMetadata"
    block_hashes: dict[int, "BlockHashWithGroupId"]
    prefix_start_tokens: int
    prefix_end_tokens: int
    epoch: int = 0
    cache_salt: str = ""


def _content_key(
    op: PendingStoreOp,
) -> tuple[str, int, tuple["BlockHashWithGroupId", ...]]:
    """Content identity of an operation, independent of its request.

    Two operations with equal keys cover the same token range with the same
    cached content: the block-hash chain encodes the token prefix, and the
    salt separates cache namespaces.
    """
    return (op.cache_salt, op.prefix_end_tokens, tuple(op.block_hashes.values()))


def _contiguous_front_run(ops: list[PendingStoreOp]) -> list[PendingStoreOp]:
    """Front slice of ops up to (excluding) the first token-range hole.

    Deduplication can leave a hole in a request's pending list: the missing
    chunk is buffered under another request. An emitted batch is coalesced
    into a single store operation with one contiguous token range, so it
    must never span a hole; ops past the hole stay pending and are emitted
    in a later batch once the front run's completion receipt arrives.
    """
    for index in range(1, len(ops)):
        if ops[index].prefix_start_tokens != ops[index - 1].prefix_end_tokens:
            return ops[:index]
    return ops


DEFAULT_HORIZON_STEPS = 2.5


@dataclass(frozen=True)
class LazyOffloadPolicyConfig:
    """Tunables of the eviction-aware drain policy.

    Attributes:
        horizon_steps: How many scheduler steps of estimated block
            consumption to treat as "imminent eviction". Larger values drain
            earlier (closer to eager, fewer drops); smaller values drain
            later (better filtering, more drops).
        danger_floor_max_blocks: Cap, in blocks, on the adaptive danger
            floor; 0 (the default) disables it. The rate model forecasts
            the *mean* consumption, so an allocation burst larger than the
            danger window destroys waiting operations without ever being
            seen as due -- and an unannounced burst is exactly the case
            :meth:`EvictionAwareStoreQueue.announce_allocation` cannot
            cover. The floor closes that gap reactively: a drain interval
            that loses operations to eviction raises the floor to the
            recent peak step allocation (at least doubling the standing
            requirement). The raised floor holds flat for two measured
            loss intervals -- bursts recur on a cadence, and a floor that
            decays between two bursts pays the leading edge of every one
            -- and only then decays back toward the rate model over tens
            of seconds. Depth is
            ``max(rate model, floor, announced)``, so the floor widens the
            window after a measured loss instead of degrading emission
            timing to eager for the whole run. The cap bounds the
            free-queue read the widened window costs on the scheduler
            path. Sizing sensor: ``LazyOffloadCounters.danger_floor_raises``
            alongside ``dropped_evicted`` (raises without further drops
            mean the floor is absorbing the bursts; drops continuing at
            the cap mean the cap is below the workload's burst size).
        min_prefix_tokens: Break-even prefix length (gate 3), applied at
            admission: a request's operations are held outside the pending
            machine while its known prefix is shorter than this, promoted
            into it when the prefix grows past the threshold, and dropped
            if the request finishes below it. 0 disables the gate.
        max_drain_per_step: Upper bound on operations emitted per step, to
            bound the D2H burst. Must be >= 1. There is no safe static
            lower bound: a prefilling request buffers about one operation
            per step, so a cap below the number of concurrently prefilling
            requests cannot keep up and the backlog is lost to eviction
            rather than merely delayed. Sizing it therefore needs the
            workload, and the runtime sensor for having sized it wrong is
            ``LazyOffloadCounters.throttled_drains``.
        max_pending_ops: Upper bound on how many operations may wait for
            their eviction date at once; 0 leaves the backlog unbounded.
            The danger depth is a forecast built from an EMA of per-step
            allocation, so it cannot see a single admission that consumes
            thousands of blocks at once -- the eviction that destroys a
            waiting operation and the allocation that pays for the forecast
            are the same event. Bounding the backlog bounds what one such
            burst can destroy: above the cap the oldest operations are
            emitted regardless of their rank, at ``max_drain_per_step``
            per step. It costs the filtering the wait would have bought
            (content evicted from the GPU after the operation was emitted
            is stored either way), so size it against
            ``LazyOffloadCounters.dropped_evicted``: a backlog deep enough
            to lose operations is deeper than the workload can defend.
        max_drain_blocks_per_step: Upper bound on GPU blocks emitted per
            step, bounding the D2H burst in bytes where
            ``max_drain_per_step`` bounds it in operations (a deferred
            backlog coalesces, so an op-count cap alone lets one step
            submit an arbitrarily long prefix as a single copy). The bound
            is soft: the operation that crosses it is still emitted --
            progress must not depend on an operation fitting under the cap
            -- and the overshoot is charged, so everything after it waits
            for the next step. 0 leaves the volume unbounded. The sizing
            sensor is ``LazyOffloadCounters.throttled_drains``, shared
            with the op-count cap.
        idle_drain_max_ops: Operations the drain may emit on an idle step
            beyond what eviction pressure calls for, oldest request first.
            Pressure times an emission to the moment its blocks are about
            to be reallocated -- which is when a prefill burst is
            allocating, so the copy lands in phase with the burst. Idle
            emission writes waiting content down in the gaps instead,
            trading filtering (content evicted after emission is stored
            either way) for staying out of the busy steps' way.
            0 disables idle draining. Effectiveness sensors:
            ``LazyOffloadCounters.idle_emitted`` and
            ``LazyOffloadCounters.idle_drain_steps``.
        idle_threshold_blocks: Allocation rate, in blocks per step, at or
            below which a step counts as idle. Compared against both the
            per-step EMA and the next step's estimate, so the first step
            of a prefill burst is never idle even before the EMA has
            caught up. Decode-only traffic allocates roughly (running
            requests / tokens per block) blocks per step, well under the
            default of 1.0 at typical concurrency, while a prefill step
            allocates tens to hundreds. Consulted only when
            ``idle_drain_max_ops`` > 0.
        degrade_l1_residence_secs: L1 residence time, in seconds, below
            which the policy considers degrading to immediate emission.
            An opt-in second trigger for the degradation controller, whose
            first trigger -- deferral losing a material share of its own
            intake to eviction -- always runs and needs no configuration.
            Deferral re-times stores toward eviction danger, and under L1
            churn that timing coincides with allocation pressure -- the
            deferred pins collide with the allocator exactly when it can
            least afford them. Residence (capacity over the windowed
            eviction byte rate, fed by
            :meth:`EvictionAwareStoreQueue.observe_l1_pressure`) below this
            threshold says that cost is being paid. Whether it should
            degrade is workload-dependent and measurement decides it: on
            agentic multi-turn replay the polarity runs the other way,
            deferral earning its keep at short residence and costing at
            long, so 0 (the default) leaves this trigger off.
            Neither signal degrades by itself: each starts a
            bounded trial of immediate emission, committed only if the
            trial's volume stays neutral against the deferred
            baseline -- degradation may change the timing of stores, never
            their volume. A trial that increases volume (deferral was
            filtering stores out, not merely re-timing them) reverts and
            enters a cooldown; a committed degradation is re-checked with
            deferred probes (periodic, or early once residence recovers
            past the hysteresis factor) and lifts only when a probe shows
            filtering value has returned.
            Effectiveness sensors: the
            ``degraded_*`` and ``degrade_*`` counters on
            :class:`LazyOffloadCounters`.
    """

    horizon_steps: float = DEFAULT_HORIZON_STEPS
    danger_floor_max_blocks: int = 0
    min_prefix_tokens: int = 0
    max_drain_per_step: int = 64
    max_pending_ops: int = 0
    max_drain_blocks_per_step: int = 0
    idle_drain_max_ops: int = 0
    idle_threshold_blocks: float = 1.0
    degrade_l1_residence_secs: float = 0.0

    def __post_init__(self) -> None:
        """Validate field ranges.

        Raises:
            ValueError: If any field is outside its documented range.
        """
        if self.horizon_steps <= 0:
            raise ValueError(f"horizon_steps must be > 0, got {self.horizon_steps}")
        if self.danger_floor_max_blocks < 0:
            raise ValueError(
                "danger_floor_max_blocks must be >= 0, got "
                f"{self.danger_floor_max_blocks}"
            )
        if self.min_prefix_tokens < 0:
            raise ValueError(
                f"min_prefix_tokens must be >= 0, got {self.min_prefix_tokens}"
            )
        if self.max_drain_per_step < 1:
            raise ValueError(
                f"max_drain_per_step must be >= 1, got {self.max_drain_per_step}"
            )
        if self.max_pending_ops < 0:
            raise ValueError(
                f"max_pending_ops must be >= 0, got {self.max_pending_ops}"
            )
        if self.max_drain_blocks_per_step < 0:
            raise ValueError(
                "max_drain_blocks_per_step must be >= 0, got "
                f"{self.max_drain_blocks_per_step}"
            )
        if self.idle_drain_max_ops < 0:
            raise ValueError(
                f"idle_drain_max_ops must be >= 0, got {self.idle_drain_max_ops}"
            )
        if self.idle_threshold_blocks <= 0:
            raise ValueError(
                f"idle_threshold_blocks must be > 0, got {self.idle_threshold_blocks}"
            )
        if self.degrade_l1_residence_secs < 0:
            raise ValueError(
                "degrade_l1_residence_secs must be >= 0, got "
                f"{self.degrade_l1_residence_secs}"
            )


@dataclass
class LazyOffloadCounters:
    """Cumulative policy counters for observability.

    ``dropped_evicted`` is the gate-1 quality sensor (drop rate): operations
    lost because their blocks were evicted before the policy drained them.
    ``rejected_short_prefix`` counts gate-3 rejections: held operations whose
    request finished below the break-even prefix length, plus chains that
    eviction truncated back below it before emission.

    ``throttled_drains`` is the sizing sensor for
    :attr:`LazyOffloadPolicyConfig.max_drain_per_step`: drains that left a
    due operation unemitted because the cap ran out. One of these is
    harmless -- the operation is emitted a step later -- but a cap below
    the number of concurrently prefilling requests never works the backlog
    off, so the count rising alongside ``dropped_evicted`` is the signature
    of a cap set too low. Counted per drain, not per operation, so it is
    comparable with the number of steps rather than with the other
    counters.

    ``covered_prefix_advances`` and ``covered_prefix_tokens_skipped`` are
    the effectiveness sensors of
    :meth:`EvictionAwareStoreQueue.covered_prefix_tokens`: how often, and
    by how many tokens, a request's store range was shortened because a
    still-buffered operation already covers that prefix. They stand
    outside the admission ledger on purpose -- the skipped range never
    becomes an operation, so it is counted in neither ``admitted`` nor any
    drop counter -- and measure work the
    deferral no longer costs the next request over the same prefix.

    ``backlog_emitted`` is the activity sensor for
    :attr:`LazyOffloadPolicyConfig.max_pending_ops`: operations emitted
    because the backlog exceeded the cap rather than because their blocks
    came under eviction pressure. Zero means the cap never bound and the
    policy behaved exactly as an unbounded backlog would; a share of
    ``emitted`` rising towards 1 means the cap, not the eviction forecast,
    is timing the stores. It is a subset of ``emitted``, not a separate
    outcome, so it does not enter the admission ledger's arithmetic.

    ``idle_emitted`` and ``idle_drain_steps`` are the effectiveness
    sensors for :attr:`LazyOffloadPolicyConfig.idle_drain_max_ops`:
    operations emitted because the step was idle rather than because their
    blocks came under eviction pressure, and the drains in which at least
    one such emission happened. ``idle_emitted`` is a subset of
    ``emitted``, like ``backlog_emitted``; zero alongside a standing
    backlog means the workload never presents an idle step, or the
    threshold sits below its decode-only allocation rate.

    ``announced_bursts`` counts allocation bursts announced from outside
    the per-step forecast (:meth:`EvictionAwareStoreQueue.announce_allocation`),
    one per announced request. The forecast cannot see a single admission
    that consumes thousands of blocks at once, so an external caller that
    knows such an admission is imminent announces it and the danger depth
    floors at the announced width until the announcement is retracted.
    Zero alongside a rising ``dropped_evicted`` under a high external hit
    rate is the signature of the announcement wiring being disconnected,
    not of the policy choosing to wait.

    ``danger_floor_raises`` counts the loss-driven raises of the adaptive
    danger floor (:attr:`LazyOffloadPolicyConfig.danger_floor_max_blocks`):
    drain intervals whose measured eviction loss widened the danger window
    beyond the rate model. Like ``throttled_drains`` it counts events, not
    operations, so it stays out of the admission ledger's arithmetic.
    Raises after which ``dropped_evicted`` goes quiet are the floor
    absorbing unannounced bursts; raises alongside a still-rising
    ``dropped_evicted`` mean the cap sits below the workload's burst size.

    ``degraded_emitted``, ``degraded_drain_steps``,
    ``degrade_transitions``, ``degrade_trials``, ``degrade_commits``,
    ``degrade_reverts``, ``degrade_probes`` and
    ``degrade_probe_recoveries`` are the sensors for
    :attr:`LazyOffloadPolicyConfig.degrade_l1_residence_secs`: operations
    emitted because the policy was in an immediate-emission regime rather
    than under eviction pressure (a subset of ``emitted``, like
    ``backlog_emitted``), the drains in which at least one such emission
    happened, how often the emission behavior flipped between deferred
    and immediate, and the controller's decision ledger -- trials
    started, trials committed because emitted volume stayed neutral,
    trials reverted because it did not, deferred probes run while
    degraded, and probes that restored deferral because filtering value
    had returned. ``degrade_reverts`` ticking once per cooldown is the
    signature of a workload whose deferral filters volume (the
    controller keeps asking and keeps being told no); a commit with no
    subsequent probe recoveries is the signature of one whose deferral
    only re-times stores.

    ``drain_steps``, ``free_queue_blocks_read``, ``requests_validated``,
    ``blocks_validated`` and ``covered_blocks_probed`` are the cost sensors
    for the per-step decision
    itself, which runs on the scheduler's critical path and is therefore
    paid by every token's decode latency. Divided by ``drain_steps`` they
    give the mean free-queue depth a step walks and the mean number of
    block-hash comparisons it makes; both are properties of the workload
    and the pending backlog, not of anything the policy is configured with,
    so a rise in either is the signature of the decision loop -- rather
    than the offload itself -- becoming the cost.
    """

    admitted: int = 0
    emitted: int = 0
    dropped_evicted: int = 0
    rejected_short_prefix: int = 0
    rejected_unhashed: int = 0
    rejected_prefix_broken: int = 0
    dropped_on_request_drop: int = 0
    dropped_failed_store: int = 0
    dropped_id_reuse: int = 0
    deduplicated: int = 0
    covered_prefix_advances: int = 0
    covered_prefix_tokens_skipped: int = 0
    backlog_emitted: int = 0
    idle_emitted: int = 0
    idle_drain_steps: int = 0
    announced_bursts: int = 0
    danger_floor_raises: int = 0
    degraded_emitted: int = 0
    degraded_drain_steps: int = 0
    degrade_transitions: int = 0
    degrade_trials: int = 0
    degrade_commits: int = 0
    degrade_reverts: int = 0
    degrade_probes: int = 0
    degrade_probe_recoveries: int = 0
    throttled_drains: int = 0
    drain_steps: int = 0
    free_queue_blocks_read: int = 0
    requests_validated: int = 0
    blocks_validated: int = 0
    covered_blocks_probed: int = 0

    def decisions(self) -> tuple[int, ...]:
        """The counters that only a policy decision moves.

        The cost sensors advance on every drain whether or not the policy
        decided anything, so a caller watching the counters for change --
        the ledger log does -- has to watch these instead, or it never goes
        quiet on an engine that is merely running.

        Returns:
            Every counter except the five per-step cost sensors, in
            declaration order.
        """
        return (
            self.admitted,
            self.emitted,
            self.dropped_evicted,
            self.rejected_short_prefix,
            self.rejected_unhashed,
            self.rejected_prefix_broken,
            self.dropped_on_request_drop,
            self.dropped_failed_store,
            self.dropped_id_reuse,
            self.deduplicated,
            self.covered_prefix_advances,
            self.covered_prefix_tokens_skipped,
            self.backlog_emitted,
            self.idle_emitted,
            self.idle_drain_steps,
            self.danger_floor_raises,
            self.degraded_emitted,
            self.degraded_drain_steps,
            self.degrade_transitions,
            self.degrade_trials,
            self.degrade_commits,
            self.degrade_reverts,
            self.degrade_probes,
            self.degrade_probe_recoveries,
            self.throttled_drains,
        )


@dataclass
class DrainResult:
    """Operations released by one :meth:`EvictionAwareStoreQueue.collect_due`.

    Attributes:
        to_store: Operations to submit now, ordered by eviction imminence
            across requests and by prefix order within a request. The
            connector must pin (``touch``) their blocks before the store and
            unpin after completion.
        dropped_evicted: Operations whose data was lost (block evicted or
            reallocated before drain), including later same-request
            operations dropped for prefix closure.
        dropped_short_prefix: Operations dropped by gate 3: held operations
            whose request finished below the break-even prefix length, and
            pending chains eviction truncated back below it by the time
            their blocks came due.
        emptied_requests: Requests whose pending operations became empty in
            this drain. The controller combines this fact with request phase
            and submitted-batch state before ending a session.
        ops_held_back: Operations this drain found due but did not emit
            because the emission budget (``max_drain_per_step`` or
            ``max_drain_blocks_per_step``) ran out. They stay pending and
            are emitted by a later drain if their blocks survive that long.
            Counts only the segment the cap cut, so it is a lower bound:
            candidates the loop never reached are not counted, their
            due-ness being unevaluated.
    """

    to_store: list[PendingStoreOp] = field(default_factory=list)
    dropped_evicted: list[PendingStoreOp] = field(default_factory=list)
    dropped_short_prefix: list[PendingStoreOp] = field(default_factory=list)
    emptied_requests: list[str] = field(default_factory=list)
    ops_held_back: int = 0


class _DrainBudget:
    """One drain's shared emission allowance: operations and blocks.

    Every emission path of a drain -- pressure, backlog, idle -- spends
    from the same budget, so the per-step caps bound the step's total D2H
    submission no matter which path asked. The block bound is soft: the
    operation that crosses it is still taken, because progress must not
    depend on an operation fitting under the cap, and the overshoot is
    charged so everything after it waits for the next step.
    """

    def __init__(self, max_ops: int, max_blocks: int) -> None:
        """Create the budget of a single drain.

        Args:
            max_ops: Operations the drain may emit. Must be >= 1.
            max_blocks: GPU blocks the drain may emit; 0 means unbounded.
        """
        self._ops_left = max_ops
        self._blocks_left = max_blocks if max_blocks > 0 else None

    def exhausted(self) -> bool:
        """Whether nothing more may be emitted by this drain."""
        return self._ops_left <= 0 or (
            self._blocks_left is not None and self._blocks_left <= 0
        )

    def take(self, ops: list[PendingStoreOp]) -> list[PendingStoreOp]:
        """Take the front slice of ``ops`` the budget still allows.

        Charges the budget for what it returns: the slice ends before the
        first operation the remaining op count cannot pay for or that
        starts at or past the block bound. While the budget is not
        exhausted the first operation is always taken, even one larger
        than the whole block bound (see the class docstring).

        Args:
            ops: Candidate operations in emission order.

        Returns:
            The operations to emit now, a front slice of ``ops``.
        """
        taken: list[PendingStoreOp] = []
        for op in ops:
            if self.exhausted():
                break
            taken.append(op)
            self._ops_left -= 1
            if self._blocks_left is not None:
                self._blocks_left -= len(op.block_hashes)
        return taken


class _PendingOperations:
    """Own pending operations and every index derived from them."""

    def __init__(self) -> None:
        self._by_request: dict[str, list[PendingStoreOp]] = {}
        self._content: dict[
            tuple[str, int, tuple["BlockHashWithGroupId", ...]], PendingStoreOp
        ] = {}
        self._requests_by_block: dict[int, set[str]] = {}
        self._request_block_refs: dict[tuple[str, int], int] = {}
        self._request_order: dict[str, int] = {}
        self._next_request_order = 0
        self._requests_to_validate: set[str] = set()
        # Block content that some live pending op already stages, refcounted
        # by (salt, block id, hash snapshot). This is what makes a deferred
        # store visible to the *next* request over the same prefix: that
        # request's LMCache lookup misses (the covering op has not been
        # emitted, so the server has never heard of it) and would otherwise
        # re-stage the whole shared prefix from token 0. The hash is part of
        # the key rather than a stored value so a stale snapshot can never
        # be mistaken for current content.
        self._covered_blocks: dict[tuple[str, int, "BlockHashWithGroupId"], int] = {}

    def __bool__(self) -> bool:
        return bool(self._by_request)

    def contains_request(self, request_id: str) -> bool:
        return request_id in self._by_request

    def get(self, request_id: str) -> list[PendingStoreOp] | None:
        return self._by_request.get(request_id)

    def covering_op(self, op: PendingStoreOp) -> PendingStoreOp | None:
        return self._content.get(_content_key(op))

    def covers_block(
        self,
        cache_salt: str,
        block_id: int,
        block_hash: "BlockHashWithGroupId",
    ) -> bool:
        """Whether a live pending op already stages this exact block content.

        Args:
            cache_salt: Cache salt of the asking request. Two requests with
                the same block content but different salts store under
                different keys, so neither covers the other.
            block_id: GPU block id to test.
            block_hash: The block's *current* prefix-cache hash. A pending
                op whose snapshot no longer matches the pool does not cover
                anything: its own data is already lost.

        Returns:
            True when some pending operation covers this block content.
        """
        return (cache_salt, block_id, block_hash) in self._covered_blocks

    def add(self, op: PendingStoreOp) -> None:
        """Add one admitted operation and all of its index entries."""
        if op.request_id not in self._by_request:
            self._request_order[op.request_id] = self._next_request_order
            self._next_request_order += 1
        self._by_request.setdefault(op.request_id, []).append(op)
        self._content[_content_key(op)] = op
        for block_id, block_hash in op.block_hashes.items():
            covered_key = (op.cache_salt, block_id, block_hash)
            self._covered_blocks[covered_key] = (
                self._covered_blocks.get(covered_key, 0) + 1
            )
            ref_key = (op.request_id, block_id)
            refs = self._request_block_refs.get(ref_key, 0) + 1
            self._request_block_refs[ref_key] = refs
            if refs == 1:
                self._requests_by_block.setdefault(block_id, set()).add(op.request_id)

    def _forget(self, ops: list[PendingStoreOp]) -> None:
        """Remove content and block index entries for departed ops."""
        for op in ops:
            key = _content_key(op)
            if self._content.get(key) is op:
                del self._content[key]
            for block_id, block_hash in op.block_hashes.items():
                covered_key = (op.cache_salt, block_id, block_hash)
                covered_refs = self._covered_blocks.get(covered_key, 0) - 1
                if covered_refs > 0:
                    self._covered_blocks[covered_key] = covered_refs
                else:
                    self._covered_blocks.pop(covered_key, None)
                ref_key = (op.request_id, block_id)
                refs = self._request_block_refs[ref_key] - 1
                if refs > 0:
                    self._request_block_refs[ref_key] = refs
                    continue
                del self._request_block_refs[ref_key]
                requests = self._requests_by_block[block_id]
                requests.discard(op.request_id)
                if not requests:
                    del self._requests_by_block[block_id]

    def pop_request(self, request_id: str) -> list[PendingStoreOp]:
        """Atomically remove a request and all entries derived from its ops."""
        departed = self._by_request.pop(request_id, [])
        self._forget(departed)
        self._finish_request(request_id)
        return departed

    def replace_request(
        self,
        request_id: str,
        departed: list[PendingStoreOp],
        remaining: list[PendingStoreOp],
    ) -> None:
        """Atomically remove a front/suffix and install the surviving list."""
        self._forget(departed)
        if remaining:
            self._by_request[request_id] = remaining
            return
        self._by_request.pop(request_id, None)
        self._finish_request(request_id)

    def num_ops(self) -> int:
        return sum(len(ops) for ops in self._by_request.values())

    def _finish_request(self, request_id: str) -> None:
        self._request_order.pop(request_id, None)
        self._requests_to_validate.discard(request_id)

    def observe_allocations(
        self,
        allocated_block_ids: set[int] | None,
    ) -> None:
        """Mark requests whose snapshots require validation this step."""
        if allocated_block_ids is None:
            self._requests_to_validate.update(self._by_request)
            return
        for block_id in allocated_block_ids:
            self._requests_to_validate.update(self._requests_by_block.get(block_id, ()))

    def requests_for_blocks(self, block_ids: Iterable[int]) -> set[str]:
        return {
            request_id
            for block_id in block_ids
            for request_id in self._requests_by_block.get(block_id, ())
        }

    def requests_to_check(self, block_ids: Iterable[int]) -> set[str]:
        return self._requests_to_validate | self.requests_for_blocks(block_ids)

    def validation_complete(self, request_id: str) -> None:
        self._requests_to_validate.discard(request_id)

    def admission_order(self, request_id: str) -> int:
        return self._request_order[request_id]

    def requests_in_admission_order(self) -> list[str]:
        """Pending request ids, oldest admission first.

        Returns:
            Every request holding pending operations, ordered by when it
            first entered the queue. The backlog drain uses it to emit the
            longest-waiting content first.
        """
        return sorted(self._by_request, key=self._request_order.__getitem__)


class EvictionAwareStoreQueue:
    """Buffers store operations and releases them by eviction imminence.

    Gate 1 realization: an operation is emitted when any of its blocks sits
    within the *danger depth* of the free queue -- the number of blocks the
    engine is expected to consume within ``horizon_steps`` steps, estimated
    from an EMA of observed per-step allocation and a one-step feedforward
    supplied by the connector. An idle engine (no allocation pressure) never
    triggers a drain; operations whose blocks are evicted before they come
    due are dropped and counted, never stored stale.

    Admission deduplicates by content: an operation whose salt, range, and
    block-hash chain match a pending operation of another request is not
    buffered again. This bounds the queue by the amount of unique cached
    content on the GPU -- without it, every request over a hot shared
    prefix (blocks that never enter the free queue, so never come due)
    would buffer its own copy indefinitely. A hit is validated against the
    pool: the covering op -- and every earlier pending op of its request,
    whose loss would prefix-close over the cover on the next drain -- must
    still hold its admission-time snapshot; otherwise the new copy is
    admitted instead and takes over the content key. Deduplication is still
    optimistic past that check: if the covering operation is dropped later,
    chunks the deduplicated request stores past that point are unreachable
    until a future request re-buffers the missing prefix -- wasted storage,
    never corruption.
    A deduplicated chunk also leaves a hole in its request's pending list;
    emission never spans a hole (each batch is one contiguous store), so
    the ops on each side of it go out in separate batches.

    Gate 3 realization, at admission: while a request's known prefix is
    below ``min_prefix_tokens``, its operations are held outside the
    pending machine -- unindexed, so the per-step validation and free-queue
    walk never pay for them. The chunk that lifts the prefix past the
    threshold promotes the whole held chain into the pending machine (after
    one snapshot check, because evictions during the wait were unobserved);
    a request that finishes below the threshold has its held chain dropped
    by the next :meth:`collect_due`. Held and pending are mutually
    exclusive per request: nothing emits before promotion, and every chunk
    after promotion ends past the threshold, so it never holds again while
    the store epoch lasts.

    Not thread-safe: all methods must be called from the scheduler thread
    (the vLLM connector scheduler-side call pattern).
    """

    def __init__(self, config: LazyOffloadPolicyConfig, pool: BlockPoolReader) -> None:
        """Create an empty queue.

        Args:
            config: Policy tunables.
            pool: Read-only view of the GPU block pool.
        """
        self._config = config
        self._pool = pool
        # Primary pending storage and every derived secondary index share one
        # owner so departure paths cannot update one without the other.
        self._pending_ops = _PendingOperations()
        # Gate-3 holding pen: chains below the break-even prefix length,
        # keyed by request. Deliberately index-free -- the point of holding
        # is that these ops cost nothing on the per-step path.
        self._held_short: dict[str, list[PendingStoreOp]] = {}
        # Prefix validity is a policy concern. Request phase, epochs, and
        # submitted batches are owned by the controller.
        self._broken_prefixes: set[str] = set()
        self._blocks_per_step_ema: float = 0.0
        self._ema_initialized = False
        self._next_step_estimate = 0
        # Announced imminent allocations, request id -> expected blocks.
        # While non-empty, the danger depth floors at their sum.
        self._announced_blocks: dict[str, int] = {}
        # Adaptive danger floor: extra depth demanded by measured loss.
        # Raised when a drain interval lost operations to eviction, decayed
        # while intervals stay loss-free; 0 while the feature is off
        # (danger_floor_max_blocks == 0) or the workload is quiescent.
        self._danger_floor_blocks = 0.0
        self._dropped_evicted_at_floor_check = 0
        self._recent_step_allocs: deque[int] = deque(maxlen=_RECENT_ALLOC_STEPS)
        # Hold bookkeeping: drains seen, the drain of the last raise, and
        # the smoothed interval between raises (0.0 until two raises have
        # been observed). Decay starts only after the hold expires.
        self._floor_drain_index = 0
        self._floor_last_raise_drain = 0
        self._floor_raise_gap_ema = 0.0
        # Adaptive-degradation controller state (observe_l1_pressure). The
        # controller runs on the pressure-sample heartbeat: each accepted
        # snapshot appends (time, cumulative evicted bytes, cumulative
        # ledger blocks, cumulative admitted ops, cumulative dropped ops)
        # to a sliding history, and the windowed rates that drive the
        # regime machine are read off that history. Eviction totals are
        # normalized against server restarts as they arrive.
        self._pressure_history: deque[tuple[float, int, int, int, int]] = deque()
        self._evicted_norm_total = 0
        self._evicted_last_raw = 0
        self._evicted_baseline_recorded = False
        # Volume ledger: blocks that left the backlog, split by fate.
        # Deferral is volume-neutral only when what it loses is charged
        # against it, so the controller's rates read their sum.
        self._emitted_blocks_total = 0
        self._lost_blocks_total = 0
        self._regime = _DegradeRegime.NORMAL
        self._regime_entered_at = 0.0
        self._regime_entered_volume = 0
        self._trial_baseline_rate = 0.0
        self._last_probe_at = 0.0
        self._probe_failures = 0
        self._cooldown_until = 0.0
        self._counters = LazyOffloadCounters()

    def covered_prefix_tokens(
        self,
        cache_salt: str,
        allocated_block_ids: dict[int, list[int]],
        group_tokens_per_block: list[int],
        tokens_per_chunk: int,
        from_tokens: int,
    ) -> int:
        """Length of the request prefix a live pending operation already stages.

        A request whose prefix is served from vLLM's prefix cache learns
        nothing from its LMCache lookup about content that is merely
        *deferred*: the covering operation is still buffered here, the
        server has never been told about it, so the lookup misses and the
        request would stage the whole shared prefix again from
        ``from_tokens``. The redundant range is filtered out again further
        down (the server reserves new keys only), but only after the
        request has paid for hashing it, one reservation round-trip per
        chunk, and one oversized atomic store on the transfer thread that
        retrieves share. Answering it here is what keeps a deferred store
        from costing the *next* request anything.

        Blocks are walked from ``from_tokens`` rather than from zero: the
        caller's watermark never moves backwards, so a request pays for
        each of its blocks at most once across its whole lifetime instead
        of once per scheduler step.

        Args:
            cache_salt: The asking request's cache salt.
            allocated_block_ids: Per-engine-group GPU block ids in prefix
                order, as tracked by the caller.
            group_tokens_per_block: Tokens covered by one block of each
                engine group. Must be non-empty for a positive answer.
            tokens_per_chunk: LMCache chunk size; the result is floored to
                it because a store range must be chunk-aligned.
            from_tokens: Prefix length already accounted for by the caller.
                Blocks below it are not probed.

        Returns:
            A chunk-aligned prefix length, at least ``from_tokens``, that a
            pending operation covers in every engine group. Equal to
            ``from_tokens`` when nothing further is covered.

        Raises:
            ValueError: If ``tokens_per_chunk`` is not positive or
                ``from_tokens`` is negative.
        """
        if tokens_per_chunk <= 0:
            raise ValueError(f"tokens_per_chunk must be > 0, got {tokens_per_chunk}")
        if from_tokens < 0:
            raise ValueError(f"from_tokens must be >= 0, got {from_tokens}")
        if not group_tokens_per_block or not self._pending_ops:
            return from_tokens
        covered_tokens = -1
        for engine_group_idx, tokens_per_block in enumerate(group_tokens_per_block):
            block_ids = allocated_block_ids.get(engine_group_idx, [])
            # Blocks below the caller's watermark are already accounted for;
            # a partially covered block cannot extend the run either way.
            index = from_tokens // tokens_per_block
            group_tokens = index * tokens_per_block
            while index < len(block_ids):
                block_id = block_ids[index]
                block_hash = self._pool.block_hash(block_id)
                self._counters.covered_blocks_probed += 1
                if block_hash is None or not self._pending_ops.covers_block(
                    cache_salt, block_id, block_hash
                ):
                    break
                index += 1
                group_tokens += tokens_per_block
            covered_tokens = (
                group_tokens
                if covered_tokens < 0
                else min(covered_tokens, group_tokens)
            )
            if covered_tokens <= from_tokens:
                return from_tokens
        aligned = covered_tokens // tokens_per_chunk * tokens_per_chunk
        if aligned <= from_tokens:
            return from_tokens
        self._counters.covered_prefix_advances += 1
        self._counters.covered_prefix_tokens_skipped += aligned - from_tokens
        return aligned

    def admit(self, op: PendingStoreOp) -> AdmitResult:
        """Admit a store operation into the pending queue.

        Args:
            op: The operation to buffer. ``op.block_hashes`` must cover every
                GPU block of the operation's token range.

        Returns:
            The admission outcome; see :class:`AdmitResult` for the action
            the caller must take on each value.
        """
        existing = self._held_short.get(op.request_id) or self._pending_ops.get(
            op.request_id
        )
        if existing is not None and existing[0].epoch != op.epoch:
            raise RuntimeError(
                f"request {op.request_id!r} mixed store epochs "
                f"{existing[0].epoch} and {op.epoch}"
            )
        if op.request_id in self._broken_prefixes:
            self._counters.rejected_prefix_broken += 1
            return AdmitResult.REJECTED_PREFIX_BROKEN
        if any(block_hash is None for block_hash in op.block_hashes.values()):
            # The caller's tracker has already advanced past this range, so
            # the request's later chunks would be stored without their prefix
            # (unreachable): reject them like any other broken chain.
            self._broken_prefixes.add(op.request_id)
            self._counters.rejected_unhashed += 1
            return AdmitResult.REJECTED_UNHASHED_BLOCK
        if op.prefix_end_tokens < self._config.min_prefix_tokens:
            # Gate 3 (economy): the chain is below break-even. Hold it out
            # of the pending machine until the request grows past the
            # threshold or finishes below it. Counted admitted now so the
            # ledger closes: admitted == pending + held + emitted + drops.
            self._held_short.setdefault(op.request_id, []).append(op)
            self._counters.admitted += 1
            return AdmitResult.ADMITTED
        if op.request_id in self._held_short and not self._promote_held(op.request_id):
            # A held block was recycled while the chain waited below the
            # threshold: stored without that prefix, this op would be
            # unreachable.
            self._counters.rejected_prefix_broken += 1
            return AdmitResult.REJECTED_PREFIX_BROKEN
        covering = self._pending_ops.covering_op(op)
        if covering is not None and self._chain_intact(covering):
            self._counters.deduplicated += 1
            return AdmitResult.DEDUPLICATED
        # No covering op, or it is doomed (its own blocks were recycled, or
        # an earlier sibling's were, so the next drain prefix-closes over
        # it): buffer the live copy and make it the new cover. The doomed
        # op stays pending and is dropped by collect_due().
        self._pending_ops.add(op)
        self._counters.admitted += 1
        return AdmitResult.ADMITTED

    def _promote_held(self, request_id: str) -> bool:
        """Move the request's held chain into the pending machine.

        Held ops are invisible to the per-step loss check, so an eviction
        that recycled one of their blocks was never observed: the chain is
        validated here instead. A lost block kills the whole chain, not
        just its suffix -- the intact front is still below the break-even
        length (every held op is) and the break stops it from ever growing
        past it, so promoting it would only recreate the sub-break-even
        queue residents this gate exists to remove.

        Promoted ops skip the content-deduplication check on purpose: they
        entered the admission ledger when they were held, and deduplicating
        one away here would leave that entry matched by neither a pending
        op nor a drop counter. The cost is a rare duplicate store when
        another request buffered identical content during the wait --
        wasted device-to-host bandwidth, never corruption.

        Args:
            request_id: Request whose held chain must move. The caller has
                already checked that one exists.

        Returns:
            False when the chain lost a block while held: nothing is
            promoted, the chain is dropped, and further admissions of this
            request are rejected.
        """
        held = self._held_short.pop(request_id)
        first_lost = len(held)
        for index, held_op in enumerate(held):
            if not self._snapshot_intact(held_op):
                first_lost = index
                break
        if first_lost < len(held):
            # Ledger split mirrors _drop_evicted_suffix: ops from the first
            # lost one on are gate-1 losses; the intact front is a gate-3
            # rejection (it can never reach break-even now).
            self._counters.rejected_short_prefix += first_lost
            self._counters.dropped_evicted += len(held) - first_lost
            self._note_lost(held[first_lost:])
            self._broken_prefixes.add(request_id)
            logger.info(
                "Lazy offload: dropped %d held store op(s) of request %s: "
                "a block was evicted while the chain waited below the "
                "break-even prefix length",
                len(held),
                request_id,
            )
            return False
        for held_op in held:
            self._pending_ops.add(held_op)
        return True

    def observe_step(
        self,
        new_blocks_allocated: int,
        est_next_step_blocks: int,
        allocated_block_ids: set[int] | None = None,
    ) -> None:
        """Record one scheduler step's block-consumption signals.

        Must be called once per step, before :meth:`collect_due`.

        Args:
            new_blocks_allocated: GPU blocks newly allocated in the step
                that just finished scheduling (gross allocation, counted
                from the scheduler output).
            est_next_step_blocks: Estimated blocks the next step will
                allocate (e.g. scheduled tokens divided by block size).
            allocated_block_ids: Block ids allocated or resurrected in this
                step. Requests indexed by these ids are revalidated during
                the drain. None asks for a full validation pass and is kept
                for callers that cannot provide the incremental signal.
        """
        self._pending_ops.observe_allocations(allocated_block_ids)
        self._recent_step_allocs.append(new_blocks_allocated)
        if self._ema_initialized:
            self._blocks_per_step_ema = (
                _EMA_ALPHA * new_blocks_allocated
                + (1 - _EMA_ALPHA) * self._blocks_per_step_ema
            )
        else:
            self._blocks_per_step_ema = float(new_blocks_allocated)
            self._ema_initialized = True
        self._next_step_estimate = est_next_step_blocks

    def announce_allocation(self, request_id: str, num_blocks: int) -> None:
        """Widen the danger window ahead of an announced allocation burst.

        The per-step forecast is blind to a single admission that consumes
        thousands of blocks at once: the allocation that pays for the
        forecast and the eviction that destroys a waiting operation are
        the same event (see
        :attr:`LazyOffloadPolicyConfig.max_pending_ops`). An announcement
        closes that gap from the outside: while any announcement is
        outstanding, the danger depth is at least the sum of announced
        block counts, so the drain running ahead of the burst emits the
        operations the burst would otherwise destroy. Re-announcing a
        request replaces its previous width.

        Args:
            request_id: Request whose imminent admission is announced.
            num_blocks: Blocks that admission is expected to consume.

        Raises:
            ValueError: If ``num_blocks`` is not positive.
        """
        if num_blocks <= 0:
            raise ValueError(f"num_blocks must be positive, got {num_blocks}")
        if request_id not in self._announced_blocks:
            self._counters.announced_bursts += 1
        self._announced_blocks[request_id] = num_blocks

    def retract_allocation(self, request_id: str) -> None:
        """Withdraw a request's announced allocation, if any.

        Called when the announced allocation has landed (the request was
        scheduled) or the request left the system before being scheduled.
        Unknown request ids are a no-op on purpose: retraction is wired
        into paths that see every request, announced or not.

        Args:
            request_id: Request whose announcement is withdrawn.
        """
        self._announced_blocks.pop(request_id, None)

    def observe_l1_pressure(
        self,
        monotonic_time: float,
        capacity_bytes: int,
        evicted_bytes_total: int,
    ) -> None:
        """Record one L1 pressure snapshot and run the degradation controller.

        The controller runs on this sample heartbeat. Each strictly newer
        snapshot appends (time, cumulative evicted bytes, cumulative ledger
        blocks) to a sliding history; the eviction byte rate over that
        window -- a windowed rate, never a per-sample EMA, because eviction
        arrives in bursts and per-sample smoothing flaps across any
        threshold -- gives the residence estimate ``capacity / rate``
        compared against
        :attr:`LazyOffloadPolicyConfig.degrade_l1_residence_secs`.

        The regime machine enforces the controller's invariant -- degrading
        may change the timing of stores, never their volume. NORMAL defers;
        a TRIAL opens (outside any cooldown) when the policy's own loss
        ledger shows deferral destroying a material share of its intake --
        windowed dropped operations reaching ``_MATERIAL_LOSS_SHARE`` of
        windowed admissions -- or, where the residence threshold is
        configured, when residence sits under it. The loss trigger needs no
        configuration and is always live: losing intake is the one way
        deferral can be strictly worse than storing eagerly, so it is
        guarded by default. The trigger may be eager because the trial
        bounds the cost of a false alarm.

        TRIAL emits immediately for a bounded window and commits to
        DEGRADED only if its ledger rate stayed within the neutrality
        factor of the deferred baseline; a volume increase means deferral
        was filtering stores out, so the trial reverts and enters a
        cooldown. The ledger counts emitted blocks plus blocks lost to
        eviction (:meth:`_volume_blocks_total`), which is what makes the
        comparison honest: measured on emissions alone, deferral that
        bleeds its backlog looks exactly like deferral that usefully
        filtered it, and the trial that would have fixed the bleed reverts.

        DEGRADED emits immediately and recovers only through a PROBE -- a
        bounded deferred window, run every probe interval and armed early
        (at a shorter retry spacing) when residence recovers past the
        hysteresis factor -- returning to NORMAL when the probe's ledger
        rate drops enough to show deferral is filtering volume again. Each
        consecutive failed probe backs the spacing off geometrically to
        ``_PROBE_BACKOFF_MAX``, bounding what a permanently unfavourable
        workload pays to keep asking. The estimate alone never lifts a
        committed degradation: bursts spacing out past the window read as
        infinite residence, and acting on that would hand the deferred
        backlog to the next burst. The volume rates compare this policy
        against itself, so a trial or probe needs no server-side
        counterfactual.

        Callers may repeat the latest snapshot every step: only a strictly
        newer ``monotonic_time`` advances the controller. A cumulative
        counter that moved backwards (server restart) contributes a zero
        delta instead of a negative rate. With the residence threshold at 0
        the residence trigger is inert and only the loss trigger can open a
        trial.

        Args:
            monotonic_time: When the snapshot was taken, on the caller's
                monotonic clock.
            capacity_bytes: The server's L1 capacity. Zero marks the
                snapshot invalid and it is ignored.
            evicted_bytes_total: The server's cumulative deleted-bytes
                counter at that time.
        """
        if capacity_bytes <= 0:
            return
        if not self._evicted_baseline_recorded:
            self._evicted_last_raw = evicted_bytes_total
            self._evicted_baseline_recorded = True
            self._pressure_history.append(
                (
                    monotonic_time,
                    0,
                    self._volume_blocks_total,
                    self._counters.admitted,
                    self._counters.dropped_evicted,
                )
            )
            return
        if monotonic_time <= self._pressure_history[-1][0]:
            return
        delta = evicted_bytes_total - self._evicted_last_raw
        self._evicted_last_raw = evicted_bytes_total
        if delta < 0:
            # The cumulative counter moved backwards: the server restarted.
            # This sample re-baselines and contributes no evictions.
            delta = 0
        self._evicted_norm_total += delta
        self._pressure_history.append(
            (
                monotonic_time,
                self._evicted_norm_total,
                self._volume_blocks_total,
                self._counters.admitted,
                self._counters.dropped_evicted,
            )
        )
        while (
            len(self._pressure_history) > 2
            and monotonic_time - self._pressure_history[1][0] >= _PRESSURE_WINDOW_SECS
        ):
            self._pressure_history.popleft()
        threshold = self._config.degrade_l1_residence_secs
        residence = self._residence_estimate(capacity_bytes)
        if self._regime is _DegradeRegime.NORMAL:
            baseline = self._trailing_volume_rate(monotonic_time)
            churning = threshold > 0 and residence < threshold
            if (
                (churning or self._loss_is_material(monotonic_time))
                and monotonic_time >= self._cooldown_until
                and baseline is not None
            ):
                self._trial_baseline_rate = baseline
                self._enter_regime(_DegradeRegime.TRIAL, monotonic_time)
                self._counters.degrade_trials += 1
                self._counters.degrade_transitions += 1
        elif self._regime is _DegradeRegime.TRIAL:
            if monotonic_time - self._regime_entered_at >= _TRIAL_SECS:
                trial_rate = self._regime_volume_rate(monotonic_time)
                if trial_rate <= _NEUTRALITY_FACTOR * self._trial_baseline_rate:
                    self._enter_regime(_DegradeRegime.DEGRADED, monotonic_time)
                    self._last_probe_at = monotonic_time
                    self._counters.degrade_commits += 1
                else:
                    self._enter_regime(_DegradeRegime.NORMAL, monotonic_time)
                    self._cooldown_until = monotonic_time + _REVERT_COOLDOWN_SECS
                    self._counters.degrade_reverts += 1
                    self._counters.degrade_transitions += 1
        elif self._regime is _DegradeRegime.DEGRADED:
            since_probe = monotonic_time - self._last_probe_at
            backoff = min(2**self._probe_failures, _PROBE_BACKOFF_MAX)
            recovered = (
                threshold > 0 and residence >= _RESIDENCE_RECOVERY_FACTOR * threshold
            )
            if since_probe >= _PROBE_INTERVAL_SECS * backoff or (
                recovered and since_probe >= _PROBE_RETRY_MIN_SECS * backoff
            ):
                baseline = self._trailing_volume_rate(monotonic_time)
                self._trial_baseline_rate = baseline if baseline is not None else 0.0
                self._enter_regime(_DegradeRegime.PROBE, monotonic_time)
                self._counters.degrade_probes += 1
                self._counters.degrade_transitions += 1
        else:
            if monotonic_time - self._regime_entered_at >= _TRIAL_SECS:
                probe_rate = self._regime_volume_rate(monotonic_time)
                if probe_rate * _NEUTRALITY_FACTOR <= self._trial_baseline_rate:
                    self._enter_regime(_DegradeRegime.NORMAL, monotonic_time)
                    self._cooldown_until = monotonic_time + _REVERT_COOLDOWN_SECS
                    self._probe_failures = 0
                    self._counters.degrade_probe_recoveries += 1
                else:
                    self._enter_regime(_DegradeRegime.DEGRADED, monotonic_time)
                    self._last_probe_at = monotonic_time
                    self._probe_failures += 1
                    self._counters.degrade_transitions += 1

    def _residence_estimate(self, capacity_bytes: int) -> float:
        """L1 residence implied by the windowed eviction rate.

        Args:
            capacity_bytes: The server's L1 capacity.

        Returns:
            ``capacity / rate`` over the pressure-history window, or
            infinity while the history spans less than the minimum or shows
            no eviction at all.
        """
        first = self._pressure_history[0]
        last = self._pressure_history[-1]
        span = last[0] - first[0]
        if span < _PRESSURE_MIN_SPAN_SECS:
            return math.inf
        rate = (last[1] - first[1]) / span
        if rate <= 0:
            return math.inf
        return capacity_bytes / rate

    @property
    def _volume_blocks_total(self) -> int:
        """Blocks this policy has taken out of the backlog, delivered or not.

        Emitted blocks plus blocks lost to eviction while deferred. The
        controller compares regimes on this sum rather than on emissions
        alone: deferral that stores less because a later op superseded an
        earlier one is filtering volume (a benefit worth keeping), while
        deferral that stores less because its backlog was evicted is
        destroying volume (the defect the controller exists to stop), and
        the two are indistinguishable in the emission count.
        """
        return self._emitted_blocks_total + self._lost_blocks_total

    def _trailing_volume_rate(self, now: float) -> float | None:
        """Ledger-block rate over the trailing trial-length window.

        The deferred (or degraded) baseline a trial (or probe) is judged
        against, read off the pressure history.

        Args:
            now: The current sample's monotonic time.

        Returns:
            Blocks per second over the last ``_TRIAL_SECS`` (or the whole
            history if shorter), or None while the history spans no time.
        """
        cutoff = now - _TRIAL_SECS
        start_time = now
        start_volume = self._volume_blocks_total
        for time, _, volume, _, _ in reversed(self._pressure_history):
            start_time = time
            start_volume = volume
            if time <= cutoff:
                break
        span = now - start_time
        if span <= 0:
            return None
        return (self._volume_blocks_total - start_volume) / span

    def _loss_is_material(self, now: float) -> bool:
        """Whether deferral's own eviction losses justify a trial.

        Admitted and dropped operation counts over the trailing
        trial-length window, read off the pressure history against the
        live counters. Losing at least ``_MATERIAL_LOSS_SHARE`` of the
        windowed intake -- with at least one loss -- is direct evidence
        that deferral is destroying its backlog rather than re-timing it,
        available well before the windowed residence estimate can cross
        the threshold.

        While the adaptive danger floor is enabled and below its cap, the
        gate stands down: the floor is the graduated response to the same
        loss (widen the window, stay deferred), and opening a trial on the
        loss that raised it would flip the run to immediate emission
        before the response it triggered has been measured. The loss stays
        in the trailing window, so if it continues once the floor is at
        its cap -- the burst size exceeds what the cap can absorb -- the
        gate opens the trial as before. At cap 0 (floor disabled) the gate
        is unconditional, exactly as it was before the floor existed.

        Args:
            now: The current sample's monotonic time.

        Returns:
            True when the windowed loss share is material.
        """
        floor_cap = self._config.danger_floor_max_blocks
        if floor_cap > 0 and self._danger_floor_blocks < float(floor_cap):
            return False
        cutoff = now - _TRIAL_SECS
        start_admitted = self._counters.admitted
        start_dropped = self._counters.dropped_evicted
        for time, _, _, admitted, dropped in reversed(self._pressure_history):
            start_admitted = admitted
            start_dropped = dropped
            if time <= cutoff:
                break
        dropped = self._counters.dropped_evicted - start_dropped
        if dropped <= 0:
            return False
        admitted = self._counters.admitted - start_admitted
        return dropped >= _MATERIAL_LOSS_SHARE * admitted

    def _regime_volume_rate(self, now: float) -> float:
        """Ledger-block rate since the current regime was entered.

        Args:
            now: The current sample's monotonic time, after the regime has
                lasted at least one sample interval.

        Returns:
            Blocks per second across the regime's lifetime so far.
        """
        span = now - self._regime_entered_at
        volume = self._volume_blocks_total - self._regime_entered_volume
        return volume / span

    def _enter_regime(self, regime: "_DegradeRegime", now: float) -> None:
        """Switch regimes, snapshotting the volume ledger.

        Args:
            regime: The regime to enter.
            now: The current sample's monotonic time.
        """
        self._regime = regime
        self._regime_entered_at = now
        self._regime_entered_volume = self._volume_blocks_total

    def _note_emitted(self, ops: list[PendingStoreOp]) -> None:
        """Advance the delivered half of the volume ledger.

        Args:
            ops: Operations just emitted by any drain path.
        """
        self._emitted_blocks_total += sum(len(op.block_hashes) for op in ops)

    def _note_lost(self, ops: list[PendingStoreOp]) -> None:
        """Advance the destroyed half of the volume ledger.

        Charged against the deferred regime that was holding the
        operations, so a trial of immediate emission is measured against
        what deferral actually took in rather than against what survived
        it.

        Args:
            ops: Operations just dropped because a covered block was
                evicted while they waited.
        """
        self._lost_blocks_total += sum(len(op.block_hashes) for op in ops)

    @property
    def degraded(self) -> bool:
        """Whether the queue currently emits immediately instead of deferring."""
        return self._regime in (_DegradeRegime.TRIAL, _DegradeRegime.DEGRADED)

    def has_pending_request(self, request_id: str) -> bool:
        """Whether this request currently owns buffered or held operations."""
        return (
            self._pending_ops.contains_request(request_id)
            or request_id in self._held_short
        )

    def drop_request(self, request_id: str) -> int:
        """Discard buffered and held operations invalidated by a tracker reset."""
        dropped = self._pending_ops.pop_request(request_id)
        dropped += self._held_short.pop(request_id, [])
        self._broken_prefixes.discard(request_id)
        self._counters.dropped_on_request_drop += len(dropped)
        return len(dropped)

    def discard_for_reuse(self, request_id: str) -> int:
        """Discard a finished predecessor's buffered policy state."""
        dropped = self._pending_ops.pop_request(request_id)
        dropped += self._held_short.pop(request_id, [])
        self._broken_prefixes.discard(request_id)
        self._counters.dropped_id_reuse += len(dropped)
        return len(dropped)

    def release_request(self, request_id: str) -> None:
        """Forget non-pending policy state after current-session teardown."""
        self._broken_prefixes.discard(request_id)

    def mark_store_failed(self, request_id: str) -> int:
        """Record that the request's in-flight store batch failed.

        The request's stored prefix chain is broken: its held-back pending
        operations are dropped (stored without the failed prefix they would
        be unreachable) and further admissions are rejected. Receipt and
        request lifecycle remain entirely controller-owned.

        The controller only calls this method for a batch from the current
        store epoch. Failures from an epoch made stale by reset or id reuse
        are filtered there because they do not break the current prefix.

        Args:
            request_id: The request whose store failed.

        Returns:
            The number of pending operations dropped.
        """
        dropped = self._pending_ops.pop_request(request_id)
        # A held chain cannot coexist with an in-flight batch (nothing emits
        # before promotion), but a departure path that forgot the pen would
        # leak it, so it is cleared here too.
        dropped += self._held_short.pop(request_id, [])
        self._counters.dropped_failed_store += len(dropped)
        self._broken_prefixes.add(request_id)
        return len(dropped)

    def num_pending_ops(self) -> int:
        """Return the total number of buffered store operations."""
        return self._pending_ops.num_ops()

    def num_held_ops(self) -> int:
        """Return the number of operations held below the break-even length."""
        return sum(len(ops) for ops in self._held_short.values())

    def stats(self) -> LazyOffloadCounters:
        """Return a copy of the cumulative policy counters."""
        return replace(self._counters)

    def collect_due(
        self,
        blocked_request_ids: set[str] | None = None,
        finished_request_ids: set[str] | None = None,
    ) -> DrainResult:
        """Release the operations whose blocks face imminent eviction.

        For every pending request, first drops the suffix of its operation
        list starting at the first operation whose data is already lost
        (current block hash differs from the admission snapshot) -- storing
        a later chunk without its prefix would be unreachable. Then, if any
        surviving operation has a block within the danger depth of the free
        queue, the request's operations are released from the front up to
        the last due one (prefix closure), subject to gate 3. The released
        segment is additionally cut at the first deduplication hole: the
        batch is coalesced into one contiguous store operation, so ops past
        the hole wait for a later batch.

        Emitting a segment pins its blocks out of the free queue, which
        moves every block behind them toward the head by the segment's size
        before the next step's allocation runs. Each candidate is therefore
        checked against ``danger_depth`` extended by the blocks emitted
        earlier in this call, so a request that an emission teleports into
        the danger window drains now instead of losing the race to the next
        allocation. The first emission still requires a plain
        ``danger_depth`` hit: an idle system never starts draining.

        The free-queue read follows that threshold instead of anticipating
        it: the window opens at the danger depth and widens only by a shift
        an emission has already caused, so a step reads the ranks its
        decisions compare rather than the ranks a full-budget drain could
        have compared. Whether an emitted block shifts the queue is asked of
        the pool directly, not of the window, so a pin deeper than the
        window still counts and the widening cannot stall behind itself.

        After the pressure pass, budget left over goes to the backlog cap
        (``max_pending_ops``) and then, when the step's allocation rate is
        at or below ``idle_threshold_blocks``, to idle draining: up to
        ``idle_drain_max_ops`` of the oldest operations are emitted so the
        backlog is worked off in the gaps between bursts instead of in
        phase with them. All paths spend from one budget, capped in
        operations (``max_drain_per_step``) and, when configured, in
        blocks (``max_drain_blocks_per_step``).

        Args:
            blocked_request_ids: Requests that already have a store batch in
                flight. They are left pending, and any validation this
                step's allocations asked for stays pending with them.
            finished_request_ids: Requests the controller has seen finish
                generation. A finished request's known prefix can no longer
                grow, so a chain still held below the gate-3 break-even
                length is dropped here (pending operations of finished
                requests are untouched: waiting out their eviction clock is
                the point of lazy offload).

        Returns:
            The operations to store and to drop this step; see
            :class:`DrainResult`.
        """
        # Settle the loss delta since the previous drain first: a raise must
        # widen this very drain's window (the burst that caused the loss may
        # still be running), and the decay must tick even on the early
        # returns below or a raised floor would outlive an emptied queue.
        self._update_danger_floor()
        result = DrainResult()
        for request_id in sorted(finished_request_ids or ()):
            # A finished request's announced admission can no longer land.
            self._announced_blocks.pop(request_id, None)
            held = self._held_short.pop(request_id, None)
            if held is None:
                continue
            # Gate 3: the request finished below the break-even length.
            # Held and pending are mutually exclusive, so the request is
            # emptied by this drop and its session becomes releasable.
            result.dropped_short_prefix.extend(held)
            result.emptied_requests.append(request_id)
            self._counters.rejected_short_prefix += len(held)
            self._broken_prefixes.add(request_id)
        blocked_request_ids = blocked_request_ids or set()
        if not self._pending_ops:
            return result
        self._counters.drain_steps += 1

        danger_depth = self._danger_depth()
        # A zero danger depth makes nothing due, so the free queue is not
        # walked at all -- the loss check below reads block hashes, not
        # ranks, and still runs.
        window = _FreeQueueWindow(
            self._pool.free_queue_block_ids() if danger_depth > 0 else iter(())
        )
        window.extend_to(danger_depth)

        # Requests due now, ascending by eviction imminence, and the cursor
        # of the first one this drain has not decided yet.
        candidates: list[tuple[int, str, list[PendingStoreOp]]] = []
        cursor = 0
        candidate_ids: set[str] = set()
        # Survivors of the loss check, kept so that a request the window
        # reveals later is not validated a second time in the same step.
        surviving_by_request: dict[str, list[PendingStoreOp]] = {}

        def discover(request_ids: set[str]) -> None:
            """Validate these requests and queue the ones now due.

            Args:
                request_ids: Requests whose outcome may have changed --
                    touched by this step's allocations, or holding a block
                    the window just revealed. Ones already queued as
                    candidates are skipped.
            """
            fresh: list[tuple[int, str, list[PendingStoreOp]]] = []
            for request_id in request_ids:
                if request_id in candidate_ids:
                    continue
                surviving = surviving_by_request.get(request_id)
                if surviving is None:
                    ops = self._pending_ops.get(request_id)
                    if not ops:
                        self._pending_ops.validation_complete(request_id)
                        surviving_by_request[request_id] = []
                        continue
                    if request_id in blocked_request_ids:
                        # One in-flight store batch per request (worker
                        # constraint). Keep an allocation-triggered
                        # validation pending: after the receipt, the
                        # held-back ops still need their snapshots checked
                        # even if their recycled blocks are no longer free.
                        continue
                    surviving = self._drop_evicted_suffix(request_id, ops, result)
                    self._pending_ops.validation_complete(request_id)
                    surviving_by_request[request_id] = surviving
                if not surviving:
                    continue
                op_ranks = [
                    rank
                    for op in surviving
                    for block_id in op.block_hashes
                    if (rank := window.ranks.get(block_id)) is not None
                ]
                if not op_ranks:
                    # No block inside the window: the request cannot be due
                    # yet. A later widening can still bring one into view.
                    continue
                fresh.append((min(op_ranks), request_id, surviving))
                candidate_ids.add(request_id)
            if not fresh:
                return
            # Most imminent first. Sorting only the undecided tail keeps the
            # order the emission loop relies on as the window widens.
            candidates.extend(fresh)
            candidates[cursor:] = sorted(
                candidates[cursor:],
                key=lambda cand: (
                    cand[0],
                    self._pending_ops.admission_order(cand[1]),
                ),
            )

        # Only requests touched by this step's allocations or represented in
        # the window can have changed outcome. The reverse index avoids a
        # full pending-queue scan on every scheduler step.
        discover(self._pending_ops.requests_to_check(window.ranks))

        budget = _DrainBudget(
            self._config.max_drain_per_step,
            self._config.max_drain_blocks_per_step,
        )
        # Blocks this drain has pinned out of the free queue, and so the
        # distance every block behind them moves toward the head. A shared
        # block shifts the queue only on its first pin; a block that was
        # already out of the queue does not shift it at all.
        shift_blocks = 0
        pinned_free_blocks: set[int] = set()
        # Requests this drain has already emitted for. The pressure loop
        # visits each request at most once, but the backlog drain iterates
        # independently and must not put a second batch in flight.
        emitted_request_ids: set[str] = set()
        while not budget.exhausted():
            threshold = danger_depth + shift_blocks
            if window.depth() < threshold:
                revealed = window.extend_to(threshold)
                if revealed:
                    discover(self._pending_ops.requests_for_blocks(revealed))
            if cursor >= len(candidates):
                break
            min_rank, request_id, surviving = candidates[cursor]
            if min_rank >= threshold:
                # Candidates are rank-ordered and the threshold only grows
                # with emissions, so no later candidate can be due either.
                break
            cursor += 1
            segment = self._due_front_segment(surviving, window.ranks, threshold)
            if segment is None:
                continue
            _, due_ops = segment
            # Never emit across a deduplication hole: the batch is coalesced
            # into one contiguous store. The request keeps its due urgency;
            # the post-hole ops follow in a later batch.
            due_ops = _contiguous_front_run(due_ops)
            if self._fails_economy_gate(surviving):
                # Gate 3: the whole known prefix is below break-even. The due
                # front is about to die, which breaks the prefix chain for
                # the rest -- drop everything, not just the due segment.
                # Dropped blocks stay in the free queue, so they do not
                # extend the emission shift.
                result.dropped_short_prefix.extend(surviving)
                self._counters.rejected_short_prefix += len(surviving)
                self._broken_prefixes.add(request_id)
                self._replace_pending(request_id, surviving, [], result)
                continue
            emitted = budget.take(due_ops)
            result.ops_held_back += len(due_ops) - len(emitted)
            result.to_store.extend(emitted)
            self._counters.emitted += len(emitted)
            self._note_emitted(emitted)
            newly_pinned = {
                block_id
                for op in emitted
                for block_id in op.block_hashes
                if block_id not in pinned_free_blocks and self._pool.is_free(block_id)
            }
            pinned_free_blocks.update(newly_pinned)
            shift_blocks += len(newly_pinned)
            remaining = surviving[len(emitted) :]
            self._replace_pending(request_id, emitted, remaining, result)
            emitted_request_ids.add(request_id)
        self._counters.free_queue_blocks_read += window.depth()
        skip_request_ids = blocked_request_ids | emitted_request_ids
        if self.degraded:
            # Immediate-emission regime: flush everything pending instead of
            # working the backlog and idle policies -- both are subsumed.
            if not budget.exhausted():
                self._drain_degraded(
                    budget, skip_request_ids, surviving_by_request, result
                )
        else:
            if not budget.exhausted() and self._config.max_pending_ops > 0:
                self._drain_backlog(
                    budget, skip_request_ids, surviving_by_request, result
                )
            if (
                not budget.exhausted()
                and self._config.idle_drain_max_ops > 0
                and self._step_is_idle()
            ):
                self._drain_idle(budget, skip_request_ids, surviving_by_request, result)
        if result.ops_held_back:
            self._counters.throttled_drains += 1
        return result

    def _drain_backlog(
        self,
        budget: _DrainBudget,
        skip_request_ids: set[str],
        surviving_by_request: dict[str, list[PendingStoreOp]],
        result: DrainResult,
    ) -> None:
        """Emit the oldest operations while the backlog exceeds its cap.

        The danger depth forecasts eviction from an EMA of per-step
        allocation, so one admission that consumes thousands of blocks --
        a large external cache hit, whose blocks vLLM allocates in a single
        step -- destroys waiting operations before any forecast built from
        the preceding steps could have widened to cover them. The forecast
        cannot be fixed by looking further ahead, because the burst *is*
        the step it would have to predict; what can be bounded is how much
        content is exposed to one. This drain does that, emitting the
        longest-waiting operations regardless of their free-queue rank until
        the backlog is back at ``max_pending_ops``.

        Requests are taken in admission order and each contributes the
        contiguous front run of its surviving operations, so prefix closure
        and the one-batch-per-request constraint hold exactly as they do for
        a pressure-driven emission.

        Args:
            budget: The drain's shared emission budget, not yet exhausted.
            skip_request_ids: Requests that must not emit -- ones with a
                batch already in flight, and ones this drain already
                emitted for. Requests this method emits for are added, so
                a later emission path cannot put a second batch of theirs
                in flight.
            surviving_by_request: Loss-check results this drain already
                computed, extended in place for requests it reaches first.
                Reusing it keeps a request's snapshots validated once per
                step.
            result: The drain result to extend with emissions and drops.
        """
        cap = self._config.max_pending_ops
        for request_id in self._pending_ops.requests_in_admission_order():
            if budget.exhausted() or self._pending_ops.num_ops() <= cap:
                return
            if request_id in skip_request_ids:
                continue
            surviving = surviving_by_request.get(request_id)
            if surviving is None:
                ops = self._pending_ops.get(request_id)
                if not ops:
                    continue
                surviving = self._drop_evicted_suffix(request_id, ops, result)
                self._pending_ops.validation_complete(request_id)
                surviving_by_request[request_id] = surviving
            if not surviving:
                continue
            if self._fails_economy_gate(surviving):
                # Same backstop as the pressure path: a chain eviction
                # truncated back below break-even is not worth storing.
                result.dropped_short_prefix.extend(surviving)
                self._counters.rejected_short_prefix += len(surviving)
                self._broken_prefixes.add(request_id)
                self._replace_pending(request_id, surviving, [], result)
                continue
            # Emit only what the overflow calls for: the cap is a bound on
            # the backlog, not an instruction to empty it.
            overflow = self._pending_ops.num_ops() - cap
            due_ops = _contiguous_front_run(surviving)
            emitted = budget.take(due_ops[:overflow])
            if not emitted:
                continue
            result.to_store.extend(emitted)
            self._counters.emitted += len(emitted)
            self._counters.backlog_emitted += len(emitted)
            self._note_emitted(emitted)
            self._replace_pending(
                request_id, emitted, surviving[len(emitted) :], result
            )
            skip_request_ids.add(request_id)

    def _drain_idle(
        self,
        budget: "_DrainBudget",
        skip_request_ids: set[str],
        surviving_by_request: dict[str, list[PendingStoreOp]],
        result: DrainResult,
    ) -> None:
        """Emit the oldest waiting operations while the step is idle.

        Eviction pressure times an emission to the moment its blocks are
        about to be reallocated -- which is exactly when a prefill burst
        is allocating, so the D2H copy lands in phase with the burst it
        waited for. An idle step is the opposite moment: nothing is being
        allocated, writing waiting content down costs the step nothing,
        and every operation emitted now is one a later burst does not have
        to flush in phase with itself.

        Requests are taken in admission order and each contributes the
        contiguous front run of its surviving operations, exactly as in
        the backlog drain: prefix closure and one batch per request hold
        unchanged. What an idle emission gives up is filtering -- content
        evicted after the emission is stored either way -- which is why
        the per-step allowance exists instead of the idle path emptying
        the backlog outright.

        Args:
            budget: The drain's shared emission budget, not yet exhausted.
            skip_request_ids: Requests that must not emit -- ones with a
                batch already in flight, and ones this drain already
                emitted for. Requests this method emits for are added.
            surviving_by_request: Loss-check results this drain already
                computed, extended in place for requests it reaches first.
                Reusing it keeps a request's snapshots validated once per
                step.
            result: The drain result to extend with emissions and drops.
        """
        allowance = self._config.idle_drain_max_ops
        emitted_any = False
        for request_id in self._pending_ops.requests_in_admission_order():
            if allowance <= 0 or budget.exhausted():
                break
            if request_id in skip_request_ids:
                continue
            surviving = surviving_by_request.get(request_id)
            if surviving is None:
                ops = self._pending_ops.get(request_id)
                if not ops:
                    continue
                surviving = self._drop_evicted_suffix(request_id, ops, result)
                self._pending_ops.validation_complete(request_id)
                surviving_by_request[request_id] = surviving
            if not surviving:
                continue
            if self._fails_economy_gate(surviving):
                # Same backstop as the other paths: a chain eviction
                # truncated back below break-even can never regrow (its
                # prefix is marked broken) and is not worth storing.
                result.dropped_short_prefix.extend(surviving)
                self._counters.rejected_short_prefix += len(surviving)
                self._broken_prefixes.add(request_id)
                self._replace_pending(request_id, surviving, [], result)
                continue
            due_ops = _contiguous_front_run(surviving)
            emitted = budget.take(due_ops[:allowance])
            if not emitted:
                continue
            allowance -= len(emitted)
            result.to_store.extend(emitted)
            self._counters.emitted += len(emitted)
            self._counters.idle_emitted += len(emitted)
            self._note_emitted(emitted)
            self._replace_pending(
                request_id, emitted, surviving[len(emitted) :], result
            )
            skip_request_ids.add(request_id)
            emitted_any = True
        if emitted_any:
            self._counters.idle_drain_steps += 1

    def _drain_degraded(
        self,
        budget: "_DrainBudget",
        skip_request_ids: set[str],
        surviving_by_request: dict[str, list[PendingStoreOp]],
        result: DrainResult,
    ) -> None:
        """Emit every waiting operation: the immediate-emission regime.

        Runs instead of the backlog and idle drains while the L1 residence
        signal says deferral has no dividend (see
        :meth:`observe_l1_pressure`). Requests are taken in admission order
        and each contributes the contiguous front run of its surviving
        operations under the shared per-step budget -- validation, prefix
        closure, the dedup-hole cut, the economy backstop, and one batch
        per request all hold exactly as in the other drain paths. An
        operation admitted while degraded is therefore emitted on the first
        drain after its admission, while its request is still running and
        its blocks are not yet in the free queue, which is what makes the
        regime cost-equivalent to not deferring at all.

        Args:
            budget: The drain's shared emission budget, not yet exhausted.
            skip_request_ids: Requests that must not emit -- ones with a
                batch already in flight, and ones this drain already
                emitted for. Requests this method emits for are added.
            surviving_by_request: Loss-check results this drain already
                computed, extended in place for requests it reaches first.
            result: The drain result to extend with emissions and drops.
        """
        emitted_any = False
        for request_id in self._pending_ops.requests_in_admission_order():
            if budget.exhausted():
                break
            if request_id in skip_request_ids:
                continue
            surviving = surviving_by_request.get(request_id)
            if surviving is None:
                ops = self._pending_ops.get(request_id)
                if not ops:
                    continue
                surviving = self._drop_evicted_suffix(request_id, ops, result)
                self._pending_ops.validation_complete(request_id)
                surviving_by_request[request_id] = surviving
            if not surviving:
                continue
            if self._fails_economy_gate(surviving):
                result.dropped_short_prefix.extend(surviving)
                self._counters.rejected_short_prefix += len(surviving)
                self._broken_prefixes.add(request_id)
                self._replace_pending(request_id, surviving, [], result)
                continue
            due_ops = _contiguous_front_run(surviving)
            emitted = budget.take(due_ops)
            result.ops_held_back += len(due_ops) - len(emitted)
            if not emitted:
                continue
            result.to_store.extend(emitted)
            self._counters.emitted += len(emitted)
            self._counters.degraded_emitted += len(emitted)
            self._note_emitted(emitted)
            self._replace_pending(
                request_id, emitted, surviving[len(emitted) :], result
            )
            skip_request_ids.add(request_id)
            emitted_any = True
        if emitted_any:
            self._counters.degraded_drain_steps += 1

    def _step_is_idle(self) -> bool:
        """Whether the last observed step ran at an idle allocation rate.

        Both the smoothed rate and the next step's estimate must sit at or
        below the threshold: the estimate vetoes the first step of a burst
        before the EMA has caught up, and the EMA vetoes the trailing
        steps of one after the estimate has already fallen.
        """
        rate = max(self._blocks_per_step_ema, float(self._next_step_estimate))
        return rate <= self._config.idle_threshold_blocks

    def _rate_depth(self) -> int:
        """Free-queue depth the rate model alone considers at risk.

        Expected consumption below half a block over the whole horizon is
        treated as idle (depth 0): the EMA decays asymptotically after a
        burst and would otherwise keep a ceil'd depth of 1 forever.
        """
        per_step = max(self._blocks_per_step_ema, float(self._next_step_estimate))
        horizon_blocks = per_step * self._config.horizon_steps
        return 0 if horizon_blocks < 0.5 else math.ceil(horizon_blocks)

    def _danger_depth(self) -> int:
        """Free-queue depth considered at risk within the horizon.

        The maximum of three demands. The rate model covers steady
        consumption (:meth:`_rate_depth`). The adaptive danger floor covers
        the burst sizes measured loss has proven the rate model blind to;
        it is 0 while the feature is off or no loss is recent
        (:meth:`_update_danger_floor`). An outstanding announcement
        (:meth:`announce_allocation`) floors the depth at the announced
        block sum regardless of either model: the announced blocks are
        about to be consumed no matter what the recent per-step rate says.
        """
        depth = max(self._rate_depth(), math.ceil(self._danger_floor_blocks))
        if not self._announced_blocks:
            return depth
        return max(depth, sum(self._announced_blocks.values()))

    def _update_danger_floor(self) -> None:
        """Adapt the danger floor from the drain-to-drain loss delta.

        Runs once per :meth:`collect_due`, before the depth is read. A loss
        since the previous drain means the forecast (rate model plus
        announcements) was beaten -- in the measured dominant mode, by an
        unannounced allocation burst -- so the floor jumps to the recent
        peak step allocation and at least doubles on consecutive losses,
        capped at ``danger_floor_max_blocks``.

        A raised floor first *holds*: bursts recur on a cadence, and a
        floor that decays between two bursts pays the leading edge of
        every one (i60F measured 58 raises and 272 lost operations from
        exactly that cycle). The hold spans
        :data:`_DANGER_FLOOR_HOLD_GAPS` times the smoothed measured
        interval between losses, floored at
        :data:`_DANGER_FLOOR_MIN_HOLD_DRAINS`; every loss restarts it.
        Only after the hold expires do loss-free drains decay the floor
        exponentially (:data:`_DANGER_FLOOR_DECAY`), so the widened window
        and its free-queue read cost outlive the cadence they answer, not
        the workload. At cap 0 the floor stays 0 and the depth is the
        rate model's alone.
        """
        cap = self._config.danger_floor_max_blocks
        if cap <= 0:
            return
        self._floor_drain_index += 1
        lost = self._counters.dropped_evicted - self._dropped_evicted_at_floor_check
        self._dropped_evicted_at_floor_check = self._counters.dropped_evicted
        if lost > 0:
            # Every loss restarts the hold, whether or not it moves the
            # floor: the cadence being measured is "how often do bursts
            # beat the forecast", not "how often does the cap bind".
            if self._floor_last_raise_drain > 0:
                gap = float(self._floor_drain_index - self._floor_last_raise_drain)
                if self._floor_raise_gap_ema > 0.0:
                    self._floor_raise_gap_ema = (
                        _EMA_ALPHA * gap + (1 - _EMA_ALPHA) * self._floor_raise_gap_ema
                    )
                else:
                    self._floor_raise_gap_ema = gap
            self._floor_last_raise_drain = self._floor_drain_index
            raised = min(
                float(cap),
                max(
                    _DANGER_FLOOR_GROWTH * self._danger_floor_blocks,
                    float(max(self._recent_step_allocs, default=0)),
                    _DANGER_FLOOR_GROWTH * self._rate_depth(),
                ),
            )
            if raised > self._danger_floor_blocks:
                self._danger_floor_blocks = raised
                self._counters.danger_floor_raises += 1
            return
        if self._danger_floor_blocks <= 0.0:
            return
        hold = max(
            float(_DANGER_FLOOR_MIN_HOLD_DRAINS),
            _DANGER_FLOOR_HOLD_GAPS * self._floor_raise_gap_ema,
        )
        if self._floor_drain_index - self._floor_last_raise_drain <= hold:
            return
        self._danger_floor_blocks *= _DANGER_FLOOR_DECAY
        if self._danger_floor_blocks < 1.0:
            self._danger_floor_blocks = 0.0

    def _drop_evicted_suffix(
        self,
        request_id: str,
        ops: list[PendingStoreOp],
        result: DrainResult,
    ) -> list[PendingStoreOp]:
        """Drop ops from the first one whose data was lost; return survivors.

        A hash mismatch on any covered block means the block was evicted (or
        reallocated); the op and every later op of the request are dropped
        for prefix closure, and further admissions are rejected.
        """
        self._counters.requests_validated += 1
        first_lost = len(ops)
        for index, op in enumerate(ops):
            if not self._snapshot_intact(op):
                first_lost = index
                break
        if first_lost == len(ops):
            return ops
        dropped = ops[first_lost:]
        result.dropped_evicted.extend(dropped)
        self._counters.dropped_evicted += len(dropped)
        self._note_lost(dropped)
        self._broken_prefixes.add(request_id)
        surviving = ops[:first_lost]
        self._replace_pending(request_id, dropped, surviving, result)
        return surviving

    def _due_front_segment(
        self,
        ops: list[PendingStoreOp],
        ranks: dict[int, int],
        danger_depth: int,
    ) -> tuple[int, list[PendingStoreOp]] | None:
        """Find the front segment of ops to release for one request.

        An op is due when any of its blocks is within ``danger_depth`` of
        the free-queue head. Blocks absent from ``ranks`` (in use or pinned)
        are not at risk. The released segment runs from the front to the
        last due op, so a stored chunk never lacks its stored prefix.

        Returns:
            (min rank across the segment's due blocks, the segment), or
            None when no op is due.
        """
        if danger_depth <= 0:
            return None
        last_due = -1
        min_rank = danger_depth
        for index, op in enumerate(ops):
            op_ranks = [
                rank
                for block_id in op.block_hashes
                if (rank := ranks.get(block_id)) is not None
            ]
            due_ranks = [rank for rank in op_ranks if rank < danger_depth]
            if due_ranks:
                last_due = index
                min_rank = min(min_rank, min(due_ranks))
        if last_due < 0:
            return None
        return min_rank, ops[: last_due + 1]

    def _fails_economy_gate(self, ops: list[PendingStoreOp]) -> bool:
        """Gate 3 backstop: did the chain fall back below break-even length?

        Admission holds sub-break-even chains out of the pending machine,
        so every chain in it once ended past the threshold. This only fires
        when eviction truncated a promoted chain back below it -- storing
        the stub would cost more than recomputing it.
        """
        if self._config.min_prefix_tokens == 0:
            return False
        known_prefix = ops[-1].prefix_end_tokens
        return known_prefix < self._config.min_prefix_tokens

    def _snapshot_intact(self, op: PendingStoreOp) -> bool:
        """Whether every covered block still holds its admission-time hash.

        A mismatch on any block means it was evicted (or reallocated): the
        operation's data is lost and it must not be stored or deduplicated
        against.
        """
        checked = 0
        for block_id, snapshot in op.block_hashes.items():
            checked += 1
            if self._pool.block_hash(block_id) != snapshot:
                self._counters.blocks_validated += checked
                return False
        self._counters.blocks_validated += checked
        return True

    def _chain_intact(self, op: PendingStoreOp) -> bool:
        """Whether the op and its pending prefix chain are all intact.

        A valid deduplication cover must be more than intact itself: if an
        earlier pending op of its request has lost a block, the next drain
        drops the cover too (prefix closure), so it must not absorb a live
        copy of the content. Later siblings do not matter: their loss
        prefix-closes from their own position, leaving the cover storable.
        """
        for sibling in self._pending_ops.get(op.request_id) or []:
            if not self._snapshot_intact(sibling):
                return False
            if sibling is op:
                return True
        # Unreachable while the content-index invariant holds (every
        # cover is a pending op of its request); a missing cover is treated
        # as doomed, the safe direction.
        return False

    def _replace_pending(
        self,
        request_id: str,
        departed: list[PendingStoreOp],
        remaining: list[PendingStoreOp],
        result: DrainResult,
    ) -> None:
        """Replace pending ops and report requests whose buffer became empty."""
        self._pending_ops.replace_request(request_id, departed, remaining)
        if not remaining:
            result.emptied_requests.append(request_id)
