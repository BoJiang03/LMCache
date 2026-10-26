# SPDX-License-Identifier: Apache-2.0
"""
Thread pool with affinity routing.

Tasks submitted with the same ``affinity_key`` always execute on the same
worker thread.  Within each worker, tasks execute sequentially: ``FIFO`` tasks
in submission order, and an ``AHEAD`` task before every ``FIFO`` task that has
not started yet (``AHEAD`` tasks keep submission order among themselves).

This is used for GPU-bound request handlers (STORE / RETRIEVE) so that all
operations for a given vLLM instance land on one thread, eliminating the need
for per-instance locks on the shared temporary GPU buffer.  A RETRIEVE is
submitted ``AHEAD`` so that a burst of queued STOREs does not delay a load.
"""

# Standard
from concurrent.futures import Future
from enum import Enum
import itertools
import queue
import threading

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

# Sentinel object to signal worker shutdown
_SHUTDOWN = object()


class AffinityOrder(Enum):
    """Where a task joins its worker's queue.

    The value is the queue rank: a lower rank is served first.

    Attributes:
        AHEAD: Runs before every queued ``FIFO`` task of the worker, after
            the worker's running task and its earlier queued ``AHEAD`` tasks.
        FIFO: Runs after every task already queued on the worker.
    """

    AHEAD = 0
    FIFO = 1


class AffinityThreadPool:
    """Thread pool that routes tasks to workers by affinity key.

    Not thread-safe: the request transport must serialize calls to ``submit()``.
    ZMQ submits from its main loop, while gRPC protects submission with a lock.

    Args:
        max_workers: Number of worker threads.
        thread_name_prefix: Prefix for worker thread names.
    """

    def __init__(
        self,
        max_workers: int,
        thread_name_prefix: str = "affinity",
    ) -> None:
        self._num_workers = max_workers
        # Entries are (order rank, submission sequence, task); the unique
        # sequence keeps each rank FIFO and never lets two tasks compare.
        self._queues: list[queue.PriorityQueue] = [
            queue.PriorityQueue() for _ in range(max_workers)
        ]
        self._sequence = itertools.count()
        self._threads: list[threading.Thread] = []
        # Maps an affinity_key -> the worker slot (thread index) bound to it.
        self._key_to_slot: dict[int, int] = {}
        self._next_slot = 0
        self._overflow_warned = False
        for i in range(max_workers):
            t = threading.Thread(
                target=self._worker,
                args=(self._queues[i],),
                daemon=True,
                name=f"{thread_name_prefix}-{i}",
            )
            t.start()
            self._threads.append(t)

        logger.info(
            "Created AffinityThreadPool '%s' with %d worker slots: up to %d "
            "distinct affinity keys each bind to their own thread before slots "
            "are shared. Compare this against the number of clients expected to "
            "connect to confirm routing.",
            thread_name_prefix,
            max_workers,
            max_workers,
        )

    # ------------------------------------------------------------------
    # Worker loop
    # ------------------------------------------------------------------

    @staticmethod
    def _worker(q: queue.PriorityQueue) -> None:
        while True:
            _, _, item = q.get()
            if item is _SHUTDOWN:
                break
            future, fn, args, kwargs = item
            if future.set_running_or_notify_cancel():
                try:
                    result = fn(*args, **kwargs)
                    future.set_result(result)
                except BaseException as exc:
                    future.set_exception(exc)

    # ------------------------------------------------------------------
    # Routing
    # ------------------------------------------------------------------

    def _slot_for_key(self, affinity_key: int) -> int:
        """Return the worker slot bound to ``affinity_key``, assigning on first use.

        Returns:
            The worker slot (an index in ``[0, _num_workers)``) for the key.
        """
        slot = self._key_to_slot.get(affinity_key)
        if slot is not None:
            return slot

        # First time we see this key -- bind it to the next free slot.
        slot = self._next_slot % self._num_workers
        is_overflow = self._next_slot >= self._num_workers
        self._key_to_slot[affinity_key] = slot
        self._next_slot += 1

        logger.info(
            "AffinityThreadPool: affinity_key=%d assigned to worker "
            "slot %d of %d (thread %s); %d distinct key(s) now bound",
            affinity_key,
            slot,
            self._num_workers,
            self._threads[slot].name,
            len(self._key_to_slot),
        )
        if is_overflow and not self._overflow_warned:
            self._overflow_warned = True
            logger.warning(
                "AffinityThreadPool: affinity_key=%d wrapped onto worker slot "
                "%d (only %d workers), so it shares a thread with an earlier "
                "key and their tasks are serialized. Increase the worker count "
                "to give each client its own thread.",
                affinity_key,
                slot,
                self._num_workers,
            )
        return slot

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def submit(
        self,
        fn,
        *args,
        affinity_key: int = 0,
        order: AffinityOrder = AffinityOrder.FIFO,
        **kwargs,
    ) -> Future:
        """Submit *fn* for execution on the worker bound to *affinity_key*.

        Args:
            fn: Callable to run as ``fn(*args, **kwargs)``.
            *args: Positional arguments for ``fn``.
            affinity_key: Key selecting the worker; equal keys share one.
            order: Where the task joins the worker's queue; see
                :class:`AffinityOrder`. A running task is never preempted.
            **kwargs: Keyword arguments for ``fn``.

        Returns:
            A :class:`concurrent.futures.Future` for ``fn``'s result.
        """
        future: Future = Future()
        slot = self._slot_for_key(affinity_key)
        self._queues[slot].put(
            (order.value, next(self._sequence), (future, fn, args, kwargs))
        )
        return future

    def shutdown(self, wait: bool = True) -> None:
        """Shut down the pool.

        Sends a shutdown sentinel to every worker, queued as a ``FIFO`` task.
        If *wait* is true, blocks until all workers have exited.
        """
        for q in self._queues:
            q.put((AffinityOrder.FIFO.value, next(self._sequence), _SHUTDOWN))
        if wait:
            for t in self._threads:
                t.join()
