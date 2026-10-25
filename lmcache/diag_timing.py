# SPDX-License-Identifier: Apache-2.0
"""DIAG (not for merge): windowed timing aggregates for lazy offload + XR.

Two collectors, each logging one line per window (default 10 s):

* ``SchedulerDiag`` (``sched_diag()``): the vLLM connector's scheduler side.
  Single-threaded (vLLM's scheduler thread), so it keeps exact per-window
  sample lists and flushes from ``build_connector_meta``. INFO, prefix
  ``DIAG sched``.
* ``ServerDiag`` (``server_diag()``): the LMCache MP server. Handlers run on
  many threads, so each thread owns cumulative counters plus a log-bucket
  histogram (4 buckets per octave, so percentiles are within ~9%), and a
  daemon thread logs per-window deltas. No lock on the record path. WARNING,
  prefix ``DIAG server``, because the server runs at WARNING.
"""

# Standard
from collections.abc import Callable
from typing import Any
import math
import threading
import time
import weakref

# First Party
from lmcache.logging import init_logger

logger = init_logger(__name__)

WINDOW_S = 10.0

# Log histogram: bucket i covers [2^(i/4 - 20), 2^((i+1)/4 - 20)) seconds,
# i.e. ~1 us .. ~128 s; out-of-range values clamp to the end buckets.
_OCTAVE_OFFSET = 20
_SUB = 4
_NUM_BUCKETS = (_OCTAVE_OFFSET + 8) * _SUB


def _bucket(value: float) -> int:
    """Return the histogram bucket of a duration in seconds."""
    if value <= 0.0:
        return 0
    mantissa, exponent = math.frexp(value)
    idx = (exponent + _OCTAVE_OFFSET - 1) * _SUB + int((mantissa - 0.5) * 2 * _SUB)
    if idx < 0:
        return 0
    if idx >= _NUM_BUCKETS:
        return _NUM_BUCKETS - 1
    return idx


def _bucket_mid(idx: int) -> float:
    """Return the geometric midpoint, in seconds, of histogram bucket ``idx``."""
    octave, sub = divmod(idx, _SUB)
    lo = 2.0 ** (octave - _OCTAVE_OFFSET) * (1.0 + sub / _SUB)
    hi = 2.0 ** (octave - _OCTAVE_OFFSET) * (1.0 + (sub + 1) / _SUB)
    return math.sqrt(lo * hi)


def _hist_quantile(hist: list[int], q: float) -> float:
    """Return the ``q`` quantile, in seconds, of a bucket-count histogram."""
    total = sum(hist)
    if total == 0:
        return 0.0
    rank = q * total
    seen = 0
    for idx, count in enumerate(hist):
        seen += count
        if seen >= rank and count:
            return _bucket_mid(idx)
    return _bucket_mid(len(hist) - 1)


def _quantile(sorted_values: list[float], q: float) -> float:
    """Return the ``q`` quantile of an already sorted list (0 when empty)."""
    if not sorted_values:
        return 0.0
    return sorted_values[min(len(sorted_values) - 1, int(q * len(sorted_values)))]


def _ms(values: list[float], qs: tuple[float, ...] = (0.5, 0.9)) -> str:
    """Render ``n/mean/quantiles`` of second-valued samples in milliseconds."""
    if not values:
        return "n=0"
    s = sorted(values)
    parts = [f"n={len(s)}", f"mean={sum(s) / len(s) * 1e3:.2f}"]
    parts += [f"p{int(q * 100)}={_quantile(s, q) * 1e3:.2f}" for q in qs]
    return " ".join(parts)


# Scheduler-side connector methods, in log order. "drain" and "admit" are the
# lazy offload parts of build_connector_meta and get_num_new_matched_tokens.
SCHED_METHODS = ("gnmt", "admit", "usaa", "bcm", "drain", "uco", "rf", "onr")
_NESTED = ("admit", "drain")


class SchedulerDiag:
    """Per-window scheduler-side aggregates; scheduler thread only."""

    def __init__(self, window_s: float = WINDOW_S) -> None:
        self.window_s = window_s
        self._n = len(SCHED_METHODS)
        self._index = {name: i for i, name in enumerate(SCHED_METHODS)}
        # admit and drain are nested in gnmt and bcm, so not in the total.
        self._nested = {self._index[name] for name in _NESTED}
        self._reset_window(time.monotonic())
        self._last_step_at = 0.0
        self._step_cpu = [0.0] * self._n
        self._step_wall = [0.0] * self._n
        # request_id -> RETRIEVE issue time; bounded by in-flight loads.
        self._load_issued: dict[str, float] = {}
        self.pending_ops_fn: Callable[[], tuple[int, int]] | None = None
        self.last_line = ""

    def add(self, name: str, cpu: float, wall: float) -> None:
        """Charge one call of a connector method to the current step.

        Args:
            name: One of ``SCHED_METHODS``.
            cpu: Thread CPU seconds spent in the call.
            wall: Wall seconds spent in the call.
        """
        i = self._index[name]
        self._step_cpu[i] += cpu
        self._step_wall[i] += wall
        self._calls[i] += 1

    def step_begin(self, now: float) -> None:
        """Record the interval since the previous ``build_connector_meta``."""
        if self._last_step_at:
            self._intervals.append(now - self._last_step_at)
        self._last_step_at = now

    def step_end(self, now: float) -> None:
        """Close the step's per-method sums; flush the window if it is due."""
        tot_cpu = 0.0
        tot_wall = 0.0
        for i in range(self._n):
            c = self._step_cpu[i]
            w = self._step_wall[i]
            self._cpu[i].append(c)
            self._wall[i].append(w)
            if i not in self._nested:
                tot_cpu += c
                tot_wall += w
            self._step_cpu[i] = 0.0
            self._step_wall[i] = 0.0
        self._tot_cpu.append(tot_cpu)
        self._tot_wall.append(tot_wall)
        self._steps += 1
        if now - self._window_start >= self.window_s:
            self.flush(now)

    def record_lookup(
        self,
        submit_to_reply: float,
        reply_to_consume: float,
        ack: float,
        status_rtt: float,
        polls: int,
    ) -> None:
        """Record one lookup resolved by ``check_lookup_result``.

        Args:
            submit_to_reply: LOOKUP sent -> last final status reply received.
            reply_to_consume: That reply received -> consumed by the call.
            ack: LOOKUP sent -> every LOOKUP ack received.
            status_rtt: Summed send -> reply time of its status queries.
            polls: Number of status queries sent for it.
        """
        self._lk_reply.append(submit_to_reply)
        self._lk_consume.append(reply_to_consume)
        self._lk_ack.append(ack)
        self._lk_rtt.append(status_rtt)
        self._lk_polls += polls

    def load_issued(self, request_id: str, now: float) -> None:
        """Note that a RETRIEVE for ``request_id`` went into step metadata."""
        self._load_issued[request_id] = now

    def loads_finished(self, request_ids: Any, now: float) -> None:
        """Record issue -> finished_recving for each finished load."""
        for request_id in request_ids:
            issued = self._load_issued.pop(request_id, None)
            if issued is not None:
                self._loads.append(now - issued)

    def forget_request(self, request_id: str) -> None:
        """Drop per-request state of a finished request."""
        self._load_issued.pop(request_id, None)

    def flush(self, now: float) -> str:
        """Log and return the window's line, then start a new window."""
        elapsed = now - self._window_start
        iv = sorted(self._intervals)
        steps = max(1, self._steps)
        cpu_parts = []
        wall_parts = []
        tot_cpu = self._tot_cpu
        tot_wall = self._tot_wall
        for i, name in enumerate(SCHED_METHODS):
            c = self._cpu[i]
            w = self._wall[i]
            cpu_parts.append(
                f"{name}={sum(c) / steps * 1e6:.0f}/"
                f"{_quantile(sorted(c), 0.99) * 1e6:.0f}"
            )
            wall_parts.append(
                f"{name}={sum(w) / steps * 1e6:.0f}/"
                f"{_quantile(sorted(w), 0.99) * 1e6:.0f}"
            )
        cpu_parts.append(
            f"total={sum(tot_cpu) / steps * 1e6:.0f}/"
            f"{_quantile(sorted(tot_cpu), 0.99) * 1e6:.0f}"
        )
        wall_parts.append(
            f"total={sum(tot_wall) / steps * 1e6:.0f}/"
            f"{_quantile(sorted(tot_wall), 0.99) * 1e6:.0f}"
        )
        calls = " ".join(
            f"{name}={self._calls[i]}" for i, name in enumerate(SCHED_METHODS)
        )
        nlk = len(self._lk_reply)
        pending, inflight = (
            self.pending_ops_fn() if self.pending_ops_fn is not None else (-1, -1)
        )
        line = (
            f"DIAG sched window={elapsed:.1f}s steps={self._steps} "
            f"interval_ms mean={sum(iv) / max(1, len(iv)) * 1e3:.2f} "
            f"p50={_quantile(iv, 0.5) * 1e3:.2f} p99={_quantile(iv, 0.99) * 1e3:.2f} "
            f"max={(iv[-1] if iv else 0.0) * 1e3:.2f} "
            f"| cpu_us/step mean/p99 {' '.join(cpu_parts)} "
            f"| wall_us/step mean/p99 {' '.join(wall_parts)} "
            f"| calls {calls} "
            f"| lookup submit_reply_ms {_ms(self._lk_reply)} "
            f"reply_consume_ms {_ms(self._lk_consume)} "
            f"ack_ms {_ms(self._lk_ack)} status_rtt_ms {_ms(self._lk_rtt)} "
            f"polls_per_lookup={self._lk_polls / max(1, nlk):.2f} "
            f"| load_ms {_ms(self._loads)} loads_inflight={len(self._load_issued)} "
            f"| lazy pending_ops={pending} inflight_stores={inflight}"
        )
        logger.info("%s", line)
        self.last_line = line
        self._reset_window(now)
        return line

    def _reset_window(self, now: float) -> None:
        self._window_start = now
        self._steps = 0
        self._intervals: list[float] = []
        self._cpu: list[list[float]] = [[] for _ in range(self._n)]
        self._wall: list[list[float]] = [[] for _ in range(self._n)]
        self._tot_cpu: list[float] = []
        self._tot_wall: list[float] = []
        self._calls = [0] * self._n
        self._lk_reply: list[float] = []
        self._lk_consume: list[float] = []
        self._lk_ack: list[float] = []
        self._lk_rtt: list[float] = []
        self._lk_polls = 0
        self._loads: list[float] = []


_SCHED = SchedulerDiag()


def sched_diag() -> SchedulerDiag:
    """Return the process-wide scheduler-side collector."""
    return _SCHED


def timed(diag: SchedulerDiag, name: str, func: Callable[..., Any]) -> Any:
    """Wrap a bound method so each call charges thread CPU and wall to ``name``.

    Args:
        diag: The collector to charge.
        name: One of ``SCHED_METHODS``.
        func: The bound method to wrap.

    Returns:
        A callable with the same signature and return value as ``func``.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        w0 = time.monotonic()
        c0 = time.thread_time()
        try:
            return func(*args, **kwargs)
        finally:
            diag.add(name, time.thread_time() - c0, time.monotonic() - w0)

    return wrapper


def timed_step(diag: SchedulerDiag, func: Callable[..., Any]) -> Any:
    """Wrap ``build_connector_meta``: it delimits steps and is charged as "bcm".

    Args:
        diag: The collector to charge.
        func: The bound ``build_connector_meta``.

    Returns:
        A callable with the same signature and return value as ``func``.
    """

    def wrapper(*args: Any, **kwargs: Any) -> Any:
        w0 = time.monotonic()
        c0 = time.thread_time()
        diag.step_begin(w0)
        try:
            return func(*args, **kwargs)
        finally:
            w1 = time.monotonic()
            diag.add("bcm", time.thread_time() - c0, w1 - w0)
            diag.step_end(w1)

    return wrapper


class _ThreadStats:
    """One thread's cumulative server counters: name -> [count, sum, qsum, hist]."""

    def __init__(self) -> None:
        self.stats: dict[str, list[Any]] = {}


class ServerDiag:
    """Per-window server aggregates; lock-free per-thread recording."""

    def __init__(self, window_s: float = WINDOW_S) -> None:
        self.window_s = window_s
        self._tls = threading.local()
        self._threads: list[_ThreadStats] = []
        self._threads_lock = threading.Lock()
        self._prev: dict[str, list[Any]] = {}
        self._gauges: dict[str, Callable[[], int]] = {}
        self._counters: dict[str, int] = {}
        self._prev_counters: dict[str, int] = {}
        self._reporter: threading.Thread | None = None
        self._window_start = time.monotonic()
        self.last_line = ""

    def record(self, name: str, value: float, qwait: float = 0.0) -> None:
        """Record one sample of ``name`` (seconds) and an optional queue wait."""
        try:
            stats = self._tls.s.stats
        except AttributeError:
            stats = self._register_thread().stats
        row = stats.get(name)
        if row is None:
            row = [0, 0.0, 0.0, [0] * _NUM_BUCKETS]
            stats[name] = row
        row[0] += 1
        row[1] += value
        row[2] += qwait
        row[3][_bucket(value)] += 1

    def count(self, name: str) -> None:
        """Increment a cumulative counter (callers already hold a lock)."""
        self._counters[name] = self._counters.get(name, 0) + 1

    def set_gauge(self, name: str, owner: object, fn: Callable[[Any], int]) -> None:
        """Report ``fn(owner)`` as a snapshot gauge while ``owner`` lives."""
        ref = weakref.ref(owner)

        def read() -> int:
            obj = ref()
            return -1 if obj is None else fn(obj)

        self._gauges[name] = read

    def flush(self, now: float | None = None) -> str:
        """Log and return the window's line (deltas since the last flush)."""
        now = time.monotonic() if now is None else now
        with self._threads_lock:
            threads = list(self._threads)
        total: dict[str, list[Any]] = {}
        for ts in threads:
            for name, row in list(ts.stats.items()):
                acc = total.get(name)
                if acc is None:
                    acc = [0, 0.0, 0.0, [0] * _NUM_BUCKETS]
                    total[name] = acc
                acc[0] += row[0]
                acc[1] += row[1]
                acc[2] += row[2]
                hist = acc[3]
                for i, c in enumerate(row[3]):
                    hist[i] += c
        parts = []
        for name in sorted(total):
            cur = total[name]
            prev = self._prev.get(name, [0, 0.0, 0.0, [0] * _NUM_BUCKETS])
            n = cur[0] - prev[0]
            if n <= 0:
                continue
            s = cur[1] - prev[1]
            q = cur[2] - prev[2]
            hist = [a - b for a, b in zip(cur[3], prev[3], strict=True)]
            parts.append(
                f"{name}:n={n},tot_ms={s * 1e3:.1f},mean_ms={s / n * 1e3:.3f},"
                f"p90_ms={_hist_quantile(hist, 0.9) * 1e3:.3f},"
                f"qwait_ms={q / n * 1e3:.3f}"
            )
        self._prev = total
        counters = dict(self._counters)
        counter_parts = [
            f"{k}={v - self._prev_counters.get(k, 0)}/{v}"
            for k, v in sorted(counters.items())
        ]
        self._prev_counters = counters
        gauge_parts = [f"{k}={fn()}" for k, fn in sorted(self._gauges.items())]
        line = (
            f"DIAG server window={now - self._window_start:.1f}s "
            f"{' '.join(parts) or 'idle'} | counters(window/total) "
            f"{' '.join(counter_parts)} | {' '.join(gauge_parts)}"
        )
        self._window_start = now
        logger.warning("%s", line)
        self.last_line = line
        return line

    def _register_thread(self) -> _ThreadStats:
        ts = _ThreadStats()
        self._tls.s = ts
        with self._threads_lock:
            self._threads.append(ts)
            if self._reporter is None:
                self._reporter = threading.Thread(
                    target=self._report_loop, daemon=True, name="lmcache-diag"
                )
                self._reporter.start()
        return ts

    def _report_loop(self) -> None:
        while True:
            time.sleep(self.window_s)
            try:
                self.flush()
            except Exception:
                logger.exception("DIAG server flush failed")


_SERVER = ServerDiag()


def server_diag() -> ServerDiag:
    """Return the process-wide server-side collector."""
    return _SERVER
