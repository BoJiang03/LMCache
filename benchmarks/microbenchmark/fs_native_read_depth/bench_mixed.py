"""Store latency while loads saturate the array: does num_workers substitute
for a read pool?

WHY THIS EXISTS
Raising num_workers does buy read concurrency, and on this array it very
likely matches the read pool on bandwidth alone.  So bandwidth cannot settle
the question.  What separates them is structural, and it is in
ConnectorBase::start_workers: dedicated per-op lanes are created ONLY from
worker_pool_config_.per_op_workers, and the fs connector's pybind does not
accept per_op_workers at all.  So on fs every op -- loads and stores alike --
lands in one shared queue served by num_workers threads.

That gives three arms with a real difference:

  A  workers=4,  no pool      the default: loads are slow AND occupy all 4
  B  workers=64, no pool      "just raise num_workers": loads are fast, but
                              they occupy all 64, so a store queues behind
                              whatever loads are already enqueued
  C  workers=4,  depth=64     do_batch_get hands the read to the reader
                              threads and returns, so the 4 worker threads
                              never touch a load and a store is served at once

LMCache runs exactly this mix: prefill stores KV while other requests load it.

MEASURED: read bandwidth over the window, and store completion latency
(submit -> completion) sampled one store at a time, which is what a store
actually experiences when loads are in flight.

FALSIFIER: if B's store latency is comparable to C's, the isolation argument
fails and num_workers really is an adequate substitute on this backend.
"""

# Standard
from typing import Any
import argparse
import os
import select
import shutil
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# First Party
from benchN import ALIGN, PinnedPool, key_for  # noqa: E402

STORE_KEY_BASE = 90_000_000  # never collides with the read corpus


class Completer(threading.Thread):
    """Single owner of drain_completions(), dispatching results by future id.

    drain_completions() returns every finished future, so two threads calling
    it would consume each other's results.  One thread polls, records a
    completion timestamp per future, and wakes whoever is waiting.
    """

    def __init__(self, client: Any) -> None:
        super().__init__(daemon=True)
        self._client = client
        self._done: dict[int, tuple[float, bool, str]] = {}
        self._cv = threading.Condition()
        self._stop = False

    def run(self) -> None:
        while not self._stop:
            got = self._client.drain_completions()
            if got:
                now = time.monotonic()
                with self._cv:
                    for fid, ok, msg, _per in got:
                        self._done[fid] = (now, ok, msg)
                    self._cv.notify_all()
            else:
                select.select([self._client.event_fd()], [], [], 0.002)

    def stop(self) -> None:
        self._stop = True

    def reap(self, pending: set[int], target: int) -> None:
        """Block until at most ``target`` of ``pending`` remain outstanding."""
        while len(pending) > target:
            with self._cv:
                ready = pending & self._done.keys()
                if not ready:
                    self._cv.wait(0.05)
                    ready = pending & self._done.keys()
                for fid in list(ready):
                    _t, ok, msg = self._done.pop(fid)
                    pending.discard(fid)
                    if not ok:
                        raise RuntimeError(f"future {fid} failed: {msg}")

    def wait_one(self, fid: int) -> float:
        """Return the completion timestamp for a single future."""
        with self._cv:
            while fid not in self._done:
                self._cv.wait(0.05)
            t, ok, msg = self._done.pop(fid)
        if not ok:
            raise RuntimeError(f"store future {fid} failed: {msg}")
        return t


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", required=True)
    ap.add_argument("--obj-kib", type=int, default=6144)
    ap.add_argument("--corpus-gib", type=int, default=64)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=512)
    ap.add_argument("--outstanding", type=int, default=3)
    ap.add_argument("--io-depth", type=int, default=0)
    ap.add_argument("--budget-mib", type=int, default=0)
    ap.add_argument("--duration-s", type=float, default=45.0)
    ap.add_argument("--store-batch", type=int, default=4)
    ap.add_argument("--store-gap-ms", type=float, default=50.0)
    ap.add_argument("--arm", default="?")
    ap.add_argument("--reuse-corpus", action="store_true")
    ap.add_argument("--keep-corpus", action="store_true")
    args = ap.parse_args()

    obj = args.obj_kib << 10
    n_obj = (args.corpus_gib << 30) // obj

    # First Party
    from lmcache.lmcache_fs import LMCacheFSClient

    os.makedirs(args.base_path, exist_ok=True)
    read_pool = PinnedPool(args.batch * (args.outstanding + 1), obj, True)
    store_pool = PinnedPool(args.store_batch, obj, True)

    client = LMCacheFSClient(args.base_path, args.workers, "", True, 0,
                             args.io_depth, args.budget_mib << 20)
    print(f"arm={args.arm} workers={args.workers} io_depth={args.io_depth} "
          f"budget_mib={args.budget_mib} obj={args.obj_kib}KiB "
          f"corpus={args.corpus_gib}GiB n_obj={n_obj}", flush=True)

    comp = Completer(client)
    comp.start()
    try:
        if not args.reuse_corpus:
            pending: set[int] = set()
            b = 0
            for start in range(0, n_obj, args.batch):
                keys = [key_for(i)
                        for i in range(start, min(start + args.batch, n_obj))]
                views = [read_pool.view(b + j) for j in range(len(keys))]
                b += len(keys)
                comp.reap(pending, args.outstanding - 1)
                pending.add(client.submit_batch_set(keys, views))
            comp.reap(pending, 0)
            os.sync()
            print("corpus written", flush=True)

        stop_at = time.monotonic() + args.duration_s
        lat: list[float] = []
        read_bytes = [0]

        def storer() -> None:
            n = 0
            while time.monotonic() < stop_at:
                keys = [key_for(STORE_KEY_BASE + n + j)
                        for j in range(args.store_batch)]
                views = [store_pool.view(j) for j in range(args.store_batch)]
                n += args.store_batch
                t0 = time.monotonic()
                fid = client.submit_batch_set(keys, views)
                lat.append(comp.wait_one(fid) - t0)
                time.sleep(args.store_gap_ms / 1000.0)

        th = threading.Thread(target=storer, daemon=True)
        t0 = time.monotonic()
        th.start()

        pending = set()
        b = 0
        start = 0
        while time.monotonic() < stop_at:
            end = min(start + args.batch, n_obj)
            keys = [key_for(i) for i in range(start, end)]
            views = [read_pool.view(b + j) for j in range(len(keys))]
            b += len(keys)
            comp.reap(pending, args.outstanding - 1)
            pending.add(client.submit_batch_get(keys, views))
            read_bytes[0] += len(keys) * obj
            start = 0 if end >= n_obj else end
        comp.reap(pending, 0)
        wall = time.monotonic() - t0
        th.join(timeout=10)

        gbs = read_bytes[0] / 1e9 / wall
        if lat:
            s = sorted(lat)
            p50 = 1000 * s[len(s) // 2]
            p95 = 1000 * s[int(len(s) * 0.95)]
            mx = 1000 * s[-1]
            mean = 1000 * statistics.fmean(s)
        else:
            p50 = p95 = mx = mean = float("nan")
        print(f"RESULT arm={args.arm} read={gbs:.2f} GB/s  wall={wall:.1f}s  "
              f"stores={len(lat)}  store_lat_ms p50={p50:.1f} mean={mean:.1f} "
              f"p95={p95:.1f} max={mx:.1f}", flush=True)
    finally:
        comp.stop()
        client.close()
        if not args.keep_corpus:
            shutil.rmtree(args.base_path, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
