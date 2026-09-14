"""Load latency under load: the quantity num_workers cannot get right.

WHY BANDWIDTH CANNOT SETTLE THIS
Raising num_workers does buy read bandwidth, so a bandwidth table will very
likely show num_workers matching the read pool.  The cost of raising it is
not bandwidth, it is latency, and by Little's law the two are not independent:

    bytes in flight = bandwidth x latency

Once the device is saturated the bandwidth term is pinned, so every extra byte
in flight is paid for entirely in latency, buying nothing.  On the legacy path
bytes in flight is num_workers x object_size, which SCALES WITH OBJECT SIZE.
On the pooled path it is a constant the operator set in bytes.

So a single num_workers must choose: large enough for small objects to reach
bandwidth, or small enough for large objects not to queue.  A single byte
budget does not have to choose, because bytes is the quantity that sets both.

WHAT THIS MEASURES
A saturating read stream, plus a single-object read PROBE issued one at a time
alongside it.  The probe's submit-to-completion time is what one load actually
waits when the connector is busy, which end to end is TTFT.

FALSIFIER: if probe latency does not rise with num_workers at large objects,
Little's law is not binding here and the argument above is wrong.
"""

# Standard
from typing import Any
import argparse
import os
import shutil
import statistics
import sys
import threading
import time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
# First Party
from bench_mixed import Completer  # noqa: E402
from benchN import PinnedPool, key_for  # noqa: E402


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
    ap.add_argument("--duration-s", type=float, default=30.0)
    ap.add_argument("--probe-gap-ms", type=float, default=100.0)
    ap.add_argument("--arm", default="?")
    ap.add_argument("--reuse-corpus", action="store_true")
    ap.add_argument("--keep-corpus", action="store_true")
    args = ap.parse_args()

    obj = args.obj_kib << 10
    n_obj = (args.corpus_gib << 30) // obj
    if n_obj < args.batch + 8:
        raise ValueError("corpus too small for this batch")

    # First Party
    from lmcache.lmcache_fs import LMCacheFSClient

    os.makedirs(args.base_path, exist_ok=True)
    stream_pool = PinnedPool(args.batch * (args.outstanding + 1), obj, True)
    probe_pool = PinnedPool(1, obj, True)

    client = LMCacheFSClient(args.base_path, args.workers, "", True, 0,
                             args.io_depth, args.budget_mib << 20)
    comp = Completer(client)
    comp.start()
    try:
        if not args.reuse_corpus:
            pending: set[int] = set()
            b = 0
            for start in range(0, n_obj, args.batch):
                keys = [key_for(i)
                        for i in range(start, min(start + args.batch, n_obj))]
                views = [stream_pool.view(b + j) for j in range(len(keys))]
                b += len(keys)
                comp.reap(pending, args.outstanding - 1)
                pending.add(client.submit_batch_set(keys, views))
            comp.reap(pending, 0)
            os.sync()

        stop_at = time.monotonic() + args.duration_s
        lat: list[float] = []

        def prober() -> None:
            i = 0
            # Let the stream reach steady state before the first probe.
            time.sleep(2.0)
            while time.monotonic() < stop_at:
                k = [key_for(i % n_obj)]
                i += 7  # stride, so probes do not walk the stream's cursor
                t0 = time.monotonic()
                fid = client.submit_batch_get(k, [probe_pool.view(0)])
                lat.append(comp.wait_one(fid) - t0)
                time.sleep(args.probe_gap_ms / 1000.0)

        th = threading.Thread(target=prober, daemon=True)
        pending = set()
        b = 0
        start = 0
        got = 0
        t0 = time.monotonic()
        th.start()
        while time.monotonic() < stop_at:
            end = min(start + args.batch, n_obj)
            keys = [key_for(i) for i in range(start, end)]
            views = [stream_pool.view(b + j) for j in range(len(keys))]
            b += len(keys)
            comp.reap(pending, args.outstanding - 1)
            pending.add(client.submit_batch_get(keys, views))
            got += len(keys) * obj
            start = 0 if end >= n_obj else end
        comp.reap(pending, 0)
        wall = time.monotonic() - t0
        th.join(timeout=10)

        inflight_mib = (args.budget_mib if args.io_depth
                        else args.workers * args.obj_kib / 1024)
        if lat:
            s = sorted(lat)
            p50, p95, mx = (1000 * s[len(s) // 2], 1000 * s[int(len(s) * .95)],
                            1000 * s[-1])
            mean = 1000 * statistics.fmean(s)
        else:
            p50 = p95 = mx = mean = float("nan")
        print(f"RESULT arm={args.arm} obj={args.obj_kib}KiB workers={args.workers} "
              f"depth={args.io_depth} budget={args.budget_mib}MiB "
              f"inflight_cap={inflight_mib:.0f}MiB "
              f"stream={got / 1e9 / wall:.2f} GB/s probes={len(lat)} "
              f"probe_ms p50={p50:.1f} mean={mean:.1f} p95={p95:.1f} max={mx:.1f}",
              flush=True)
    finally:
        comp.stop()
        client.close()
        if not args.keep_corpus:
            shutil.rmtree(args.base_path, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
