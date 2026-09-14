"""Measure the fs_native L2->L1 read path in isolation: no vLLM, no GPU compute.

Why this exists
---------------
Two numbers disagreed about how fast LMCache moves bytes out of L2:

  * ``lmcache_mp_l2_load_throughput`` reported **28.83 GB/s** in the P1 run.
  * The block layer reported 562.5 GiB moved with the array busy 11.5 s,
    i.e. **~52 GB/s** while busy.

The metric is not wrong, it is answering a different question -- its own
docstring says the sample is ``total_bytes / (completed_ts - submitted_ts)``
and that the interval "spans adapter queue, network, and disk time -- not just
transfer".  Under concurrency that per-request figure is a *latency* measure
and is strictly below aggregate bandwidth.  Neither number, on its own, tells
you whether the transfer mechanism is leaving bandwidth on the table, because
both were taken with a 100-request vLLM workload sitting on top.

This driver removes vLLM.  It talks to ``LMCacheFSClient`` -- the same C++
connector the ``mp_l2``/``mp_ceiling`` arms use, same ``use_odirect=True``,
same object sizes -- and times a full-corpus read into pinned host memory,
bracketing the timed region with ``/proc/diskstats`` so achieved bandwidth can
be stated against device busy time rather than against wall clock alone.

What it reports
---------------
  bytes, wall time, GB/s vs wall, device busy time, GB/s while busy, and the
  same per-member busy spread the run harness prints.  Compare GB/s to the
  independently measured array ceiling (fio, O_DIRECT, 53.5 GB/s).

Re-reading the same corpus each pass is safe: a pass is >= 100 GiB, far larger
than the drives' combined internal buffers, so no pass is served from drive
DRAM.  (fio hit exactly that trap when its jobs shared LBAs; see record
2026-09-10/5 section 1.2.)

usage:
  python benchB.py --base-path $WORK/l2bench \
      --obj-kib 98304 --corpus-gib 200 --workers 4 --passes 3
"""

# Standard
from typing import NamedTuple
import argparse
import ctypes
import os
import select
import shutil
import sys
import time

SECTOR = 512
ALIGN = 4096


class DiskSnap(NamedTuple):
    """One /proc/diskstats reading for an array and its members."""

    t: float
    rd_sectors: int
    rd_ticks_ms: int
    io_ticks_ms: int
    members: dict[str, tuple[int, int]]  # name -> (rd_ticks_ms, io_ticks_ms)


def _members_of(dev: str) -> list[str]:
    """Return the member block devices backing ``dev``, or [] if not an array."""
    try:
        return sorted(os.listdir(f"/sys/block/{dev}/slaves"))
    except OSError:
        return []


def snap_disk(dev: str, members: list[str]) -> DiskSnap:
    """Read /proc/diskstats for ``dev`` and each of ``members``.

    Fields used (1-indexed as in the kernel's documentation): 3 name,
    6 rd_sectors, 7 rd_ticks, 13 io_ticks.  ``rd_ticks`` is summed per-request
    service time and exceeds wall clock at queue depth; ``io_ticks`` is
    wall-clock ms with at least one request in flight, which is the busy-time
    number this driver reports against.
    """
    want = {dev, *members}
    rd_sectors = rd_ticks = io_ticks = 0
    per: dict[str, tuple[int, int]] = {}
    with open("/proc/diskstats") as fh:
        for line in fh:
            f = line.split()
            if len(f) < 13 or f[2] not in want:
                continue
            if f[2] == dev:
                rd_sectors, rd_ticks, io_ticks = int(f[5]), int(f[6]), int(f[12])
            else:
                per[f[2]] = (int(f[6]), int(f[12]))
    return DiskSnap(time.monotonic(), rd_sectors, rd_ticks, io_ticks, per)


class PinnedPool:
    """A ring of page-aligned host buffers, optionally CUDA-pinned.

    O_DIRECT requires the destination address to be block aligned, and the
    connector fails the request outright rather than silently falling back
    (``test_odirect_fails_for_misaligned_buffer``), so alignment is enforced
    here by over-allocating and slicing to the next 4096 boundary.
    """

    def __init__(self, count: int, size: int, pin: bool) -> None:
        self._views: list[memoryview] = []
        self._keep: list[object] = []
        for _ in range(count):
            if pin:
                # Third Party
                import torch

                t = torch.empty(size + ALIGN, dtype=torch.uint8).pin_memory()
                self._keep.append(t)
                mv = memoryview(t.numpy())
            else:
                raw = bytearray(size + ALIGN)
                self._keep.append(raw)
                mv = memoryview(raw)
            addr = ctypes.addressof(ctypes.c_char.from_buffer(mv))
            off = (-addr) % ALIGN
            view = mv[off : off + size]
            if ctypes.addressof(ctypes.c_char.from_buffer(view)) % ALIGN:
                raise RuntimeError("failed to align a buffer to 4096")
            self._views.append(view)

    def __len__(self) -> int:
        return len(self._views)

    def view(self, i: int) -> memoryview:
        return self._views[i % len(self._views)]


def key_for(i: int) -> str:
    """Build a connector key for object ``i`` in the synthetic corpus."""
    return f"l2bench@00000000@{i:016x}"


def drain_until(client: object, pending: set[int], target: int) -> None:
    """Block until at most ``target`` futures remain outstanding.

    Raises:
        RuntimeError: if any completed future reports failure.
    """
    while len(pending) > target:
        for fid, ok, msg, _per in client.drain_completions():
            if fid in pending:
                pending.discard(fid)
                if not ok:
                    raise RuntimeError(f"future {fid} failed: {msg}")
        if len(pending) > target:
            select.select([client.event_fd()], [], [], 0.05)


def run_phase(
    client: object,
    method: str,
    n_obj: int,
    pool: PinnedPool,
    batch: int,
    outstanding: int,
) -> None:
    """Submit ``n_obj`` single-object operations, holding ``outstanding`` futures.

    Args:
        client: an ``LMCacheFSClient``.
        method: ``submit_batch_get`` or ``submit_batch_set``.
        n_obj: number of corpus objects to touch.
        pool: buffer ring; must hold at least ``batch * (outstanding + 1)``.
        batch: objects per submitted future.
        outstanding: futures kept in flight.
    """
    submit = getattr(client, method)
    pending: set[int] = set()
    b = 0
    for start in range(0, n_obj, batch):
        keys = [key_for(i) for i in range(start, min(start + batch, n_obj))]
        views = [pool.view(b + j) for j in range(len(keys))]
        b += len(keys)
        drain_until(client, pending, outstanding - 1)
        pending.add(submit(keys, views))
    drain_until(client, pending, 0)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-path", required=True)
    ap.add_argument("--obj-kib", type=int, default=96 << 10)
    ap.add_argument("--corpus-gib", type=int, default=200)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--outstanding", type=int, default=2)
    ap.add_argument("--passes", type=int, default=3)
    ap.add_argument("--dev", default="md1")
    ap.add_argument("--no-pin", action="store_true")
    ap.add_argument("--io-depth", type=int, default=0,
                    help="FSConnector read_io_depth; 0 = legacy read path")
    ap.add_argument("--no-odirect", action="store_true",
                    help="read through the page cache instead of O_DIRECT, "
                         "which changes the storage latency by orders of "
                         "magnitude on the same array")
    ap.add_argument("--budget-mib", type=int, default=0,
                    help="FSConnector read_max_bytes_in_flight in MiB; "
                         "0 = measure it during the run")
    ap.add_argument("--tile-kib", type=int, default=0,
                    help="FSConnector read_tile_bytes in KiB; 0 = no split")
    ap.add_argument("--keep-corpus", action="store_true")
    ap.add_argument("--reuse-corpus", action="store_true")
    args = ap.parse_args()

    obj = args.obj_kib << 10
    if obj % ALIGN:
        raise ValueError("--obj-kib must keep the object 4096-aligned")
    n_obj = (args.corpus_gib << 30) // obj
    if n_obj < args.batch:
        raise ValueError("corpus is smaller than one batch")
    total = n_obj * obj

    # First Party
    from lmcache.lmcache_fs import LMCacheFSClient

    os.makedirs(args.base_path, exist_ok=True)
    members = _members_of(args.dev)
    pool = PinnedPool(args.batch * (args.outstanding + 1), obj, not args.no_pin)

    print(
        f"corpus {n_obj} x {args.obj_kib / 1024:g} MiB = {total / 2**30:.1f} GiB   "
        f"workers={args.workers} batch={args.batch} "
        f"outstanding={args.outstanding} pinned={not args.no_pin} "
        f"io_depth={args.io_depth} tile_kib={args.tile_kib} "
        f"budget_mib={args.budget_mib} odirect={not args.no_odirect}",
        flush=True,
    )

    client = LMCacheFSClient(args.base_path, args.workers, "",
                             not args.no_odirect, 0,
                             args.io_depth,
                             args.budget_mib << 20)
    try:
        if not args.reuse_corpus:
            t = time.monotonic()
            run_phase(client, "submit_batch_set", n_obj, pool,
                      args.batch, args.outstanding)
            os.sync()
            print(f"write: {total / 1e9:.1f} GB in {time.monotonic() - t:.2f}s "
                  f"({total / 1e9 / (time.monotonic() - t):.2f} GB/s)", flush=True)

        for p in range(args.passes):
            a = snap_disk(args.dev, members)
            t0 = time.monotonic()
            run_phase(client, "submit_batch_get", n_obj, pool,
                      args.batch, args.outstanding)
            wall = time.monotonic() - t0
            b = snap_disk(args.dev, members)

            dev_bytes = (b.rd_sectors - a.rd_sectors) * SECTOR
            busy = (b.io_ticks_ms - a.io_ticks_ms) / 1000.0
            rdt = (b.rd_ticks_ms - a.rd_ticks_ms) / 1000.0
            mb = [
                (b.members[k][1] - a.members[k][1]) / 1000.0
                for k in sorted(b.members)
                if k in a.members
            ]
            print(
                f"pass {p}: wall={wall:.2f}s  app={total / 1e9:.1f} GB "
                f"({total / 1e9 / wall:.2f} GB/s)  "
                f"dev={dev_bytes / 1e9:.1f} GB "
                f"({dev_bytes / 1e9 / wall:.2f} GB/s vs wall)",
                flush=True,
            )
            if busy > 0:
                print(
                    f"         busy={busy:.2f}s ({100 * busy / wall:.1f}% of wall)"
                    f"  -> {dev_bytes / 1e9 / busy:.2f} GB/s while busy"
                    f"   rd_ticks_sum={rdt:.1f}s",
                    flush=True,
                )
            if mb:
                print(
                    f"         member busy%: min={100 * min(mb) / wall:.1f} "
                    f"mean={100 * (sum(mb) / len(mb)) / wall:.1f} "
                    f"max={100 * max(mb) / wall:.1f}  (n={len(mb)})",
                    flush=True,
                )
            if hasattr(client, "read_budget_bytes"):
                budget = client.read_budget_bytes()
                if budget:
                    print(
                        f"         budget={budget / 2**20:.1f} MiB in flight"
                        f"  ({budget / args.workers / (args.obj_kib << 10):.1f}"
                        f" objects per worker)",
                        flush=True,
                    )
    finally:
        client.close()
        if not args.keep_corpus:
            shutil.rmtree(args.base_path, ignore_errors=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
