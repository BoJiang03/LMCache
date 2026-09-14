"""Reduce a bench2 log to one row per object size.

Each size ran warmup, L1, P1, L2, P2.  A row is trustworthy only if the two
legacy blocks agree with each other and the two pooled blocks agree with each
other; that spread is the drift, and an effect smaller than the drift is not
an effect.  Bench1 failed exactly this test at 24 MiB.
"""
import collections
import re
import statistics
import sys

BPT_KIB = 24  # bytes per token per rank, gpt-oss-20b TP=2 -> chunk_size = MiB*1024/24


def parse(path):
    sizes, obj, arm = {}, None, None
    qd = collections.defaultdict(dict)
    last_wall = [0.0]
    for line in open(path):
        # Take the size from the arm line in KiB, not from the ======== header:
        # the header integer-divides, so 1536 KiB prints there as "1 MiB".
        m = re.match(r"### (\w+) obj=(\d+)", line)
        if m:
            arm, obj = m.group(1), int(m.group(2))
            sizes.setdefault(obj, {})
            continue
        m = re.search(r"wall=([\d.]+)s.*?\(([\d.]+) GB/s\)\s+dev=", line)
        if m and obj is not None and arm and arm != "warmup_discard":
            sizes[obj].setdefault(arm, []).append(float(m.group(2)))
            last_wall[0] = float(m.group(1))
            continue
        # rd_ticks_sum is the sum of per-request service time across the array.
        # Divided by wall it is the mean number of reads the devices had in
        # flight, which is the quantity this PR is actually about.
        m = re.search(r"rd_ticks_sum=([\d.]+)s", line)
        if m and obj is not None and arm and arm != "warmup_discard" and last_wall[0]:
            qd[obj].setdefault(arm, []).append(float(m.group(1)) / last_wall[0])
    return sizes, qd


def main(path):
    sizes, qd = parse(path)
    print(f"{'obj':>8} {'chunk':>6} {'legacy':>8} {'pooled':>8} {'gain':>6} "
          f"{'L drift':>8} {'P drift':>8} {'L q':>6} {'P q':>6}  verdict")
    for obj in sorted(sizes):
        a = sizes[obj]
        if not all(k in a for k in ("L1", "P1", "L2", "P2")):
            continue
        leg = statistics.median(a["L1"] + a["L2"])
        pool = statistics.median(a["P1"] + a["P2"])
        ldrift = abs(statistics.median(a["L1"]) - statistics.median(a["L2"]))
        pdrift = abs(statistics.median(a["P1"]) - statistics.median(a["P2"]))
        gain = pool / leg
        effect = abs(pool - leg)
        noise = max(ldrift, pdrift)
        if effect < noise:
            verdict = "NO CLAIM (effect < drift)"
        elif effect < 2 * noise:
            verdict = "weak (effect < 2x drift)"
        else:
            verdict = "solid"
        q = qd.get(obj, {})
        lq = statistics.median(q.get('L1', [0]) + q.get('L2', [0]))
        pq = statistics.median(q.get('P1', [0]) + q.get('P2', [0]))
        lqs = f'{lq:.0f}' if lq else '-'
        pqs = f'{pq:.0f}' if pq else '-'
        cs = round(obj / BPT_KIB)
        print(f"{obj / 1024:>6.1f} MiB {cs:>6} {leg:>8.2f} {pool:>8.2f} {gain:>5.2f}x "
              f"{ldrift:>8.2f} {pdrift:>8.2f} {lqs:>6} {pqs:>6}  {verdict}")


if __name__ == "__main__":
    for p in sys.argv[1:]:
        print(f"\n=== {p} ===")
        main(p)
