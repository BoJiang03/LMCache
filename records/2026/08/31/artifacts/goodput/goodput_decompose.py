#!/usr/bin/env python3
"""Engine time budget, decode step cost, hit timeline and L1 residency.

All four analyses read only artifacts the sweep already writes:
``<arm>_samples.log`` (periodic prometheus scrapes of both the engine and the
MP server) and ``<arm>_server.log`` (vLLM's own throughput lines).

Usage: goodput_decompose.py <sweep-dir> <arm> [<arm> ...]
"""

# Standard
import re
import sys

_LOGLINE = re.compile(
    r"Avg prompt throughput: ([\d.]+) tokens/s, "
    r"Avg generation throughput: ([\d.]+) tokens/s, "
    r"Running: (\d+) reqs, Waiting: (\d+) reqs"
)
_SAMPLE_KEYS = {
    "vllm:num_requests_running": "B",
    "vllm:generation_tokens_total": "gen",
    "lmcache_mp_l1_usage_ratio": "ur",
    "lmcache_mp_l1_memory_usage_bytes": "mem",
    "lmcache_mp_l1_write_chunks_total": "wr",
    "lmcache_mp_lookup_requested_tokens_total": "rq",
    "lmcache_mp_lookup_hit_tokens_total": "hi",
}


def read_samples(path: str) -> list[dict]:
    """Parse one ``<arm>_samples.log`` into a list of scrape dicts."""
    out: list[dict] = []
    cur: dict = {}
    for line in open(path, errors="ignore"):
        if line.startswith("=== t="):
            if cur:
                out.append(cur)
            cur = {"t": int(line.split("=")[-1])}
            continue
        key = line.split("{")[0].split(" ")[0]
        try:
            value = float(line.rsplit(" ", 1)[1])
        except (IndexError, ValueError):
            continue
        if key in _SAMPLE_KEYS:
            cur[_SAMPLE_KEYS[key]] = value
        elif key == "vllm:prompt_tokens_by_source_total":
            for src, field in (
                ("local_compute", "cp"),
                ("local_cache_hit", "l0"),
                ("external_kv_transfer", "ex"),
            ):
                if f'source="{src}"' in line:
                    cur[field] = value
    if cur:
        out.append(cur)
    return out


def _solve(rows: list[list[float]], rhs: list[float]) -> list[float]:
    """Least squares by normal equations with Gauss-Jordan elimination."""
    n = len(rows[0])
    m = [
        [sum(r[i] * r[j] for r in rows) for j in range(n)]
        + [sum(r[i] * y for r, y in zip(rows, rhs))]
        for i in range(n)
    ]
    for i in range(n):
        p = max(range(i, n), key=lambda r: abs(m[r][i]))
        m[i], m[p] = m[p], m[i]
        for r in range(n):
            if r != i and m[i][i]:
                f = m[r][i] / m[i][i]
                for c in range(i, n + 1):
                    m[r][c] -= f * m[i][c]
    return [m[i][n] / m[i][i] for i in range(n)]


def budget(samples: list[dict]) -> None:
    """Split engine wall time into prefill, decode and external transfer.

    Regresses the scrape interval on (computed prefill tokens, decode steps,
    externally transferred tokens); the fitted coefficients are the unit costs
    and their products with the totals are the shares.
    """
    a, y = [], []
    for lo, hi in zip(samples, samples[1:]):
        dt = hi["t"] - lo["t"]
        if not 0 < dt <= 30:
            continue
        comp = hi.get("cp", 0) - lo.get("cp", 0)
        ext = hi.get("ex", 0) - lo.get("ex", 0)
        gen = hi.get("gen", 0) - lo.get("gen", 0)
        b = max(hi.get("B", 0), 1)
        if comp == 0 and gen == 0 and ext == 0:
            continue
        a.append([comp, gen / b, ext])
        y.append(float(dt))
    c = _solve(a, y)
    total = sum(y)
    share = [c[i] * sum(r[i] for r in a) / total for i in range(3)]
    print(
        f"  budget  prefill {share[0] * 100:4.1f}%  decode {share[1] * 100:4.1f}%  "
        f"ext {share[2] * 100:4.1f}%  unattributed {(1 - sum(share)) * 100:4.1f}%"
        f"   ({c[0] * 1e6:.0f} us/prefill-token, {c[1] * 1e3:.1f} ms/decode-step)"
    )


def decode_steps(path: str) -> None:
    """Report decode step cost from windows with no prefill and a stable B."""
    rows = []
    for line in open(path, errors="ignore"):
        m = _LOGLINE.search(line)
        if m:
            rows.append((float(m.group(1)), float(m.group(2)), int(m.group(3))))
    out = []
    for (p0, _g0, b0), (p1, g1, b1) in zip(rows, rows[1:]):
        if p1 == 0.0 and b0 >= 6 and b1 >= 6 and abs(b0 - b1) <= 1 and g1 > 0:
            out.append((b1, g1 / b1, 1000 * b1 / g1))
    out.sort()
    print(
        "  decode  "
        + ", ".join(f"B={b} {r:.0f}tok/s/u {t:.0f}ms" for b, r, t in out[:12])
    )


def hits(samples: list[dict], window: int = 200) -> None:
    """Print the instantaneous L0 / L1 / recompute split over time."""
    t0 = samples[0]["t"]
    buckets: dict[int, list[float]] = {}
    for lo, hi in zip(samples, samples[1:]):
        d = buckets.setdefault(int((lo["t"] - t0) // window), [0.0, 0.0, 0.0])
        for i, field in enumerate(("l0", "ex", "cp")):
            d[i] += hi.get(field, 0) - lo.get(field, 0)
    for k in sorted(buckets):
        l0, ex, cp = buckets[k]
        total = l0 + ex + cp
        if total < 200000:
            continue
        print(
            f"  hits    t+{k * window:>5}s  L0 {l0 / total * 100:5.1f}%  "
            f"L1 {ex / total * 100:5.1f}%  recompute {cp / total * 100:5.1f}%"
        )


def residency(samples: list[dict]) -> None:
    """Report bytes per stored token, L1 turnover rate and residency time."""
    full = [s for s in samples if "mem" in s and "wr" in s]
    if len(full) < 2:
        return
    grow = [
        (b["mem"] - a["mem"]) / (b["wr"] - a["wr"])
        for a, b in zip(full, full[1:])
        if b.get("ur", 1) < 0.6 and b["mem"] > a["mem"] and b["wr"] > a["wr"]
    ]
    grow.sort()
    if not grow:
        return
    per_unit = grow[len(grow) // 2]
    per_token = per_unit * 8 / 256
    # Only intervals that actually stored: a sample log can carry a long idle
    # head, and averaging the write rate over it understates the turnover.
    active = [
        (b["t"] - a["t"], b["wr"] - a["wr"])
        for a, b in zip(full, full[1:])
        if b["wr"] > a["wr"]
    ]
    stored = sum(w for _, w in active) / 8 * 256
    seconds = sum(dt for dt, _ in active)
    resident = full[-1]["mem"] / per_token
    last = full[-1]
    print(
        f"  L1      {per_token:.0f} B/token stored, resident {last['mem'] / 1e9:.0f} GB "
        f"= {resident / 1e6:.2f}M tokens, wrote {stored / 1e6:.1f}M tokens in "
        f"{seconds}s -> residency {resident / (stored / seconds):.0f}s"
    )


def main() -> None:
    """Run all four analyses for each arm named on the command line."""
    root = sys.argv[1]
    for arm in sys.argv[2:]:
        print(f"== {arm}")
        samples = read_samples(f"{root}/{arm}_samples.log")
        budget(samples)
        decode_steps(f"{root}/{arm}_server.log")
        residency(samples)
        hits(samples)


if __name__ == "__main__":
    main()
