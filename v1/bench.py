"""Decode-latency-under-P2P benchmark for the DP2/EP2 DeepSeek-V4-Flash deployment.

prime: send a pool of long random prefixes to DP rank 0 (node 26) so that its
       LMCache server holds their KV.
run:   keep `--decode-conc` streaming decode requests on each DP rank and
       measure inter-token latency (ITL).  Optionally, in parallel, send
       pool prefixes (each once) to DP rank 1 (node 16):
         --p2p-mode hit   -> rank 1's LMCache pulls the KV from node 26 (P2P)
         --p2p-mode miss  -> fresh prefixes, full prefill, no P2P (compute control)
       Local rail TX/RX counters are sampled throughout.
"""

import argparse
import asyncio
import itertools
import json
import random
import statistics
import time

import aiohttp

URL = "http://127.0.0.1:8300/v1/completions"
MODEL = "/raid/bo/p2pbench/models/DeepSeek-V4-Flash"
RAILS = {"rail1": "roce_rail1", "rail7": "roce_rail7", "rail8": "roce_rail8"}
VOCAB_LO, VOCAB_HI = 1000, 100000


def prefix(seed, n):
    rng = random.Random(seed)
    return [rng.randrange(VOCAB_LO, VOCAB_HI) for _ in range(n)]


def pct(xs, p):
    if not xs:
        return None
    xs = sorted(xs)
    return xs[min(len(xs) - 1, int(p / 100 * len(xs)))]


async def one_shot(sess, rank, tokens):
    body = {"model": MODEL, "prompt": tokens, "max_tokens": 1, "temperature": 0}
    t0 = time.perf_counter()
    async with sess.post(URL, json=body, headers={"X-data-parallel-rank": str(rank)}) as r:
        js = await r.json()
    dt = time.perf_counter() - t0
    details = (js.get("usage") or {}).get("prompt_tokens_details") or {}
    return dt, details.get("cached_tokens", 0)


async def prime(args):
    sem = asyncio.Semaphore(4)
    async with aiohttp.ClientSession(timeout=aiohttp.ClientTimeout(total=600)) as sess:

        async def go(i):
            async with sem:
                return await one_shot(sess, 0, prefix(args.seed + i, args.prefix_len))

        t0 = time.time()
        res = await asyncio.gather(*(go(i) for i in range(args.pool)))
    print(f"primed {args.pool} prefixes x {args.prefix_len} tok on rank 0 "
          f"in {time.time() - t0:.1f}s; mean latency {statistics.mean(r[0] for r in res):.2f}s")


async def decode_worker(sess, rank, args, stop, itls, ttfts, wid):
    n = 0
    while not stop.is_set():
        toks = prefix(10_000_000 + rank * 100_000 + wid * 1000 + n, args.decode_prompt_len)
        n += 1
        body = {"model": MODEL, "prompt": toks, "max_tokens": args.decode_max_tokens,
                "ignore_eos": True, "temperature": 0, "stream": True}
        t0 = time.perf_counter()
        last = None
        async with sess.post(URL, json=body, headers={"X-data-parallel-rank": str(rank)}) as r:
            async for line in r.content:
                if not line.startswith(b"data:") or line.strip() == b"data: [DONE]":
                    continue
                now = time.perf_counter()
                if last is None:
                    ttfts.append(now - t0)
                elif args.warmup_until <= now:
                    itls.append((now, now - last))
                last = now
                if stop.is_set():
                    break


async def p2p_worker(sess, args, stop, q, out):
    # hit: cycle over the primed pool; node 16's small L1 has evicted a prefix
    # long before it comes round again, so every request is a P2P fetch.
    # miss: a fresh prefix every time.
    while not stop.is_set():
        i = next(q)
        seed = args.seed + i % args.pool if args.p2p_mode == "hit" else 20_000_000 + args.seed + i
        dt, cached = await one_shot(sess, 1, prefix(seed, args.prefix_len))
        out.append((time.perf_counter(), dt, cached))


def read_counters():
    c = {}
    for name, dev in RAILS.items():
        base = f"/sys/class/infiniband/{dev}/ports/1/counters/"
        c[name] = tuple(int(open(base + f).read()) * 4 for f in ("port_xmit_data", "port_rcv_data"))
    return c


async def sampler(stop, samples):
    while not stop.is_set():
        samples.append((time.perf_counter(), read_counters()))
        await asyncio.sleep(0.5)


async def run(args):
    stop = asyncio.Event()
    itls = {0: [], 1: []}
    ttfts = {0: [], 1: []}
    p2p = []
    samples = []
    q = itertools.count(args.pool_offset)
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=None)) as sess:
        tasks = [asyncio.create_task(sampler(stop, samples))]
        for rank in (0, 1):
            for w in range(args.decode_conc):
                tasks.append(asyncio.create_task(
                    decode_worker(sess, rank, args, stop, itls[rank], ttfts[rank], w)))
        t_start = time.perf_counter()
        args.warmup_until = t_start + args.warmup
        await asyncio.sleep(args.warmup)
        if args.p2p_mode != "none":
            for _ in range(args.p2p_conc):
                tasks.append(asyncio.create_task(p2p_worker(sess, args, stop, q, p2p)))
        await asyncio.sleep(args.duration)
        stop.set()
        await asyncio.wait(tasks, timeout=60)
    t_end = args.warmup_until + args.duration

    rails = {}
    win = [s for s in samples if args.warmup_until <= s[0] <= t_end]
    if len(win) >= 2:
        span = win[-1][0] - win[0][0]
        for name in RAILS:
            tx = (win[-1][1][name][0] - win[0][1][name][0]) * 8 / span / 1e9
            rx = (win[-1][1][name][1] - win[0][1][name][1]) * 8 / span / 1e9
            peak_tx = max((b[1][name][0] - a[1][name][0]) * 8 / (b[0] - a[0]) / 1e9
                          for a, b in zip(win, win[1:]))
            rails[name] = {"avg_tx_gbps": round(tx, 2), "avg_rx_gbps": round(rx, 2),
                           "peak_tx_gbps_500ms": round(peak_tx, 2)}

    summary = {"label": args.label, "p2p_mode": args.p2p_mode, "decode_conc": args.decode_conc,
               "prefix_len": args.prefix_len, "duration": args.duration, "rails": rails}
    for rank in (0, 1):
        xs = [d * 1000 for t, d in itls[rank] if t <= t_end]
        summary[f"itl_rank{rank}_ms"] = {
            "n": len(xs), "mean": round(statistics.mean(xs), 2) if xs else None,
            **{f"p{p}": round(pct(xs, p), 2) if xs else None for p in (50, 90, 99, 99.9)}}
    if p2p:
        lat = [d * 1000 for _, d, _ in p2p]
        summary["p2p_req"] = {
            "n": len(p2p), "per_s": round(len(p2p) / args.duration, 2),
            "full_hit": sum(c >= args.prefix_len - 256 for _, _, c in p2p),
            "mean_cached": round(statistics.mean(c for _, _, c in p2p)),
            "lat_ms_p50": round(pct(lat, 50), 1), "lat_ms_p99": round(pct(lat, 99), 1)}
    print(json.dumps(summary, indent=1))
    with open(args.out, "a") as f:
        f.write(json.dumps(summary) + "\n")
    with open(args.out + f".{args.label}.raw.json", "w") as f:
        json.dump({"itl": {r: itls[r] for r in itls}, "p2p": p2p,
                   "samples": [(t, c) for t, c in samples]}, f)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("phase", choices=["prime", "run"])
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--pool", type=int, default=200)
    ap.add_argument("--pool-offset", type=int, default=0)
    ap.add_argument("--prefix-len", type=int, default=16384)
    ap.add_argument("--decode-conc", type=int, default=32)
    ap.add_argument("--decode-prompt-len", type=int, default=128)
    ap.add_argument("--decode-max-tokens", type=int, default=1024)
    ap.add_argument("--p2p-mode", choices=["none", "hit", "miss"], default="none")
    ap.add_argument("--p2p-conc", type=int, default=4)
    ap.add_argument("--warmup", type=float, default=20)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--label", default="run")
    ap.add_argument("--out", default="/raid/bo/dpep/results.jsonl")
    args = ap.parse_args()
    asyncio.run(prime(args) if args.phase == "prime" else run(args))


if __name__ == "__main__":
    main()
