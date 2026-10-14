"""Controlled decode ITL probe against the two nodes' API servers.

N decode streams (short prompts, ignore_eos) split evenly across nodes; optionally
a background loop sends long prompts to one node so we can see how they affect
everyone's ITL.  ITL is measured client-side per streamed chunk and normalised
per output token (DSpark emits several tokens per chunk).

  python decode_probe.py --streams 32 --out-tokens 512
  python decode_probe.py --streams 32 --bg-prompt-tokens 30000 --bg-node 26 --bg-mode miss
  python decode_probe.py --streams 32 --bg-prompt-tokens 30000 --bg-node 16 --bg-mode p2p --bg-pool 40

bg modes: miss = fresh prompt each time (full prefill); hit = the same prompt
every time (local cache hit); p2p = a pool of prompts prefilled on the other
node before the measurement, each sent once to --bg-node (loaded over P2P).
"""
import argparse, asyncio, json, random, statistics, time
import aiohttp

NODES = {"26": "http://10.2.201.26:8100", "16": "http://10.2.201.16:8100"}
MODEL = "dsv4flash"
WORDS = ["alpha", "beta", "gamma", "delta", "omega", "kappa", "sigma", "theta"]


def prompt(n_words, seed):
    r = random.Random(seed)
    return " ".join(r.choice(WORDS) + str(r.randint(0, 999)) for _ in range(n_words))


async def stream(session, url, text, max_tokens):
    body = {"model": MODEL, "messages": [{"role": "user", "content": text}],
            "max_tokens": max_tokens, "ignore_eos": True, "stream": True,
            "stream_options": {"include_usage": True}}
    t0 = time.perf_counter(); first = None; last = None; gaps = []; ntok = 0
    async with session.post(url + "/v1/chat/completions", json=body) as resp:
        async for raw in resp.content:
            line = raw.decode().strip()
            if not line.startswith("data:") or line == "data: [DONE]":
                continue
            d = json.loads(line[5:])
            if d.get("usage"):
                ntok = d["usage"]["completion_tokens"]
            if not d.get("choices"):
                continue
            now = time.perf_counter()
            if first is None:
                first = now
            else:
                gaps.append(now - last)
            last = now
    return dict(ttft=first - t0, decode_s=last - first, ntok=ntok, gaps=gaps)


async def decode_worker(session, url, idx, args, results, stop):
    i = 0
    while not stop.is_set():
        r = await stream(session, url, prompt(60, 1000 * idx + i), args.out_tokens)
        r["t_end"] = time.perf_counter()
        results.append(r); i += 1


async def bg_worker(session, args, bg, stop, pool):
    url = NODES[args.bg_node]; i = 0
    while not stop.is_set():
        if args.bg_mode == "p2p":
            if i == len(pool):
                break
            seed = pool[i]
        else:
            seed = 7 if args.bg_mode == "hit" else random.randrange(1 << 30)
        t = time.perf_counter()
        r = await stream(session, url, prompt(args.bg_prompt_tokens // 2, seed), 1)
        bg.append((t, r["ttft"])); i += 1
        await asyncio.sleep(args.bg_gap)


async def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--streams", type=int, default=32)
    ap.add_argument("--out-tokens", type=int, default=512)
    ap.add_argument("--duration", type=float, default=60)
    ap.add_argument("--bg-prompt-tokens", type=int, default=0)
    ap.add_argument("--bg-node", default="26")
    ap.add_argument("--bg-mode", choices=["miss", "hit", "p2p"], default="miss")
    ap.add_argument("--bg-gap", type=float, default=0.0)
    ap.add_argument("--bg-pool", type=int, default=40)
    args = ap.parse_args()

    results, bg, stop = [], [], asyncio.Event()
    conn = aiohttp.TCPConnector(limit=0)
    async with aiohttp.ClientSession(connector=conn, timeout=aiohttp.ClientTimeout(total=3600)) as s:
        pool = []
        if args.bg_mode == "p2p":
            other = NODES["26" if args.bg_node == "16" else "16"]
            pool = [random.randrange(1 << 30) for _ in range(args.bg_pool)]
            for seed in pool:
                await stream(s, other, prompt(args.bg_prompt_tokens // 2, seed), 1)
        urls = list(NODES.values())
        tasks = [asyncio.create_task(decode_worker(s, urls[i % 2], i, args, results, stop))
                 for i in range(args.streams)]
        if args.bg_prompt_tokens:
            tasks.append(asyncio.create_task(bg_worker(s, args, bg, stop, pool)))
        await asyncio.sleep(args.duration)
        stop.set()
        await asyncio.gather(*tasks)

    itl = [r["decode_s"] / max(r["ntok"] - 1, 1) * 1000 for r in results if r["ntok"] > 1]
    gaps = sorted(g * 1000 for r in results for g in r["gaps"])
    q = lambda x, p: x[min(int(p * len(x)), len(x) - 1)]
    out = dict(streams=args.streams, bg_tokens=args.bg_prompt_tokens, bg_mode=args.bg_mode,
               reqs=len(results), itl_ms_p50=round(statistics.median(itl), 2),
               itl_ms_p90=round(q(sorted(itl), .9), 2),
               chunk_gap_ms_p50=round(q(gaps, .5), 2), chunk_gap_ms_p90=round(q(gaps, .9), 2),
               chunk_gap_ms_p99=round(q(gaps, .99), 2),
               tok_per_chunk=round(sum(r["ntok"] for r in results) / max(sum(len(r["gaps"]) + 1 for r in results), 1), 2))
    if bg:
        out.update(bg_reqs=len(bg), bg_ttft_s_p50=round(statistics.median(t for _, t in bg), 2))
    print(json.dumps(out))


asyncio.run(main())
