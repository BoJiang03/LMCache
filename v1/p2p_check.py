"""One-shot P2P sanity check: prime a prefix on DP rank 0, then fetch it on DP rank 1."""
import asyncio, sys, time
import aiohttp
from bench import one_shot, prefix

def rail7():
    b = "/sys/class/infiniband/roce_rail7/ports/1/counters/"
    return [int(open(b + f).read()) * 4 for f in ("port_xmit_data", "port_rcv_data")]

async def main(seed, n):
    toks = prefix(seed, n)
    async with aiohttp.ClientSession() as s:
        dt, c = await one_shot(s, 0, toks)
        print(f"rank0 prime: {dt*1000:.0f} ms cached={c}")
        await asyncio.sleep(3)
        a = rail7()
        dt, c = await one_shot(s, 1, toks)
        b = rail7()
        print(f"rank1 fetch: {dt*1000:.0f} ms cached={c}/{n}  rail7 tx {(b[0]-a[0])/2**20:.1f} MiB rx {(b[1]-a[1])/2**20:.1f} MiB")
        dt, c = await one_shot(s, 1, prefix(seed + 999999, n))
        print(f"rank1 cold miss (control): {dt*1000:.0f} ms cached={c}")

asyncio.run(main(int(sys.argv[1]), int(sys.argv[2])))
