import time
from lmcache.v1.multiprocess.custom_types import IPCCacheServerKey
from lmcache.v1.multiprocess.mq import msgspec_encode, msgspec_decode

def mk(n, off):
    return IPCCacheServerKey(model_name="gpt-oss-120b", world_size=8, worker_id=0,
                             num_kv_readers=1, token_ids=tuple(range(n)),
                             start=off, end=off+n, request_id="r0", cache_salt="",
                             token_offset=off)

for label, n, off in (("old  full prefix N=60000", 60000, 0),
                      ("new  delta        N=8192", 8192, 51808)):
    k = mk(n, off)
    b = msgspec_encode(k, cls=IPCCacheServerKey)
    k2 = msgspec_decode(b, cls=IPCCacheServerKey)
    assert k2 == k and k2.token_offset == k.token_offset, "round trip mismatch"
    t = time.perf_counter()
    for _ in range(50):
        msgspec_encode(k, cls=IPCCacheServerKey)
    enc = (time.perf_counter()-t)/50*1000
    src = list(range(n))
    t = time.perf_counter()
    for _ in range(50):
        tuple(src)
    tup = (time.perf_counter()-t)/50*1000
    print(f"{label}: {len(b):>7,} B   tuple {tup:.3f} ms   encode {enc:.3f} ms")

# forward compat: a payload written without the new field decodes as 0
import msgspec, dataclasses
old_style = {f.name: getattr(mk(16,0), f.name) for f in dataclasses.fields(IPCCacheServerKey)
             if f.name != "token_offset"}
raw = msgspec.msgpack.encode(old_style)
k3 = msgspec_decode(raw, cls=IPCCacheServerKey)
print(f"payload without token_offset decodes to token_offset={k3.token_offset}")
