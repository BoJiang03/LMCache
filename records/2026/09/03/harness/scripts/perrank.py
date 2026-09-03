import glob, pstats, sys, re
STEPS=600
GROUPS = {
 "spin(shm_broadcast+sched_yield)": re.compile(r"shm_broadcast\.py|sched_yield"),
 "msgpack_encode": re.compile(r"msgpack_encode"),
 "zmq(send/recv/poll)": re.compile(r"zmq/sugar"),
 "lmcache_*": re.compile(r"lmcache/"),
 "_launch_kernel": re.compile(r"_launch_kernel"),
}
for prefix, name in (("pns","nostore"), ("pmp","mp")):
    print(f"=== {name} ===")
    print(f"{'rank(pid)':>14} {'total':>8} " + " ".join(f"{k[:22]:>23}" for k in GROUPS))
    for f in sorted(glob.glob(f"/home/bo/.claude/jobs/ba4f4ca8/tmp/{prefix}.*.pstats")):
        st = pstats.Stats(f)
        acc = {k: [0.0, 0] for k in GROUPS}
        for func, (cc, nc, tt, ct, _c) in st.stats.items():
            key = f"{func[0]}:{func[2]}"
            for g, rx in GROUPS.items():
                if rx.search(key):
                    acc[g][0] += tt; acc[g][1] += nc
        pid = f.split(".")[-2]
        cells = " ".join(f"{acc[k][0]*1000/STEPS:9.3f}/{acc[k][1]/STEPS:<13.1f}" for k in GROUPS)
        print(f"{pid:>14} {st.total_tt*1000/STEPS:8.2f} {cells}")
    print()
print("cells are  ms/step / calls-per-step")
