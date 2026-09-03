import glob, pstats
STEPS, WK = 600, 8
TARGETS = {("shm_broadcast.py", 657), ("shm_broadcast.py", 176)}
def load(prefix):
    st = None
    for f in sorted(glob.glob(f"/home/bo/.claude/jobs/ba4f4ca8/tmp/{prefix}.*.pstats")):
        if st is None: st = pstats.Stats(f)
        else: st.add(f)
    return st
for p, name in (("pns","nostore"), ("pmp","mp")):
    st = load(p)
    print(f"--- {name} ---")
    for func, (cc, nc, tt, ct, callers) in st.stats.items():
        if (func[0].split("/")[-1], func[1]) not in TARGETS: continue
        print(f"  {func[0].split('/')[-1]}:{func[1]}({func[2]})  tot={tt*1000/STEPS/WK:.3f} "
              f"cum={ct*1000/STEPS/WK:.3f} calls/step/wk={nc/STEPS/WK:.1f}")
        rows=[]
        for c, v in callers.items():
            n = v[1] if isinstance(v, tuple) else v
            cum = v[3] if isinstance(v, tuple) else 0.0
            rows.append((cum, n, c))
        for cum, n, c in sorted(rows, reverse=True)[:5]:
            print(f"      <- {c[0].split('/')[-1]}:{c[1]}({c[2]})  calls/step/wk={n/STEPS/WK:.2f} cum={cum*1000/STEPS/WK:.3f}")
    print()
