import glob, pstats
STEPS, WK = 600, 8
def load(prefix):
    st = None
    for f in sorted(glob.glob(f"/home/bo/.claude/jobs/ba4f4ca8/tmp/{prefix}.*.pstats")):
        st = pstats.Stats(f) if st is None else (st.add(f) or st)
    return st
for p, name in (("pns","nostore"), ("pmp","mp")):
    st = load(p)
    print(f"=== {name}  total={st.total_tt*1000/STEPS/WK:.2f} ms/step/wk ===")
    for func, (cc, nc, tt, ct, callers) in sorted(st.stats.items(), key=lambda kv: -kv[1][3])[:14]:
        print(f"  cum={ct*1000/STEPS/WK:8.3f} tot={tt*1000/STEPS/WK:7.3f} n/step={nc/STEPS/WK:9.1f}  "
              f"{func[0].split('/')[-1]}:{func[1]}({func[2]})")
    print()
