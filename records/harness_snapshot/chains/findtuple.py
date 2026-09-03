import glob, pstats
STEPS, WK = 600, 8
for p, name in (("pns","nostore"), ("pmp","mp")):
    st = None
    for f in sorted(glob.glob(f"/home/bo/.claude/jobs/ba4f4ca8/tmp/{p}.*.pstats")):
        st = pstats.Stats(f) if st is None else (st.add(f) or st)
    print(f"=== {name} ===")
    for func, (cc, nc, tt, ct, cl) in st.stats.items():
        s = f"{func[0]}:{func[1]}({func[2]})"
        if ("tuple" in func[2] or "IPCCacheServerKey" in s or "custom_types" in s
            or "_create_key" in s or "worker_transfer" in s or "submit_store" in s
            or "mq.py" in s or "event_ipc" in s):
            print(f"  tot={tt*1000/STEPS/WK:7.3f} cum={ct*1000/STEPS/WK:8.3f} n/step={nc/STEPS/WK:7.2f}  {func[0].split('/')[-1]}:{func[1]}({func[2]})")
    print()
