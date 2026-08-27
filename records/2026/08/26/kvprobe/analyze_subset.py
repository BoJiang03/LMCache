"""Compare pass1 vs pass2 of a subset probe run: answer flips and
decision-gap (Yes-No logprob difference) movement, split by whether the
question flipped in the full 2374 run."""
import json
import sys


def gap(lps: dict) -> float | None:
    yes = lps.get("Yes", lps.get("yes"))
    no = lps.get("No", lps.get("no"))
    if yes is None or no is None:
        return None
    return round(yes - no, 6)


def parse(text: str) -> str:
    t = text.strip().lower()
    if t.startswith("yes"):
        return "yes"
    if t.startswith("no"):
        return "no"
    return ""


for path in sys.argv[1:]:
    r = json.load(open(path))
    p1, p2 = r["passes"]["pass1"], r["passes"]["pass2"]
    flips, gap_moves, text_diffs = [], [], []
    cached1 = sum(x["num_cached_tokens"] for x in p1)
    cached2 = sum(x["num_cached_tokens"] for x in p2)
    for a, b in zip(p1, p2):
        idx, was = a["orig_index"], a["flipped_in_full_run"]
        if a["text"] != b["text"]:
            text_diffs.append(idx)
        if parse(a["text"]) != parse(b["text"]):
            flips.append((idx, was, parse(a["text"]), parse(b["text"])))
        g1, g2 = gap(a["lps"]), gap(b["lps"])
        if g1 is not None and g2 is not None and g1 != g2:
            gap_moves.append((idx, was, g1, g2, round(g2 - g1, 6)))
    print(f"=== {path} ({r['backend']}) {len(p1)} items ===")
    print(f"  cached tokens pass1={cached1} pass2={cached2}")
    print(f"  text diffs: {len(text_diffs)}  answer flips: {len(flips)}")
    for f in flips:
        print(f"    flip idx={f[0]} full_run_flipped={f[1]} {f[2]}->{f[3]}")
    print(f"  decision-gap moves: {len(gap_moves)}")
    for g in gap_moves[:25]:
        print(f"    gap idx={g[0]} full={g[1]} {g[2]} -> {g[3]} (d={g[4]})")
