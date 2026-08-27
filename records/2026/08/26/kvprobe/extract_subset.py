"""Extract an MME item subset for the KV-dedup probe: the union of flipped
indices across the three qwen runs, their same-image siblings, plus evenly
spaced extra pairs for statistics. CPU-only; writes JSON with orig_index."""
import json
import sys

sys.path.insert(0, "/home/bo/LMCache-worktrees/multi_modal_verify/tests/e2e_mm")
from benchmark_parity import MMEBenchmark  # noqa: E402

FLIPPED = sorted({58, 241, 299, 329, 352, 392, 396, 449, 465, 531, 741, 781,
                  823, 949, 996, 1072, 1482, 1576, 1618, 24})

bench = MMEBenchmark()
items = bench.load_items(0)
print(f"loaded {len(items)} items", flush=True)

by_qid: dict[str, list[int]] = {}
for i, it in enumerate(items):
    by_qid.setdefault(it["qid"], []).append(i)

selected: set[int] = set()
for idx in FLIPPED:
    selected.update(by_qid[items[idx]["qid"]])
# 35 extra pairs, evenly spaced, skipping any qid already in.
extra_pairs = 0
for i in range(0, len(items), 66):
    qid = items[i]["qid"]
    if any(j in selected for j in by_qid[qid]):
        continue
    selected.update(by_qid[qid])
    extra_pairs += 1
    if extra_pairs >= 35:
        break

out = [{"orig_index": i, "flipped": i in FLIPPED, **items[i]}
       for i in sorted(selected)]
path = sys.argv[1]
with open(path, "w") as f:
    json.dump(out, f)
print(f"wrote {len(out)} items ({extra_pairs} extra pairs) to {path}", flush=True)
