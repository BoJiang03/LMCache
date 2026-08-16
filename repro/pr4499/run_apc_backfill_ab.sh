#!/usr/bin/env bash
set -euo pipefail

CODE_SHA="f4c77ed4808e00cd90047daaf7d6d0455ea6f3dd"
ROOT="$(git rev-parse --show-toplevel)"
HERE="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PYTHON="${SMOKE_PYTHON:-python3}"
RESULTS="${1:-$HERE/results/apc-ab}"
BASELINE="$(mktemp -d)/lmcache-no-eager-apc-backfill"
mkdir -p "$RESULTS"

cleanup() {
  git -C "$ROOT" worktree remove --force "$BASELINE" >/dev/null 2>&1 || true
  rm -rf "$(dirname "$BASELINE")"
}
trap cleanup EXIT

if ! git -C "$ROOT" diff --quiet "$CODE_SHA" -- lmcache; then
  echo "Production files differ from the recorded code-under-test $CODE_SHA" >&2
  exit 2
fi

git -C "$ROOT" worktree add --detach "$BASELINE" "$CODE_SHA"
"$PYTHON" - "$BASELINE/lmcache/integration/vllm/lmcache_mp_connector.py" <<'PY'
from pathlib import Path
import sys

path = Path(sys.argv[1])
source = path.read_text()
old = '''        tracker.num_vllm_hit_tokens = (
            num_computed_tokens
            // self._hit_alignment_tokens
            * self._hit_alignment_tokens
        )
'''
new = '''        if self.lazy_offload:
            tracker.num_vllm_hit_tokens = (
                num_computed_tokens
                // self._hit_alignment_tokens
                * self._hit_alignment_tokens
            )
'''
if source.count(old) != 1:
    raise SystemExit("expected exactly one eager APC-accounting block")
path.write_text(source.replace(old, new))
PY

echo "=== Baseline production diff (the only A/B variable) ==="
git -C "$BASELINE" diff -- lmcache/integration/vllm/lmcache_mp_connector.py

echo "=== A: PR behavior ==="
SMOKE_REPO="$ROOT" "$PYTHON" "$HERE/apc_backfill.py" \
  --repetitions 5 --output "$RESULTS/with_backfill.json"

echo "=== B: same code with eager APC accounting disabled ==="
SMOKE_REPO="$BASELINE" "$PYTHON" "$HERE/apc_backfill.py" \
  --repetitions 5 --output "$RESULTS/without_backfill.json"

"$PYTHON" - "$RESULTS" <<'PY'
from pathlib import Path
import json
import sys

results = Path(sys.argv[1])
with_fix = json.loads((results / "with_backfill.json").read_text())
without = json.loads((results / "without_backfill.json").read_text())
new = with_fix["third_request_median_seconds"]
old = without["third_request_median_seconds"]
summary = {
    "latency_reduction_percent": (old - new) / old * 100,
    "speedup": old / new,
    "with_backfill_median_seconds": new,
    "without_backfill_median_seconds": old,
    "with_backfill_rebuilt_objects": with_fix["rebuilt_objects"],
    "without_backfill_rebuilt_objects": without["rebuilt_objects"],
    "with_backfill_retrieved_ranges": with_fix["retrieved_token_ranges"],
    "without_backfill_retrieved_ranges": without["retrieved_token_ranges"],
}
(results / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
print(json.dumps(summary, indent=2))
PY
