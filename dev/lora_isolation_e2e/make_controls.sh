#!/usr/bin/env bash
# Build the two control trees from this checkout (native .so files included)
# and roll back fix files to BASE_REF:
#   prefix    - both files rolled back: the bug as on #5843
#   utilsonly - only the lazy coalesce rolled back: tags lost under lazy offload
# Usage: make_controls.sh [BASE_REF]  (default: the #5843 head)
set -eu
source "$(dirname "${BASH_SOURCE[0]}")/env.sh"
BASE_REF=${1:-3c6003ee5}
UTILS=lmcache/integration/vllm/utils.py
LAZY=lmcache/integration/vllm/lazy_offload_manager.py
mkdir -p "$CONTROLS"
for name in prefix utilsonly; do
  rsync -a --exclude build --exclude=.git --exclude dev/lora_isolation_e2e "$REPO/" "$CONTROLS/$name/"
done
git -C "$REPO" show "$BASE_REF:$UTILS" > "$CONTROLS/prefix/$UTILS"
git -C "$REPO" show "$BASE_REF:$LAZY" > "$CONTROLS/prefix/$LAZY"
git -C "$REPO" show "$BASE_REF:$LAZY" > "$CONTROLS/utilsonly/$LAZY"
echo "controls in $CONTROLS"
