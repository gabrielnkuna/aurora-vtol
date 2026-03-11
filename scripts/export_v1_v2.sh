#!/usr/bin/env bash
set -euo pipefail

# Website repo location (cloned from https://github.com/gabrielnkuna/flying-saucer.git)
WEB_REPO="${WEB_REPO:-$HOME/flying-saucer}"
OUT_DIR="$WEB_REPO/client/public/aurora/allocator"
mkdir -p "$OUT_DIR"

echo "[export] Website repo: $WEB_REPO"
echo "[export] Output dir:   $OUT_DIR"

uv run aurora alloc demo --version v1 --dir-deg 90 --fxy 3000 --mz-nm 0 \
  | tee "$OUT_DIR/v1_demo.json" >/dev/null

uv run aurora alloc demo --version v2 --dir-deg 90 --fxy 3000 --mz-nm 0 \
  | tee "$OUT_DIR/v2_demo_mz0.json" >/dev/null

uv run aurora alloc demo --version v2 --dir-deg 90 --fxy 3000 --mz-nm 2000 \
  | tee "$OUT_DIR/v2_demo_mz2000.json" >/dev/null

echo "[ok] Exported: v1_demo.json, v2_demo_mz0.json, v2_demo_mz2000.json"
