#!/usr/bin/env bash
set -euo pipefail

WEB_REPO="${WEB_REPO:-$HOME/flying-saucer}"
OUT_DIR="$WEB_REPO/client/public/aurora/allocator"
mkdir -p "$OUT_DIR"

TRACE_TMP="runs/trace_repel_demo_wall.json"

echo "[export] Generating V4 trace: $TRACE_TMP"

uv run aurora alloc repel --ox 40 --oy 0 --radius-m 30 --k 600 \
  --fxy-max 8000 --init-vx 2 --total-s 12 \
  --alpha-rate-deg-s 350 --plenum-tau-s 0.08 \
  --trace-out "$TRACE_TMP" \
  >/dev/null

cp -f "$TRACE_TMP" "$OUT_DIR/v4_repel_wall.json"
echo "[ok] Exported: v4_repel_wall.json"
