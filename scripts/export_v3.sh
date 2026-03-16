#!/usr/bin/env bash
set -euo pipefail

WEB_REPO="${WEB_REPO:-$HOME/flying-saucer}"
OUT_DIR="$WEB_REPO/client/public/aurora-vtol/allocator"
mkdir -p "$OUT_DIR"

echo "[export] Output dir: $OUT_DIR"

uv run aurora-vtol alloc step \
  --dir-a-deg 0 --dir-b-deg 180 \
  --fxy 3000 --step-time-s 3 --total-s 8 \
  | tee "$OUT_DIR/v3_step.json" >/dev/null

uv run aurora-vtol alloc step-snap \
  --dir-a-deg 0 --dir-b-deg 180 \
  --fxy 1200 \
  --step-time-s 1.6 \
  --snap-stop-s 1.3 \
  --brake-gain 2.2 \
  --alpha-rate-deg-s 500 \
  --plenum-tau-s 0.05 \
  --total-s 8 \
  | tee "$OUT_DIR/v3_step_snap.json" >/dev/null

echo "[ok] Exported: v3_step.json, v3_step_snap.json"
