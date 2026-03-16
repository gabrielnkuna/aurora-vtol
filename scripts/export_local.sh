#!/usr/bin/env bash
set -euo pipefail

# Export the various demo/step/trace JSONs into the local `runs/` directory
# rather than pushing them to the website repo.  This is convenient for
# offline work and visualization inside this repository.

OUT_DIR="runs"
mkdir -p "$OUT_DIR"

echo "[export-local] Output dir: $OUT_DIR"

uv run aurora-vtol alloc demo --version v1 --dir-deg 90 --fxy 3000 --mz-nm 0 \
  | tee "$OUT_DIR/v1_demo.json" >/dev/null

uv run aurora-vtol alloc demo --version v2 --dir-deg 90 --fxy 3000 --mz-nm 0 \
  | tee "$OUT_DIR/v2_demo_mz0.json" >/dev/null

uv run aurora-vtol alloc demo --version v2 --dir-deg 90 --fxy 3000 --mz-nm 2000 \
  | tee "$OUT_DIR/v2_demo_mz2000.json" >/dev/null

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

uv run aurora-vtol alloc repel --ox 40 --oy 0 --radius-m 30 --k 600 \
  --fxy-max 8000 --init-vx 2 --total-s 12 \
  --alpha-rate-deg-s 350 --plenum-tau-s 0.08 \
  --trace-out "$OUT_DIR/v4_repel_wall.json" \
  >/dev/null

echo "[ok] Exported to runs/: v1_demo.json, v2_demo_mz0.json, v2_demo_mz2000.json, v3_step.json, v3_step_snap.json, v4_repel_wall.json"
