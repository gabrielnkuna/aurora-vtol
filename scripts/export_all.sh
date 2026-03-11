#!/usr/bin/env bash
set -euo pipefail
./scripts/export_v1_v2.sh
./scripts/export_v3.sh
./scripts/export_v4.sh
echo "[ok] All allocator artifacts exported to website repo."
