#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PYTHON:-python}"
DATA="${MVTEC_LOCO_ROOT:-$ROOT/data/mvtec_loco}"
MASKS="${CSAD_MASK_ROOT:-$ROOT/data/csad_component_masks}"
WORK="${WORK_DIR:-$ROOT/work}"

mkdir -p "$WORK"

echo "[1/3] PiDiNet first-scale edges"
"$PY" -u "$ROOT/src/prepare_pidinet_edges.py" \
  --data-root "$DATA" \
  --output "$WORK/pidinet_edges" \
  --checkpoint "$ROOT/checkpoints/table5_pidinet.pth"

echo "[2/3] Unified five-category DeepSIM (300 epochs)"
"$PY" -u "$ROOT/src/train_deepsim_unified.py" \
  --data-root "$DATA" \
  --edges-root "$WORK/pidinet_edges" \
  --output "$WORK/deepsim_unified"

echo "[3/3] Online edge manipulation + localization (LAD, 300 epochs)"
"$PY" -u "$ROOT/src/train_localization_online.py" \
  --data-root "$DATA" \
  --edges-root "$WORK/pidinet_edges" \
  --mask-root "$MASKS" \
  --batch-size 12 \
  --deepsim-checkpoint "$WORK/deepsim_unified/latest_net_G.pth" \
  --output "$WORK/localization"

echo "Done. See $WORK/localization/results.json"
