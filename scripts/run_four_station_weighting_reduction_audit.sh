#!/usr/bin/env bash
# Workbook 61: train-only weighting / reduction feasibility.
# Does not train, does not load transfer, and does not open test.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
CONTROL="$ROOT/outputs/mc24_four_station_dustbin_aware_route_v1"
OUT="$CONTROL/weighting_reduction_feasibility_v1"
CKPT="$CONTROL/checkpoint"
CONTRACT="$ROOT/configs/physical_four_station_weighting_reduction_feasibility.yaml"
MANIFEST="$ROOT/outputs/mc24_four_station_relative_association_retrain_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"

if [[ -f "$OUT/summary.json" ]]; then
  echo "skip: $OUT/summary.json already exists"
  exit 0
fi
if [[ -d "$OUT" ]]; then
  rm -rf "$OUT"
fi

python scripts/audit_four_station_weighting_reduction.py \
  --contract "$CONTRACT" \
  --synthetic-manifest "$MANIFEST" \
  --candidate-frozen-output "$CKPT" \
  --output-dir "$OUT" \
  --device cuda \
  --batch-size 32

echo AUDIT_EXIT=$?
