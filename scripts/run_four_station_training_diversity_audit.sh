#!/usr/bin/env bash
# Workbook 63: train-side + new-source diversity audit.
# Does not train, does not retune workbook 62, and does not open test
# or reserved blind sources.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/outputs/mc24_four_station_hard_aware_reduction_v1/training_diversity_audit_v1"
CONTRACT="$ROOT/configs/physical_four_station_training_diversity_feasibility.yaml"
CKPT="$ROOT/outputs/mc24_four_station_hard_aware_reduction_v1/checkpoint"
TRAIN="$ROOT/outputs/mc24_four_station_relative_association_retrain_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
TRANSFER="$ROOT/outputs/mc24_four_station_relative_transfer_validation_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"

if [[ -f "$OUT/decision.json" ]]; then
  echo "skip: $OUT/decision.json already exists"
  exit 0
fi

python scripts/audit_four_station_training_diversity.py \
  --contract "$CONTRACT" \
  --train-manifest "$TRAIN" \
  --transfer-manifest "$TRANSFER" \
  --candidate-frozen-output "$CKPT" \
  --output-dir "$OUT" \
  --device cuda \
  --batch-size 32

echo AUDIT_EXIT=$?
