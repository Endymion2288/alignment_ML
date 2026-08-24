#!/usr/bin/env bash
# Workbook 60: solver-generated hard-negative mining audit.
# Does not train, does not retune the operating point, and does not open test.
set -euo pipefail

cd "$(dirname "$0")/.."
ROOT="$PWD"
CONTROL="$ROOT/outputs/mc24_four_station_dustbin_aware_route_v1"
OUT="$CONTROL/solver_hard_negative_mining_v1"
CKPT="$CONTROL/checkpoint"
CONTRACT="$ROOT/configs/physical_four_station_solver_hard_negative_feasibility.yaml"

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 transfer|train|all" >&2
  exit 2
fi
stage="$1"

run_split() {
  local split="$1"
  local manifest="$2"
  local split_dir="$OUT/$split"
  if [[ -f "$split_dir/summary.json" ]]; then
    echo "skip $split: $split_dir/summary.json already exists"
    return 0
  fi
  if [[ -d "$split_dir" ]]; then
    # Incomplete previous attempt (empty or crashed before summary).
    rm -rf "$split_dir"
  fi
  python scripts/audit_four_station_solver_hard_negatives.py \
    --contract "$CONTRACT" \
    --split "$split" \
    --synthetic-manifest "$manifest" \
    --candidate-frozen-output "$CKPT" \
    --output-dir "$OUT" \
    --device cuda \
    --batch-size 32
}

case "$stage" in
transfer)
  run_split transfer \
    "$ROOT/outputs/mc24_four_station_relative_transfer_validation_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
  ;;
train)
  run_split train \
    "$ROOT/outputs/mc24_four_station_relative_association_retrain_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
  ;;
all)
  run_split transfer \
    "$ROOT/outputs/mc24_four_station_relative_transfer_validation_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
  run_split train \
    "$ROOT/outputs/mc24_four_station_relative_association_retrain_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
  ;;
*)
  echo "unknown stage: $stage" >&2
  exit 2
  ;;
esac
echo AUDIT_EXIT=$?
