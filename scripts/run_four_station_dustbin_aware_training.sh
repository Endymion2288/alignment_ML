#!/usr/bin/env bash
# Workbook 59: dustbin-aware route-margin V2 control.
# Development / transfer validation are not used to set weights or the OP.
# Sealed test is never opened.  GPU-only for train/inference.
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 STAGE" >&2
  exit 2
fi
stage="$1"

cd "$(dirname "$0")/.."
ROOT="$PWD"
TRAIN_BANK="$ROOT/outputs/mc24_four_station_relative_association_retrain_v1"
TRAIN_OVERLAY="$TRAIN_BANK/overlay_synthetic_v1/synthetic_corpus_manifest.json"
TRANSFER="$ROOT/outputs/mc24_four_station_relative_transfer_validation_v1"
CONTROL="$ROOT/outputs/mc24_four_station_dustbin_aware_route_v1"
WB54="$TRAIN_BANK/retrained_v2"
WB56_SCALE="$ROOT/outputs/mc24_four_station_gauge_consistent_route_v1/workbook54_transfer_score_scale.json"
WB56_ASSOC="$ROOT/outputs/mc24_four_station_gauge_consistent_route_v1/workbook54_transfer_control"
PAYLOADS=(
  iteration_00_reference
  iteration_00_hard_s3_ry
  iteration_00_hard_s3_ry_plus_common
  iteration_00_draw_00
  iteration_00_draw_00_plus_common
  iteration_00_draw_01
  iteration_00_draw_01_plus_common
)

case "$stage" in

train)
  python scripts/train_four_station_gauge_consistent_v2.py \
    --config configs/geometry_aware_transformer_v2_four_station_dustbin_aware.yaml \
    --objective-contract configs/physical_four_station_dustbin_aware_route_training.yaml \
    --synthetic-manifest "$TRAIN_OVERLAY" \
    --output-dir "$CONTROL/checkpoint" \
    --q-over-p-mode 0
  ;;

infer-transfer)
  MAN="$TRANSFER/overlay_synthetic_v1/synthetic_corpus_manifest.json"
  for payload in "${PAYLOADS[@]}"; do
    python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \
      --split validation --device cuda \
      --synthetic-manifest "$MAN" \
      --frozen-output "$CONTROL/checkpoint" \
      --payload-id "$payload" \
      --output-dir "$CONTROL/transfer_association/$payload"
    python scripts/summarize_four_station_frozen_association.py \
      --association-dir "$CONTROL/transfer_association/$payload" \
      --synthetic-manifest "$MAN" \
      --payload-id "$payload" \
      --split validation \
      --output-json "$CONTROL/transfer_association/${payload}_diagnostics.json"
  done
  ;;

score-scale)
  MAN="$TRANSFER/overlay_synthetic_v1/synthetic_corpus_manifest.json"
  python scripts/audit_four_station_transfer_score_scale.py \
    --synthetic-manifest "$MAN" \
    --frozen-output "$CONTROL/checkpoint" \
    --iteration-manifest "$TRANSFER/iteration_manifest.json" \
    --split validation \
    --device cuda \
    --output-json "$CONTROL/transfer_score_scale.json"
  ;;

mechanism)
  python scripts/audit_four_station_dustbin_aware_mechanism.py \
    --synthetic-manifest "$TRANSFER/overlay_synthetic_v1/synthetic_corpus_manifest.json" \
    --candidate-frozen-output "$CONTROL/checkpoint" \
    --output-json "$CONTROL/transfer_mechanism.json" \
    --device cuda \
    --batch-size 32
  ;;

assess)
  python scripts/evaluate_four_station_dustbin_aware_transfer.py \
    --gates configs/physical_four_station_dustbin_aware_route_training.yaml \
    --iteration-manifest "$TRANSFER/iteration_manifest.json" \
    --split validation \
    --candidate-score-scale-json "$CONTROL/transfer_score_scale.json" \
    --control-score-scale-json "$WB56_SCALE" \
    --mechanism-json "$CONTROL/transfer_mechanism.json" \
    --output-json "$CONTROL/transfer_gate_decision.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_reference_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_hard_s3_ry_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_hard_s3_ry_plus_common_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_draw_00_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_draw_00_plus_common_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_draw_01_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_draw_01_plus_common_diagnostics.json" \
    --control-diagnostic-json "$WB56_ASSOC/iteration_00_reference_diagnostics.json" \
    --control-diagnostic-json "$WB56_ASSOC/iteration_00_hard_s3_ry_diagnostics.json" \
    --control-diagnostic-json "$WB56_ASSOC/iteration_00_hard_s3_ry_plus_common_diagnostics.json" \
    --control-diagnostic-json "$WB56_ASSOC/iteration_00_draw_00_diagnostics.json" \
    --control-diagnostic-json "$WB56_ASSOC/iteration_00_draw_00_plus_common_diagnostics.json" \
    --control-diagnostic-json "$WB56_ASSOC/iteration_00_draw_01_diagnostics.json" \
    --control-diagnostic-json "$WB56_ASSOC/iteration_00_draw_01_plus_common_diagnostics.json"
  ;;

*)
  echo "unknown stage: $stage" >&2
  exit 2
  ;;
esac
