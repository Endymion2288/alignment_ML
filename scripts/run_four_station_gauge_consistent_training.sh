#!/usr/bin/env bash
# Workbook 56: one pre-registered V2 gauge-consistent route-training control.
# Development validation is not a gate and is not used to set loss weights.
# Transfer production is the only production association gate.  Sealed test
# is never opened.  GPU-only for train/inference.
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
CONTROL="$ROOT/outputs/mc24_four_station_gauge_consistent_route_v1"
WB54="$TRAIN_BANK/retrained_v2"
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

audit-sources)
  python scripts/audit_four_station_transfer_sources.py \
    --output-json "$CONTROL/transfer_source_audit.json"
  ;;

prepare-transfer)
  python scripts/prepare_four_station_relative_curriculum.py \
    --source-config configs/physical_curriculum_four_station_relative_transfer_sources.yaml \
    --iteration-template configs/physical_refit_four_station_relative_transfer_curriculum.yaml \
    --output-root "$TRANSFER" \
    --iteration 0 \
    --nevents 50
  ;;

submit-transfer)
  python scripts/submit_multisource_multidof_iteration_condor.py \
    --iteration-manifest "$TRANSFER/iteration_manifest.json" \
    --submit-dir "$TRANSFER/condor" \
    --request-memory-mb 6000 \
    --job-flavour tomorrow \
    --schedd-mode eossubmit \
    --submit
  ;;

assemble-transfer)
  python scripts/assemble_multisource_multidof_iteration_manifest.py \
    --iteration-manifest "$TRANSFER/iteration_manifest.json" \
    --materialization-config configs/physical_alignment_iteration_four_station_relative_transfer.yaml \
    --output "$TRANSFER/physical_corpus_manifest.json"
  ;;

overlay-transfer)
  python scripts/materialize_pooled_curriculum_synthetics.py \
    --config configs/physical_alignment_iteration_four_station_relative_transfer.yaml \
    --physical-manifest "$TRANSFER/physical_corpus_manifest.json" \
    --output-dir "$TRANSFER/overlay_synthetic_v1" \
    --split validation
  ;;

train)
  python scripts/train_four_station_gauge_consistent_v2.py \
    --config configs/geometry_aware_transformer_v2_four_station_gauge_consistent.yaml \
    --objective-contract configs/physical_four_station_gauge_consistent_route_training.yaml \
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
    python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \
      --split validation --device cuda \
      --synthetic-manifest "$MAN" \
      --frozen-output "$WB54" \
      --payload-id "$payload" \
      --output-dir "$CONTROL/workbook54_transfer_control/$payload"
    python scripts/summarize_four_station_frozen_association.py \
      --association-dir "$CONTROL/workbook54_transfer_control/$payload" \
      --synthetic-manifest "$MAN" \
      --payload-id "$payload" \
      --split validation \
      --output-json "$CONTROL/workbook54_transfer_control/${payload}_diagnostics.json"
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
  python scripts/audit_four_station_transfer_score_scale.py \
    --synthetic-manifest "$MAN" \
    --frozen-output "$WB54" \
    --iteration-manifest "$TRANSFER/iteration_manifest.json" \
    --split validation \
    --device cuda \
    --output-json "$CONTROL/workbook54_transfer_score_scale.json"
  ;;

assess)
  python scripts/evaluate_four_station_gauge_consistent_transfer.py \
    --gates configs/physical_four_station_gauge_consistent_route_training.yaml \
    --iteration-manifest "$TRANSFER/iteration_manifest.json" \
    --split validation \
    --candidate-score-scale-json "$CONTROL/transfer_score_scale.json" \
    --control-score-scale-json "$CONTROL/workbook54_transfer_score_scale.json" \
    --output-json "$CONTROL/transfer_gate_decision.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_reference_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_hard_s3_ry_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_hard_s3_ry_plus_common_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_draw_00_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_draw_00_plus_common_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_draw_01_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/transfer_association/iteration_00_draw_01_plus_common_diagnostics.json" \
    --control-diagnostic-json "$CONTROL/workbook54_transfer_control/iteration_00_reference_diagnostics.json" \
    --control-diagnostic-json "$CONTROL/workbook54_transfer_control/iteration_00_hard_s3_ry_diagnostics.json" \
    --control-diagnostic-json "$CONTROL/workbook54_transfer_control/iteration_00_hard_s3_ry_plus_common_diagnostics.json" \
    --control-diagnostic-json "$CONTROL/workbook54_transfer_control/iteration_00_draw_00_diagnostics.json" \
    --control-diagnostic-json "$CONTROL/workbook54_transfer_control/iteration_00_draw_00_plus_common_diagnostics.json" \
    --control-diagnostic-json "$CONTROL/workbook54_transfer_control/iteration_00_draw_01_diagnostics.json" \
    --control-diagnostic-json "$CONTROL/workbook54_transfer_control/iteration_00_draw_01_plus_common_diagnostics.json"
  ;;

*)
  echo "unknown stage: $stage" >&2
  exit 2
  ;;
esac
