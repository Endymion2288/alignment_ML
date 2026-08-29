#!/usr/bin/env bash
# Workbook 64: source-disjoint six-source V2, frozen workbook-62 objective.
# Do not retune loss, weights, reduction, margin, threshold, penalty,
# Platt, candidate builder, solver, or architecture.  Do not open 15D
# WLS unless the reserved blind gate fully passes.  Sealed test stays
# closed.  GPU-only training.  Training never reads blind data.
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 STAGE" >&2
  exit 2
fi
stage="$1"

cd "$(dirname "$0")/.."
ROOT="$PWD"
EXISTING="$ROOT/outputs/mc24_four_station_relative_association_retrain_v1"
NEW_TRAIN="$ROOT/outputs/mc24_four_station_source_diversity_new_train_v1"
TRAIN="$ROOT/outputs/mc24_four_station_source_diversity_train_v1"
BLIND="$ROOT/outputs/mc24_four_station_source_diversity_blind_v1"
CONTROL="$ROOT/outputs/mc24_four_station_source_diversity_v1"
WB62="$ROOT/outputs/mc24_four_station_hard_aware_reduction_v1/checkpoint"
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

audit-new-sources)
  python scripts/audit_four_station_source_diversity_new_sources.py \
    --source-config configs/physical_curriculum_four_station_diversity_new_train_sources.yaml \
    --output-dir "$CONTROL/new_source_audit_v1"
  ;;

audit-blind-unused)
  python scripts/audit_four_station_reserved_blind_unused.py \
    --repo-root "$ROOT" \
    --output-json "$CONTROL/blind_unused_audit.json"
  ;;

prepare-train)
  python scripts/prepare_four_station_relative_curriculum.py \
    --source-config configs/physical_curriculum_four_station_diversity_new_train_sources.yaml \
    --iteration-template configs/physical_refit_four_station_relative_association_curriculum.yaml \
    --output-root "$NEW_TRAIN" \
    --iteration 0 \
    --nevents 50
  ;;

submit-train)
  python scripts/submit_multisource_multidof_iteration_condor.py \
    --iteration-manifest "$NEW_TRAIN/iteration_manifest.json" \
    --submit-dir "$NEW_TRAIN/condor" \
    --request-memory-mb 6000 \
    --job-flavour tomorrow \
    --schedd-mode eossubmit \
    --submit
  ;;

assemble-train)
  python scripts/assemble_multisource_multidof_iteration_manifest.py \
    --iteration-manifest "$NEW_TRAIN/iteration_manifest.json" \
    --materialization-config configs/physical_alignment_iteration_four_station_diversity_train.yaml \
    --output "$NEW_TRAIN/physical_corpus_manifest.json"
  ;;

merge-train)
  python scripts/merge_four_station_source_diversity_train_corpus.py \
    --existing-train-corpus "$EXISTING/physical_corpus_manifest.json" \
    --new-train-corpus "$NEW_TRAIN/physical_corpus_manifest.json" \
    --materialization-config configs/physical_alignment_iteration_four_station_diversity_train.yaml \
    --output "$TRAIN/physical_corpus_manifest.json"
  ;;

overlay-train)
  python scripts/materialize_pooled_curriculum_synthetics.py \
    --config configs/physical_alignment_iteration_four_station_diversity_train.yaml \
    --physical-manifest "$TRAIN/physical_corpus_manifest.json" \
    --output-dir "$TRAIN/overlay_synthetic_v1" \
    --split train
  ;;

coverage-sanity)
  python scripts/audit_four_station_source_diversity_coverage_sanity.py \
    --new-train-manifest "$TRAIN/overlay_synthetic_v1/synthetic_corpus_manifest.json" \
    --candidate-frozen-output "$WB62" \
    --output-dir "$TRAIN/coverage_sanity_v1" \
    --device cuda
  ;;

train)
  python scripts/train_four_station_gauge_consistent_v2.py \
    --config configs/geometry_aware_transformer_v2_four_station_source_diversity.yaml \
    --objective-contract configs/physical_four_station_diversity_training.yaml \
    --synthetic-manifest "$TRAIN/overlay_synthetic_v1/synthetic_corpus_manifest.json" \
    --output-dir "$CONTROL/checkpoint" \
    --q-over-p-mode 0
  ;;

freeze-checkpoint)
  python scripts/freeze_four_station_source_diversity_checkpoint.py \
    --checkpoint-dir "$CONTROL/checkpoint" \
    --output-json "$CONTROL/checkpoint_freeze.json"
  ;;

prepare-blind)
  python scripts/prepare_four_station_relative_curriculum.py \
    --source-config configs/physical_curriculum_four_station_diversity_blind_production_sources.yaml \
    --iteration-template configs/physical_refit_four_station_relative_association_curriculum_blind.yaml \
    --output-root "$BLIND" \
    --iteration 0 \
    --nevents 50
  ;;

submit-blind)
  python scripts/submit_multisource_multidof_iteration_condor.py \
    --iteration-manifest "$BLIND/iteration_manifest.json" \
    --submit-dir "$BLIND/condor" \
    --request-memory-mb 6000 \
    --job-flavour tomorrow \
    --schedd-mode eossubmit \
    --submit
  ;;

assemble-blind)
  python scripts/assemble_multisource_multidof_iteration_manifest.py \
    --iteration-manifest "$BLIND/iteration_manifest.json" \
    --materialization-config configs/physical_alignment_iteration_four_station_diversity_blind.yaml \
    --output "$BLIND/physical_corpus_manifest.json"
  ;;

overlay-blind)
  python scripts/materialize_pooled_curriculum_synthetics.py \
    --config configs/physical_alignment_iteration_four_station_diversity_blind.yaml \
    --physical-manifest "$BLIND/physical_corpus_manifest.json" \
    --output-dir "$BLIND/overlay_synthetic_v1" \
    --split validation
  ;;

infer-blind)
  MAN="$BLIND/overlay_synthetic_v1/synthetic_corpus_manifest.json"
  for payload in "${PAYLOADS[@]}"; do
    python scripts/run_frozen_association_backbone.py --backbone v2 --q-over-p-mode 0 \
      --split validation --device cuda \
      --synthetic-manifest "$MAN" \
      --frozen-output "$CONTROL/checkpoint" \
      --payload-id "$payload" \
      --output-dir "$CONTROL/blind_association/$payload"
    python scripts/summarize_four_station_frozen_association.py \
      --association-dir "$CONTROL/blind_association/$payload" \
      --synthetic-manifest "$MAN" \
      --payload-id "$payload" \
      --split validation \
      --output-json "$CONTROL/blind_association/${payload}_diagnostics.json"
  done
  ;;

mechanism-blind)
  python scripts/audit_four_station_source_diversity_blind_mechanism.py \
    --synthetic-manifest "$BLIND/overlay_synthetic_v1/synthetic_corpus_manifest.json" \
    --candidate-frozen-output "$CONTROL/checkpoint" \
    --output-json "$CONTROL/blind_mechanism.json" \
    --device cuda \
    --batch-size 32
  ;;

assess)
  python scripts/evaluate_four_station_source_diversity_blind.py \
    --gates configs/physical_four_station_diversity_training.yaml \
    --iteration-manifest "$BLIND/iteration_manifest.json" \
    --output-json "$CONTROL/blind_gate_decision.json" \
    --mechanism-json "$CONTROL/blind_mechanism.json" \
    --split validation \
    --candidate-diagnostic-json "$CONTROL/blind_association/iteration_00_reference_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/blind_association/iteration_00_hard_s3_ry_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/blind_association/iteration_00_hard_s3_ry_plus_common_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/blind_association/iteration_00_draw_00_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/blind_association/iteration_00_draw_00_plus_common_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/blind_association/iteration_00_draw_01_diagnostics.json" \
    --candidate-diagnostic-json "$CONTROL/blind_association/iteration_00_draw_01_plus_common_diagnostics.json"
  ;;

*)
  echo "unknown stage: $stage" >&2
  exit 2
  ;;
esac
