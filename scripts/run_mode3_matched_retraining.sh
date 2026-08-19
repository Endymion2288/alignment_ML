#!/usr/bin/env bash
# Matched-retraining validation of the mode-3 (fixed-q/p seed, suppressed
# q/p covariance transport) propagation hypothesis against the canonical
# mode-0 control.
#
# Stage list (run in order; each stage refuses to overwrite non-empty outputs):
#   refresh-manifests  refresh v3 per-point completion flags after Condor
#   verify-bank        re-produced banks reproduce the production banks exactly
#   overlay            re-materialize both overlays from the new banks
#   verify-overlay     re-materialized overlays contain identical synthetic events
#   mode3-candidates   export mode-3 candidate graphs for the identical events
#   train-mlp          retrain MLP + route on the mode-3 corpus
#   train-v1-full      retrain V1 full-context on the mode-3 corpus
#   train-v1-nocontext retrain V1 no-context on the mode-3 corpus
#   train-v2           retrain V2 BCE route-query on the mode-3 corpus
#   backbone           frozen-backbone inference arms on the iteration-1 overlay
#   coverage           candidate truth-pair recall audits on the training corpus
#   bad-edge           known bad 0->1 mismatch score/rank audits
#   closure            mode-3 physical_edge_deduplicated route-selected update
#   verdict            consolidated promotion-criteria evaluation
#
# The sealed test split is never touched: every loader restricts to
# train/validation and the training corpus manifests certify it.

set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 STAGE" >&2
  exit 2
fi
stage="$1"

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/outputs/mc24_mode3_matched_retraining_v1"

# Production (mode-0) references.
V3_BANK0="$ROOT/outputs/mc24_v3_expanded_trainval_physical_v1"
IT1_BANK0="$ROOT/outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_v1"
V3_OVERLAY0="$ROOT/outputs/mc24_v3_expanded_trainval_synthetic_v1"
IT1_OVERLAY0="$ROOT/outputs/mc24_multidof_ift_iteration01_anchor_trainval_synthetic_v1"
MLP0="$ROOT/outputs/mc24_v3_expanded_trainval_mlp_control_v1"
V1FULL0="$ROOT/outputs/mc24_v3_expanded_trainval_v1_geometry_refreeze_v1"
V1NOCTX0="$ROOT/outputs/mc24_v3_expanded_trainval_v1_no_context_control_v1"
V2_0="$ROOT/outputs/mc24_v3_expanded_trainval_v2_bce_control_v1"
BACKBONE0="$ROOT/outputs/mc24_multidof_ift_iteration01_v2_frozen_backbone_validation_v1"
CLOSURE0="$ROOT/outputs/mc24_multidof_ift_iteration01_route_selected_update_validation_physical_edge_deduplicated_v1"

# Mode-3 re-production / retraining artifacts.
V3_BANK3="$ROOT/outputs/mc24_v3_expanded_trainval_physical_mode3_v1"
IT1_BANK3="$ROOT/outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_mode3_v1"
V3_OVERLAY3BANK="$ROOT/outputs/mc24_v3_expanded_trainval_synthetic_mode3bank_v1"
IT1_OVERLAY3BANK="$ROOT/outputs/mc24_multidof_ift_iteration01_anchor_trainval_synthetic_mode3bank_v1"
V3_OVERLAY3="$ROOT/outputs/mc24_v3_expanded_trainval_synthetic_mode3_v1"
IT1_OVERLAY3="$ROOT/outputs/mc24_multidof_ift_iteration01_anchor_trainval_synthetic_mode3_v1"
MLP3="$ROOT/outputs/mc24_v3_expanded_trainval_mode3_mlp_control_v1"
V1FULL3="$ROOT/outputs/mc24_v3_expanded_trainval_mode3_v1_full_context_v1"
V1NOCTX3="$ROOT/outputs/mc24_v3_expanded_trainval_mode3_v1_no_context_v1"
V2_3="$ROOT/outputs/mc24_v3_expanded_trainval_mode3_v2_bce_control_v1"

case "$stage" in

refresh-manifests)
  # The v3 curriculum manifest embeds per-point completion flags written at
  # prepare time; refresh them from the on-disk Condor outputs in a single
  # process (workers never touch the shared manifest).  The iteration-1 bank
  # additionally needs its corpus-style manifest (physical_corpus_manifest.json)
  # for the overlay materializer; it is assembled read-only from the iteration
  # manifest, exactly as for the original iteration-1 bank.
  python scripts/build_physical_curriculum_corpus.py \
    --config configs/physical_curriculum_v3_expanded_trainval.yaml \
    --output-dir "$V3_BANK3" \
    --prepare-only --resume
  python scripts/assemble_multisource_multidof_iteration_manifest.py \
    --iteration-manifest "$IT1_BANK3/iteration_manifest.json" \
    --materialization-config configs/physical_alignment_iteration_trainval.yaml \
    --output "$IT1_BANK3/physical_corpus_manifest.json"
  ;;

verify-bank)
  python scripts/verify_mode3_bank_identity.py \
    --production-manifest "$V3_BANK0/physical_corpus_manifest.json" \
    --mode3-manifest "$V3_BANK3/physical_corpus_manifest.json" \
    --output "$OUT/verify_bank_v3.json"
  python scripts/verify_mode3_bank_identity.py \
    --production-manifest "$IT1_BANK0/iteration_manifest.json" \
    --mode3-manifest "$IT1_BANK3/iteration_manifest.json" \
    --output "$OUT/verify_bank_iter1.json"
  ;;

overlay)
  # The v3 production overlay predates the condition-axis seed identity and
  # was materialized in two per-split runs (group indices, hence synthetic
  # run ids and provenance namespaces, restart per split).  Exact
  # reproduction therefore requires the legacy magnitude-only seed identity
  # and the same per-split invocation sequence.  The iteration-1 overlay was
  # materialized in one current-convention run and must NOT use the legacy
  # flag.
  python scripts/materialize_pooled_curriculum_synthetics.py \
    --physical-manifest "$V3_BANK3/physical_corpus_manifest.json" \
    --config configs/physical_curriculum_v3_expanded_trainval.yaml \
    --legacy-seed-identity --split train \
    --output-dir "$V3_OVERLAY3BANK"
  python scripts/materialize_pooled_curriculum_synthetics.py \
    --physical-manifest "$V3_BANK3/physical_corpus_manifest.json" \
    --config configs/physical_curriculum_v3_expanded_trainval.yaml \
    --legacy-seed-identity --split validation --resume \
    --output-dir "$V3_OVERLAY3BANK"
  python scripts/materialize_pooled_curriculum_synthetics.py \
    --physical-manifest "$IT1_BANK3/physical_corpus_manifest.json" \
    --config configs/physical_alignment_iteration_trainval.yaml \
    --output-dir "$IT1_OVERLAY3BANK"
  ;;

verify-overlay)
  python scripts/verify_mode3_overlay_identity.py \
    --original-overlay "$V3_OVERLAY0" \
    --rematerialized-overlay "$V3_OVERLAY3BANK" \
    --output "$OUT/verify_overlay_v3.json"
  python scripts/verify_mode3_overlay_identity.py \
    --original-overlay "$IT1_OVERLAY0" \
    --rematerialized-overlay "$IT1_OVERLAY3BANK" \
    --output "$OUT/verify_overlay_iter1.json"
  ;;

mode3-candidates)
  python scripts/build_mode3_suppression_pilot.py mode3-overlay \
    --overlay-root "$V3_OVERLAY3BANK" \
    --output-root "$V3_OVERLAY3"
  python scripts/build_mode3_suppression_pilot.py mode3-overlay \
    --overlay-root "$IT1_OVERLAY3BANK" \
    --output-root "$IT1_OVERLAY3"
  ;;

train-mlp)
  python scripts/run_global_assignment_mlp_baseline.py \
    --synthetic-manifest "$V3_OVERLAY3/synthetic_corpus_manifest.json" \
    --config configs/physical_global_assignment_mlp_v3_expanded_control.yaml \
    --validation-only --q-over-p-mode 3 \
    --output-dir "$MLP3"
  ;;

train-v1-full)
  python scripts/train_geometry_aware_transformer_v1.py \
    --synthetic-manifest "$V3_OVERLAY3/synthetic_corpus_manifest.json" \
    --config configs/geometry_aware_transformer_v1_expanded_control_mode3.yaml \
    --ablation geometry_aware_transformer --q-over-p-mode 3 \
    --output-dir "$V1FULL3"
  ;;

train-v1-nocontext)
  python scripts/train_geometry_aware_transformer_v1.py \
    --synthetic-manifest "$V3_OVERLAY3/synthetic_corpus_manifest.json" \
    --config configs/geometry_aware_transformer_v1_expanded_no_context_control_mode3.yaml \
    --ablation geometry_aware_no_multistation_context --q-over-p-mode 3 \
    --output-dir "$V1NOCTX3"
  ;;

train-v2)
  python scripts/train_route_aware_transformer_v2.py \
    --synthetic-manifest "$V3_OVERLAY3/synthetic_corpus_manifest.json" \
    --config configs/geometry_aware_transformer_v2_expanded_bce_control_mode3.yaml \
    --q-over-p-mode 3 \
    --output-dir "$V2_3"
  ;;

backbone)
  # mode3-data + mode3-model (V2) and the mode3-data + frozen-mode0-model
  # domain-shift control, anchor payload of the iteration-1 mode-3 overlay
  # validation split, matching the per-payload layout of the historical
  # mode-0 backbone.  The V1 retrained models are compared through their own
  # train/validation route metrics; the historical mode-0 V1 artifacts are not
  # loadable by the frozen backbone (no per-ablation operating point), so no
  # matched V1 backbone arm exists.
  python scripts/run_frozen_association_backbone.py \
    --synthetic-manifest "$IT1_OVERLAY3/synthetic_corpus_manifest.json" \
    --backbone v2 --frozen-output "$V2_3" \
    --split validation --q-over-p-mode 3 \
    --payload-id iteration_01_anchor \
    --output-dir "$OUT/backbone_mode3data_mode3model_v2/iteration_01_anchor"
  python scripts/run_frozen_association_backbone.py \
    --synthetic-manifest "$IT1_OVERLAY3/synthetic_corpus_manifest.json" \
    --backbone v2 --frozen-output "$V2_0" \
    --split validation --q-over-p-mode 3 \
    --payload-id iteration_01_anchor \
    --output-dir "$OUT/backbone_mode3data_frozen_mode0model_v2/iteration_01_anchor"
  ;;

coverage)
  python scripts/audit_field_candidate_coverage.py \
    --synthetic-manifest "$V3_OVERLAY0/synthetic_corpus_manifest.json" \
    --split train --q-over-p-mode 0 \
    --output-dir "$OUT/coverage/mode0"
  python scripts/audit_field_candidate_coverage.py \
    --synthetic-manifest "$V3_OVERLAY3/synthetic_corpus_manifest.json" \
    --split train --q-over-p-mode 3 \
    --output-dir "$OUT/coverage/mode3"
  ;;

pareto)
  # Validation-only matched operating-point scan.  Checkpoint, feature
  # standardizers, calibration, candidate graph and the V2 route architecture
  # stay frozen; only the route unmatched penalty (primary) and the frozen
  # per-pair threshold scale (secondary probe) vary, via inference-time
  # overrides recorded in each output's frozen_backbone metadata.
  PENALTIES=(-2.5 -2.0 -1.5 -1.25 -1.0 -0.75 -0.5 -0.35 -0.2 -0.1 0.0)
  for penalty in "${PENALTIES[@]}"; do
    tag="penalty_$(echo "$penalty" | sed 's/-/m/; s/\./p/')"
    python scripts/run_frozen_association_backbone.py \
      --synthetic-manifest "$IT1_OVERLAY0/synthetic_corpus_manifest.json" \
      --backbone v2 --frozen-output "$V2_0" \
      --split validation --q-over-p-mode 0 \
      --payload-id iteration_01_anchor \
      --unmatched-penalty-override "$penalty" \
      --output-dir "$OUT/pareto/mode0_data_mode0_model/$tag/iteration_01_anchor"
    python scripts/run_frozen_association_backbone.py \
      --synthetic-manifest "$IT1_OVERLAY3/synthetic_corpus_manifest.json" \
      --backbone v2 --frozen-output "$V2_3" \
      --split validation --q-over-p-mode 3 \
      --payload-id iteration_01_anchor \
      --unmatched-penalty-override "$penalty" \
      --output-dir "$OUT/pareto/mode3_data_mode3_model/$tag/iteration_01_anchor"
  done
  # Secondary threshold-axis probes at each arm's own frozen penalty.
  for scale in 10 100; do
    python scripts/run_frozen_association_backbone.py \
      --synthetic-manifest "$IT1_OVERLAY0/synthetic_corpus_manifest.json" \
      --backbone v2 --frozen-output "$V2_0" \
      --split validation --q-over-p-mode 0 \
      --payload-id iteration_01_anchor \
      --threshold-scale-override "$scale" \
      --output-dir "$OUT/pareto/mode0_data_mode0_model/thrscale_${scale}/iteration_01_anchor"
    python scripts/run_frozen_association_backbone.py \
      --synthetic-manifest "$IT1_OVERLAY3/synthetic_corpus_manifest.json" \
      --backbone v2 --frozen-output "$V2_3" \
      --split validation --q-over-p-mode 3 \
      --payload-id iteration_01_anchor \
      --threshold-scale-override "$scale" \
      --output-dir "$OUT/pareto/mode3_data_mode3_model/thrscale_${scale}/iteration_01_anchor"
  done
  python scripts/evaluate_mode3_pareto_frontier.py \
    --arm mode0_data_mode0_model:"$OUT/pareto/mode0_data_mode0_model" \
    --arm mode3_data_mode3_model:"$OUT/pareto/mode3_data_mode3_model" \
    --reference-efficiency 0.7426 \
    --output "$OUT/pareto/mode3_pareto_frontier.json"
  ;;

bad-edge)
  # Matched control arm on the historical mode-0 overlay plus the two mode-3
  # arms (retrained mode-3 model and frozen mode-0 domain-shift control).
  python scripts/audit_mode3_bad_edge_score.py \
    --synthetic-manifest "$IT1_OVERLAY0/synthetic_corpus_manifest.json" \
    --model-artifact "$V2_0" --q-over-p-mode 0 \
    --output "$OUT/bad_edge_mode0_data_mode0_model.json"
  python scripts/audit_mode3_bad_edge_score.py \
    --synthetic-manifest "$IT1_OVERLAY3/synthetic_corpus_manifest.json" \
    --model-artifact "$V2_3" --q-over-p-mode 3 \
    --output "$OUT/bad_edge_mode3data_mode3model.json"
  python scripts/audit_mode3_bad_edge_score.py \
    --synthetic-manifest "$IT1_OVERLAY3/synthetic_corpus_manifest.json" \
    --model-artifact "$V2_0" --q-over-p-mode 3 \
    --output "$OUT/bad_edge_mode3data_frozen_mode0model.json"
  ;;

closure)
  SCAN_ROOT="$IT1_BANK3/sources/mc24_100047_00000_00049/physical_scan"
  SAMPLES="$IT1_OVERLAY3/samples/validation"
  python scripts/run_route_selected_multidof_update.py \
    --scan-root "$SCAN_ROOT" \
    --anchor-point iteration_01_anchor \
    --anchor-association-output "$OUT/backbone_mode3data_mode3model_v2/iteration_01_anchor" \
    --target-point iteration_01_reference \
    --target-association-output "$SAMPLES/iteration_01_reference" \
    --positive-association-output ift_dx_mm:"$SAMPLES/iteration_01_fd_ift_dx_mm_p" \
    --positive-association-output ift_dy_mm:"$SAMPLES/iteration_01_fd_ift_dy_mm_p" \
    --positive-association-output ift_ry_mrad:"$SAMPLES/iteration_01_fd_ift_ry_mrad_p" \
    --negative-association-output ift_dx_mm:"$SAMPLES/iteration_01_fd_ift_dx_mm_m" \
    --negative-association-output ift_dy_mm:"$SAMPLES/iteration_01_fd_ift_dy_mm_m" \
    --negative-association-output ift_ry_mrad:"$SAMPLES/iteration_01_fd_ift_ry_mrad_m" \
    --observation-kind anchor_selected_field_edge \
    --observation-statistics physical_edge_deduplicated \
    --q-over-p-mode 3 \
    --capture-tolerance ift_dx_mm:0.1 --capture-tolerance ift_dy_mm:0.1 --capture-tolerance ift_ry_mrad:1.0 \
    --require-full-rank \
    --output-dir "$OUT/closure_mode3_dedup"
  ;;

verdict)
  python scripts/evaluate_mode3_matched_retraining.py \
    --coverage-mode0 "$OUT/coverage/mode0/coverage_audit.json" \
    --coverage-mode3 "$OUT/coverage/mode3/coverage_audit.json" \
    --model-metrics mlp:mode0:"$MLP0/metrics.json" \
    --model-metrics mlp:mode3:"$MLP3/metrics.json" \
    --model-metrics v1full:mode0:"$V1FULL0/validation_metrics.json" \
    --model-metrics v1full:mode3:"$V1FULL3/geometry_aware_transformer/validation_metrics.json" \
    --model-metrics v1nocontext:mode0:"$V1NOCTX0/geometry_aware_no_multistation_context/validation_metrics.json" \
    --model-metrics v1nocontext:mode3:"$V1NOCTX3/geometry_aware_no_multistation_context/validation_metrics.json" \
    --model-metrics v2:mode0:"$V2_0/validation_results.json" \
    --model-metrics v2:mode3:"$V2_3/validation_results.json" \
    --backbone-anchor mode0_data_mode0_model:"$BACKBONE0/iteration_01_anchor/association_summary.json" \
    --backbone-anchor mode3_data_mode3_model:"$OUT/backbone_mode3data_mode3model_v2/iteration_01_anchor/association_summary.json" \
    --backbone-anchor mode3_data_frozen_mode0_model:"$OUT/backbone_mode3data_frozen_mode0model_v2/iteration_01_anchor/association_summary.json" \
    --bad-edge mode0_data_mode0_model:"$OUT/bad_edge_mode0_data_mode0_model.json" \
    --bad-edge mode3_data_mode3_model:"$OUT/bad_edge_mode3data_mode3model.json" \
    --bad-edge mode3_data_frozen_mode0_model:"$OUT/bad_edge_mode3data_frozen_mode0model.json" \
    --closure-mode0 "$CLOSURE0/route_selected_update.json" \
    --closure-mode3 "$OUT/closure_mode3_dedup/route_selected_update.json" \
    --output "$OUT/mode3_matched_retraining_verdict.json"
  ;;

*)
  echo "unknown stage: $stage" >&2
  exit 2
  ;;
esac
