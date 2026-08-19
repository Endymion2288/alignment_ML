#!/bin/bash
# Fixed-q/p-seed covariance-suppression (mode-3) pilot pipeline.
#
# Stages after the pilot physical production has completed:
#   manifest   -> pilot physical corpus manifest (3 train sources x 8 points)
#   overlay    -> pilot multi-track overlay with canonical mode-0 candidates
#   mode3      -> mode-3 candidate export on the identical overlay events
#   budget     -> same-edge uncertainty budget per payload point (modes 0/1/2/3)
#   coverage   -> candidate-coverage audit for both variants
#   backbone   -> frozen V2 inference on both variants (train split)
#   closure    -> route-selected dx/dy/Ry closure for both variants
#   verdict    -> joint pilot gate evaluation
#
# Everything runs train-only; the frozen V2 checkpoint/calibration/route policy
# and the physical_edge_deduplicated observation semantics are never changed.
set -euo pipefail
cd /eos/home-x/xcheng/FASER/alignment_ML

PILOT_PHYS=outputs/mc24_mode3_suppression_pilot_physical_v1
PILOT_MANIFEST=$PILOT_PHYS/physical_corpus_manifest.json
OVERLAY0=outputs/mc24_mode3_suppression_pilot_synthetic_v1
OVERLAY3=outputs/mc24_mode3_suppression_pilot_synthetic_mode3_v1
EVAL=outputs/mc24_mode3_suppression_pilot_eval_v1
FROZEN_V2=outputs/mc24_v3_expanded_trainval_v2_bce_control_v1
SCAN_ROOT=$PILOT_PHYS/sources/mc24_100043_00200_00299/physical_scan
POINTS="iteration_01_reference iteration_01_anchor iteration_01_fd_ift_dx_mm_p iteration_01_fd_ift_dx_mm_m iteration_01_fd_ift_dy_mm_p iteration_01_fd_ift_dy_mm_m iteration_01_fd_ift_ry_mrad_p iteration_01_fd_ift_ry_mrad_m"

stage="${1:-all}"

if [[ "$stage" == "manifest" || "$stage" == "all" ]]; then
  python scripts/build_mode3_suppression_pilot.py manifest \
    --production-manifest outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_v1/physical_corpus_manifest.json \
    --pilot-physical-root "$PILOT_PHYS" \
    --source-id mc24_100043_00200_00299 \
    --source-id mc24_100043_00300_00399 \
    --source-id mc24_100044_00500_00599 \
    --output "$PILOT_MANIFEST"
fi

if [[ "$stage" == "overlay" || "$stage" == "all" ]]; then
  python scripts/materialize_pooled_curriculum_synthetics.py \
    --physical-manifest "$PILOT_MANIFEST" \
    --config configs/mode3_suppression_pilot_overlay.yaml \
    --output-dir "$OVERLAY0"
fi

if [[ "$stage" == "mode3" || "$stage" == "all" ]]; then
  python scripts/build_mode3_suppression_pilot.py mode3-overlay \
    --overlay-root "$OVERLAY0" \
    --output-root "$OVERLAY3"
fi

if [[ "$stage" == "budget" || "$stage" == "all" ]]; then
  for point in $POINTS; do
    python scripts/audit_propagation_uncertainty_budget.py \
      --iteration-manifest "$PILOT_MANIFEST" \
      --point "$point" \
      --split train \
      --output-dir "$EVAL/budget/$point"
  done
fi

if [[ "$stage" == "coverage" || "$stage" == "all" ]]; then
  python scripts/audit_field_candidate_coverage.py \
    --synthetic-manifest "$OVERLAY0/synthetic_corpus_manifest.json" \
    --split train \
    --output-dir "$EVAL/coverage/mode0"
  python scripts/audit_field_candidate_coverage.py \
    --synthetic-manifest "$OVERLAY3/synthetic_corpus_manifest.json" \
    --split train \
    --q-over-p-mode 3 \
    --output-dir "$EVAL/coverage/mode3"
fi

if [[ "$stage" == "backbone" || "$stage" == "all" ]]; then
  for mode in 0 3; do
    manifest="$OVERLAY0/synthetic_corpus_manifest.json"
    if [[ "$mode" == "3" ]]; then manifest="$OVERLAY3/synthetic_corpus_manifest.json"; fi
    for point in $POINTS; do
      python scripts/run_frozen_association_backbone.py \
        --synthetic-manifest "$manifest" \
        --backbone v2 \
        --frozen-output "$FROZEN_V2" \
        --split train \
        --q-over-p-mode "$mode" \
        --payload-id "$point" \
        --output-dir "$EVAL/backbone_mode$mode/$point"
    done
  done
fi

if [[ "$stage" == "closure" || "$stage" == "all" ]]; then
  run_closure () {
    local mode="$1" samples="$2"
    python scripts/run_route_selected_multidof_update.py \
      --scan-root "$SCAN_ROOT" \
      --anchor-point iteration_01_anchor \
      --anchor-association-output "$EVAL/backbone_mode$mode/iteration_01_anchor" \
      --target-point iteration_01_reference \
      --target-association-output "$samples/train/iteration_01_reference" \
      --positive-association-output ift_dx_mm:"$samples/train/iteration_01_fd_ift_dx_mm_p" \
      --positive-association-output ift_dy_mm:"$samples/train/iteration_01_fd_ift_dy_mm_p" \
      --positive-association-output ift_ry_mrad:"$samples/train/iteration_01_fd_ift_ry_mrad_p" \
      --negative-association-output ift_dx_mm:"$samples/train/iteration_01_fd_ift_dx_mm_m" \
      --negative-association-output ift_dy_mm:"$samples/train/iteration_01_fd_ift_dy_mm_m" \
      --negative-association-output ift_ry_mrad:"$samples/train/iteration_01_fd_ift_ry_mrad_m" \
      --observation-kind anchor_selected_field_edge \
      --observation-statistics physical_edge_deduplicated \
      --q-over-p-mode "$mode" \
      --capture-tolerance ift_dx_mm:0.1 --capture-tolerance ift_dy_mm:0.1 --capture-tolerance ift_ry_mrad:1.0 \
      --require-full-rank \
      --output-dir "$EVAL/closure_mode$mode"
  }
  run_closure 0 "$OVERLAY0/samples"
  run_closure 3 "$OVERLAY3/samples"
fi

if [[ "$stage" == "verdict" || "$stage" == "all" ]]; then
  python scripts/evaluate_mode3_suppression_pilot.py \
    --pilot-physical-manifest "$PILOT_MANIFEST" \
    --points $POINTS \
    --coverage-mode0 "$EVAL/coverage/mode0" \
    --coverage-mode3 "$EVAL/coverage/mode3" \
    --backbone-mode0 "$EVAL/backbone_mode0" \
    --backbone-mode3 "$EVAL/backbone_mode3" \
    --closure-mode0 "$EVAL/closure_mode0" \
    --closure-mode3 "$EVAL/closure_mode3" \
    --output-dir "$EVAL/verdict"
fi
echo "STAGE $stage DONE"
