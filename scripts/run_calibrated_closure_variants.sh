#!/bin/bash
# Calibrated-covariance closure variants for the frozen iteration-1 banks.
# Read-only re-analysis: no retraining, no repropagation, no new iterations.
set -u
cd /eos/home-x/xcheng/FASER/alignment_ML
CAL=outputs/mc24_multidof_ift_pull_calibration_train_v1/mode0_covariance_calibration_train_frozen.json
SCAN=outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_v1/sources/mc24_100043_00200_00299/physical_scan
VSCAN=outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_v1/sources/mc24_100047_00000_00049/physical_scan
SYN=outputs/mc24_multidof_ift_iteration01_anchor_trainval_synthetic_v1/samples

run_update () {
  local split="$1" tag="$2"; shift 2
  local extra=("$@")
  local backbone scan_root sample_root
  if [ "$split" = train ]; then
    backbone=outputs/mc24_multidof_ift_iteration01_v2_frozen_backbone_train_v1
    scan_root="$SCAN"; sample_root="$SYN/train"
  else
    backbone=outputs/mc24_multidof_ift_iteration01_v2_frozen_backbone_validation_v1
    scan_root="$VSCAN"; sample_root="$SYN/validation"
  fi
  python scripts/run_route_selected_multidof_update.py \
    --scan-root "$scan_root" \
    --anchor-point iteration_01_anchor \
    --anchor-association-output "$backbone/iteration_01_anchor" \
    --target-point iteration_01_reference \
    --target-association-output "$sample_root/iteration_01_reference" \
    --positive-association-output ift_dx_mm:"$sample_root/iteration_01_fd_ift_dx_mm_p" \
    --positive-association-output ift_dy_mm:"$sample_root/iteration_01_fd_ift_dy_mm_p" \
    --positive-association-output ift_ry_mrad:"$sample_root/iteration_01_fd_ift_ry_mrad_p" \
    --negative-association-output ift_dx_mm:"$sample_root/iteration_01_fd_ift_dx_mm_m" \
    --negative-association-output ift_dy_mm:"$sample_root/iteration_01_fd_ift_dy_mm_m" \
    --negative-association-output ift_ry_mrad:"$sample_root/iteration_01_fd_ift_ry_mrad_m" \
    --observation-kind anchor_selected_field_edge \
    --capture-tolerance ift_dx_mm:0.1 --capture-tolerance ift_dy_mm:0.1 --capture-tolerance ift_ry_mrad:1.0 \
    --require-full-rank "${extra[@]}" \
    --output-dir "outputs/mc24_multidof_ift_iteration01_route_selected_update_${split}_${tag}" 2>&1 | tail -2
  echo "=== ${split}/${tag} done ==="
}

run_update train calibrated_v2 --covariance-calibration "$CAL" --anchor-payload-sample "$SYN/train/iteration_01_anchor"
run_update train calibrated_huber_v2 --covariance-calibration "$CAL" --huber-k 2.5 --anchor-payload-sample "$SYN/train/iteration_01_anchor"
run_update validation calibrated_v2 --covariance-calibration "$CAL" --anchor-payload-sample "$SYN/validation/iteration_01_anchor"
run_update validation calibrated_huber_v2 --covariance-calibration "$CAL" --huber-k 2.5 --anchor-payload-sample "$SYN/validation/iteration_01_anchor"
echo "ALL VARIANTS DONE"
