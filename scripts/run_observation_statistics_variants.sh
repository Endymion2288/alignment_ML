#!/bin/bash
# Observation-statistics variants for the frozen iteration-1 route-selected update.
# replica_weighted control already exists (update_v1 / update_validation_v1).
set -u
cd /eos/home-x/xcheng/FASER/alignment_ML
SYN=outputs/mc24_multidof_ift_iteration01_anchor_trainval_synthetic_v1/samples
SCAN=outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_v1/sources/mc24_100043_00200_00299/physical_scan
VSCAN=outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_v1/sources/mc24_100047_00000_00049/physical_scan

run_variant () {
  local split="$1" semantics="$2"
  local backbone scan_root sample_root
  backbone="outputs/mc24_multidof_ift_iteration01_v2_frozen_backbone_${split}_v1"
  sample_root="$SYN/$split"
  if [ "$split" = train ]; then scan_root="$SCAN"; else scan_root="$VSCAN"; fi
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
    --observation-statistics "$semantics" \
    --capture-tolerance ift_dx_mm:0.1 --capture-tolerance ift_dy_mm:0.1 --capture-tolerance ift_ry_mrad:1.0 \
    --require-full-rank \
    --output-dir "outputs/mc24_multidof_ift_iteration01_route_selected_update_${split}_${semantics}_v1" 2>&1 | tail -1
  echo "=== ${split} ${semantics} done ==="
}

for split in train validation; do
  run_variant "$split" physical_edge_deduplicated
  run_variant "$split" physical_edge_inverse_multiplicity_weighted
done
echo "ALL SEMANTICS DONE"
