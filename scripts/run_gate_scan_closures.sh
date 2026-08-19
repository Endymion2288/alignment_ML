#!/bin/bash
# Frozen route-selected multi-DoF update per candidate gate.
# Anchor route sets come from the gated frozen-backbone outputs; everything
# else (FD probes, covariances, solver, tolerances) is unchanged.
set -u
cd /eos/home-x/xcheng/FASER/alignment_ML
SYN=outputs/mc24_multidof_ift_iteration01_anchor_trainval_synthetic_v1/samples
SCAN=outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_v1/sources/mc24_100043_00200_00299/physical_scan
VSCAN=outputs/mc24_multidof_ift_iteration01_anchor_trainval_physical_v1/sources/mc24_100047_00000_00049/physical_scan

run_gate () {
  local split="$1" gate="$2"
  local backbone scan_root sample_root anchor_out
  if [ "$split" = train ]; then scan_root="$SCAN"; else scan_root="$VSCAN"; fi
  sample_root="$SYN/$split"
  if [ "$gate" = ungated ]; then
    anchor_out="outputs/mc24_multidof_ift_iteration01_v2_frozen_backbone_${split}_v1/iteration_01_anchor"
  else
    anchor_out="outputs/mc24_multidof_ift_iteration01_v2_backbone_gate${gate}_${split}_v1"
  fi
  python scripts/run_route_selected_multidof_update.py \
    --scan-root "$scan_root" \
    --anchor-point iteration_01_anchor \
    --anchor-association-output "$anchor_out" \
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
    --require-full-rank \
    --output-dir "outputs/mc24_multidof_ift_iteration01_route_selected_update_${split}_gate${gate}_v1" 2>&1 | tail -1
  echo "=== closure ${split} gate${gate} done ==="
}

for split in train validation; do
  for gate in 25 50 100 200 500 ungated; do
    run_gate "$split" "$gate"
  done
done
echo "ALL CLOSURES DONE"
