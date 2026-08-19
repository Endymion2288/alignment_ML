#!/bin/bash
# Frozen-backbone candidate-gate scan on the iteration-1 multi-track overlays.
# Only the candidate chi2 gate changes; checkpoint/calibration/route policy frozen.
set -u
cd /eos/home-x/xcheng/FASER/alignment_ML
MANIFEST=outputs/mc24_multidof_ift_iteration01_anchor_trainval_synthetic_v1/synthetic_corpus_manifest.json
FROZEN=outputs/mc24_v3_expanded_trainval_v2_bce_control_v1
for split in train validation; do
  for gate in 25 50 100 200 500; do
    out=outputs/mc24_multidof_ift_iteration01_v2_backbone_gate${gate}_${split}_v1
    if [ -d "$out" ]; then echo "skip $out"; continue; fi
    python scripts/run_frozen_association_backbone.py \
      --synthetic-manifest "$MANIFEST" \
      --backbone v2 --frozen-output "$FROZEN" \
      --split "$split" --payload-id iteration_01_anchor \
      --candidate-chi2-gate "$gate" \
      --output-dir "$out" --device auto 2>&1 | tail -1
    echo "=== gate${gate} ${split} done ==="
  done
done
echo "ALL GATES DONE"
