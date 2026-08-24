#!/usr/bin/env bash
set -e
cd /eos/home-x/xcheng/FASER/alignment_ML_4station_branch
source scripts/setup_environment.sh ml
export CUDA_VISIBLE_DEVICES=0
export PYTHONUNBUFFERED=1
python scripts/audit_four_station_route_utility_identifiability.py \
  --synthetic-manifest outputs/mc24_four_station_relative_transfer_validation_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json \
  --candidate-frozen-output outputs/mc24_four_station_gauge_consistent_route_v1/checkpoint \
  --control-frozen-output outputs/mc24_four_station_relative_association_retrain_v1/retrained_v2 \
  --output-dir outputs/mc24_four_station_gauge_consistent_route_v1/route_utility_identifiability_v1 \
  --device cuda \
  --batch-size 32
echo AUDIT_EXIT=$?
