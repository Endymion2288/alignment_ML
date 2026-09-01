#!/usr/bin/env bash
# Condor GPU worker for Workbook 69: RelativeRoute V4 head-only training.
# Runs either Arm 1 (absolute control) or Arm 2 (relative primary)
# depending on the second argument.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 PROJECT_ROOT arm1|arm2" >&2
  exit 2
fi

project_root="$1"
arm="$2"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi

unset CUDA_VISIBLE_DEVICES || true
set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

# Verify GPU availability
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("RelativeRoute V4 head-only training is GPU-only; CUDA is unavailable on this slot")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
PY

base_checkpoint="outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt"
synthetic_manifest="outputs/mc24_four_station_source_diversity_train_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"

if [[ "$arm" == "arm1" ]]; then
  echo "=== Running Arm 1: Absolute-Route Control ==="
  python -u scripts/train_relative_route_v4_head_only.py \
    --config configs/absolute_route_control_head_only_train.yaml \
    --synthetic-manifest "$synthetic_manifest" \
    --base-checkpoint "$base_checkpoint" \
    --output-dir outputs/mc24_four_station_relative_route_v4_head_only/absolute_control \
    --device cuda
elif [[ "$arm" == "arm2" ]]; then
  echo "=== Running Arm 2: RelativeRoute V4 Primary ==="
  python -u scripts/train_relative_route_v4_head_only.py \
    --config configs/relative_route_v4_head_only_train.yaml \
    --synthetic-manifest "$synthetic_manifest" \
    --base-checkpoint "$base_checkpoint" \
    --output-dir outputs/mc24_four_station_relative_route_v4_head_only/relative_primary \
    --device cuda
else
  echo "Unknown arm: $arm. Use arm1 or arm2." >&2
  exit 2
fi

echo "=== $arm training completed successfully ==="
