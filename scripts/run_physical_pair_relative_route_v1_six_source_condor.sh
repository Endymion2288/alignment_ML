#!/usr/bin/env bash
# Condor GPU worker for Workbook 75: full six-source training of the frozen
# Physical Pair-Relative Route Encoder V1.  Does not retune architecture,
# loss, B, solver, or OP.  Never opens development / Final Blind / sealed.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 PROJECT_ROOT" >&2
  exit 2
fi

project_root="$1"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi

unset CUDA_VISIBLE_DEVICES || true
set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

echo "hostname $(hostname)"
echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_commit $(git rev-parse HEAD)"
echo "git_status_porcelain_tracked:"
git status --porcelain --untracked-files=no || true
echo "cvmfs_lcg /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Workbook 75 six-source training is GPU-only; CUDA is unavailable on this slot")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
PY

config="configs/wb75_physical_pair_relative_bounded_primary_six_source.yaml"
base_checkpoint="outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt"
synthetic_manifest="outputs/mc24_four_station_source_diversity_train_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
output_dir="outputs/mc24_four_station_physical_pair_relative_route_v1_six_source/training/primary"

echo "=== Workbook 75 full six-source Physical Pair-Relative training ==="
echo "config=$config"
echo "synthetic_manifest=$synthetic_manifest"
echo "base_checkpoint=$base_checkpoint"
echo "output_dir=$output_dir"

python -u scripts/train_relative_route_v4_head_only.py \
  --config "$config" \
  --synthetic-manifest "$synthetic_manifest" \
  --base-checkpoint "$base_checkpoint" \
  --output-dir "$output_dir" \
  --device cuda

echo "=== Workbook 75 six-source training completed successfully ==="
