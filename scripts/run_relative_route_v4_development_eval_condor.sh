#!/usr/bin/env bash
# Condor GPU worker for Workbook 70: Arm 0/1/2 reserved-blind development eval.
# Does not train.  Does not open 00800_00849 or sealed test.
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
echo "git_status_porcelain:"
git status --porcelain || true
echo "cvmfs_lcg /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh"

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("Workbook 70 development evaluation is GPU-only; CUDA is unavailable on this slot")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
PY

python -u scripts/evaluate_relative_route_v4_development.py \
  --contract configs/relative_route_v4_head_only_development_eval.yaml \
  --gates configs/physical_four_station_diversity_training.yaml \
  --synthetic-manifest outputs/mc24_four_station_source_diversity_blind_v1/overlay_synthetic_v1/synthetic_corpus_manifest.json \
  --iteration-manifest outputs/mc24_four_station_source_diversity_blind_v1/iteration_manifest.json \
  --workbook64-frozen-output outputs/mc24_four_station_source_diversity_v1/checkpoint \
  --arm1-checkpoint outputs/mc24_four_station_relative_route_v4_head_only/absolute_control/checkpoint_last.pt \
  --arm2-checkpoint outputs/mc24_four_station_relative_route_v4_head_only/relative_primary/checkpoint_last.pt \
  --output-dir outputs/mc24_four_station_relative_route_v4_head_only/development_eval \
  --device cuda \
  --batch-size 32

echo "=== Workbook 70 development evaluation completed successfully ==="
