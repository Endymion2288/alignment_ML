#!/usr/bin/env bash
# Condor GPU worker for Workbook 72: V5A source-transfer CV evaluation.
# Runs the paired Control/Primary evaluation on both held-out families and
# emits the pre-registered source-transfer gate.  Requires the four fold
# checkpoints to already exist (produced by the training jobs).
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
    raise SystemExit("V5A source-transfer evaluation is GPU-only; CUDA is unavailable on this slot")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
PY

echo "=== V5A source-transfer evaluation (both folds) ==="
python -u scripts/evaluate_relative_route_v5a_source_transfer.py --device cuda

echo "=== V5A source-transfer evaluation completed successfully ==="
