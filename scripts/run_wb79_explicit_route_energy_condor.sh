#!/usr/bin/env bash
# Condor CPU worker for Workbook 79: Explicit Route Energy Model v1.
# Arm B is a width-64 MLP; W64 Arm A is frozen inference.  LCG_110_cuda
# PyTorch 2.11 has no CC 7.0 kernels, so V100 GPU slots fail.  Formal
# training/eval therefore runs on CPU.  Never opens Final Blind / sealed / 00350.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 PROJECT_ROOT holdout_family1|holdout_family2" >&2
  exit 2
fi

project_root="$1"
fold="$2"
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
echo "fold ${fold}"

python - <<'PY'
import torch
print("torch", torch.__version__, flush=True)
print("cuda_available", torch.cuda.is_available(), flush=True)
print("device cpu", flush=True)
PY

python -u scripts/train_eval_wb79_explicit_route_energy.py --fold "$fold" --device cpu
echo "=== Workbook 79 fold ${fold} completed successfully ==="
