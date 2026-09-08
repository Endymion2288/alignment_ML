#!/usr/bin/env bash
# Condor CPU worker for Workbook 80: Arm B failure-mechanism audit.
# Does not retrain V5A / W64 / Arm B.  Ablation C is not wp4 Arm C.
# Formal path is CPU for the same LCG / V100 / H100 VO reasons as WB79.
# Never opens Final Blind / sealed / 00350.
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

python -u scripts/train_eval_wb80_failure_mechanism.py --fold "$fold" --device cpu
echo "=== Workbook 80 fold ${fold} completed successfully ==="
