#!/usr/bin/env bash
# Condor GPU worker for workbook 59.  Interactive lxplus GPUs are shared.
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

python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("dustbin-aware V2 is GPU-only; CUDA is unavailable on this slot")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
PY

bash scripts/run_four_station_dustbin_aware_training.sh train
bash scripts/run_four_station_dustbin_aware_training.sh infer-transfer
bash scripts/run_four_station_dustbin_aware_training.sh score-scale
bash scripts/run_four_station_dustbin_aware_training.sh mechanism
bash scripts/run_four_station_dustbin_aware_training.sh assess
