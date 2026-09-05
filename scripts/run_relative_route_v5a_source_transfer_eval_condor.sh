#!/usr/bin/env bash
# Condor worker for Workbook 72: V5A source-transfer CV evaluation.
# Runs the paired Control/Primary evaluation on both held-out families and
# emits the pre-registered source-transfer gate.  Requires the four fold
# checkpoints to already exist (produced by the training jobs).
#
#   device = cpu  -> CPU slot (abundant, schedules immediately); inference is a
#                    small frozen/transformer forward pass, solver is CPU-bound.
#   device = cuda -> GPU slot.
set -euo pipefail

if [[ "$#" -lt 1 || "$#" -gt 2 ]]; then
  echo "Usage: $0 PROJECT_ROOT [cpu|cuda]" >&2
  exit 2
fi

project_root="$1"
device="${2:-cpu}"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi

if [[ "$device" == "cuda" ]]; then
  unset CUDA_VISIBLE_DEVICES || true
fi
set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

echo "hostname $(hostname)"
echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_commit $(git rev-parse HEAD)"
echo "git_status_porcelain:"
git status --porcelain || true
echo "device $device"
echo "cvmfs_lcg /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh"

if [[ "$device" == "cuda" ]]; then
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("CUDA device requested but unavailable on this slot")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
PY
else
python - <<'PY'
import torch
print("cpu eval, torch", torch.__version__, flush=True)
PY
fi

echo "=== V5A source-transfer evaluation (both folds, device=$device) ==="
python -u scripts/evaluate_relative_route_v5a_source_transfer.py --device "$device"

echo "=== V5A source-transfer evaluation completed successfully ==="
