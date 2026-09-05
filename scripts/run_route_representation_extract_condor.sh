#!/usr/bin/env bash
# Condor worker for Workbook 73: route-representation source-domain audit.
# Extracts per-route representation levels R0-R7 / R_phys for one Workbook-72
# source-pure family corpus using a fixed reference absolute head.  Diagnostic
# only -- no production model is trained.
#
#   $1 = PROJECT_ROOT
#   $2 = family1 | family2
#   $3 = cpu | cuda   (default cpu; small frozen/head forward, CPU is enough)
set -euo pipefail

if [[ "$#" -lt 2 || "$#" -gt 4 ]]; then
  echo "Usage: $0 PROJECT_ROOT {family1|family2} [cpu|cuda] [head_checkpoint]" >&2
  exit 2
fi

project_root="$1"
family="$2"
device="${3:-cpu}"
head_ckpt="${4:-}"

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

out_dir="outputs/mc24_four_station_route_representation_domain_audit_v1/extract"

echo "hostname $(hostname)"
echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_commit $(git rev-parse HEAD)"
echo "git_status_porcelain:"
git status --porcelain || true
echo "family $family"
echo "device $device"

python - <<PY
import torch
print("torch", torch.__version__, "cuda_available", torch.cuda.is_available(), flush=True)
if "$device" == "cuda":
    assert torch.cuda.is_available(), "CUDA requested but unavailable"
PY

extra_args=()
if [[ -n "$head_ckpt" ]]; then
  extra_args+=(--head-checkpoint "$head_ckpt")
  echo "head_override $head_ckpt"
fi

echo "=== Workbook 73 representation extraction: $family (device=$device) ==="
python -u scripts/extract_route_representations.py \
  --family "$family" \
  --output-dir "$out_dir" \
  --device "$device" \
  --fake-keep-per-mille 120 \
  "${extra_args[@]}"

echo "=== Workbook 73 extraction completed: $family ==="
