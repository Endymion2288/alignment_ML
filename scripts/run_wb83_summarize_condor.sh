#!/usr/bin/env bash
# Condor CPU summary / fail-closed gate for Workbook 83.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 PROJECT_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

project_root="$1"
output_root="$2"
unset CUDA_VISIBLE_DEVICES || true
set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"
python -u scripts/summarize_wb83_qualification.py --output-root "$output_root"
echo "=== Workbook 83 summary completed ==="
