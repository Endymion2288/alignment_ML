#!/usr/bin/env bash
# Residual-blind occupancy preflight for one expansion (run, segment).
# Does not inspect residuals, V2 scores, or alignment parameters.

set -euo pipefail

if [[ "$#" -ne 4 ]]; then
  echo "Usage: $0 OUTPUT_DIR RUN SEGMENT PROJECT_ROOT" >&2
  exit 2
fi

output_dir="$1"
run="$2"
segment="$3"
project_root="$4"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi

set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"
python scripts/run_real_data_occupancy_preflight.py \
  --run "$run" \
  --segment "$segment" \
  --output-dir "$output_dir" \
  --role monitoring
