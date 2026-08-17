#!/usr/bin/env bash
# Run one frozen physical refit scan inside an isolated Condor worker.
#
# The Python driver owns every payload/refit/export step.  This wrapper only
# establishes the ML environment required by the driver and keeps a local
# interactive session from being mistaken for completed reconstruction.

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 SCAN_CONFIG OUTPUT_DIR PROJECT_ROOT" >&2
  exit 2
fi

scan_config="$1"
output_dir="$2"
project_root="$3"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi
if [[ ! -f "$scan_config" ]]; then
  echo "Physical scan configuration is absent: $scan_config" >&2
  exit 2
fi

# The LCG setup script probes optional environment variables.  Preserve strict
# behavior for this worker while allowing the external setup script to run.
set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

python scripts/run_physical_refit_capture_scan.py \
  --config "$scan_config" \
  --output-dir "$output_dir" \
  --resume
