#!/usr/bin/env bash
# Run one configured train/validation source through every real payload point.
# This worker deliberately does not refresh the shared corpus manifest; a
# single post-job aggregation step owns that write and avoids Condor races.

set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 PHYSICAL_OUTPUT_DIR SOURCE_ID PROJECT_ROOT" >&2
  exit 2
fi

physical_output_dir="$1"
source_id="$2"
project_root="$3"
source_root="$physical_output_dir/sources/$source_id"
config="$source_root/physical_scan_config.yaml"
scan_root="$source_root/physical_scan"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi

if [[ ! -f "$config" ]]; then
  echo "Configured source scan is absent: $config" >&2
  exit 2
fi

# LCG view setup probes optional environment variables such as COMPILER.  Keep
# strict shell behavior for the worker itself, but do not impose nounset on an
# externally maintained setup script.
set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"
python scripts/run_physical_refit_capture_scan.py \
  --config "$config" \
  --output-dir "$scan_root" \
  --resume
