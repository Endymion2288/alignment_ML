#!/usr/bin/env bash
set -eo pipefail
ulimit -c 0
project_dir="$1"
output_dir="$2"
cd "$project_dir"
source scripts/setup_environment.sh calypso
if [[ -n "${3:-}" ]]; then
  python scripts/run_wb91_physical_validation.py --output-root "$output_dir" --reuse-build-root "$3"
else
  python scripts/run_wb91_physical_validation.py --output-root "$output_dir"
fi
