#!/usr/bin/env bash
set -euo pipefail
ulimit -c 0
project_dir="$1"
output_dir="$2"
cd "$project_dir"
source scripts/setup_environment.sh calypso
python scripts/run_wb91_physical_validation.py --output-root "$output_dir"
