#!/usr/bin/env bash
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 PROJECT_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

project_root="$1"
output_root="$2"

set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

python scripts/run_calypso_physical_production.py \
  --output-root "$output_root" \
  qa
