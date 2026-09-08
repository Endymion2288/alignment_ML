#!/usr/bin/env bash
# One Calypso physical-replica shard.  Infrastructure retries only; do not
# change scientific configuration on failure.
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 PROJECT_ROOT OUTPUT_ROOT SHARD_ID" >&2
  exit 2
fi

project_root="$1"
output_root="$2"
shard_id="$3"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "setup script missing: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi

set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

python scripts/run_calypso_physical_production.py \
  --output-root "$output_root" \
  worker --shard-id "$shard_id"
