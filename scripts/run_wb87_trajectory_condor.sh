#!/usr/bin/env bash
# HTCondor worker for one frozen WB87 trajectory.  Does not touch WB86.
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 PROJECT_ROOT OUTPUT_ROOT INDEX" >&2
  exit 2
fi

project_root="$1"
output_root="$2"
index="$3"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi

unset CUDA_VISIBLE_DEVICES || true
set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

echo "hostname $(hostname)"
echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "index $index"
echo "git_commit $(git rev-parse HEAD)"

python -u scripts/run_wb87_trajectory_event.py \
  --output-root "$output_root" \
  --index "$index"

echo "=== WB87 trajectory $index completed ==="
