#!/usr/bin/env bash
# HTCondor worker for the WB85b official qualification-runner e2e smoke.
set -euo pipefail

if [[ "$#" -lt 2 ]]; then
  echo "Usage: $0 PROJECT_ROOT OUTPUT_ROOT" >&2
  exit 2
fi

project_root="$1"
output_root="$2"

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
echo "git_commit $(git rev-parse HEAD)"
echo "git_status_porcelain:"
git status --porcelain || true

python -u scripts/run_wb85b_official_runner_smoke.py --output-root "$output_root"
echo "=== WB85b official runner smoke completed ==="
