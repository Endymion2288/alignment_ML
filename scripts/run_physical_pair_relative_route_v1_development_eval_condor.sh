#!/usr/bin/env bash
# Condor CPU worker for Workbook 75 Stage 2: frozen development evaluation.
# Never opens Final Blind / sealed test.  CPU-only.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 PROJECT_ROOT" >&2
  exit 2
fi

project_root="$1"
if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent" >&2
  exit 2
fi

set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

echo "hostname $(hostname)"
echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_commit $(git rev-parse HEAD)"
echo "git_status_porcelain_tracked:"
git status --porcelain --untracked-files=no || true

python -u scripts/evaluate_physical_pair_relative_route_v1_development.py --device cpu
echo "=== Workbook 75 development evaluation completed successfully ==="
