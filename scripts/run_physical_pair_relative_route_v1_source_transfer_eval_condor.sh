#!/usr/bin/env bash
# Condor CPU worker for Workbook 74: Physical Pair-Relative Route Encoder V1
# source-transfer evaluation.  Runs the full 2-fold evaluation (both folds, 3
# arms each) on the held-out family corpora, computes the pre-registered
# Gate A/B/C, and writes evaluation.json / source_transfer_summary.json.
#
# CPU-only: Workbook 72 proved full evaluation is CPU-feasible; do not hold a
# GPU for evaluation.
set -euo pipefail

if [[ "$#" -ne 1 ]]; then
  echo "Usage: $0 PROJECT_ROOT" >&2
  exit 2
fi

project_root="$1"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi

set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

echo "hostname $(hostname)"
echo "date_utc $(date -u +%Y-%m-%dT%H:%M:%SZ)"
echo "git_commit $(git rev-parse HEAD)"
echo "git_status_porcelain:"
git status --porcelain || true

echo "=== Workbook 74 Physical Pair-Relative source-transfer evaluation (CPU) ==="
python -u scripts/evaluate_physical_pair_relative_route_v1_source_transfer.py \
  --device cpu

echo "=== evaluation completed successfully ==="
