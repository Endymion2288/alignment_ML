#!/usr/bin/env bash
# Condor CPU worker for Workbook 83 independent truth-only replicas.
# Does not train association models, open Blind/sealed/00350, or use overlays.
set -euo pipefail

if [[ "$#" -lt 5 ]]; then
  echo "Usage: $0 PROJECT_ROOT OUTPUT_ROOT CELL_ID REPLICA_BEGIN REPLICA_END" >&2
  exit 2
fi

project_root="$1"
output_root="$2"
cell_id="$3"
replica_begin="$4"
replica_end="$5"

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
echo "cell ${cell_id} replicas ${replica_begin}:${replica_end}"

python -u scripts/run_wb83_replica_worker.py \
  --output-root "$output_root" \
  --cell-id "$cell_id" \
  --replica-begin "$replica_begin" \
  --replica-end "$replica_end"
echo "=== Workbook 83 cell ${cell_id} completed ==="
