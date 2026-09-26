#!/usr/bin/env bash
# HTCondor worker for WB86 physical qualification DAG stages.
set -euo pipefail

if [[ "$#" -lt 3 ]]; then
  echo "Usage: $0 PROJECT_ROOT OUTPUT_ROOT STAGE [INDEX]" >&2
  exit 2
fi

project_root="$1"
output_root="$2"
stage="$3"
index="${4:-}"

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
echo "stage $stage"
echo "index ${index:-none}"
echo "git_commit $(git rev-parse HEAD)"

case "$stage" in
  snapshot)
    python -u scripts/run_wb86_snapshot_verify.py --output-root "$output_root"
    ;;
  corpus)
    python -u scripts/run_wb86_corpus_verify.py --output-root "$output_root"
    ;;
  event)
    if [[ -z "$index" ]]; then
      echo "event stage requires INDEX" >&2
      exit 2
    fi
    python -u scripts/run_wb86_event.py --output-root "$output_root" --index "$index"
    ;;
  integrity)
    python -u scripts/run_wb86_result_integrity.py --output-root "$output_root"
    ;;
  summary)
    python -u scripts/run_wb86_global_summary.py --output-root "$output_root"
    ;;
  gate)
    python -u scripts/run_wb86_frozen_qualification_gate.py --output-root "$output_root"
    ;;
  closure)
    python -u scripts/run_wb86_closure.py --output-root "$output_root"
    ;;
  *)
    echo "unknown stage: $stage" >&2
    exit 2
    ;;
esac

echo "=== WB86 stage $stage completed ==="
