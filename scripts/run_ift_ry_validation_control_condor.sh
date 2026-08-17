#!/usr/bin/env bash
# Run one frozen IFT-R_y validation control on an allocated Condor GPU.

set -euo pipefail

if [[ "$#" -ne 5 && "$#" -ne 6 && "$#" -ne 7 ]]; then
  echo "Usage: $0 {mlp|mlp_route|v1|v2} CONFIG SYNTHETIC_MANIFEST OUTPUT_DIR PROJECT_ROOT [MLP_CHECKPOINT|__none__] [MLP_CALIBRATION|__none__]" >&2
  exit 2
fi

kind="$1"
config="$2"
synthetic_manifest="$3"
output_dir="$4"
project_root="$5"
mlp_checkpoint="${6:-__none__}"
mlp_calibration="${7:-__none__}"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi
if [[ ! -f "$config" || ! -f "$synthetic_manifest" ]]; then
  echo "Configured validation input is absent" >&2
  exit 2
fi

set +u
source "$project_root/scripts/setup_environment.sh" ml
set -u
cd "$project_root"

case "$kind" in
  mlp)
    command=(python scripts/run_global_assignment_mlp_baseline.py \
      --synthetic-manifest "$synthetic_manifest" \
      --config "$config" \
      --output-dir "$output_dir" \
      --validation-only)
    if [[ "$mlp_checkpoint" != "__none__" ]]; then
      if [[ ! -f "$mlp_checkpoint" ]]; then
        echo "MLP checkpoint is absent: $mlp_checkpoint" >&2
        exit 2
      fi
      command+=(--checkpoint "$mlp_checkpoint")
    fi
    exec "${command[@]}"
    ;;
  mlp_route)
    if [[ "$mlp_checkpoint" == "__none__" || "$mlp_calibration" == "__none__" ]]; then
      echo "MLP route control requires both a checkpoint and frozen calibration" >&2
      exit 2
    fi
    if [[ ! -f "$mlp_checkpoint" || ! -f "$mlp_calibration" ]]; then
      echo "MLP route checkpoint or calibration is absent" >&2
      exit 2
    fi
    exec python scripts/evaluate_pairwise_mlp_route_validation.py \
      --synthetic-manifest "$synthetic_manifest" \
      --checkpoint "$mlp_checkpoint" \
      --frozen-calibration "$mlp_calibration" \
      --config "$config" \
      --output-dir "$output_dir" \
      --device cuda
    ;;
  v1)
    exec python scripts/train_geometry_aware_transformer_v1.py \
      --synthetic-manifest "$synthetic_manifest" \
      --config "$config" \
      --output-dir "$output_dir"
    ;;
  v2)
    exec python scripts/train_route_aware_transformer_v2.py \
      --synthetic-manifest "$synthetic_manifest" \
      --config "$config" \
      --output-dir "$output_dir"
    ;;
  *)
    echo "Unsupported validation control: $kind" >&2
    exit 2
    ;;
esac
