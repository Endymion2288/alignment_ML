#!/usr/bin/env bash
# Condor GPU worker for Workbook 72: V5A source-transfer CV head-only training.
# Runs one (fold, arm) job.  Each fold trains on the fold's TRAIN-family
# regenerated corpus only; the held-out family never enters training.
#
#   fold = holdout_family1  -> train on family2 corpus (hold out family1)
#   fold = holdout_family2  -> train on family1 corpus (hold out family2)
#   arm  = control          -> V4A absolute UNBOUNDED (route_correction_bound=null)
#   arm  = primary          -> V5A absolute BOUNDED B=4 (route_correction_bound=4.0)
set -euo pipefail

if [[ "$#" -ne 3 ]]; then
  echo "Usage: $0 PROJECT_ROOT holdout_family1|holdout_family2 control|primary" >&2
  exit 2
fi

project_root="$1"
fold="$2"
arm="$3"

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
echo "cvmfs_lcg /cvmfs/sft.cern.ch/lcg/views/LCG_110_cuda/x86_64-el9-gcc13-opt/setup.sh"

# Verify GPU availability
python - <<'PY'
import torch
if not torch.cuda.is_available():
    raise SystemExit("V5A source-transfer training is GPU-only; CUDA is unavailable on this slot")
print("cuda_device", torch.cuda.get_device_name(0), flush=True)
print("torch", torch.__version__, "cuda", torch.version.cuda, flush=True)
PY

base_checkpoint="outputs/mc24_four_station_source_diversity_v1/checkpoint/route_aware_transformer_v2.pt"
corpus_root="outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/corpora"
train_root="outputs/mc24_four_station_relative_route_v5a_source_transfer_v1/training"

# Fold -> TRAIN-family corpus + expected (train-side) sources.  The held-out
# family is the OTHER family and must be absent from training.
case "$fold" in
  holdout_family1)
    synthetic_manifest="$corpus_root/family2/overlay_synthetic_v1/synthetic_corpus_manifest.json"
    expected_sources="mc24_100047_00100_00149,mc24_100048_00100_00149"
    ;;
  holdout_family2)
    synthetic_manifest="$corpus_root/family1/overlay_synthetic_v1/synthetic_corpus_manifest.json"
    expected_sources="mc24_100043_00200_00299,mc24_100043_00300_00399,mc24_100044_00200_00299,mc24_100044_00300_00399"
    ;;
  *)
    echo "Unknown fold: $fold. Use holdout_family1 or holdout_family2." >&2
    exit 2
    ;;
esac

case "$arm" in
  control)
    config="configs/v5a_absolute_unbounded_control_source_transfer.yaml"
    ;;
  primary)
    config="configs/v5a_absolute_bounded_primary_source_transfer.yaml"
    ;;
  *)
    echo "Unknown arm: $arm. Use control or primary." >&2
    exit 2
    ;;
esac

output_dir="$train_root/$fold/$arm"

echo "=== V5A source-transfer training: fold=$fold arm=$arm ==="
echo "config=$config"
echo "synthetic_manifest=$synthetic_manifest"
echo "expected_train_sources=$expected_sources"
echo "output_dir=$output_dir"

python -u scripts/train_relative_route_v4_head_only.py \
  --config "$config" \
  --synthetic-manifest "$synthetic_manifest" \
  --base-checkpoint "$base_checkpoint" \
  --output-dir "$output_dir" \
  --expected-train-sources "$expected_sources" \
  --device cuda

echo "=== fold=$fold arm=$arm training completed successfully ==="
