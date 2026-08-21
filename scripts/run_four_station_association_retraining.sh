#!/usr/bin/env bash
# Four-station matched V2 association retraining.  Each stage refuses to
# overwrite non-empty outputs.  Sealed test is never opened.  Workbook-52
# held-out overlays are forbidden operating-point inputs.  15-DoF unknown-
# association WLS stays closed until the source-disjoint association gate
# passes.
#
# Stages, in order:
#   prepare            write the 15-DoF relative + left-SE(3) physical plan
#   submit             Condor-dispatch real /Tracker/Align -> Acts(mode 0)
#   assemble           corpus manifest after every source completes
#   overlay            frozen-V2 overlay recipe on train then validation
#   control-frozen-v2  historical V2 on the new overlay (domain-shift control)
#   train-v2           GPU-only matched V2 retraining; OP on validation only
#   assess             association + gauge-invariance gates; no 15-D WLS
set -euo pipefail

if [[ "$#" -lt 1 ]]; then
  echo "Usage: $0 STAGE" >&2
  exit 2
fi
stage="$1"

cd "$(dirname "$0")/.."
ROOT="$PWD"
OUT="$ROOT/outputs/mc24_four_station_relative_association_retrain_v1"
FROZEN_V2="/eos/home-x/xcheng/FASER/alignment_ML/outputs/mc24_v3_expanded_trainval_v2_bce_control_v1"

case "$stage" in

prepare)
  python scripts/prepare_four_station_relative_curriculum.py \
    --source-config configs/physical_curriculum_four_station_relative_association_sources.yaml \
    --iteration-template configs/physical_refit_four_station_relative_association_curriculum.yaml \
    --output-root "$OUT" \
    --iteration 0 \
    --nevents 50
  ;;

submit)
  python scripts/submit_multisource_multidof_iteration_condor.py \
    --iteration-manifest "$OUT/iteration_manifest.json" \
    --submit-dir "$OUT/condor" \
    --request-memory-mb 6000 \
    --job-flavour tomorrow \
    --schedd-mode eossubmit \
    --submit
  ;;

assemble)
  python scripts/assemble_multisource_multidof_iteration_manifest.py \
    --iteration-manifest "$OUT/iteration_manifest.json" \
    --materialization-config configs/physical_alignment_iteration_four_station_relative_trainval.yaml \
    --output "$OUT/physical_corpus_manifest.json"
  ;;

overlay)
  python scripts/materialize_pooled_curriculum_synthetics.py \
    --config configs/physical_alignment_iteration_four_station_relative_trainval.yaml \
    --physical-manifest "$OUT/physical_corpus_manifest.json" \
    --output-dir "$OUT/overlay_synthetic_v1" \
    --split train
  python scripts/materialize_pooled_curriculum_synthetics.py \
    --config configs/physical_alignment_iteration_four_station_relative_trainval.yaml \
    --physical-manifest "$OUT/physical_corpus_manifest.json" \
    --output-dir "$OUT/overlay_synthetic_v1" \
    --split validation \
    --resume
  ;;

control-frozen-v2)
  echo "Run frozen historical V2 on the new overlay with the frozen OP." >&2
  echo "Do not pass workbook-52 overlay paths.  Split must be train or validation." >&2
  exit 2
  ;;

train-v2)
  python scripts/train_route_aware_transformer_v2.py \
    --config configs/geometry_aware_transformer_v2_four_station_relative_matched.yaml \
    --synthetic-manifest "$OUT/overlay_synthetic_v1/synthetic_curriculum_manifest.json" \
    --output-dir "$OUT/retrained_v2" \
    --q-over-p-mode 0
  ;;

assess)
  echo "Association gates live in configs/physical_four_station_association_retraining_gates.yaml." >&2
  echo "Do not open 15-DoF unknown-association WLS until those gates pass." >&2
  exit 2
  ;;

*)
  echo "unknown stage: $stage" >&2
  exit 2
  ;;
esac
