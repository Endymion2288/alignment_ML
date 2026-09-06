#!/usr/bin/env bash
# Task A2 worker: dump persisted CKF 5x5 from one existing xAOD.
# No SegmentFit refit, no alignment payload, no truth q/p seed.

set -euo pipefail

if [[ "$#" -ne 5 ]]; then
  echo "Usage: $0 INPUT_XAOD SOURCE_ID OUTPUT_JSONL NEVENTS PROJECT_ROOT" >&2
  exit 2
fi

input_xaod="$1"
source_id="$2"
output_jsonl="$3"
nevents="$4"
project_root="$5"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi
if [[ ! -f "$input_xaod" ]]; then
  echo "Input xAOD is not a file: $input_xaod" >&2
  exit 2
fi

unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME \
  Athena_SET_UP Athena_EXTONLY_SET_UP Athena_RELONLY_SET_UP \
  AthenaExternals_SET_UP AthenaExternals_EXTONLY_SET_UP AthenaExternals_RELONLY_SET_UP \
  Calypso_SET_UP Calypso_EXTONLY_SET_UP Calypso_RELONLY_SET_UP
set +u
source "$project_root/scripts/setup_environment.sh" calypso
set -u
mkdir -p "$(dirname "$output_jsonl")"
cd "$(dirname "$output_jsonl")"
python "$project_root/scripts/dump_ckf_qoverp_covariance.py" \
  --input-xaod "$input_xaod" \
  --source-id "$source_id" \
  --output "$output_jsonl" \
  --nevents "$nevents"
