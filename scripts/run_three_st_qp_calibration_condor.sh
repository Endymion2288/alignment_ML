#!/usr/bin/env bash
# Yasu Stage 2 worker: dump truth-matched WithoutIFT q/p from one xAOD.
# Truth is a calibration reference only.  No refit, no alignment payload.

set -euo pipefail

if [[ "$#" -ne 7 ]]; then
  echo "Usage: $0 INPUT_XAOD SOURCE_ID OUTPUT_JSONL EVENT_JSONL NEVENTS PROJECT_ROOT CAMPAIGN" >&2
  exit 2
fi

input_xaod="$1"
source_id="$2"
output_jsonl="$3"
event_jsonl="$4"
nevents="$5"
project_root="$6"
campaign="$7"

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
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$project_root/build/three_st_qp_calibration_dump${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$(dirname "$output_jsonl")" "$(dirname "$event_jsonl")"
cd "$(dirname "$output_jsonl")"
python "$project_root/scripts/dump_ckf_three_st_qp_calibration.py" \
  --input-xaod "$input_xaod" \
  --source-id "$source_id" \
  --output "$output_jsonl" \
  --event-output "$event_jsonl" \
  --nevents "$nevents" \
  --campaign "$campaign"
