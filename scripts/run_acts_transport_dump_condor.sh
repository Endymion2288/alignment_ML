#!/usr/bin/env bash
# Task B0 worker: C++ ACTS C0/C1 dump on persisted CKF 5x5.
# No SegmentFit refit, no alignment payload, no truth q/p seed, no Q tuning.

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
plugin="$project_root/build/acts_transport_dump/libCkfActsTransportDump.so"
if [[ ! -f "$plugin" ]]; then
  echo "C++ dump helper is absent: $plugin" >&2
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
export LD_LIBRARY_PATH="$project_root/build/acts_transport_dump${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$(dirname "$output_jsonl")"
cd "$(dirname "$output_jsonl")"
python "$project_root/scripts/dump_ckf_acts_transport_covariance.py" \
  --input-xaod "$input_xaod" \
  --source-id "$source_id" \
  --output "$output_jsonl" \
  --nevents "$nevents"
