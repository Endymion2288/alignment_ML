#!/usr/bin/env bash
# Task B14M-S worker: diagnose FAIL identities plus negative controls.
# Arguments contain no /eos paths so standard CERN schedds accept the submit file.
set -euo pipefail

if [[ "$#" -ne 2 ]]; then
  echo "Usage: $0 SOURCE_ID PART" >&2
  exit 2
fi

source_id="$1"
part="$2"
project_root="/eos/home-x/xcheng/FASER/alignment_ML"

case "$source_id" in
  mc24_100043_00400_00499)
    input_xaod="/eos/experiment/faser/data0/sim/mc24/particle_gun/100043/rec/s0013-r0022/FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100043-00400-00499-s0013-r0022-xAOD.root"
    nevents=50
    select_events="0,37"
    basin_source="$project_root/outputs/leave_target_out_dump_v1/b14m_reopen_smoke/mc24_100043_00400_00499/ckf_leave_target_out_b14m_reopen.jsonl"
    ;;
  mc24_100048_00000_00049)
    input_xaod="/eos/experiment/faser/data0/sim/mc24/particle_gun/100048/rec/s0013-r0022/FaserMC-MC24_PG_mupl_fasernu_5mrad_flukaE-100048-00000-00049-s0013-r0022-xAOD.root"
    nevents=100
    select_events="86"
    basin_source="$project_root/outputs/leave_target_out_dump_v1/b14m_reopen_smoke/mc24_100048_00000_00049/ckf_leave_target_out_b14m_reopen.jsonl"
    ;;
  *)
    echo "unknown source_id: $source_id" >&2
    exit 2
    ;;
esac

output_jsonl="$project_root/outputs/leave_target_out_dump_v1/b14ms_basin_smoke/${source_id}/part_${part}.jsonl"

if [[ ! -f "$project_root/scripts/setup_environment.sh" ]]; then
  echo "Project setup script is absent: $project_root/scripts/setup_environment.sh" >&2
  exit 2
fi
if [[ ! -f "$input_xaod" ]]; then
  echo "Input xAOD is not a file: $input_xaod" >&2
  exit 2
fi
if [[ ! -f "$basin_source" ]]; then
  echo "WB129 basin source is absent: $basin_source" >&2
  exit 2
fi
plugin="$project_root/build/leave_target_out_dump/libCkfLeaveTargetOutDump.so"
if [[ ! -f "$plugin" ]]; then
  echo "Independent LTO helper is absent: $plugin" >&2
  exit 2
fi
if [[ -f "$output_jsonl" ]]; then
  echo "Refusing to overwrite existing basin dump: $output_jsonl" >&2
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
export LD_LIBRARY_PATH="$project_root/build/leave_target_out_dump${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
mkdir -p "$(dirname "$output_jsonl")"
cd "$(dirname "$output_jsonl")"
echo "=== B14M-S basin diagnosis part=$part source=$source_id events=$select_events ==="
python "$project_root/scripts/dump_ckf_leave_target_out.py" \
  --input-xaod "$input_xaod" \
  --source-id "$source_id" \
  --output "$output_jsonl" \
  --nevents "$nevents" \
  --skip-events 0 \
  --enable-profile-likelihood \
  --profile-only \
  --enable-profile-numerics \
  --enable-profile-seed-invariance \
  --enable-field-gradient-variational-repair \
  --enable-profile-basin-diagnosis \
  --basin-source-jsonl "$basin_source" \
  --select-event-ids "$select_events"
