#!/usr/bin/env bash
# Yasu Stage 1 smoke: 20 events on one construction and one validation source.
# Does not overwrite dumps.  Does not start a new MC campaign.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CONSTRUCTION_XAOD="/eos/experiment/faser/data0/sim/mc24/particle_gun/100043/rec/s0013-r0022/FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100043-00400-00499-s0013-r0022-xAOD.root"
VALIDATION_XAOD="/eos/experiment/faser/data0/sim/mc24/particle_gun/100048/rec/s0013-r0022/FaserMC-MC24_PG_mupl_fasernu_5mrad_flukaE-100048-00000-00049-s0013-r0022-xAOD.root"
CONSTRUCTION_ID="mc24_100043_00400_00499"
VALIDATION_ID="mc24_100048_00000_00049"
DUMP_ROOT="$ROOT/outputs/three_st_to_ift_prediction_v1/dumps"

if [[ ! -f "$CONSTRUCTION_XAOD" ]]; then
  echo "missing construction xAOD: $CONSTRUCTION_XAOD" >&2
  exit 2
fi
if [[ ! -f "$VALIDATION_XAOD" ]]; then
  echo "missing validation xAOD: $VALIDATION_XAOD" >&2
  exit 2
fi

dump_one() {
  local xaod="$1" source_id="$2"
  local outdir="$DUMP_ROOT/$source_id"
  local tracks="$outdir/three_st_to_ift_prediction.jsonl"
  local events="$outdir/three_st_to_ift_events.jsonl"
  mkdir -p "$outdir"
  if [[ -f "$tracks" || -f "$events" ]]; then
    echo "refusing to overwrite $outdir" >&2
    exit 2
  fi
  echo "=== 3ST→IFT prediction dump source=$source_id ==="
  (
    cd "$outdir"
    python "$ROOT/scripts/dump_ckf_three_st_to_ift_prediction.py" \
      --input-xaod "$xaod" \
      --source-id "$source_id" \
      --output "$tracks" \
      --event-output "$events" \
      --nevents 20 \
      --skip-events 0
  )
}

unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME \
  Athena_SET_UP Athena_EXTONLY_SET_UP Athena_RELONLY_SET_UP \
  AthenaExternals_SET_UP AthenaExternals_EXTONLY_SET_UP AthenaExternals_RELONLY_SET_UP \
  Calypso_SET_UP Calypso_EXTONLY_SET_UP Calypso_RELONLY_SET_UP
set +u
source "$ROOT/scripts/setup_environment.sh" calypso
set -u
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/build/three_st_to_ift_dump${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

dump_one "$CONSTRUCTION_XAOD" "$CONSTRUCTION_ID"
dump_one "$VALIDATION_XAOD" "$VALIDATION_ID"
echo "Yasu Stage 1 smoke dumps complete"
ls -l "$DUMP_ROOT"/*/*.jsonl
