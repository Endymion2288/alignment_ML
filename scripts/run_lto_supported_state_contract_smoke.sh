#!/usr/bin/env bash
# Task B14S directional seed-sensitivity smoke.
# Pre-registered directions x/y/tx/ty/q_over_p and scales 0.1 / 1 / 10.
# Does not overwrite WB109 dumps and does not choose a production scale.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XAOD="/eos/experiment/faser/data0/sim/mc24/particle_gun/100043/rec/s0013-r0022/FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100043-00400-00499-s0013-r0022-xAOD.root"
LTO_SOURCE_ID="mc24_100043_00400_00499"
OFFICIAL_DUMP="$ROOT/outputs/leave_target_out_dump_v1/dumps/${LTO_SOURCE_ID}/ckf_leave_target_out.jsonl"
SMOKE_ROOT="$ROOT/outputs/leave_target_out_dump_v1/b14s_smoke/${LTO_SOURCE_ID}"

if [[ ! -f "$XAOD" ]]; then
  echo "missing xAOD: $XAOD" >&2
  exit 2
fi
if [[ ! -f "$OFFICIAL_DUMP" ]]; then
  echo "missing official WB109 dump: $OFFICIAL_DUMP" >&2
  exit 2
fi

unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME \
  Athena_SET_UP Athena_EXTONLY_SET_UP Athena_RELONLY_SET_UP \
  AthenaExternals_SET_UP AthenaExternals_EXTONLY_SET_UP AthenaExternals_RELONLY_SET_UP \
  Calypso_SET_UP Calypso_EXTONLY_SET_UP Calypso_RELONLY_SET_UP
set +u
source "$ROOT/scripts/setup_environment.sh" calypso
set -u
# Re-assert after Athena/LCG, which overwrite generic names such as SOURCE.
LTO_SOURCE_ID="mc24_100043_00400_00499"
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/build/leave_target_out_dump${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_part() {
  local skip="$1" nevents="$2" part="$3"
  local partfile="$SMOKE_ROOT/${part}.jsonl"
  mkdir -p "$SMOKE_ROOT"
  if [[ -f "$partfile" ]]; then
    echo "refusing to overwrite $partfile" >&2
    exit 2
  fi
  echo "=== B14S directional smoke skip=$skip nevents=$nevents out=$partfile ==="
  (
    cd "$SMOKE_ROOT"
    python "$ROOT/scripts/dump_ckf_leave_target_out.py" \
      --input-xaod "$XAOD" \
      --source-id "$LTO_SOURCE_ID" \
      --output "$partfile" \
      --nevents "$nevents" \
      --skip-events "$skip" \
      --directional-seed-campaign
  )
}

run_part 0 2 part_evt0_1
run_part 37 1 part_evt37
final="$SMOKE_ROOT/ckf_leave_target_out.jsonl"
if [[ -f "$final" ]]; then
  echo "refusing to overwrite $final" >&2
  exit 2
fi
cat "$SMOKE_ROOT/part_evt0_1.jsonl" "$SMOKE_ROOT/part_evt37.jsonl" > "$final"
echo "concatenated $final lines=$(wc -l < "$final")"
echo "B14S directional smoke complete"
ls -l "$SMOKE_ROOT"/*.jsonl
