#!/usr/bin/env bash
# Task B14K map-smoothness smoke.
# Pre-registered: 100048/86 and kinematic control 100048/44.
# Control was selected from WB109 native-state kinematics only.
# Does not overwrite WB109–WB117 dumps and does not submit 1989 rows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XAOD48="/eos/experiment/faser/data0/sim/mc24/particle_gun/100048/rec/s0013-r0022/FaserMC-MC24_PG_mupl_fasernu_5mrad_flukaE-100048-00000-00049-s0013-r0022-xAOD.root"
SRC48="mc24_100048_00000_00049"
OFFICIAL="$ROOT/outputs/leave_target_out_dump_v1/dumps/${SRC48}/ckf_leave_target_out.jsonl"
WB117="$ROOT/outputs/leave_target_out_dump_v1/b14j_smoke/${SRC48}/ckf_leave_target_out_jacobian_continuity.jsonl"
SMOKE_ROOT="$ROOT/outputs/leave_target_out_dump_v1/b14k_smoke"

if [[ ! -f "$XAOD48" ]]; then
  echo "missing xAOD: $XAOD48" >&2
  exit 2
fi
if [[ ! -f "$OFFICIAL" ]]; then
  echo "missing official WB109 dump: $OFFICIAL" >&2
  exit 2
fi
if [[ ! -f "$WB117" ]]; then
  echo "missing WB117 smoke dump: $WB117" >&2
  exit 2
fi

unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME \
  Athena_SET_UP Athena_EXTONLY_SET_UP Athena_RELONLY_SET_UP \
  AthenaExternals_SET_UP AthenaExternals_EXTONLY_SET_UP AthenaExternals_RELONLY_SET_UP \
  Calypso_SET_UP Calypso_EXTONLY_SET_UP Calypso_RELONLY_SET_UP
set +u
source "$ROOT/scripts/setup_environment.sh" calypso
set -u
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/build/leave_target_out_dump${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

run_job() {
  local xaod="$1" source_id="$2" events="$3" nevents="$4" part="$5"
  local outdir="$SMOKE_ROOT/${source_id}"
  local partfile="$outdir/${part}.jsonl"
  mkdir -p "$outdir"
  if [[ -f "$partfile" ]]; then
    echo "refusing to overwrite $partfile" >&2
    exit 2
  fi
  echo "=== B14K map smoothness smoke source=$source_id events=$events out=$partfile ==="
  (
    cd "$outdir"
    python "$ROOT/scripts/dump_ckf_leave_target_out.py" \
      --input-xaod "$xaod" \
      --source-id "$source_id" \
      --output "$partfile" \
      --nevents "$nevents" \
      --skip-events 0 \
      --enable-profile-likelihood \
      --profile-only \
      --enable-profile-numerics \
      --profile-evaluate-only \
      --enable-profile-transport \
      --enable-map-smoothness \
      --map-smoothness-repeats 3 \
      --select-event-ids "$events"
  )
}

run_job "$XAOD48" "$SRC48" "86,44" 100 part_evt86_44

outdir="$SMOKE_ROOT/${SRC48}"
final="$outdir/ckf_leave_target_out_map_smoothness.jsonl"
if [[ -f "$final" ]]; then
  echo "refusing to overwrite $final" >&2
  exit 2
fi
cat "$outdir"/part_*.jsonl > "$final"
echo "concatenated $final lines=$(wc -l < "$final")"
echo "B14K map smoothness smoke complete"
ls -l "$SMOKE_ROOT"/*/*.jsonl
