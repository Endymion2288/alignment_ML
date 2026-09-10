#!/usr/bin/env bash
# Task B14U official supporting-plane Jacobian smoke.
# Pre-registered: 100043/0,1,37 (FD PASS controls) and 100048/86 (focus).
# Does not overwrite WB109–WB119 dumps and does not submit 1989 rows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XAOD43="/eos/experiment/faser/data0/sim/mc24/particle_gun/100043/rec/s0013-r0022/FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100043-00400-00499-s0013-r0022-xAOD.root"
XAOD48="/eos/experiment/faser/data0/sim/mc24/particle_gun/100048/rec/s0013-r0022/FaserMC-MC24_PG_mupl_fasernu_5mrad_flukaE-100048-00000-00049-s0013-r0022-xAOD.root"
SRC43="mc24_100043_00400_00499"
SRC48="mc24_100048_00000_00049"
OFFICIAL="$ROOT/outputs/leave_target_out_dump_v1/dumps/${SRC43}/ckf_leave_target_out.jsonl"
WB119="$ROOT/outputs/leave_target_out_dump_v1/b14l_smoke/${SRC43}/ckf_leave_target_out_derivative_contract.jsonl"
SMOKE_ROOT="$ROOT/outputs/leave_target_out_dump_v1/b14u_smoke"

if [[ ! -f "$XAOD43" ]]; then
  echo "missing xAOD: $XAOD43" >&2
  exit 2
fi
if [[ ! -f "$OFFICIAL" ]]; then
  echo "missing official WB109 dump: $OFFICIAL" >&2
  exit 2
fi
if [[ ! -f "$WB119" ]]; then
  echo "missing WB119 smoke dump: $WB119" >&2
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
  echo "=== B14U official Jacobian smoke source=$source_id events=$events out=$partfile ==="
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
      --enable-official-supporting-plane-jacobian \
      --select-event-ids "$events"
  )
}

run_job "$XAOD43" "$SRC43" "0,1,37" 50 part_evt0_1_37
if [[ -f "$XAOD48" ]]; then
  run_job "$XAOD48" "$SRC48" "86" 100 part_evt86
fi

for source_id in "$SRC43" "$SRC48"; do
  outdir="$SMOKE_ROOT/${source_id}"
  final="$outdir/ckf_leave_target_out_official_jacobian.jsonl"
  if [[ ! -d "$outdir" ]]; then
    continue
  fi
  if [[ -f "$final" ]]; then
    echo "refusing to overwrite $final" >&2
    exit 2
  fi
  cat "$outdir"/part_*.jsonl > "$final"
  echo "concatenated $final lines=$(wc -l < "$final")"
done
echo "B14U official supporting-plane Jacobian smoke complete"
ls -l "$SMOKE_ROOT"/*/*.jsonl
