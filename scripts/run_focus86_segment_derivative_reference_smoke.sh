#!/usr/bin/env bash
# Task B14Y smoke.  Rebuilds the diagnostic helper, then dumps 100048/86
# required-hop hop-start FD + independent DOPRI5 reference.
# Reuses frozen WB123 100043 dumps for control 0/1/37.  Does not overwrite
# WB109–WB123 dumps.  Does not change FieldGradientDefaultExtension.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XAOD48="/eos/experiment/faser/data0/sim/mc24/particle_gun/100048/rec/s0013-r0022/FaserMC-MC24_PG_mupl_fasernu_5mrad_flukaE-100048-00000-00049-s0013-r0022-xAOD.root"
SRC48="mc24_100048_00000_00049"
WB12343="$ROOT/outputs/leave_target_out_dump_v1/b14x_smoke/mc24_100043_00400_00499/ckf_leave_target_out_official_jacobian.jsonl"
WB109="$ROOT/outputs/leave_target_out_dump_v1/dumps/mc24_100043_00400_00499/ckf_leave_target_out.jsonl"
SMOKE_ROOT="$ROOT/outputs/leave_target_out_dump_v1/b14y_smoke"

if [[ ! -f "$XAOD48" ]]; then
  echo "missing xAOD: $XAOD48" >&2
  exit 2
fi
if [[ ! -f "$WB109" ]]; then
  echo "missing official WB109 dump: $WB109" >&2
  exit 2
fi
if [[ ! -f "$WB12343" ]]; then
  echo "missing frozen WB123 dump: $WB12343" >&2
  exit 2
fi

bash "$ROOT/scripts/build_ckf_leave_target_out_dump.sh"

unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME \
  Athena_SET_UP Athena_EXTONLY_SET_UP Athena_RELONLY_SET_UP \
  AthenaExternals_SET_UP AthenaExternals_EXTONLY_SET_UP AthenaExternals_RELONLY_SET_UP \
  Calypso_SET_UP Calypso_EXTONLY_SET_UP Calypso_RELONLY_SET_UP
set +u
source "$ROOT/scripts/setup_environment.sh" calypso
set -u
export PYTHONPATH="$ROOT${PYTHONPATH:+:$PYTHONPATH}"
export LD_LIBRARY_PATH="$ROOT/build/leave_target_out_dump${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"

outdir="$SMOKE_ROOT/${SRC48}"
partfile="$outdir/part_evt86.jsonl"
mkdir -p "$outdir"
if [[ -f "$partfile" ]]; then
  echo "refusing to overwrite $partfile" >&2
  exit 2
fi
echo "=== B14Y focus-86 segment-reference smoke source=$SRC48 events=86 out=$partfile ==="
(
  cd "$outdir"
  python "$ROOT/scripts/dump_ckf_leave_target_out.py" \
    --input-xaod "$XAOD48" \
    --source-id "$SRC48" \
    --output "$partfile" \
    --nevents 100 \
    --skip-events 0 \
    --enable-profile-likelihood \
    --profile-only \
    --enable-profile-numerics \
    --profile-evaluate-only \
    --enable-profile-transport \
    --enable-official-supporting-plane-jacobian \
    --enable-field-gradient-variational-repair \
    --enable-focus86-segment-reference \
    --select-event-ids "86"
)

final="$outdir/ckf_leave_target_out_official_jacobian.jsonl"
if [[ -f "$final" ]]; then
  echo "refusing to overwrite $final" >&2
  exit 2
fi
cat "$outdir"/part_*.jsonl > "$final"
echo "concatenated $final lines=$(wc -l < "$final")"
echo "B14Y focus-86 segment reference smoke complete"
ls -l "$SMOKE_ROOT"/*/*.jsonl
echo "official audit is scripts/audit_focus86_segment_derivative_reference.py in the ML env"
