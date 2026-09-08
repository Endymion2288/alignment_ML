#!/usr/bin/env bash
# Task B14R seed-scale smoke.  Pre-registered scales 0.1 / 1 / 10 only.
# Does not overwrite WB109 dumps and does not choose a production scale.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
XAOD="/eos/experiment/faser/data0/sim/mc24/particle_gun/100043/rec/s0013-r0022/FaserMC-MC24_PG_mumi_fasernu_5mrad_flukaE-100043-00400-00499-s0013-r0022-xAOD.root"
LTO_SOURCE_ID="mc24_100043_00400_00499"
OFFICIAL_DUMP="$ROOT/outputs/leave_target_out_dump_v1/dumps/${LTO_SOURCE_ID}/ckf_leave_target_out.jsonl"

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
  local scale="$1" label="$2" skip="$3" nevents="$4" part="$5"
  local outdir="$ROOT/outputs/leave_target_out_dump_v1/b14r_smoke/${label}/${LTO_SOURCE_ID}"
  local partfile="$outdir/${part}.jsonl"
  mkdir -p "$outdir"
  if [[ -f "$partfile" ]]; then
    echo "refusing to overwrite $partfile" >&2
    exit 2
  fi
  echo "=== B14R smoke scale=$scale skip=$skip nevents=$nevents out=$partfile ==="
  (
    cd "$outdir"
    python "$ROOT/scripts/dump_ckf_leave_target_out.py" \
      --input-xaod "$XAOD" \
      --source-id "$LTO_SOURCE_ID" \
      --output "$partfile" \
      --nevents "$nevents" \
      --skip-events "$skip" \
      --seed-covariance-scale "$scale"
  )
}

for spec in 0.1:scale_0p1 1.0:scale_1p0 10.0:scale_10p0; do
  scale="${spec%%:*}"
  label="${spec##*:}"
  run_part "$scale" "$label" 0 2 part_evt0_1
  run_part "$scale" "$label" 37 1 part_evt37
  outdir="$ROOT/outputs/leave_target_out_dump_v1/b14r_smoke/${label}/${LTO_SOURCE_ID}"
  final="$outdir/ckf_leave_target_out.jsonl"
  if [[ -f "$final" ]]; then
    echo "refusing to overwrite $final" >&2
    exit 2
  fi
  cat "$outdir/part_evt0_1.jsonl" "$outdir/part_evt37.jsonl" > "$final"
  echo "concatenated $final lines=$(wc -l < "$final")"
done
echo "B14R smoke complete"
ls -l "$ROOT/outputs/leave_target_out_dump_v1/b14r_smoke"/scale_*/"${LTO_SOURCE_ID}"/*.jsonl
