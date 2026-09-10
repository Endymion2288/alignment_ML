#!/usr/bin/env bash
# Task B14W smoke.  Reuses frozen WB120 dumps.  Does not rerun Athena.
set -euo pipefail
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D43="$ROOT/outputs/leave_target_out_dump_v1/b14u_smoke/mc24_100043_00400_00499/ckf_leave_target_out_official_jacobian.jsonl"
D48="$ROOT/outputs/leave_target_out_dump_v1/b14u_smoke/mc24_100048_00000_00049/ckf_leave_target_out_official_jacobian.jsonl"
H43="1c2daf7fda4e585d62624d1705934c3d621a69d560881f3d417d8a96bb1343ab"
H48="831a2a3fc9ff6e0aa519895d642a5de576c42a3a947700fab48df58bb798a19d"
if [[ ! -f "$D43" || ! -f "$D48" ]]; then
  echo "missing frozen WB120 dumps" >&2
  exit 2
fi
got43="$(sha256sum "$D43" | awk '{print $1}')"
got48="$(sha256sum "$D48" | awk '{print $1}')"
if [[ "$got43" != "$H43" || "$got48" != "$H48" ]]; then
  echo "WB120 dump hash mismatch" >&2
  exit 2
fi
echo "B14W reuses frozen WB120 dumps; Athena is not rerun"
echo "100043 $got43"
echo "100048 $got48"
python "$ROOT/scripts/run_independent_derivative_reference.py"
echo "B14W independent-reference smoke complete"
