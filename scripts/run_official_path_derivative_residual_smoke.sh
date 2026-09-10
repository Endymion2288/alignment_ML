#!/usr/bin/env bash
# Task B14V residual-contract smoke.
# Reuses frozen WB120 official-path Jacobian dumps.  Does not rerun Athena,
# does not overwrite WB109–WB120 dumps, and does not submit 1989 rows.
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
D43="$ROOT/outputs/leave_target_out_dump_v1/b14u_smoke/mc24_100043_00400_00499/ckf_leave_target_out_official_jacobian.jsonl"
D48="$ROOT/outputs/leave_target_out_dump_v1/b14u_smoke/mc24_100048_00000_00049/ckf_leave_target_out_official_jacobian.jsonl"
H43="1c2daf7fda4e585d62624d1705934c3d621a69d560881f3d417d8a96bb1343ab"
H48="831a2a3fc9ff6e0aa519895d642a5de576c42a3a947700fab48df58bb798a19d"

if [[ ! -f "$D43" ]]; then
  echo "missing frozen WB120 dump: $D43" >&2
  exit 2
fi
if [[ ! -f "$D48" ]]; then
  echo "missing frozen WB120 dump: $D48" >&2
  exit 2
fi

got43="$(sha256sum "$D43" | awk '{print $1}')"
got48="$(sha256sum "$D48" | awk '{print $1}')"
if [[ "$got43" != "$H43" ]]; then
  echo "WB120 100043 dump hash mismatch: $got43" >&2
  exit 2
fi
if [[ "$got48" != "$H48" ]]; then
  echo "WB120 100048 dump hash mismatch: $got48" >&2
  exit 2
fi

echo "B14V reuses frozen WB120 dumps; Athena is not rerun"
echo "100043 dump $D43 sha256=$got43 bytes=$(wc -c < "$D43") lines=$(wc -l < "$D43")"
echo "100048 dump $D48 sha256=$got48 bytes=$(wc -c < "$D48") lines=$(wc -l < "$D48")"
python "$ROOT/scripts/run_official_path_derivative_residual.py"
echo "B14V residual-contract smoke complete"
