#!/usr/bin/env bash
# Local inspect-only dump smoke for Yasu-S2J.  Does not submit HTCondor
# and does not overwrite WB125 contract dumps.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
nevents="${1:-5}"

unset PYTHONPATH LD_LIBRARY_PATH ROOTSYS ROOT_INCLUDE_PATH PYTHONHOME \
  Athena_SET_UP Athena_EXTONLY_SET_UP Athena_RELONLY_SET_UP \
  AthenaExternals_SET_UP AthenaExternals_EXTONLY_SET_UP AthenaExternals_RELONLY_SET_UP \
  Calypso_SET_UP Calypso_EXTONLY_SET_UP Calypso_RELONLY_SET_UP
set +u
source "$project_root/scripts/setup_environment.sh" calypso
set -u

plugin_dir="$project_root/build/three_st_qp_measurement_bending_dump"
export LD_LIBRARY_PATH="$plugin_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JOBOPTSEARCHPATH="$plugin_dir${JOBOPTSEARCHPATH:+:$JOBOPTSEARCHPATH}"
export PYTHONPATH="$project_root${PYTHONPATH:+:$PYTHONPATH}"

python - "$project_root" "$nevents" <<'PY'
import math
import subprocess
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
nevents = int(sys.argv[2])
config = yaml.safe_load((root / "configs/three_st_qp_measurement_bending_batch_v1.yaml").read_text())
spec = config["mc_data"]["construction_sources"][0]
source_id = spec["source_id"]
out_dir = root / config["dump_root"] / "local_smoke" / source_id
out_dir.mkdir(parents=True, exist_ok=True)
track = out_dir / config["dump_filename"]
event = out_dir / config["event_filename"]
if track.exists() or event.exists():
    raise FileExistsError(f"refusing to overwrite {track} / {event}")
cmd = [
    "python",
    str(root / "scripts" / "dump_ckf_three_st_qp_measurement_bending.py"),
    "--input-xaod",
    spec["input_xaod"],
    "--output",
    str(track),
    "--event-output",
    str(event),
    "--source-id",
    source_id,
    "--nevents",
    str(nevents),
    "--skip-events",
    "0",
    "--campaign",
    "local_smoke",
    "--config",
    str(root / "configs/three_st_qp_measurement_bending_batch_v1.yaml"),
]
print("+", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

import json
n_tracks = 0
n_nan = 0
n_complete = 0
n_truth = 0
stations = set()
for line in track.read_text(encoding="utf-8").splitlines():
    row = json.loads(line)
    if row.get("kind") != "track":
        continue
    n_tracks += 1
    if row.get("truth_reference_available"):
        n_truth += 1
    cents = row.get("station_centroids") or {}
    complete = True
    for st in ("1", "2", "3"):
        item = cents.get(st) or {}
        stations.add(st)
        for key in ("x_mm", "y_mm", "z_mm"):
            value = item.get(key)
            if value is None or (isinstance(value, float) and math.isnan(value)):
                n_nan += 1
                complete = False
    if complete and all(st in cents for st in ("1", "2", "3")):
        n_complete += 1
print(
    f"local_smoke tracks={n_tracks} complete_centroids={n_complete} "
    f"nan={n_nan} truth_ref={n_truth} stations={sorted(stations)}"
)
if n_tracks == 0 or n_nan or n_complete == 0 or n_truth == 0:
    raise SystemExit("local smoke dump failed consistency checks")
print("local smoke dump is consistent with WB125 station/truth contract")
PY
