#!/usr/bin/env bash
# Local measurement-bending dump on the two frozen xAODs.
# Does not submit HTCondor and does not rewrite official collections.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"
campaign="${1:-contract}"
if [[ "$campaign" == "smoke" || "$campaign" == "contract" ]]; then
  nevents=20
elif [[ "$campaign" == "batch" ]]; then
  echo "S2I batch/HTCondor is not authorized; use contract" >&2
  exit 2
else
  echo "usage: $0 contract|smoke" >&2
  exit 2
fi

set +u
source "$project_root/scripts/setup_environment.sh" calypso
set -u

plugin_dir="$project_root/build/three_st_qp_measurement_bending_dump"
export LD_LIBRARY_PATH="$plugin_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JOBOPTSEARCHPATH="$plugin_dir${JOBOPTSEARCHPATH:+:$JOBOPTSEARCHPATH}"

python - "$project_root" "$campaign" "$nevents" <<'PY'
import subprocess
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
campaign = sys.argv[2]
nevents = int(sys.argv[3])
config = yaml.safe_load((root / "configs/three_st_qp_measurement_bending_v1.yaml").read_text())
dump = root / "scripts" / "dump_ckf_three_st_qp_measurement_bending.py"
sources = (
    config["mc_data"]["construction_sources"]
    + config["mc_data"]["validation_sources"]
)
for spec in sources:
    source_id = spec["source_id"]
    out_dir = root / config["dump_root"] / campaign / source_id
    out_dir.mkdir(parents=True, exist_ok=True)
    track = out_dir / config["dump_filename"]
    event = out_dir / config["event_filename"]
    if track.exists() or event.exists():
        raise FileExistsError(f"refusing to overwrite {track} / {event}")
    cmd = [
        "python",
        str(dump),
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
        campaign,
        "--config",
        str(root / "configs/three_st_qp_measurement_bending_v1.yaml"),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
print(f"finished {campaign} measurement-bending dumps", flush=True)
PY
