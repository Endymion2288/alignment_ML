#!/usr/bin/env bash
# Local Yasu-S3A Jacobian dump on the two frozen xAODs.
# Does not submit HTCondor and does not rewrite official collections.
# Focus identities are frozen in the YAML before any Jacobian number is
# inspected.  Each source is dumped through the highest frozen skip_index.

set -euo pipefail

script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
project_root="$(cd "$script_dir/.." && pwd)"

set +u
source "$project_root/scripts/setup_environment.sh" calypso
set -u

plugin_dir="$project_root/build/yasu_s3a_jacobian_dump"
export LD_LIBRARY_PATH="$plugin_dir${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
export JOBOPTSEARCHPATH="$plugin_dir${JOBOPTSEARCHPATH:+:$JOBOPTSEARCHPATH}"

python - "$project_root" <<'PY'
import subprocess
import sys
from pathlib import Path

import yaml

root = Path(sys.argv[1])
config_path = root / "configs/yasu_s3a_jacobian_closure_v1.yaml"
config = yaml.safe_load(config_path.read_text())
if config.get("htcondor_authorized"):
    raise SystemExit("S3A must not submit HTCondor")
if config.get("use_qp_bending_proxy"):
    raise SystemExit("S3A must not use qp_bending_proxy")
dump = root / "scripts" / "dump_ckf_yasu_s3a_jacobian.py"
sources = (
    config["mc_data"]["construction_sources"]
    + config["mc_data"]["validation_sources"]
)
for spec in sources:
    source_id = spec["source_id"]
    skips = [
        int(row["skip_index"])
        for row in config["focus_identities"]
        if row["source_id"] == source_id
    ]
    if not skips:
        raise SystemExit(f"no frozen focus identities for {source_id}")
    nevents = max(skips) + 1
    out_dir = root / config["dump_root"] / source_id
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
        "--file-sha256",
        spec["file_sha256"],
        "--nevents",
        str(nevents),
        "--skip-events",
        "0",
        "--config",
        str(config_path),
    ]
    print("+", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
print("finished Yasu-S3A Jacobian dumps", flush=True)
PY
