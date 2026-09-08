#!/usr/bin/env python3
"""Freeze WB83 geometry matrix, weak mode, replica manifest, and JSON contract.

Must run before replica workers.  Does not open overlays / 00350 / Blind.
"""

from __future__ import annotations

import argparse
import os
import socket
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from alignment.wb83_qualification import (
    EXPERIMENT,
    build_geometry_matrix,
    build_replica_manifest,
    freeze_weak_mode,
    measurement_likelihood_contract_json,
    refuse_wb83_path,
    write_json,
)


def _git_sha() -> str:
    return subprocess.run(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout.strip()


def _git_porcelain() -> str:
    return subprocess.run(["git", "status", "--porcelain"], cwd=PROJECT_ROOT, check=True, capture_output=True, text=True).stdout


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        default=str(PROJECT_ROOT / f"outputs/mc24_four_station_{EXPERIMENT}"),
    )
    args = parser.parse_args()
    output = Path(args.output_root)
    refuse_wb83_path(output)
    weak = freeze_weak_mode()
    matrix = build_geometry_matrix(weak)
    manifest = build_replica_manifest(matrix)
    snapshot = {
        "stage": "SNAPSHOT",
        "git_sha": _git_sha(),
        "git_status_porcelain": _git_porcelain(),
        "hostname": socket.gethostname(),
        "utc": datetime.now(timezone.utc).isoformat(),
        "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
        "matrix_sha256": matrix["matrix_sha256"],
        "calypso_revision": None,
        "field_material_config": "toy_uniform_By_v1",
        "alignment_oracle_qualified": False,
    }
    write_json(output / "measurement_likelihood_contract.json", measurement_likelihood_contract_json())
    write_json(output / "geometry_qualification_matrix.json", matrix)
    write_json(output / "replica_manifest.json", manifest)
    write_json(output / "weak_mode_report.json", weak)
    write_json(output / "snapshot.json", snapshot)
    print(json_summary(matrix, manifest, weak))


def json_summary(matrix, manifest, weak) -> str:
    import json

    return json.dumps(
        {
            "matrix_sha256": matrix["matrix_sha256"],
            "n_cells": len(matrix["cells"]),
            "n_declared_rows": manifest["n_declared_rows"],
            "weak_condition_number": weak["condition_number"],
            "full_rank_claim": weak["full_rank_claim"],
            "alignment_oracle_qualified": False,
        },
        indent=2,
    )


if __name__ == "__main__":
    main()
