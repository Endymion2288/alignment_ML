#!/usr/bin/env python3
"""Write the Nov-2022 provenance and station-ry contract reports.

Does not train, refit, write geometry, map Kabsch/dz-vs-x to station ry,
construct covariance from population scatter, or enter Fisher.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.module_level_residual_poc import json_ready
from alignment.nov22_metrology_provenance_station_ry import (
    SCHEMA_VERSION,
    build_all_reports,
    load_contract_config,
)
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root, sha256_file
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/nov22_metrology_provenance_station_ry_contract_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_contract_config(args.config)
    root = project_root()
    output = resolve_under_root(root, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    reports = build_all_reports(config)
    shutil.copy2(config["config_path"], output / "config.yaml")
    extras = {
        "created_utc": created,
        "schema_version": SCHEMA_VERSION,
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "source_sha256": reports["nov22_raw_metrology_provenance"]["source_file"]["sha256"],
        "git_sha": reports["nov22_raw_metrology_provenance"]["git_sha"],
    }
    for filename, key in (
        ("nov22_raw_metrology_provenance.json", "nov22_raw_metrology_provenance"),
        ("calypso_station_ry_transform_contract.json", "calypso_station_ry_transform_contract"),
        ("station_ry_software_fd.json", "station_ry_software_fd"),
        ("nov22_iov_provenance.json", "nov22_iov_provenance"),
        ("official_constraint_slots.json", "official_constraint_slots"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = dict(reports[key])
        payload.update(extras)
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    fd_result = reports["station_ry_software_fd"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "closest_pivot": fd_result["closest_pivot"],
                "have_validated_ry_mapping": decision["have_validated_ry_mapping"],
                "have_measurement_covariance": decision["have_measurement_covariance"],
                "have_matching_year_conditions_iov": decision["have_matching_year_conditions_iov"],
                "enough_to_enter_fisher": decision["enough_to_enter_fisher"],
                "geometry_candidate": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
