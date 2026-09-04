#!/usr/bin/env python3
"""Write the cad_survey_nov22 parser, frame, and covariance-audit reports.

Does not train, refit, write geometry, map unconfirmed tilts to station ry,
treat population Sigma as a Gaussian prior, or emit an alignment payload.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.cad_survey_nov22 import SCHEMA_VERSION, build_all_reports, load_audit_config
from alignment.module_level_residual_poc import json_ready
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
        default=str(project_root() / "configs/cad_survey_nov22_frame_covariance_audit_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_audit_config(args.config)
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
        "source_sha256": reports["cad_survey_nov22_provenance"]["source_file"]["sha256"],
        "git_sha": reports["cad_survey_nov22_provenance"]["git_sha"],
    }
    for filename, key in (
        ("cad_survey_nov22_provenance.json", "cad_survey_nov22_provenance"),
        ("cad_survey_nov22_parsed.json", "cad_survey_nov22_parsed"),
        ("cad_survey_nov22_geometry.json", "cad_survey_nov22_geometry"),
        ("cad_survey_nov22_frame_reconciliation.json", "cad_survey_nov22_frame_reconciliation"),
        ("cad_survey_nov22_covariance_audit.json", "cad_survey_nov22_covariance_audit"),
        ("official_constraint_slots.json", "official_constraint_slots"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = dict(reports[key])
        payload.update(extras)
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    geometry = reports["cad_survey_nov22_geometry"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "C_dx_mm": geometry["C_dx_mm"],
                "reproduces_entry64_C_dx_slide": geometry["reproduces_entry64_C_dx_slide"],
                "have_validated_ry_mapping": decision["have_validated_ry_mapping"],
                "have_measurement_covariance": decision["have_measurement_covariance"],
                "have_matching_year_conditions_iov": decision["have_matching_year_conditions_iov"],
                "geometry_candidate": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
