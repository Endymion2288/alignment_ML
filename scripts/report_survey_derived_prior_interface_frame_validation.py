#!/usr/bin/env python3
"""Write the survey-derived prior interface and frame-validation reports.

Does not train, refit, write geometry, or emit an alignment payload.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from alignment.module_level_residual_poc import json_ready
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.survey_derived_prior_interface import (
    SCHEMA_VERSION,
    build_all_reports,
    load_interface_config,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _write_json(path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/survey_derived_prior_interface_frame_validation_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_interface_config(args.config)
    output = resolve_under_root(project_root(), str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    reports = build_all_reports(config)
    for filename, key in (
        ("survey_candidate_database.json", "survey_candidate_database"),
        ("survey_frame_reconciliation_validation.json", "survey_frame_reconciliation_validation"),
        ("track_fisher_external_prior_feasibility.json", "track_fisher_external_prior_feasibility"),
        ("metrology_covariance_request.json", "metrology_covariance_request"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = dict(reports[key])
        payload["created_utc"] = created
        payload["schema_version"] = SCHEMA_VERSION
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "possess_information_requirement": decision[
                    "survey_candidates_possess_degeneracy_breaking_information_requirement"
                ],
                "have_validated_ry_mapping": decision["have_validated_ry_mapping"],
                "have_measurement_covariance": decision["have_measurement_covariance"],
                "geometry_candidate": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
