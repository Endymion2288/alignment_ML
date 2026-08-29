#!/usr/bin/env python3
"""Write the survey-summary reconstruction and frame-reconciliation reports.

Does not train, refit, write geometry, or emit an alignment payload.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from typing import Any, Mapping

from alignment.module_level_residual_poc import json_ready
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.survey_summary_reconstruction import (
    SCHEMA_VERSION,
    build_all_reports,
    load_reconstruction_config,
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
        default=str(project_root() / "configs/survey_summary_reconstruction_frame_reconciliation_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_reconstruction_config(args.config)
    output = resolve_under_root(project_root(), str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    reports = build_all_reports(config)
    for filename, key in (
        ("survey_2021_summary.json", "survey_2021_summary"),
        ("survey_2022_summary.json", "survey_2022_summary"),
        ("ift_layer_contrast_reconstruction.json", "ift_layer_contrast_reconstruction"),
        ("survey_calypso_frame_reconciliation.json", "survey_calypso_frame_reconciliation"),
        ("slide_summary_prior_feasibility.json", "slide_summary_prior_feasibility"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = dict(reports[key])
        payload["created_utc"] = created
        payload["schema_version"] = SCHEMA_VERSION
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    contrast = reports["ift_layer_contrast_reconstruction"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "C_dx_slide_mm": contrast["C_dx_slide_mm"],
                "layer_mean_coherent_tilt_candidate_mrad": contrast[
                    "layer_mean_coherent_tilt_candidate_mrad"
                ],
                "layers_0_1_2_confirmed_as_ift": contrast["layers_0_1_2_confirmed_as_ift"],
                "mutually_coordinate_consistent": decision["answer"]["mutually_coordinate_consistent"],
                "remaining_uncertainty": decision["answer"]["remaining_uncertainty"],
                "geometry_candidate": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
