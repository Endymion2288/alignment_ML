#!/usr/bin/env python3
"""Write the survey/metrology ingestion + IOV infrastructure reports.

Does not rebuild the 2024 r0022 Jacobian, invent survey numbers, or emit
an alignment payload.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.module_level_residual_poc import json_ready
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.survey_metrology_iov_infrastructure import (
    SCHEMA_VERSION,
    calypso_detector_hierarchy,
    conditions_provenance_matrix,
    decide_next_stage,
    empty_constraint_catalog,
    fisher_validation,
    iov_records,
    load_infrastructure_config,
    metrology_requirement_table,
    common_audit_state,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _write_json(path: Path, payload: Mapping[str, Any]) -> None:
    assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/survey_metrology_iov_alignment_infrastructure_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_infrastructure_config(args.config)
    root = project_root()
    output = resolve_under_root(root, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    state = common_audit_state()

    catalog = empty_constraint_catalog(config)
    schema = {
        **state,
        "created_utc": created,
        "schema_version": SCHEMA_VERSION,
        "accepted_kinds": [
            "station_rigid_transform",
            "layer_relative_transform",
            "C_dx",
            "linear_equality",
        ],
        "required_fields": [
            "constraint_id",
            "kind",
            "detector_hierarchy",
            "frame",
            "reference_object",
            "parameter_names",
            "value",
            "sigma",
            "covariance",
            "units",
            "year",
            "iov_id",
            "source_file",
            "independent_of_track_residual",
            "provenance",
            "availability",
        ],
        "availability_states": ["unavailable", "feasibility_only", "measured"],
        "forbidden_as_measurement": [
            "geomdb_layerpitch",
            "design_stereo",
            "write_alignment_neutral_pool",
            "software_dz_gauge_5mm",
            "nominal_station_z",
        ],
        "calypso_hierarchy": calypso_detector_hierarchy(config),
        "empty_slots": catalog,
        "n_measured": 0,
        "ingest_rule": (
            "A real survey must set availability=measured, independent_of_track_residual=true, "
            "value, sigma/covariance, source_file, year, and iov_id. Design sizes and the "
            "software dz gauge stay out of value/sigma. feasibility_only is unit-test only."
        ),
    }
    _write_json(output / "external_constraint_schema.json", schema)

    records = iov_records(config)
    years = {}
    for row in records:
        years.setdefault(int(row["year"]), []).append(row)
    state_manifest = {
        **state,
        "created_utc": created,
        "form": config["parameterization"]["form"],
        "do_not_free_fill_or_run_initially": True,
        "do_not_share_one_constant_set_across_years": True,
        "do_not_share_one_constant_set_across_conditions_tags": True,
        "immutable_static_geometry": [
            "strip_pitch",
            "stereo_angle",
            "ift_layer_pitch_and_cassette_drawing",
            "nominal_station_z",
        ],
        "iov_specific_default": ["station_dx", "station_dy", "station_rx", "station_ry", "station_rz"],
        "C_dx_policy": "common_static_candidate_until_opening_or_thermal_evidence_but_external_metrology_may_override",
        "dz_policy": "survey_or_gauge_constrained_not_floated_from_tracks",
        "years": [
            {
                "year": year,
                "iovs": iovs,
                "do_not_merge_with_other_years": True,
            }
            for year, iovs in sorted(years.items())
        ],
        "did_rebuild_2024_r0022_jacobian": False,
    }
    _write_json(output / "iov_alignment_state_manifest.json", state_manifest)

    validation = fisher_validation(config)
    validation["created_utc"] = created
    _write_json(output / "track_plus_prior_fisher_validation.json", validation)

    table = metrology_requirement_table(config)
    table["created_utc"] = created
    _write_json(output / "metrology_requirement_table.json", table)

    matrix = conditions_provenance_matrix(records)
    matrix["created_utc"] = created
    _write_json(output / "conditions_provenance_matrix.json", matrix)

    decision = decide_next_stage(catalog=catalog, validation=validation, matrix=matrix)
    decision["created_utc"] = created
    decision["schema_version"] = SCHEMA_VERSION
    _write_json(output / "next_stage_decision.json", decision)
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "combiner_ready": decision["combiner_ready"],
                "n_empty_slots": len(catalog),
                "n_iovs": len(records),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
