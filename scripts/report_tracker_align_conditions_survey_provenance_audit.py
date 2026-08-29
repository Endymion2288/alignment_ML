#!/usr/bin/env python3
"""Write the /Tracker/Align conditions dump and survey-provenance reports.

Does not train, refit, write geometry, or emit an alignment payload.
"""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.module_level_residual_poc import json_ready
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root
from alignment.tracker_align_conditions_survey_provenance import (
    SCHEMA_VERSION,
    build_all_reports,
    load_audit_config,
    load_raw_pool_dump,
    query_cool_align_metadata,
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
        default=str(project_root() / "configs/tracker_align_conditions_survey_provenance_audit_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_audit_config(args.config)
    root = project_root()
    output = resolve_under_root(root, str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    raw = load_raw_pool_dump(resolve_under_root(root, str(config["raw_pool_dump"])))
    cool = query_cool_align_metadata(config["dbrelease"]["sqlite"])
    created = datetime.now(timezone.utc).isoformat()
    reports = build_all_reports(config, raw, cool)

    names = {
        "OFLCOND-FASER-04": "tracker_align_dump_oflcond_faser_04.json",
        "OFLCOND-FASER-05": "tracker_align_dump_oflcond_faser_05.json",
        "OFLCOND-FASER-06": "tracker_align_dump_oflcond_faser_06.json",
    }
    for tag, filename in names.items():
        payload = dict(reports["dumps"][tag])
        payload["created_utc"] = created
        _write_json(output / filename, payload)

    for filename, key in (
        ("conditions_channel_diff_report.json", "diff"),
        ("ift_station0_ry_cdx_conditions_summary.json", "summary"),
        ("survey_conditions_numeric_comparison.json", "comparison"),
        ("alignment_payload_provenance_report.json", "provenance"),
        ("next_stage_decision.json", "decision"),
    ):
        payload = dict(reports[key])
        payload["created_utc"] = created
        payload["schema_version"] = SCHEMA_VERSION
        _write_json(output / filename, payload)

    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": reports["decision"]["decision"],
                "current_reco_ift_plane_ry_mrad": reports["summary"]["ift_plane_ry_used_by_reconstruction_mrad"],
                "current_reco_C_dx_cond_mm": reports["summary"]["C_dx_cond_mm"],
                "enough_provenance_for_independent_survey_constraint": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
