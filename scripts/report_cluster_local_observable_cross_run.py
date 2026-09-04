#!/usr/bin/env python3
"""Write cluster-local observable cross-run identifiability reports.

Phase 1 is r14973 versus r14974 only.  Does not rescue tracklet-level
observables, retune S or rank_tolerance, reopen Frozen V2, run three-arm,
write geometry, or emit an alignment payload.  Entry 58 remains a
negative-control metric.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.cluster_local_observable_cross_run import (
    SCHEMA_VERSION,
    build_all_reports,
    load_campaign_config,
    refuse_forbidden_operations,
)
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
        default=str(project_root() / "configs/cluster_local_observable_cross_run_identifiability_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_campaign_config(args.config)
    output = resolve_under_root(project_root(), str(config["output_dir"]))
    output.mkdir(parents=True, exist_ok=True)
    created = datetime.now(timezone.utc).isoformat()
    reports = build_all_reports(config)
    shutil.copy2(config["config_path"], output / "config.yaml")
    extras = {
        "created_utc": created,
        "schema_version": SCHEMA_VERSION,
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "forbidden_operations": refuse_forbidden_operations(),
    }
    for filename, key in (
        ("parameter_definition.json", "parameter_definition"),
        ("reference_run.json", "reference_run"),
        ("transfer_run.json", "transfer_run"),
        ("wrong_dump_control.json", "wrong_dump_control"),
        ("cross_run_subspace.json", "cross_run_subspace"),
        ("entry_58_reproduction.json", "entry_58_reproduction"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = reports[key]
        if isinstance(payload, Mapping):
            payload = dict(payload)
            payload.update(extras)
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    compare = reports["cross_run_subspace"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "gates": decision["gates"],
                "all_four_gates_passed": decision["all_four_gates_passed"],
                "three_arm_authorized": False,
                "frozen_v2_alignment_loop_authorized": False,
                "real_data_correction_authorized": False,
                "geometry_write_allowed": False,
                "authorize_more_runs": False,
                "inherited_entry_58_decision": decision["inherited_entry_58_decision"],
                "entry_58_reproduction_decision": decision["entry_58_reproduction_decision"],
                "reference_rank": compare.get("left_rank"),
                "transfer_rank": compare.get("right_rank"),
                "max_principal_angle_deg": compare.get("max_principal_angle_deg"),
                "projector_frobenius_distance": compare.get("projector_frobenius_distance"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
