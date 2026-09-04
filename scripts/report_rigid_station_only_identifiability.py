#!/usr/bin/env python3
"""Write rigid-station-only 5DoF identifiability reports.

Does not rescue 7D / cluster-local / stable-core observables, retune S or
rank_tolerance, drop failed sources, delete dz/C_dx from a previous 7D SVD,
retrain, change Frozen V2, run Newton, write geometry, or emit an alignment
payload.  Survey remains an external cross-check.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.module_level_residual_poc import json_ready
from alignment.operating_protocol_v1_final_closure import project_root, resolve_under_root, sha256_file
from alignment.rigid_station_only_identifiability import (
    SCHEMA_VERSION,
    build_all_reports,
    load_campaign_config,
    refuse_forbidden_operations,
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
        default=str(project_root() / "configs/rigid_station_only_tracker_alignment_identifiability_v1.yaml"),
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
        ("jacobian_validity.json", "jacobian_validity"),
        ("identifiable_basis.json", "identifiable_basis"),
        ("coverage.json", "coverage"),
        ("subspace_stability.json", "subspace_stability"),
        ("rank_loss_diagnosis.json", "rank_loss_diagnosis"),
        ("three_arm_closure.json", "three_arm_closure"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = reports.get(key)
        if payload is None:
            payload = {"skipped": True, "reason": "not_produced_because_identifiability_gate_failed"}
        if isinstance(payload, Mapping):
            payload = dict(payload)
            payload.update(extras)
        else:
            payload = {"value": payload, **extras}
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    basis = reports["identifiable_basis"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "jacobian_valid": decision["jacobian_valid"],
                "five_dof_identifiable_and_portable": decision["five_dof_identifiable_and_portable"],
                "pooled_identifiable_rank": decision["pooled_identifiable_rank"],
                "pooled_rank_is_not_portability": True,
                "three_arm_authorized": False,
                "frozen_v2_alignment_loop_authorized": False,
                "real_data_correction_authorized": False,
                "geometry_write_allowed": False,
                "n_parameters": 5,
                "null_dimension": basis["null_dimension"],
                "per_source_ranks": [row["identifiable_rank"] for row in basis["per_source"]],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
