#!/usr/bin/env python3
"""Write tracker-only identifiable-subspace and three-arm closure reports.

Does not retrain, change the frozen association policy, restack 2024 r0022
collision-like tracks, run full-parameter Newton, write geometry, or emit
an alignment payload.  Survey remains an external cross-check.
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
from alignment.tracker_only_identifiable_subspace import (
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
        default=str(project_root() / "configs/tracker_only_identifiable_subspace_three_arm_closure_v1.yaml"),
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
        ("identifiable_basis.json", "identifiable_basis"),
        ("subspace_stability.json", "subspace_stability"),
        ("three_arm_closure.json", "three_arm_closure"),
        ("cluster_local_diagnostic.json", "cluster_local_diagnostic"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = reports.get(key)
        if payload is None:
            payload = {"skipped": True, "reason": "not_produced_because_basis_unstable_or_disabled"}
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
                "identifiable_rank": decision["identifiable_rank"],
                "identifiable_basis_stable": decision["identifiable_basis_stable"],
                "null_injection_leakage_gate": decision["null_injection_leakage_gate"],
                "mixed_injection_projected_closure": decision["mixed_injection_projected_closure"],
                "authorize_frozen_v2_unknown_association_closure": decision[
                    "authorize_frozen_v2_unknown_association_closure"
                ],
                "geometry_write_allowed": False,
                "n_parameters": basis["pooled"]["n_parameters"],
                "null_dimension": basis["null_dimension"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
