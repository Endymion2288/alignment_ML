#!/usr/bin/env python3
"""Write residual-blind physically-distinct track-coverage inventory reports.

Stage 1 only.  Does not construct A = W^{1/2} J S, SVD, rank, or an
alignment payload.  Does not retune S or rank_tolerance, drop sources,
delete rz, restack 2024 r0022, reopen sealed test, or skip to gauge
constraint until the inventory closes.
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
from alignment.physically_distinct_track_coverage import (
    SCHEMA_VERSION,
    build_all_reports,
    load_inventory_config,
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
        default=str(
            project_root()
            / "configs/physically_distinct_track_coverage_identifiability_feasibility_v1.yaml"
        ),
    )
    args = parser.parse_args()
    config = load_inventory_config(args.config)
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
        "stage": "residual_blind_coverage_inventory",
        "fd_identifiability_executed": False,
        "svd_or_rank_computed": False,
    }
    for filename, key in (
        ("selection_contract.json", "selection_contract"),
        ("canonical_coverage.json", "canonical_coverage"),
        ("candidate_coverage.json", "candidate_coverage"),
        ("inherited_real_data.json", "inherited_real_data"),
        ("eos_layout.json", "eos_layout"),
        ("admission.json", "admission"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = reports.get(key)
        if isinstance(payload, Mapping):
            payload = dict(payload)
            payload.update(extras)
        else:
            payload = {"value": payload, **extras}
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    admission = reports["admission"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "admitted_populations": decision.get("admitted_populations"),
                "export_authorized_populations": decision.get("export_authorized_populations"),
                "n_admitted": admission.get("n_admitted"),
                "n_export_authorized": admission.get("n_export_authorized"),
                "fd_identifiability_executed": False,
                "svd_or_rank_computed": False,
                "three_arm_authorized": False,
                "frozen_v2_alignment_loop_authorized": False,
                "real_data_correction_authorized": False,
                "geometry_write_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
