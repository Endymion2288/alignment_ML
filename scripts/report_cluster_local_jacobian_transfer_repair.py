#!/usr/bin/env python3
"""Write cluster-local transfer exact-join repair reports.

Does not fuzzy-match, nearest-neighbour join, reselect the population,
reopen entry 58, merge into the stable-core gate, write geometry, or
emit an alignment payload.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.cluster_local_jacobian_transfer_repair import (
    SCHEMA_VERSION,
    build_all_reports,
    load_repair_config,
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
        default=str(project_root() / "configs/cluster_local_jacobian_transfer_repair_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_repair_config(args.config)
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
        ("reference_join.json", "reference_join"),
        ("transfer_join.json", "transfer_join"),
        ("wrong_dump_control.json", "wrong_dump_control"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = reports[key]
        if isinstance(payload, Mapping):
            payload = dict(payload)
            payload.update(extras)
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "exact_join_restored": decision["exact_join_restored"],
                "wrong_dump_control_still_zero": decision["wrong_dump_control_still_zero"],
                "n_reference_measurements": decision["n_reference_measurements"],
                "n_transfer_measurements": decision["n_transfer_measurements"],
                "n_wrong_dump_measurements": decision["n_wrong_dump_measurements"],
                "authorize_cluster_local_identifiability_campaign": False,
                "not_a_stable_core_gate": True,
                "geometry_write_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
