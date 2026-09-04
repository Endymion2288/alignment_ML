#!/usr/bin/env python3
"""Write cross-source stable-core hypothesis and independent-validation reports.

Does not reopen the workbook-68 V1 failure, retune S or rank_tolerance,
truncate the rank-6 source, open sealed test, run Newton, write geometry,
or emit an alignment payload.
"""

from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping

from alignment.cross_source_stable_core import (
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
        default=str(project_root() / "configs/cross_source_stable_core_identifiable_subspace_v1.yaml"),
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
        ("hypothesis_core.json", "hypothesis_core"),
        ("loso_stability.json", "loso_stability"),
        ("source_bootstrap.json", "source_bootstrap"),
        ("event_bootstrap.json", "event_bootstrap"),
        ("independent_validation.json", "independent_validation"),
        ("three_arm_closure.json", "three_arm_closure"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = reports.get(key)
        if payload is None:
            payload = {"skipped": True, "reason": "not_produced_because_prior_gate_failed"}
        if isinstance(payload, Mapping):
            payload = dict(payload)
            payload.update(extras)
        else:
            payload = {"value": payload, **extras}
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    core = reports["hypothesis_core"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "inherited_v1_decision": decision["inherited_v1_decision"],
                "core_dimension": decision["core_dimension"],
                "hypothesis_core_stable": decision["hypothesis_core_stable"],
                "independent_validation_pass": decision["independent_validation_pass"],
                "null_injection_leakage_gate": decision["null_injection_leakage_gate"],
                "mixed_injection_projected_closure": decision["mixed_injection_projected_closure"],
                "three_arm_authorized": decision["three_arm_authorized"],
                "authorize_frozen_v2_unknown_association_closure": False,
                "geometry_write_allowed": False,
                "native_ranks": core["native_ranks"],
                "includes_rank_six_source": core["includes_rank_six_source"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
