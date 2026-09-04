#!/usr/bin/env python3
"""Write a read-only provenance audit of workbook-69 rank 2/3/4 sources.

Does not drop sources, retune S or rank_tolerance, force rank 5, reopen
the stable-core campaign, or treat the failures as cluster-local
confirmation.
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
from alignment.tracklet_independent_failure_provenance import (
    SCHEMA_VERSION,
    build_all_reports,
    load_audit_config,
    refuse_forbidden_operations,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _write_json(path: Path, payload: object) -> None:
    if isinstance(payload, Mapping):
        assert_no_alignment_payload(payload)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(json_ready(payload), indent=2, sort_keys=True) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        default=str(project_root() / "configs/tracklet_independent_failure_provenance_audit_v1.yaml"),
    )
    args = parser.parse_args()
    config = load_audit_config(args.config)
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
        ("failure_sources.json", "failure_sources"),
        ("sibling_controls.json", "sibling_controls"),
        ("next_stage_decision.json", "next_stage_decision"),
    ):
        payload = reports[key]
        if isinstance(payload, Mapping):
            payload = dict(payload)
            payload.update(extras)
        elif isinstance(payload, list):
            payload = {"rows": payload, **extras}
        _write_json(output / filename, payload)
    decision = reports["next_stage_decision"]
    print(
        json.dumps(
            {
                "output_dir": str(output),
                "decision": decision["decision"],
                "inherited_entry_69_decision": decision["inherited_entry_69_decision"],
                "overall_classification": decision["overall_classification"],
                "native_ranks": decision["native_ranks"],
                "classifications": decision["classifications"],
                "sources_dropped": False,
                "rank_forced_to_five": False,
                "used_as_cluster_local_confirmation": False,
                "geometry_write_allowed": False,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
