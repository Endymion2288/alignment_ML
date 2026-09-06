#!/usr/bin/env python3
"""Workbook-86 Full CurvilinearUVT Branch Covariance Contract Repair driver.

Stages:

    validate-config   WB81–WB85 inheritance SHAs + frozen-flag / gate guards
    closure           WB83 Stage A three-way: baseline / WB85 beam / WB86 full
    decide            pre-registered decision tree

Not an alignment diagnostic.  No real residual, no geometry/conditions write,
no FaserActs / Stage B, no scale / chi2 / outlier / angle rejection.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from alignment.cad_survey_nov22 import git_head_sha
from alignment.gauge_fixed_real_data_diagnostic import _json_ready
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment import full_curvilinear_uvt_repair_validation as fcv
from alignment.full_curvilinear_uvt_repair_validation import DEFAULT_CONFIG


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _output_root(config: dict) -> Path:
    return resolve_under_root(project_root(), str(config["output_root"]))


def _stage_validate_config(config: dict) -> dict:
    return {
        "config_validation": {
            "kind": "config_validation",
            "config_path": config["config_path"],
            "config_sha256": sha256_file(Path(config["config_path"])),
            "git_head_sha": git_head_sha(),
            "frozen_flags_verified": True,
            "inheritance_sha256_verified": True,
            "wb83_gates_frozen": True,
            "held_out_accessed": False,
            "real_data_alignment_authorized": False,
            "geometry_write_allowed": False,
            "measurement_model_validated": False,
            "population": "mc_only_no_real_data_residual",
            "pass": True,
        }
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validate-config", "closure", "decide", "all"),
    )
    args = parser.parse_args()
    config = fcv.load_config(args.config)
    config["config_path"] = str(resolve_under_root(project_root(), args.config))
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage in ("validate-config", "all"):
        _write_json(root / "config_validation.json", _stage_validate_config(config))
    if args.stage in ("closure", "all"):
        reports = fcv.run_three_way(config)
        _write_json(
            root / "source_covariance_closure_three_way.json",
            reports["source_covariance_closure_three_way"],
        )
        _write_json(
            root / "curvilinear_branch_subset_closure.json",
            reports["curvilinear_branch_subset_closure"],
        )
        _write_json(
            root / "full_curvilinear_uvt_repair_validation.json",
            reports["full_curvilinear_uvt_repair_validation"],
        )
    if args.stage in ("decide", "all"):
        validation = json.loads(
            (root / "full_curvilinear_uvt_repair_validation.json").read_text(
                encoding="utf-8"
            )
        )["validation"]
        decision = fcv.decide(config, validation)
        summary = {
            "kind": "campaign_summary",
            "schema_version": config["schema_version"],
            "created_utc": datetime.now(timezone.utc).isoformat(),
            "git_head_sha": git_head_sha(),
            "config_path": config["config_path"],
            "workbook": 86,
            "decision": decision["decision"],
            "measurement_model_validated": False,
            "measurement_model_v2_discussion_allowed": decision[
                "measurement_model_v2_discussion_allowed"
            ],
            "propagated_covariance_validation_authorized": decision[
                "propagated_covariance_validation_authorized"
            ],
            "geometry_write_allowed": False,
            "real_data_alignment_authorized": False,
            "held_out_accessed": False,
            "faseracts_propagation_entered": False,
            "stage_b_entered": False,
            "frozen_v2_alignment_authorized": False,
        }
        _write_json(root / "covariance_repair_decision.json", decision)
        _write_json(root / "campaign_summary.json", summary)
    return 0


if __name__ == "__main__":
    sys.exit(main())
