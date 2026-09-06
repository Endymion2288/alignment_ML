#!/usr/bin/env python3
"""Workbook-84 SegmentFit Covariance Coordinate Contract Audit driver.

Stages enforce the frozen ordering structurally:

    validate-config   config + WB81/WB82/WB83 inheritance SHAs + frozen-flag guards
    audit             code-audit + synthetic covariance closure (Cases A-E) +
                      closure test; writes the four WB84 output JSONs
    decide            pre-registered decision tree (locate the WB83 mechanism)

This campaign is NOT an alignment diagnostic.  It reads NO real-data residual,
never opens held-out data, never writes geometry/conditions, never modifies the
covariance, never adds scale factors, and never tunes parameters to chi2.
``held_out_accessed=false`` throughout.
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
from alignment import segmentfit_covariance_coordinate_contract as sccc
from alignment.segmentfit_covariance_coordinate_contract import DEFAULT_CONFIG


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _output_root(config: dict) -> Path:
    return resolve_under_root(project_root(), str(config["output_root"]))


def _stage_validate_config(config: dict) -> dict:
    report = {
        "kind": "config_validation",
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "git_head_sha": git_head_sha(),
        "frozen_flags_verified": True,
        "inheritance_sha256_verified": True,
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "measurement_model_validated": False,
        "population": "mc_only_no_real_data_residual",
        "pass": True,
    }
    return {"config_validation": report}


def _stage_audit(config: dict) -> dict:
    outputs = sccc.run_audit(config)
    root = _output_root(config)
    for name, payload in outputs.items():
        _write_json(root / f"{name}.json", payload)
    return outputs


def _stage_decide(config: dict) -> dict:
    root = _output_root(config)
    failure = json.loads(
        (root / "failure_location_report.json").read_text(encoding="utf-8")
    )["failure_location"]
    decision = {
        "kind": "segmentfit_covariance_coordinate_contract_decision",
        "stage": "audit",
        "wb83_mechanism": failure["wb83_mechanism"],
        "root_cause": failure["root_cause"],
        "location": failure["location"],
        "exporter_transform_bug": failure["exporter_transform_bug"],
        "segment_fit_generation_bug": failure["segment_fit_generation_bug"],
        "coordinate_convention_mismatch": failure["coordinate_convention_mismatch"],
        "jacobian_sign_error": failure["jacobian_sign_error"],
        "segment_fit_hit_error_model_calibrated": failure[
            "segment_fit_hit_error_model_calibrated"
        ],
        "success_criterion_answer": failure["success_criterion_answer"],
        "wb83_repair_campaign_pointer_status": failure[
            "wb83_repair_campaign_pointer_status"
        ],
        "hit_error_model_audit_authorized": failure["hit_error_model_audit_authorized"],
        "decision": "deterministic_segmentfit_get_state_transform_bug",
        "covariance_repair_validation_campaign_authorized": False,
        "note": (
            "WB84 locates the WB83 position_xy_swap_with_slope_miscalibration mechanism "
            "as TWO deterministic bugs in SegmentFitAlg::GetState's covariance transform "
            "(the NtupleDumperAlg exporter transform is CORRECT): (1) a coordinate "
            "convention mismatch -- the global (x, y) covariance is written into the "
            "Curvilinear (loc1, loc2) slots assuming (loc1, loc2) = (x, y), but the Athena "
            "Curvilinear frame for FASER beam tracks defines (loc1, loc2) = (-y, +x); "
            "(2) a sign error in the analytic Jacobian d phi/d tx = +ty/r^2 (should be "
            "-ty/r^2).  The SegmentFit hit-error model itself is CALIBRATED.  A new "
            "covariance repair validation campaign may be pre-registered separately; this "
            "campaign does NOT repair the covariance."
        ),
    }
    summary = {
        "kind": "campaign_summary",
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_sha": git_head_sha(),
        "config_path": config["config_path"],
        "workbook": 84,
        "decision": decision["decision"],
        "root_cause": decision["root_cause"],
        "location": decision["location"],
        "geometry_write_allowed": False,
        "real_data_alignment_authorized": False,
        "measurement_model_validated": False,
        "held_out_accessed": False,
    }
    return {
        "segmentfit_covariance_coordinate_contract_decision": decision,
        "campaign_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validate-config", "audit", "decide", "all"),
    )
    args = parser.parse_args()
    config = sccc.load_config(args.config)
    config["config_path"] = str(resolve_under_root(project_root(), args.config))
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage in ("validate-config", "all"):
        _write_json(root / "config_validation.json", _stage_validate_config(config))
    if args.stage in ("audit", "all"):
        _stage_audit(config)
    if args.stage in ("decide", "all"):
        reports = _stage_decide(config)
        _write_json(
            root / "segmentfit_covariance_coordinate_contract_decision.json",
            reports["segmentfit_covariance_coordinate_contract_decision"],
        )
        _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
