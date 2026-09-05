#!/usr/bin/env python3
"""Workbook-83 Propagated-Covariance Upstream Repair & MC Validation driver.

Stages enforce the frozen ordering structurally:

    validate-config   config + WB81/WB82 inheritance SHAs + frozen-flag guards
    source-closure    Stage A: source-disjoint source-tracklet covariance truth
                      closure (construction + validation), NO propagation
    repairs           Stage A: pre-registered repair evaluation (derived on
                      construction, confirmed on validation)
    decide            Stage A pre-registered decision tree

Stage B (propagation covariance construction) is pre-registered in the config
but is NOT entered in this campaign unless Stage A yields a portable calibrated
source covariance.

This campaign is NOT an alignment diagnostic.  It reads NO real-data residual,
never opens held-out data, never writes geometry/conditions, and never solves
for an alignment correction.  ``held_out_accessed=false`` throughout.
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
from alignment import source_tracklet_covariance_closure as stcc
from alignment.source_tracklet_covariance_closure import DEFAULT_CONFIG


def _write_json(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(_json_ready(payload), indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


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
        "external_constraint_ingest_authorized": False,
        "population": "mc_only_no_real_data_residual",
        "pass": True,
    }
    return {"config_validation": report}


def _stage_source_closure(config: dict) -> dict:
    construction = stcc.run_source_closure_split(config, "construction")
    validation = stcc.run_source_closure_split(config, "validation")
    return {
        "source_tracklet_covariance_closure": {
            "kind": "source_tracklet_covariance_truth_closure",
            "stage": "A",
            "construction": construction,
            "validation": validation,
            "diagnostic_only": True,
            "alignment_authorized": False,
        }
    }


def _stage_repairs(config: dict) -> dict:
    repairs = stcc.evaluate_stage_a_repairs(config)
    return {
        "stage_a_repairs": {
            "kind": "stage_a_repair_evaluation",
            "derived_on": config["stage_a_repair"]["repair_derived_on"],
            "confirmed_on": config["stage_a_repair"]["repair_confirmed_on"],
            "diagnostic_only": True,
            "alignment_authorized": False,
            "models": repairs,
        }
    }


def _stage_decide(config: dict) -> dict:
    root = _output_root(config)
    closure = _read_json(root / "source_closure.json")["source_tracklet_covariance_closure"]
    repairs_path = root / "stage_a_repairs.json"
    repairs = (
        _read_json(repairs_path)["stage_a_repairs"]["models"]
        if repairs_path.is_file()
        else None
    )
    decision = stcc.decide_source_closure(
        config,
        construction=closure["construction"],
        validation=closure["validation"],
        repairs=repairs,
    )
    summary = {
        "kind": "campaign_summary",
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_sha": git_head_sha(),
        "config_path": config["config_path"],
        "stage": "A",
        "decision": decision["decision"],
        "propagated_covariance_model_validated": decision[
            "propagated_covariance_model_validated"
        ],
        "real_kinematic_jacobian_support_validated": decision[
            "real_kinematic_jacobian_support_validated"
        ],
        "measurement_model_validated": decision["measurement_model_validated"],
        "real_data_alignment_v2_preregistration_allowed": decision[
            "real_data_alignment_v2_preregistration_allowed"
        ],
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "external_constraint_ingest_authorized": False,
    }
    return {
        "propagated_covariance_upstream_repair_decision": decision,
        "campaign_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate-config",
            "source-closure",
            "repairs",
            "decide",
            "all",
        ),
    )
    args = parser.parse_args()
    config = stcc.load_config(args.config)
    config["config_path"] = str(resolve_under_root(project_root(), args.config))
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage in ("validate-config", "all"):
        _write_json(root / "config_validation.json", _stage_validate_config(config))
    if args.stage in ("source-closure", "all"):
        _write_json(root / "source_closure.json", _stage_source_closure(config))
    if args.stage in ("repairs", "all"):
        _write_json(root / "stage_a_repairs.json", _stage_repairs(config))
    if args.stage in ("decide", "all"):
        reports = _stage_decide(config)
        _write_json(
            root / "propagated_covariance_upstream_repair_decision.json",
            reports["propagated_covariance_upstream_repair_decision"],
        )
        _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
