#!/usr/bin/env python3
"""Workbook-78 gauge-fixed real-data alignment diagnostic V1 driver.

Stages enforce the frozen ordering structurally:

    validate-config   config + inheritance SHAs + tracker-information regression
    freeze-split      residual-blind split freeze (route metadata only)
    mc-control        MC-only transfer/injection/null-ensemble control
    calibrate         calibration-subset applicability audit + one-shot solve
                      + bootstrap; writes the frozen candidate artifact
    evaluate          held-out gauge-invariant evaluation (requires the frozen
                      candidate artifact and verifies its SHA before reading
                      any held-out residual)
    decide            pre-registered decision tree

``calibrate`` refuses to run unless mc-control passed.  ``evaluate`` refuses
to run unless the candidate artifact exists, is marked frozen, and its SHA
matches the manifest.  ``all`` runs the stages in the frozen order and stops
at the first stage whose pre-registered gates close the campaign.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from alignment.cad_survey_nov22 import git_head_sha
from alignment.gauge_fixed_real_data_diagnostic import (
    DECISION_OUT_OF_SUPPORT,
    DECISION_RANK_ZERO,
    DEFAULT_CONFIG_RELATIVE,
    applicability_audit,
    build_gauge_candidates,
    build_real_data_bank,
    build_transfer_model,
    decide_campaign,
    evaluate_held_out,
    freeze_candidate_artifact,
    freeze_split,
    load_config,
    load_frozen_csv_edges,
    load_tracker_information,
    mc_control,
    solve_candidate,
    _canonical_sha256,
    _json_ready,
)
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)


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
    _banks, _pooled, _subspace, _extras, regression = load_tracker_information(config)
    report = {
        "kind": "config_validation",
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "git_head_sha": git_head_sha(),
        "frozen_flags_verified": True,
        "primary_gauge": config["gauges"]["primary"],
        "inheritance_sha256_verified": True,
        "pass": True,
    }
    return {"config_validation": report, "tracker_information_regression": regression}


def _stage_freeze_split(config: dict) -> dict:
    return {"split_freeze": freeze_split(config)}


def _stage_mc_control(config: dict) -> dict:
    _banks, pooled, subspace, extras, regression = load_tracker_information(config)
    if not regression["pass"]:
        raise RuntimeError("tracker information regression failed; mc-control refused")
    transfer = build_transfer_model(pooled, extras)
    gauges = build_gauge_candidates(config, subspace)
    report = mc_control(config, pooled, subspace, extras, transfer, gauges)
    return {"mc_control": report, "transfer_model_diagnostics": transfer["diagnostics"]}


def _stage_calibrate(config: dict) -> dict:
    root = _output_root(config)
    control = _read_json(root / "mc_control.json")
    if not control["mc_control"]["pass"]:
        raise RuntimeError("mc-control did not pass; calibration solve refused")
    split = _read_json(root / "split_freeze.json")["split_freeze"]
    _banks, pooled, subspace, extras, regression = load_tracker_information(config)
    if not regression["pass"]:
        raise RuntimeError("tracker information regression failed; calibration refused")
    transfer = build_transfer_model(pooled, extras)
    gauges = build_gauge_candidates(config, subspace)
    bank = build_real_data_bank(config, roles=config["split"]["calibration_roles"])
    if not bank["csv_reproduction"]["bit_exact"]:
        audit = {
            "kind": "pre_fit_applicability_audit",
            "hard_fail": True,
            "reason": "bank/CSV reproduction mismatch",
            "csv_reproduction": bank["csv_reproduction"],
        }
        candidate = None
    else:
        audit = applicability_audit(config, bank, transfer, pooled)
        candidate = None if audit["hard_fail"] else solve_candidate(
            config, subspace, transfer, bank, gauges
        )
    artifact = None
    if candidate is not None:
        artifact = freeze_candidate_artifact(
            config,
            split=split,
            bank=bank,
            audit=audit,
            candidate=candidate,
            subspace=subspace,
            gauges=gauges,
            mc_control_report=control["mc_control"],
        )
        artifact["candidate_artifact_sha256"] = _canonical_sha256(artifact)
    return {
        "applicability_audit": audit,
        "candidate_artifact": artifact,
    }


def _stage_evaluate(config: dict) -> dict:
    root = _output_root(config)
    candidate_path = root / "candidate_artifact.json"
    if not candidate_path.is_file():
        raise RuntimeError("held-out evaluation refused: no frozen candidate artifact")
    artifact = _read_json(candidate_path)
    stored = artifact.pop("candidate_artifact_sha256", None)
    if stored != _canonical_sha256(artifact):
        raise RuntimeError("held-out evaluation refused: candidate artifact SHA mismatch")
    if not artifact.get("frozen_before_held_out_access"):
        raise RuntimeError("held-out evaluation refused: candidate not marked frozen")
    candidate = artifact["candidate"]
    if candidate.get("status") != "solved":
        raise RuntimeError(f"held-out evaluation refused: candidate status {candidate.get('status')}")
    if not candidate["within_linear_envelope"]:
        raise RuntimeError("held-out evaluation refused: candidate outside linear envelope")
    _banks, _pooled, subspace, extras, regression = load_tracker_information(config)
    if not regression["pass"]:
        raise RuntimeError("tracker information regression failed; evaluation refused")
    transfer = build_transfer_model(_pooled, extras)
    held_out_roles = list(config["split"]["held_out_roles"]) + list(
        config["split"]["report_only_roles"]
    )
    bank = build_real_data_bank(config, roles=held_out_roles)
    csv_edges = load_frozen_csv_edges(config, roles=held_out_roles)
    report = evaluate_held_out(config, subspace, transfer, artifact, bank, csv_edges)
    return {"heldout_evaluation": report}


def _stage_decide(config: dict) -> dict:
    root = _output_root(config)
    regression = _read_json(root / "tracker_information_regression.json")[
        "tracker_information_regression"
    ]
    control_path = root / "mc_control.json"
    control = _read_json(control_path)["mc_control"] if control_path.is_file() else None
    audit_path = root / "applicability_audit.json"
    audit = _read_json(audit_path)["applicability_audit"] if audit_path.is_file() else None
    candidate = None
    candidate_path = root / "candidate_artifact.json"
    if candidate_path.is_file():
        candidate = _read_json(candidate_path)["candidate"]
    held_out = None
    held_out_path = root / "heldout_evaluation.json"
    if held_out_path.is_file():
        held_out = _read_json(held_out_path)["heldout_evaluation"]
    decision = decide_campaign(
        regression=regression,
        mc_control_report=control,
        audit=audit,
        candidate=candidate,
        held_out=held_out,
    )
    summary = {
        "kind": "campaign_summary",
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_sha": git_head_sha(),
        "config_path": config["config_path"],
        "decision": decision["decision"],
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
    }
    return {"next_stage_decision": decision, "campaign_summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_RELATIVE))
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate-config",
            "freeze-split",
            "mc-control",
            "calibrate",
            "evaluate",
            "decide",
            "all",
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage in ("validate-config", "all"):
        reports = _stage_validate_config(config)
        for name, payload in reports.items():
            _write_json(root / f"{name}.json", payload)
        if not reports["tracker_information_regression"]["pass"]:
            return _finish_decide_early(config)
    if args.stage in ("freeze-split", "all"):
        _write_json(root / "split_freeze.json", _stage_freeze_split(config))
    if args.stage in ("mc-control", "all"):
        reports = _stage_mc_control(config)
        _write_json(root / "mc_control.json", reports)
        if args.stage == "all" and not reports["mc_control"]["pass"]:
            return _finish_decide_early(config)
    if args.stage in ("calibrate", "all"):
        reports = _stage_calibrate(config)
        _write_json(root / "applicability_audit.json", {"applicability_audit": reports["applicability_audit"]})
        if reports["candidate_artifact"] is not None:
            _write_json(root / "candidate_artifact.json", reports["candidate_artifact"])
        if args.stage == "all":
            candidate = reports["candidate_artifact"]
            if candidate is None or candidate["candidate"].get("status") in (
                DECISION_RANK_ZERO,
            ) or not candidate["candidate"].get("within_linear_envelope", False):
                return _finish_decide_early(config)
    if args.stage in ("evaluate", "all"):
        _write_json(root / "heldout_evaluation.json", _stage_evaluate(config))
    if args.stage in ("decide", "all"):
        reports = _stage_decide(config)
        _write_json(root / "next_stage_decision.json", reports["next_stage_decision"])
        _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


def _finish_decide_early(config: dict) -> int:
    """Close the campaign with the pre-registered failure decision."""
    root = _output_root(config)
    reports = _stage_decide(config)
    _write_json(root / "next_stage_decision.json", reports["next_stage_decision"])
    _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
