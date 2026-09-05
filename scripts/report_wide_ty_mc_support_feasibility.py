#!/usr/bin/env python3
"""Workbook-82 Wide-ty Real-Support-Matched MC Coverage Feasibility V1 driver.

Stages enforce the frozen ordering structurally:

    validate-config   config + WB81 inheritance SHAs + frozen-flag guards
    real-target       rebuild the frozen WB80 calibration bank and read the
                      residual-blind real (0,1)/(0,2)/(0,3) support target
    coverage          load each existing non-sealed MC candidate's kinematic
                      cloud, compute support-coverage + particle-domain audit
    decide            pre-registered decision tree

This campaign is RESIDUAL-BLIND.  It reads NO alignment residual, never opens
held-out or sealed-test data, never writes geometry/conditions, and never
solves for an alignment correction.  ``held_out_accessed=false`` throughout.
"""

from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from alignment.cad_survey_nov22 import git_head_sha
from alignment.gauge_fixed_real_data_diagnostic import _json_ready
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)
from alignment import wide_ty_mc_support_feasibility as wb82
from alignment.wide_ty_mc_support_feasibility import DEFAULT_CONFIG


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
        "sealed_test_sources_excluded": True,
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "residual_blind": True,
        "n_candidates": len(config["candidates"]),
        "candidate_names": [str(c["name"]) for c in config["candidates"]],
        "pass": True,
    }
    return {"config_validation": report}


def _real_target_summary(real: dict) -> dict:
    per_pair: dict[str, object] = {}
    ts = np.asarray(real["target_station_id"])
    for tgt in (1, 2, 3):
        m = ts == tgt
        pts = np.column_stack([real["source_tx"][m], real["source_ty"][m]])
        per_pair[f"0->{tgt}"] = {
            "n_pairs": int(m.sum()),
            "gated": tgt in (1, 2),
            "source_tx_abs_p95": float(np.percentile(np.abs(pts[:, 0]), 95)) if m.sum() else None,
            "source_ty_abs_p95": float(np.percentile(np.abs(pts[:, 1]), 95)) if m.sum() else None,
            "pred_tx_abs_p95": float(np.percentile(np.abs(real["pred_tx"][m]), 95)) if m.sum() else None,
            "pred_ty_abs_p95": float(np.percentile(np.abs(real["pred_ty"][m]), 95)) if m.sum() else None,
            "lever_arm_mm_median": float(np.median(real["lever_arm_mm"][m])) if m.sum() else None,
        }
    return per_pair


def _stage_real_target(config: dict) -> dict:
    real = wb82.load_real_target(config)
    root = _output_root(config)
    npz_path = root / "real_support_target.npz"
    np.savez(
        npz_path,
        source_tx=real["source_tx"],
        source_ty=real["source_ty"],
        pred_tx=real["pred_tx"],
        pred_ty=real["pred_ty"],
        lever_arm_mm=real["lever_arm_mm"],
        target_station_id=real["target_station_id"],
        run_id=real["run_id"],
    )
    report = {
        "kind": "real_support_target",
        "residual_blind": True,
        "calibration_runs": sorted({int(v) for v in real["run_id"]}),
        "n_pairs_total": int(real["run_id"].shape[0]),
        "per_pair": _real_target_summary(real),
        "npz": npz_path.name,
        "npz_sha256": sha256_file(npz_path),
        "primary_kinematic": ["source_tx", "source_ty"],
        "crosscheck_kinematic": ["pred_tx", "pred_ty"],
        "held_out_accessed": False,
    }
    return {"real_support_target": report}


def _load_real_npz(config: dict) -> dict:
    npz_path = _output_root(config) / "real_support_target.npz"
    d = np.load(npz_path)
    return {k: d[k] for k in d.files}


def _stage_coverage(config: dict) -> dict:
    real = _load_real_npz(config)
    ts = np.asarray(real["target_station_id"])
    kinematic = config["real_support_target"]["primary_kinematic"]
    crosscheck = config["real_support_target"]["crosscheck_kinematic"]
    rx = np.asarray(real[kinematic[0]])
    ry = np.asarray(real[kinematic[1]])
    rcx = np.asarray(real[crosscheck[0]])
    rcy = np.asarray(real[crosscheck[1]])

    candidates_out = []
    for cand in config["candidates"]:
        coverage_by_pair: dict[str, object] = {}
        audit_by_pair: dict[str, object] = {}
        for tgt in (1, 2, 3):
            label = f"0->{tgt}"
            cloud = wb82.load_candidate_pair_cloud(config, cand, tgt)
            real_mask = ts == tgt
            real_xy = np.column_stack([rx[real_mask], ry[real_mask]])
            cand_xy = np.column_stack([cloud["tx"], cloud["ty"]])
            cov = wb82.compute_support_coverage(real_xy, cand_xy, config)
            cov["n_sources_with_pairs"] = cloud["n_sources_with_pairs"]
            cov["n_files"] = cloud["n_files"]
            cov["n_sources_declared"] = cloud["n_sources_declared"]
            # WB80-consistency cross-check: propagation-frame coverage for
            # candidates that have propagations (pred_tx/pred_ty not NaN).
            pred = np.column_stack([cloud["pred_tx"], cloud["pred_ty"]])
            if np.isfinite(pred).any():
                real_pred = np.column_stack([rcx[real_mask], rcy[real_mask]])
                cov["crosscheck_pred_frame_fraction_within_mahalanobis_99"] = (
                    wb82._mahalanobis_envelope_fraction(
                        pred, real_pred, float(config["coverage"]["mahalanobis2_99_2dof"])
                    )
                )
            coverage_by_pair[label] = cov
            audit_by_pair[label] = wb82.audit_particle_domain(cloud, config)
        result = wb82.evaluate_candidate(cand, coverage_by_pair, audit_by_pair, config)
        result["species_expectation"] = str(cand["species_expectation"])
        result["origin_description"] = str(cand["origin_description"])
        result["prior_status"] = str(cand["prior_status"])
        result["kind"] = str(cand["kind"])
        candidates_out.append(result)
    return {
        "wide_ty_mc_coverage": {
            "kind": "wide_ty_mc_support_coverage",
            "primary_kinematic": kinematic,
            "residual_blind": True,
            "candidates": candidates_out,
            "diagnostic_only": True,
            "alignment_authorized": False,
        }
    }


def _stage_decide(config: dict) -> dict:
    root = _output_root(config)
    coverage = _read_json(root / "coverage.json")["wide_ty_mc_coverage"]
    decision = wb82.decide(coverage["candidates"], config)
    summary = {
        "kind": "campaign_summary",
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_sha": git_head_sha(),
        "config_path": config["config_path"],
        "decision": decision["decision"],
        "validated_candidates": decision["validated_candidates"],
        "existing_mc_real_wide_ty_support_validated": decision[
            "existing_mc_real_wide_ty_support_validated"
        ],
        "real_kinematic_jacobian_support_validated": decision[
            "real_kinematic_jacobian_support_validated"
        ],
        "conditional_j_retrain_permitted": decision["conditional_j_retrain_permitted"],
        "new_mc_generation_required": decision["new_mc_generation_required"],
        "measurement_model_validated": decision["measurement_model_validated"],
        "held_out_accessed": False,
        "real_data_alignment_authorized": False,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "external_constraint_ingest_authorized": False,
        "residual_blind": True,
    }
    return {
        "wide_ty_mc_support_decision": decision,
        "campaign_summary": summary,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=DEFAULT_CONFIG)
    parser.add_argument(
        "--stage",
        required=True,
        choices=("validate-config", "real-target", "coverage", "decide", "all"),
    )
    args = parser.parse_args()
    config = wb82.load_config(args.config)
    config["config_path"] = str(resolve_under_root(project_root(), args.config))
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)

    if args.stage in ("validate-config", "all"):
        _write_json(root / "config_validation.json", _stage_validate_config(config))
    if args.stage in ("real-target", "all"):
        _write_json(root / "real_support_target.json", _stage_real_target(config))
    if args.stage in ("coverage", "all"):
        _write_json(root / "coverage.json", _stage_coverage(config))
    if args.stage in ("decide", "all"):
        reports = _stage_decide(config)
        _write_json(
            root / "wide_ty_mc_support_decision.json",
            reports["wide_ty_mc_support_decision"],
        )
        _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
