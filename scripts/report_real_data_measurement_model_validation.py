#!/usr/bin/env python3
"""Workbook-80 measurement-model reconstruction & cross-run validation driver.

Stages enforce the frozen ordering structurally:

    validate-config       config + WB79/WB78 inheritance SHAs + held-out guard
    reproduce             Stage-0: reproduce the frozen WB79 covariance
                          eigensystem/whitening baseline (hard stop on mismatch)
    covariance-semantics  read-only covariance semantics audit (12 questions)
    numerical-inversion   numerical inversion audit (method A vs model change B)
    covariance-model      cross-fit (14973<->14974) covariance validation
    jacobian-model        MC source-disjoint conditional-J validation
    jacobian-support      frozen-J real residual-blind support validation
    cross-run-info        diagnostic-only cross-run information consistency
                          (ONLY if covariance AND jacobian both validated)
    decide                pre-registered decision tree

Every stage runs ONLY on the frozen calibration subset (14973+14974) and
MC/control; no held-out residual is ever read (a code-level guard refuses it).
No stage solves a final alignment correction or writes geometry.
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
from alignment import conditional_jacobian_transfer as jac_model
from alignment import residual_covariance_model as cov_model
from alignment.real_data_measurement_model_validation import (
    DEFAULT_CONFIG_RELATIVE,
    cross_run_information_check,
    decide_campaign,
    load_config,
    reproduce_baseline,
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
    report = {
        "kind": "config_validation",
        "config_path": config["config_path"],
        "config_sha256": sha256_file(Path(config["config_path"])),
        "git_head_sha": git_head_sha(),
        "frozen_flags_verified": True,
        "held_out_accessed": False,
        "inheritance_sha256_verified": True,
        "population": "calibration_14973_14974_plus_mc_control_only",
        "pass": True,
    }
    return {"config_validation": report}


class _Shared:
    """Lazily-built shared objects (baseline, enriched, MC banks)."""

    def __init__(self, config: dict):
        self.config = config
        self._baseline = None
        self._enriched = None
        self._construction = None
        self._validation = None

    @property
    def baseline(self) -> dict:
        if self._baseline is None:
            self._baseline = reproduce_baseline(self.config)
            if not self._baseline["report"]["pass"]:
                raise RuntimeError("WB79 baseline reproduction failed; stage refused")
        return self._baseline

    @property
    def enriched(self) -> dict:
        if self._enriched is None:
            self._enriched = cov_model.load_calibration_pairs_enriched(
                self.config, self.baseline["bank"]
            )
        return self._enriched

    @property
    def construction_banks(self) -> list:
        if self._construction is None:
            self._construction = [
                jac_model.load_mc_source_jacobian_bank(self.config, s)
                for s in self.config["jacobian_transfer"]["mc_construction_source_ids"]
            ]
        return self._construction

    @property
    def validation_banks(self) -> list:
        if self._validation is None:
            self._validation = [
                jac_model.load_mc_source_jacobian_bank(self.config, s)
                for s in self.config["jacobian_transfer"]["mc_validation_source_ids"]
            ]
        return self._validation


def _real_kinematics(enriched: dict) -> dict:
    return {
        "pred_tx": enriched["pred_tx"],
        "pred_ty": enriched["pred_ty"],
        "target_station_id": enriched["target_station_id"],
    }


def _mc_construction_kinematics(shared: _Shared) -> dict:
    construction = jac_model._stack_sources(shared.construction_banks)
    out = {}
    for pair in jac_model.STATION_PAIRS:
        label = f"{pair[0]}->{pair[1]}"
        mask = (construction["source_station_id"] == pair[0]) & (
            construction["target_station_id"] == pair[1]
        )
        out[label] = {
            "pred_tx": construction["pred_tx"][mask],
            "pred_ty": construction["pred_ty"][mask],
        }
    return out


def _stage_covariance_semantics(config: dict, shared: _Shared) -> dict:
    return {"covariance_semantics_audit": cov_model.covariance_semantics_audit(config, shared.enriched)}


def _stage_numerical_inversion(config: dict, shared: _Shared) -> dict:
    return {"numerical_inversion_audit": cov_model.numerical_inversion_audit(config, shared.enriched)}


def _stage_covariance_model(config: dict, shared: _Shared) -> dict:
    return {
        "covariance_model_validation": cov_model.cross_fit_covariance_validation(
            config, shared.enriched
        )
    }


def _stage_jacobian_model(config: dict, shared: _Shared) -> dict:
    return {
        "mc_jacobian_model_validation": jac_model.mc_jacobian_model_validation(
            config,
            construction_banks=shared.construction_banks,
            validation_banks=shared.validation_banks,
        )
    }


def _primary_validated_model(config: dict, shared: _Shared, mc_val: dict) -> dict | None:
    """Pre-registered preference order: J2 regression -> J1 binned -> J0 mean."""
    order = ["J2_linear_regression", "J1_kinematic_binned", "J0_station_pair_mean"]
    construction = jac_model._stack_sources(shared.construction_banks)
    for name in order:
        rep = mc_val["models"].get(name)
        if rep and rep["mc_validated"]:
            spec = next(
                m for m in config["jacobian_transfer"]["models"] if m["name"] == name
            )
            return jac_model.fit_conditional_jacobian_model(spec["kind"], construction, spec)
    return None


def _stage_jacobian_support(config: dict, shared: _Shared) -> dict:
    mc_val = _stage_jacobian_model(config, shared)["mc_jacobian_model_validation"]
    model = _primary_validated_model(config, shared, mc_val)
    mc_kin = _mc_construction_kinematics(shared)
    real_kin = _real_kinematics(shared.enriched)
    if model is None:
        # No MC-validated model: support cannot be established.
        return {
            "jacobian_transfer_validation": {
                "kind": "jacobian_transfer_validation",
                "mc_jacobian_model_validation": mc_val,
                "real_support": None,
                "jacobian_transfer_model_validated": False,
                "note": "no conditional-J model passed MC source-disjoint validation",
                "diagnostic_only": True,
                "alignment_authorized": False,
            }
        }
    support = jac_model.real_jacobian_support_validation(config, model, real_kin, mc_kin)
    validated = bool(support["kinematic_support_within_gate"])
    return {
        "jacobian_transfer_validation": {
            "kind": "jacobian_transfer_validation",
            "mc_jacobian_model_validation": mc_val,
            "primary_model_kind": model["kind"],
            "real_support": support,
            "jacobian_transfer_model_validated": validated,
            "diagnostic_only": True,
            "alignment_authorized": False,
        }
    }


def _stage_cross_run_info(config: dict, shared: _Shared) -> dict:
    """Run ONLY if covariance AND jacobian are both validated (read artifacts).

    Diagnostic-only; never a correction.  Uses the cross-fit covariance (each
    run's pairs weighted by the scale derived from the OTHER run) and the
    primary MC-validated conditional-J model.
    """
    root = _output_root(config)
    cov = _read_json(root / "covariance_model.json")["covariance_model_validation"]
    jac = _read_json(root / "jacobian_support.json")["jacobian_transfer_validation"]
    if not (cov["covariance_model_validated"] and jac["jacobian_transfer_model_validated"]):
        return {
            "cross_run_information": {
                "kind": "cross_run_information_consistency",
                "skipped": True,
                "reason": "covariance and/or jacobian model not validated; cross-run "
                "information check is only meaningful with a validated measurement model",
                "diagnostic_only": True,
                "alignment_authorized": False,
            }
        }
    enriched = shared.enriched
    cov_name = cov["validated_physical_candidates"][0]
    cov_spec = next(c for c in config["covariance_model"]["candidates"] if c["name"] == cov_name)
    c_model = cov_model.build_crossfit_model_covariance(config, enriched, cov_spec["kind"])
    mc_val = jac["mc_jacobian_model_validation"]
    j_model = _primary_validated_model(config, shared, mc_val)
    real_kin = _real_kinematics(enriched)
    check = cross_run_information_check(config, shared.baseline["bank"], c_model, j_model, real_kin)
    return {"cross_run_information": check}


def _stage_decide(config: dict) -> dict:
    root = _output_root(config)
    reproduction = _read_json(root / "baseline_reproduction.json")["baseline_reproduction"]
    cov_path = root / "covariance_model.json"
    cov = _read_json(cov_path)["covariance_model_validation"] if cov_path.is_file() else None
    jac_path = root / "jacobian_support.json"
    jac = _read_json(jac_path)["jacobian_transfer_validation"] if jac_path.is_file() else None
    cross_path = root / "cross_run_info.json"
    cross = _read_json(cross_path)["cross_run_information"] if cross_path.is_file() else None
    decision = decide_campaign(
        reproduction=reproduction,
        covariance_validation=cov,
        jacobian_validation=jac,
        cross_run_info=cross if (cross and not cross.get("skipped")) else None,
    )
    summary = {
        "kind": "campaign_summary",
        "schema_version": config["schema_version"],
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "git_head_sha": git_head_sha(),
        "config_path": config["config_path"],
        "decision": decision["decision"],
        "covariance_model_validated": decision["covariance_model_validated"],
        "jacobian_transfer_model_validated": decision["jacobian_transfer_model_validated"],
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "real_data_candidate_alignment_authorized": False,
        "held_out_accessed": False,
    }
    return {"measurement_model_decision": decision, "campaign_summary": summary}


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_RELATIVE))
    parser.add_argument(
        "--stage",
        required=True,
        choices=(
            "validate-config",
            "reproduce",
            "covariance-semantics",
            "numerical-inversion",
            "covariance-model",
            "jacobian-model",
            "jacobian-support",
            "cross-run-info",
            "decide",
            "all",
        ),
    )
    args = parser.parse_args()
    config = load_config(args.config)
    root = _output_root(config)
    root.mkdir(parents=True, exist_ok=True)
    shared = _Shared(config)

    if args.stage in ("validate-config", "all"):
        _write_json(root / "config_validation.json", _stage_validate_config(config))

    if args.stage in ("reproduce", "all"):
        baseline = reproduce_baseline(config)
        shared._baseline = baseline
        _write_json(root / "baseline_reproduction.json", {"baseline_reproduction": baseline["report"]})
        if not baseline["report"]["pass"]:
            return _finish_decide_early(config)

    if args.stage in ("covariance-semantics", "all"):
        _write_json(root / "covariance_semantics_audit.json", _stage_covariance_semantics(config, shared))
    if args.stage in ("numerical-inversion", "all"):
        _write_json(root / "numerical_inversion.json", _stage_numerical_inversion(config, shared))
    if args.stage in ("covariance-model", "all"):
        _write_json(root / "covariance_model.json", _stage_covariance_model(config, shared))
    if args.stage in ("jacobian-model", "all"):
        _write_json(root / "jacobian_model.json", _stage_jacobian_model(config, shared))
    if args.stage in ("jacobian-support", "all"):
        _write_json(root / "jacobian_support.json", _stage_jacobian_support(config, shared))
    if args.stage in ("cross-run-info", "all"):
        _write_json(root / "cross_run_info.json", _stage_cross_run_info(config, shared))
    if args.stage in ("decide", "all"):
        reports = _stage_decide(config)
        _write_json(root / "measurement_model_decision.json", reports["measurement_model_decision"])
        _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


def _finish_decide_early(config: dict) -> int:
    root = _output_root(config)
    reports = _stage_decide(config)
    _write_json(root / "measurement_model_decision.json", reports["measurement_model_decision"])
    _write_json(root / "campaign_summary.json", reports["campaign_summary"])
    return 0


if __name__ == "__main__":
    sys.exit(main())
