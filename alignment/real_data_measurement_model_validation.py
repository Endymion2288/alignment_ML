"""Workbook-80 measurement-model reconstruction & cross-run validation V1.

Orchestration for the measurement-model rebuild campaign.  This campaign ONLY
rebuilds and validates (A) the residual covariance model and (B) the MC->real
Jacobian-transfer model.  It does NOT solve a final alignment correction, does
NOT open held-out data, does NOT write geometry, does NOT iterate nonlinearly,
does NOT change the gauge, does NOT redefine V_id/V_null, does NOT modify
S/rank_tolerance, and does NOT use any external prior.

Frozen terminal permissions (all remain): ``geometry_write_allowed = false``,
``official_conditions_write_allowed = false``,
``real_data_candidate_alignment_authorized = false``,
``external_constraint_ingest_authorized = false``, ``held_out_accessed = false``.

The decision tree freezes exactly one terminal string; the pass state only
authorizes a *separate* Workbook-81 pre-registration, never an alignment solve
inside Workbook 80.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import yaml

from alignment import gauge_fixed_real_data_diagnostic as wb78
from alignment import real_data_covariance_model_adequacy as wb79
from alignment import residual_covariance_model as cov_model
from alignment import conditional_jacobian_transfer as jac_model
from alignment.cad_survey_nov22 import git_head_sha
from alignment.operating_protocol_v1_final_closure import (
    project_root,
    resolve_under_root,
    sha256_file,
)

SCHEMA_VERSION = "faser-real-data-measurement-model-reconstruction-validation-v1"
DEFAULT_CONFIG_RELATIVE = Path(
    "configs/real_data_measurement_model_reconstruction_validation_v1.yaml"
)

OBSERVABLE_NAMES = wb78.OBSERVABLE_NAMES
STATION_PAIRS = wb78.STATION_PAIRS

DECISION_VALIDATED = "measurement_model_validated_real_data_alignment_v2_preregistration_allowed"
DECISION_INCONCLUSIVE = "measurement_model_validation_inconclusive"
DECISION_COV_NOT_VALIDATED = "measurement_model_covariance_not_validated"
DECISION_JAC_NOT_VALIDATED = "measurement_model_jacobian_transfer_not_validated"
DECISION_MULTIPLE_NOT_VALIDATED = "measurement_model_multiple_components_not_validated"
DECISION_CROSS_RUN_NOT_STABLE = "measurement_model_cross_run_information_not_stable"

CONFIG_MUST_BE_FALSE = (
    "geometry_write_allowed",
    "official_conditions_write_allowed",
    "real_data_candidate_alignment_authorized",
    "external_constraint_ingest_authorized",
    "held_out_accessed",
)
CONFIG_MUST_BE_TRUE = (
    "do_not_open_held_out",
    "do_not_select_events_by_residual",
    "do_not_drop_pairs_by_condition_or_fit_quality",
    "do_not_repick_split",
    "do_not_switch_gauge",
    "do_not_tune_S_or_rank_tolerance",
    "do_not_redefine_identifiable_basis_on_real_data",
    "do_not_use_external_evidence_as_prior",
    "do_not_generate_fd_probes",
    "do_not_run_newton",
    "do_not_write_payload",
    "do_not_solve_final_alignment",
    "do_not_promote_diagonal_or_capped_or_unit_covariance",
    "do_not_choose_covariance_by_gamma_or_chi2_drop",
    "do_not_use_residual_nearest_neighbour_jacobian",
    "do_not_extrapolate_jacobian_outside_support",
    "do_not_reopen_identifiability_rescue",
    "do_not_start_nonlinear_or_trust_region_response",
)

# Held-out run ids that must never be read (hard guard).
HELD_OUT_RUN_IDS = frozenset(
    [14975, 14976, 14977, 14971, 14972, 14980, 14981, 14985, 14989, 15007]
)
CALIBRATION_RUN_IDS = frozenset([14973, 14974])


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    config_path = resolve_under_root(project_root(), str(path or DEFAULT_CONFIG_RELATIVE))
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if config.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    for key in CONFIG_MUST_BE_FALSE:
        if config.get(key) is not False:
            raise ValueError(f"config must set {key}: false")
    for key in CONFIG_MUST_BE_TRUE:
        if config.get(key) is not True:
            raise ValueError(f"config must set {key}: true")
    if list(config["eligible_external_physical_constraints"]) != []:
        raise ValueError("external constraint branch remains closed")

    inheritance = config["inheritance"]
    wb79_config_path = resolve_under_root(project_root(), str(inheritance["workbook_79_config"]))
    if sha256_file(wb79_config_path) != str(inheritance["workbook_79_config_sha256"]):
        raise ValueError("workbook-79 config SHA256 mismatch")
    wb79_root = resolve_under_root(project_root(), str(inheritance["workbook_79_output_root"]))
    for name, expected in inheritance["workbook_79_artifact_sha256"].items():
        actual = sha256_file(wb79_root / name)
        if actual != str(expected):
            raise ValueError(f"workbook-79 artifact SHA256 mismatch for {name}")

    # Load the frozen WB79 config (which loads + SHA-verifies WB78) and reuse
    # its contracts verbatim.
    wb79_config = wb79.load_config(wb79_config_path)
    config["wb79_config"] = wb79_config
    config["wb78_config"] = wb79_config["wb78_config"]
    config["config_path"] = str(config_path)
    # Frozen identifiable subspace (V_id/V_null/S), loaded via the WB78 chain.
    # Injected for the Jacobian-transfer and cross-run stages; never re-derived
    # on real data.
    _banks, _pooled, subspace, _extras, regression = wb78.load_tracker_information(
        config["wb78_config"]
    )
    if not regression["pass"]:
        raise ValueError("tracker information regression failed; WB80 refused")
    config["subspace"] = subspace
    return config


def assert_no_held_out_access(run_ids: np.ndarray) -> None:
    """Code-level guard: refuse any held-out run id."""
    seen = {int(v) for v in np.atleast_1d(run_ids)}
    bad = seen & HELD_OUT_RUN_IDS
    if bad:
        raise ValueError(f"held-out run access refused: {sorted(bad)}")


# ---------------------------------------------------------------------------
# Stage 0: reproduce the frozen WB79 covariance eigensystem/whitening baseline
# ---------------------------------------------------------------------------


def reproduce_baseline(config: Mapping[str, Any]) -> dict[str, Any]:
    """Rebuild the WB78 calibration bank + WB79 covariance audit and verify the
    frozen baseline numbers exactly.  Returns the shared baseline objects so no
    downstream stage ever rebuilds the bank differently.
    """
    baseline = config["inheritance"]["wb79_baseline"]
    rtol = float(config["inheritance"]["reproduce_rtol"])
    atol = float(config["inheritance"]["reproduce_atol"])

    # Reproduce the WB78 calibration bank + solve via the WB79 chain.
    repro = wb79.reproduce_baseline(config["wb79_config"])
    if not repro["report"]["pass"]:
        raise ValueError("WB78 baseline reproduction failed; WB80 refused")
    bank = repro["bank"]
    bank_sha = wb78.bank_sha256(bank)
    bank_sha_ok = bank_sha == str(config["inheritance"]["calibration_bank_sha256"])

    # Reproduce the WB79 covariance eigensystem/whitening audit.
    audit = wb79.covariance_eigen_audit(config["wb79_config"], bank)

    def _close(a: float, b: float) -> bool:
        return bool(np.isclose(a, b, rtol=rtol, atol=atol))

    checks = {
        "bank_sha256": bank_sha_ok,
        "n_pairs": int(audit["n_pairs"]) == int(baseline["n_pairs"]),
        "chi2_total": _close(float(audit["chi2_total"]), float(baseline["chi2_total"])),
        "fraction_chi2_from_smallest_eigenmode": _close(
            float(audit["fraction_chi2_from_smallest_eigenmode"]),
            float(baseline["fraction_chi2_from_smallest_eigenmode"]),
        ),
        "covariance_condition_median": _close(
            float(audit["covariance_condition_median"]),
            float(baseline["covariance_condition_median"]),
        ),
        "whitened_chi2_resid_per_ndof": _close(
            float(audit["whitened_chi2_resid_per_ndof"]),
            float(baseline["whitened_chi2_resid_per_ndof"]),
        ),
    }
    # Held-out guard: the calibration bank must contain only calibration runs.
    assert_no_held_out_access(wb78.concatenate_banks(bank)["run_id_arr"])

    report = {
        "kind": "wb79_baseline_reproduction",
        "bank_sha256": bank_sha,
        "chi2_total": float(audit["chi2_total"]),
        "whitened_chi2_resid_per_ndof": float(audit["whitened_chi2_resid_per_ndof"]),
        "checks": checks,
        "pass": bool(all(checks.values())),
        "held_out_accessed": False,
    }
    return {
        "report": report,
        "bank": bank,
        "pooled": repro["pooled"],
        "subspace": repro["subspace"],
        "extras": repro["extras"],
        "transfer": repro["transfer"],
        "covariance_audit": audit,
    }


# ---------------------------------------------------------------------------
# Stage: cross-run information sanity check (diagnostic-only; only if cov+J pass)
# ---------------------------------------------------------------------------


def cross_run_information_check(
    config: Mapping[str, Any],
    bank: Mapping[str, Any],
    c_model: np.ndarray,
    jacobian_model: Mapping[str, Any],
    real_kinematics: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Diagnostic-only cross-run information consistency under the rebuilt
    covariance + Jacobian models.  This is NOT an alignment correction and
    never produces a deployable candidate (alignment_authorized=false).
    """
    spec = config["cross_run_information"]
    subspace = config["subspace"]
    wb78_config = config["wb78_config"]
    rank_tolerance = float(wb78_config["solver"]["rank_tolerance"])
    arrays = wb78.concatenate_banks(bank)
    residual = np.asarray(arrays["residual"], dtype=np.float64)
    target_station = np.asarray(arrays["target_station_id"], dtype=np.int64)
    run_id = np.asarray(arrays["run_id_arr"], dtype=np.int64)
    assert_no_held_out_access(run_id)

    s_mat = np.diag(np.asarray(subspace.parameter_scales, dtype=np.float64))
    v_id = np.asarray(subspace.v_id, dtype=np.float64)

    # Build the per-pair design rows from the REBUILT conditional Jacobian.
    design = np.zeros((residual.shape[0], 4, v_id.shape[1]))
    for pair in STATION_PAIRS:
        label = f"{pair[0]}->{pair[1]}"
        mask = target_station == pair[1]
        if not np.any(mask):
            continue
        jpred = jac_model.predict_jacobian(
            jacobian_model,
            label,
            real_kinematics["pred_tx"][mask],
            real_kinematics["pred_ty"][mask],
        )
        design[mask] = jpred @ s_mat @ v_id

    per_run = {}
    betas = {}
    bases = {}
    ranks = {}
    for run in sorted(set(int(v) for v in run_id)):
        rows = np.flatnonzero(run_id == run)
        information = np.zeros((v_id.shape[1], v_id.shape[1]))
        rhs = np.zeros(v_id.shape[1])
        for i in rows:
            # Stable weight via eigendecomposition (method A; same covariance).
            w, v = np.linalg.eigh(c_model[i])
            w = np.maximum(w, 1.0e-300)
            weight = (v * (1.0 / w)) @ v.T
            a = design[i]
            information += a.T @ weight @ a
            rhs += a.T @ weight @ residual[i]
        solved = wb78._solve_informed(information, rhs, rank_tolerance=rank_tolerance)
        ranks[run] = int(solved["rank"])
        betas[run] = np.asarray(solved["beta"], dtype=np.float64)
        bases[run] = np.asarray(solved["basis"], dtype=np.float64)
        per_run[str(run)] = {
            "n_pairs": int(rows.size),
            "informed_rank": int(solved["rank"]),
            "information_eigenvalues": [float(v) for v in solved["eigenvalues"]],
            "max_abs_gamma": float(np.max(np.abs(solved["gamma"]))) if solved["rank"] else 0.0,
        }

    runs = sorted(ranks)
    comparison: dict[str, Any] = {}
    ok = True
    if len(runs) == 2:
        r1, r2 = runs
        rank_equal = ranks[r1] == ranks[r2]
        dir_angle = None
        if ranks[r1] > 0 and ranks[r2] > 0:
            v1 = bases[r1][:, -1]
            v2 = bases[r2][:, -1]
            cosine = float(abs(np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))))
            dir_angle = math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))
        gamma_angle = None
        b1, b2 = betas[r1], betas[r2]
        if np.linalg.norm(b1) > 0 and np.linalg.norm(b2) > 0:
            cosine = float(abs(np.dot(b1, b2) / (np.linalg.norm(b1) * np.linalg.norm(b2))))
            gamma_angle = math.degrees(math.acos(min(max(cosine, -1.0), 1.0)))
        comparison = {
            "rank_equal": bool(rank_equal),
            "dominant_direction_angle_deg": dir_angle,
            "gamma_direction_angle_deg": gamma_angle,
        }
        if spec.get("require_equal_rank"):
            ok = ok and rank_equal
        if dir_angle is not None:
            ok = ok and dir_angle <= float(spec["direction_max_angle_deg"])
        if gamma_angle is not None:
            ok = ok and gamma_angle <= float(spec["gamma_direction_max_angle_deg"])

    return {
        "kind": "cross_run_information_consistency",
        "per_run": per_run,
        "comparison": comparison,
        "pass": bool(ok),
        "diagnostic_only": True,
        "alignment_authorized": False,
        "not_a_correction": True,
    }


# ---------------------------------------------------------------------------
# Decision tree
# ---------------------------------------------------------------------------


def decide_campaign(
    *,
    reproduction: Mapping[str, Any],
    covariance_validation: Mapping[str, Any] | None,
    jacobian_validation: Mapping[str, Any] | None,
    cross_run_info: Mapping[str, Any] | None,
) -> dict[str, Any]:
    """Pre-registered decision tree.  Exactly one terminal string is frozen."""
    if not bool(reproduction["pass"]):
        decision = DECISION_INCONCLUSIVE
        cov_ok = False
        jac_ok = False
        cross_ok = None
    else:
        cov_ok = bool(covariance_validation["covariance_model_validated"])
        jac_ok = bool(jacobian_validation["jacobian_transfer_model_validated"])
        cross_ok = None
        if not cov_ok and not jac_ok:
            decision = DECISION_MULTIPLE_NOT_VALIDATED
        elif not cov_ok:
            decision = DECISION_COV_NOT_VALIDATED
        elif not jac_ok:
            decision = DECISION_JAC_NOT_VALIDATED
        else:
            cross_ok = bool(cross_run_info["pass"]) if cross_run_info is not None else False
            decision = DECISION_VALIDATED if cross_ok else DECISION_CROSS_RUN_NOT_STABLE

    return {
        "kind": "measurement_model_decision",
        "decision": decision,
        "covariance_model_validated": bool(cov_ok),
        "jacobian_transfer_model_validated": bool(jac_ok),
        "cross_run_information_stable": cross_ok,
        "real_data_alignment_v2_preregistration_allowed": decision == DECISION_VALIDATED,
        "geometry_write_allowed": False,
        "official_conditions_write_allowed": False,
        "real_data_candidate_alignment_authorized": False,
        "external_constraint_ingest_authorized": False,
        "held_out_accessed": False,
        "eligible_external_physical_constraints": [],
        "if_fail_continue": "residual_dq_monitoring_only",
        "next_stage_if_validated": (
            "gauge_fixed_real_data_alignment_diagnostic_v2_requires_separate_preregistration"
        ),
        "official_cool_pool_write_remains_closed": True,
    }
