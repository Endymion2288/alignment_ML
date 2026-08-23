"""Reduced real-data Station calibration mode feasibility.

Analysis only.  ``dz`` is removed from the track-driven solve and fixed
at survey 0.  Pre-declared reduced modes and the common identifiable
subspace are scored from the existing 14973/14974 finite-difference
``J_s`` and the frozen MC ``A``.  Residual reduction is never success.

A mode that floats ``dx`` or ``ry`` and still produces more than the
1.5–1.7 µm equivalent ``C_dx`` contamination is rejected immediately.
More events or a looser threshold cannot rescue that isolation failure.

This module never writes official conditions and never starts C_dx Mode.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.calibration_modes import OPERATING_BAND_UM, implied_unmodeled_cdx_um
from alignment.five_dof_sampling import DEFAULT_SCALES, FREE_PARAMETERS, SURVEY_PARAMETER
from alignment.real_data_operating_protocol import ROLE_CALIBRATION, ROLE_HELD_OUT_DQ, ROLE_HOLDOUT
from alignment.real_data_station_mode_failure_audit import (
    _inverse_covariances,
    _spectrum,
)
from alignment.real_data_station_mode_fullscale import run_to_run_stability

SCHEMA_VERSION = "faser-operating-protocol-v1-real-data-reduced-station-mode-feasibility"
DEFAULT_CONFIG_RELATIVE = (
    "configs/operating_protocol_v1_real_data_reduced_station_mode_feasibility_v1.yaml"
)

SURVEY_DZ = SURVEY_PARAMETER
TRACK_DRIVEN_PARAMETERS = FREE_PARAMETERS
LEAKAGE_PARAMETERS = ("ift_dx_mm", "ift_ry_mrad")

DECISION_V2_CANDIDATE = "real_data_station_calibration_mode_v2_candidate"
DECISION_MONITORING_ONLY = "real_data_residual_dq_monitoring_only"
DECISIONS = (DECISION_V2_CANDIDATE, DECISION_MONITORING_ONLY)

DEFAULT_TARGET_CONDITION = 1.0e4
DEFAULT_MAX_A_PROJECTION = 0.15
DEFAULT_MAX_NSIGMA = 5.0
DEFAULT_MAX_TRANSFER_RATIO = 1.10
DEFAULT_MAX_COND_SPREAD = 0.50


def load_reduced_mode_config(path: str | None = None) -> dict[str, Any]:
    from pathlib import Path

    source = Path(path).expanduser().resolve() if path is not None else (
        Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"reduced-mode config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected reduced-mode config schema: {source}")
    required_false = (
        "geometry_write_allowed",
        "official_conditions_db_write",
        "joint_station_cdx_newton",
        "new_layer_or_module_dof",
        "schur_projection_production",
        "track_driven_dz",
    )
    required_true = (
        "frozen",
        "do_not_enter_cdx_mode",
        "cdx_mode_blocked",
        "do_not_retrain_v2",
        "do_not_emit_cdx_payload",
        "residual_reduction_is_not_alignment_success",
        "do_not_open_sealed_test",
        "analysis_only",
    )
    for key in required_false:
        if payload.get(key) is not False:
            raise ValueError(f"reduced-mode config must set {key}=false")
    for key in required_true:
        if payload.get(key) is not True:
            raise ValueError(f"reduced-mode config must set {key}=true")
    if float(payload.get("survey_dz_mm", 1.0)) != 0.0:
        raise ValueError("reduced-mode config must fix survey dz at 0")
    return {"path": str(source), "schema_version": SCHEMA_VERSION, **dict(payload)}


def campaign_allows_cdx_mode(decision: str) -> bool:
    return False


def campaign_allows_geometry_write(decision: str) -> bool:
    return False


def assert_track_driven_names(names: Sequence[str]) -> tuple[str, ...]:
    resolved = tuple(str(name) for name in names)
    if not resolved:
        raise ValueError("reduced mode must float at least one track-driven parameter")
    if len(set(resolved)) != len(resolved):
        raise ValueError("reduced mode parameter names must be unique")
    if SURVEY_DZ in resolved:
        raise ValueError("dz is survey-fixed at 0 and must not enter the track-driven solve")
    unknown = [name for name in resolved if name not in TRACK_DRIVEN_PARAMETERS]
    if unknown:
        raise ValueError("unknown track-driven parameter(s): " + ", ".join(unknown))
    return resolved


def predeclared_modes(config: Mapping[str, Any] | None = None) -> dict[str, tuple[str, ...]]:
    raw = None if config is None else config.get("predeclared_modes")
    if not isinstance(raw, Mapping) or not raw:
        return {
            "three_dof_dy_rx_rz": ("ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"),
            "four_dof_dx_fixed": ("ift_dy_mm", "ift_rx_mrad", "ift_ry_mrad", "ift_rz_mrad"),
            "four_dof_ry_fixed": ("ift_dx_mm", "ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"),
            "five_dof_no_dz": TRACK_DRIVEN_PARAMETERS,
        }
    return {str(name): assert_track_driven_names(list(values)) for name, values in raw.items()}


def isolation_axes() -> tuple[str, ...]:
    """Axes with small frozen-A components.  Not assumed to be a writable mode."""
    return ("ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad")


def a_subspace_projection(
    floated: Sequence[str],
    a_native_per_mm: Mapping[str, float],
    *,
    reference: Sequence[str] = TRACK_DRIVEN_PARAMETERS,
) -> float:
    """Fraction of the 5-DoF frozen A that lies in the floated axes."""
    ref = np.asarray([float(a_native_per_mm[name]) for name in reference], dtype=np.float64)
    subset = np.asarray([float(a_native_per_mm[name]) for name in floated], dtype=np.float64)
    denom = float(np.linalg.norm(ref))
    if denom <= 0.0:
        return 0.0
    return float(np.linalg.norm(subset) / denom)


def implied_cdx_from_dx_ry(
    delta: Mapping[str, float],
    a_native_per_mm: Mapping[str, float],
) -> dict[str, float | None]:
    implied_dx = None
    implied_ry = None
    if "ift_dx_mm" in delta and a_native_per_mm.get("ift_dx_mm"):
        implied_dx = float(
            implied_unmodeled_cdx_um(
                station_dx_mm=float(delta["ift_dx_mm"]),
                A_dx=float(a_native_per_mm["ift_dx_mm"]),
            )
        )
    if "ift_ry_mrad" in delta:
        a_ry = float(a_native_per_mm["ift_ry_mrad"])
        if a_ry != 0.0:
            implied_ry = 1.0e3 * float(delta["ift_ry_mrad"]) / a_ry
    abs_values = [abs(value) for value in (implied_dx, implied_ry) if value is not None]
    return {
        "implied_C_dx_um_from_dx": implied_dx,
        "implied_C_dx_um_from_ry": implied_ry,
        "max_abs_implied_C_dx_um": None if not abs_values else float(max(abs_values)),
    }


def hard_reject_dx_ry_contamination(
    floated: Sequence[str],
    implied: Mapping[str, float | None],
    *,
    operating_hi_um: float = OPERATING_BAND_UM[1],
) -> dict[str, Any]:
    """Isolation reject for modes that still float dx or ry."""
    contains = [name for name in floated if name in LEAKAGE_PARAMETERS]
    max_abs = implied.get("max_abs_implied_C_dx_um")
    exceeds = bool(contains) and max_abs is not None and float(max_abs) > float(operating_hi_um)
    return {
        "contains_dx_or_ry": bool(contains),
        "floated_leakage_parameters": contains,
        "exceeds_operating_band": exceeds,
        "operating_band_um": list(OPERATING_BAND_UM),
        "hard_reject": exceeds,
        "cannot_rescue_by_more_events_or_looser_thresholds": True,
        "rejection_diagnostic_only": True,
        "do_not_emit_cdx_payload": True,
        "not_a_C_dx_measurement": True,
    }


def solve_reduced_self_nulling(
    derivative_native: np.ndarray,
    covariance: np.ndarray,
    residual: np.ndarray,
    *,
    all_parameter_names: Sequence[str],
    floated: Sequence[str],
    parameter_scales: Mapping[str, float] | None = None,
    rcond: float = 1.0e-10,
) -> dict[str, Any]:
    """Self-nulling WLS in a reduced track-driven subspace.  ``dz`` stays 0."""
    names = assert_track_driven_names(floated)
    full = tuple(str(name) for name in all_parameter_names)
    missing = [name for name in names if name not in full]
    if missing:
        raise ValueError("J_s is missing " + ", ".join(missing))
    jacobian = np.asarray(derivative_native, dtype=np.float64)
    resid = np.asarray(residual, dtype=np.float64)
    if resid.shape != (jacobian.shape[0], 4):
        raise ValueError("residual must have shape (n_obs, 4)")
    inverse = _inverse_covariances(covariance)
    index = [full.index(name) for name in names]
    reduced = jacobian[:, :, index]
    native = np.zeros((len(names), len(names)), dtype=np.float64)
    rhs = np.zeros(len(names), dtype=np.float64)
    chi2_before = 0.0
    for row, weight in enumerate(inverse):
        native += reduced[row].T @ weight @ reduced[row]
        rhs += reduced[row].T @ weight @ (-resid[row])
        chi2_before += float(resid[row] @ weight @ resid[row])
    native = 0.5 * (native + native.T)
    scales = np.asarray(
        [float((DEFAULT_SCALES if parameter_scales is None else parameter_scales)[name]) for name in names],
        dtype=np.float64,
    )
    scaled = 0.5 * ((np.diag(scales) @ native @ np.diag(scales)) + (np.diag(scales) @ native @ np.diag(scales)).T)
    spectrum = _spectrum(scaled, rcond=rcond)
    try:
        covariance_native = np.linalg.inv(native)
    except np.linalg.LinAlgError:
        covariance_native = np.linalg.pinv(native)
    covariance_native = 0.5 * (covariance_native + covariance_native.T)
    delta = covariance_native @ rhs
    sigma = np.sqrt(np.clip(np.diag(covariance_native), 0.0, None))
    chi2_after = 0.0
    for row, weight in enumerate(inverse):
        updated = resid[row] + reduced[row] @ delta
        chi2_after += float(updated @ weight @ updated)
    return {
        "parameter_names": list(names),
        "survey_dz_mm": 0.0,
        "track_driven_dz": False,
        "n_observations": int(jacobian.shape[0]),
        "rank": spectrum["rank"],
        "full_rank": spectrum["rank"] == len(names),
        "condition_number": spectrum["condition_number"],
        "raw_condition_number": spectrum["raw_condition_number"],
        "singular_values": list(spectrum["singular_values"]),
        "delta": {name: float(delta[index]) for index, name in enumerate(names)},
        "sigma": {name: float(sigma[index]) for index, name in enumerate(names)},
        "chi2_before": float(chi2_before),
        "chi2_after": float(chi2_after),
        "chi2_ratio": None if chi2_before <= 0.0 else float(chi2_after / chi2_before),
        "residual_reduction_is_not_alignment_success": True,
        "normal_matrix_native": native,
        "normal_matrix_scaled": scaled,
    }


def attach_leakage(
    solve: Mapping[str, Any],
    a_native_per_mm: Mapping[str, float],
    *,
    max_a_projection: float = DEFAULT_MAX_A_PROJECTION,
) -> dict[str, Any]:
    floated = tuple(solve["parameter_names"])
    implied = implied_cdx_from_dx_ry(solve["delta"], a_native_per_mm)
    isolation = hard_reject_dx_ry_contamination(floated, implied)
    projection = a_subspace_projection(floated, a_native_per_mm)
    return {
        **implied,
        **isolation,
        "a_subspace_projection": float(projection),
        "clearly_orthogonal_to_A": bool(projection <= float(max_a_projection)),
        "new_cdx_payload": None,
    }


def run_to_run_parameter_consistency(
    left: Mapping[str, float],
    right: Mapping[str, float],
    left_sigma: Mapping[str, float],
    right_sigma: Mapping[str, float],
    names: Sequence[str],
    *,
    max_nsigma: float = DEFAULT_MAX_NSIGMA,
) -> dict[str, Any]:
    nsigma: dict[str, float] = {}
    signs: dict[str, bool] = {}
    for name in names:
        spread = abs(float(left[name]) - float(right[name]))
        denom = math.hypot(float(left_sigma[name]), float(right_sigma[name]))
        nsigma[name] = float("inf") if denom <= 0.0 else float(spread / denom)
        signs[name] = bool(float(left[name]) * float(right[name]) >= 0.0)
    max_value = max(nsigma.values()) if nsigma else None
    return {
        "nsigma": nsigma,
        "sign_agreement": signs,
        "max_nsigma": max_value,
        "consistent": bool(max_value is not None and max_value <= float(max_nsigma)),
        "max_nsigma_threshold": float(max_nsigma),
        "do_not_treat_as_alignment_correctness": True,
    }


def linearized_transfer_chi2_ratio(
    derivative_native: np.ndarray,
    covariance: np.ndarray,
    residual: np.ndarray,
    *,
    all_parameter_names: Sequence[str],
    floated: Sequence[str],
    delta: Mapping[str, float],
) -> dict[str, Any]:
    """Apply another run's reduced correction to this run's ``J_s``.  Read-only."""
    names = assert_track_driven_names(floated)
    full = tuple(str(name) for name in all_parameter_names)
    jacobian = np.asarray(derivative_native, dtype=np.float64)
    resid = np.asarray(residual, dtype=np.float64)
    inverse = _inverse_covariances(covariance)
    index = [full.index(name) for name in names]
    reduced = jacobian[:, :, index]
    step = np.asarray([float(delta[name]) for name in names], dtype=np.float64)
    chi2_before = 0.0
    chi2_after = 0.0
    for row, weight in enumerate(inverse):
        chi2_before += float(resid[row] @ weight @ resid[row])
        updated = resid[row] + reduced[row] @ step
        chi2_after += float(updated @ weight @ updated)
    ratio = None if chi2_before <= 0.0 else float(chi2_after / chi2_before)
    return {
        "chi2_before": float(chi2_before),
        "chi2_after": float(chi2_after),
        "chi2_ratio": ratio,
        "residual_reduction_is_not_alignment_success": True,
        "geometry_write_allowed": False,
    }


def condition_stable(
    left: float | None,
    right: float | None,
    *,
    target: float = DEFAULT_TARGET_CONDITION,
    max_rel_spread: float = DEFAULT_MAX_COND_SPREAD,
) -> dict[str, Any]:
    values = [float(value) for value in (left, right) if value is not None and math.isfinite(float(value))]
    if len(values) != 2:
        return {"stable": False, "both_below_target": False, "rel_spread": None, "target": float(target)}
    rel = abs(values[0] - values[1]) / max(values)
    below = max(values) <= float(target)
    return {
        "stable": bool(below and rel <= float(max_rel_spread)),
        "both_below_target": below,
        "rel_spread": float(rel),
        "target": float(target),
        "max_rel_spread": float(max_rel_spread),
    }


def evaluate_reduced_mode(
    *,
    mode_name: str,
    floated: Sequence[str],
    per_run_solve: Mapping[int, Mapping[str, Any]],
    per_run_leakage: Mapping[int, Mapping[str, Any]],
    consistency: Mapping[str, Any],
    transfers: Sequence[Mapping[str, Any]],
    blind_transfer_ok: bool,
    target_condition: float = DEFAULT_TARGET_CONDITION,
    max_condition_rel_spread: float = DEFAULT_MAX_COND_SPREAD,
    max_transfer_chi2_ratio: float = DEFAULT_MAX_TRANSFER_RATIO,
    residual_used_as_success: bool = False,
) -> dict[str, Any]:
    """Admit a reduced mode as a V2 *candidate* only.  Write stays false."""
    if residual_used_as_success:
        raise ValueError("residual reduction must not be used as alignment success")
    names = assert_track_driven_names(floated)
    reasons: list[str] = []
    hard = any(bool(item.get("hard_reject")) for item in per_run_leakage.values())
    if hard:
        reasons.append("dx_or_ry_equivalent_C_dx_exceeds_operating_band")
    ordered = [per_run_solve[run] for run in sorted(per_run_solve)]
    full_rank = all(bool(item.get("full_rank")) for item in ordered)
    if not full_rank:
        reasons.append("not_full_rank_on_both_calibration_runs")
    conditions = [item.get("condition_number") or item.get("raw_condition_number") for item in ordered]
    stability = condition_stable(
        conditions[0] if conditions else None,
        conditions[1] if len(conditions) > 1 else None,
        target=target_condition,
        max_rel_spread=max_condition_rel_spread,
    )
    if not stability["stable"]:
        reasons.append("condition_not_stable_on_both_calibration_runs")
    orthogonal = all(bool(item.get("clearly_orthogonal_to_A")) for item in per_run_leakage.values())
    if not orthogonal:
        reasons.append("not_clearly_orthogonal_to_frozen_A")
    if not consistency.get("consistent"):
        reasons.append("run_to_run_parameter_estimates_inconsistent")
    transfer_ok = True
    for item in transfers:
        ratio = item.get("chi2_ratio")
        if ratio is None or float(ratio) > float(max_transfer_chi2_ratio):
            transfer_ok = False
            reasons.append("linearized_calibration_transfer_worsened")
            break
    if not blind_transfer_ok:
        reasons.append("blind_block_transfer_dq_worsened")
    admitted = not reasons
    return {
        "mode_name": mode_name,
        "floated_parameters": list(names),
        "survey_fixed_parameters": [SURVEY_DZ]
        + [name for name in TRACK_DRIVEN_PARAMETERS if name not in names],
        "survey_dz_mm": 0.0,
        "admitted_as_v2_candidate": admitted,
        "geometry_write_allowed": False,
        "cdx_mode_allowed": False,
        "hard_isolation_reject": hard,
        "cannot_rescue_by_more_events_or_looser_thresholds": bool(hard),
        "full_rank_both_runs": full_rank,
        "condition_stability": stability,
        "clearly_orthogonal_to_A": orthogonal,
        "run_to_run_consistent": bool(consistency.get("consistent")),
        "calibration_transfer_ok": transfer_ok,
        "blind_transfer_ok": bool(blind_transfer_ok),
        "residual_reduction_is_not_alignment_success": True,
        "reasons": list(dict.fromkeys(reasons)),
    }


def common_identifiable_subspace(
    per_run_five_dof: Mapping[int, Mapping[str, Any]],
    a_native_per_mm: Mapping[str, float],
    *,
    max_a_projection: float = DEFAULT_MAX_A_PROJECTION,
    target_condition: float = DEFAULT_TARGET_CONDITION,
) -> dict[str, Any]:
    """Axis-aligned intersection that is well-conditioned and A-orthogonal.

    This is a Fisher-space recommendation, not a writable geometry.
    """
    a_norm = math.sqrt(sum(float(a_native_per_mm[name]) ** 2 for name in TRACK_DRIVEN_PARAMETERS))
    axis_cosine = {
        name: (0.0 if a_norm <= 0.0 else abs(float(a_native_per_mm[name])) / a_norm)
        for name in TRACK_DRIVEN_PARAMETERS
    }
    isolation_safe = [name for name, cosine in axis_cosine.items() if cosine <= float(max_a_projection)]
    conditions = []
    ranks = []
    for payload in per_run_five_dof.values():
        conditions.append(payload.get("condition_number") or payload.get("raw_condition_number"))
        ranks.append(int(payload.get("rank") or 0))
    five_ok = all(rank == 5 for rank in ranks) and all(
        value is not None and float(value) <= float(target_condition) for value in conditions
    )
    return {
        "axis_cosine_with_A": axis_cosine,
        "isolation_safe_axes": isolation_safe,
        "recommended_floated_parameters": list(isolation_safe),
        "survey_or_external_parameters": [SURVEY_DZ]
        + [name for name in TRACK_DRIVEN_PARAMETERS if name not in isolation_safe],
        "five_dof_excluding_dz_identifiable": five_ok,
        "note": (
            "Common identifiable axes are a Fisher/normal-matrix recommendation.  "
            "They become a V2 candidate only if the reduced solve is also "
            "run-to-run consistent and transfer-stable."
        ),
        "geometry_write_allowed": False,
    }


def classify_reduced_campaign(
    mode_results: Sequence[Mapping[str, Any]],
    *,
    residual_used_as_success: bool = False,
) -> dict[str, Any]:
    if residual_used_as_success:
        raise ValueError("residual reduction must not be used as alignment success")
    admitted = [item for item in mode_results if item.get("admitted_as_v2_candidate")]
    if admitted:
        # Prefer the pre-declared isolation-safe mode if several pass.
        chosen = next(
            (item for item in admitted if item.get("mode_name") == "three_dof_dy_rx_rz"),
            admitted[0],
        )
        decision = DECISION_V2_CANDIDATE
        reasons = [
            "reduced_mode_full_rank_stable_consistent_and_isolated",
            f"candidate_mode:{chosen['mode_name']}",
        ]
        candidate = chosen["mode_name"]
        floated = list(chosen["floated_parameters"])
        external = list(chosen["survey_fixed_parameters"])
    else:
        decision = DECISION_MONITORING_ONLY
        reasons = [
            "no_reduced_mode_satisfies_identifiability_isolation_and_run_consistency",
            "real_data_cannot_produce_station_geometry_update",
        ]
        if any(item.get("hard_isolation_reject") for item in mode_results):
            reasons.append("modes_with_dx_or_ry_hard_rejected_on_frozen_A_budget")
        candidate = None
        floated = []
        external = [SURVEY_DZ, *TRACK_DRIVEN_PARAMETERS]
    return {
        "decision": decision,
        "unique_class": decision,
        "v2_candidate_mode": candidate,
        "candidate_floated_parameters": floated,
        "unidentifiable_or_contaminated_dof_assigned_to": "survey_or_external_alignment",
        "survey_or_external_parameters": external,
        "geometry_write_allowed": False,
        "cdx_mode_allowed": False,
        "official_conditions_db_write": False,
        "joint_station_cdx_newton": False,
        "new_layer_or_module_dof": False,
        "do_not_retrain_v2": True,
        "do_not_open_sealed_test": True,
        "do_not_emit_cdx_payload": True,
        "residual_reduction_is_not_alignment_success": True,
        "reasons": list(dict.fromkeys(reasons)),
        "verdict_roles": [ROLE_CALIBRATION],
        "excluded_roles": [ROLE_HOLDOUT, ROLE_HELD_OUT_DQ],
    }


def blind_transfer_from_residual_blocks(
    calibration_blocks: Sequence[Mapping[str, Any]],
    blind_blocks: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Current-geometry residual IQR transfer.  Holdout has no FD Jacobian."""
    per_block: list[dict[str, Any]] = []
    holdout_ok = True
    for blind in blind_blocks:
        role = str(blind.get("role"))
        comparisons = [run_to_run_stability(calib, blind) for calib in calibration_blocks]
        compatible = any(bool(item.get("compatible")) for item in comparisons)
        n_selected = int(blind.get("selected_routes") or 0)
        insufficient = n_selected < 10 or role == ROLE_HELD_OUT_DQ and n_selected < 10
        status = "insufficient_to_judge" if insufficient else ("compatible" if compatible else "worsened")
        if role == ROLE_HOLDOUT and status == "worsened":
            holdout_ok = False
        per_block.append(
            {
                "run": blind.get("run"),
                "role": role,
                "status": status,
                "selected_routes": n_selected,
                "used_for_verdict": False,
                "comparisons": comparisons,
                "residual_reduction_is_not_alignment_success": True,
            }
        )
    return {
        "method": "current_geometry_residual_iqr_overlap",
        "holdout_has_fd_jacobian": False,
        "linearized_update_applied": False,
        "holdout_transfer_ok": holdout_ok,
        "blocks": per_block,
        "geometry_write_allowed": False,
        "residual_reduction_is_not_alignment_success": True,
    }
