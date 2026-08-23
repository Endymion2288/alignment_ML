"""Frozen-V2 real-data Station Mode failure characterization.

Analysis only.  Reconstructs ``J_s`` from the already-captured 12 station
finite-difference probes, back-projects implied ``|C_dx|`` with the frozen
MC ``A`` operator, and classifies whether the observed
``cross_level_contaminated`` result is a statistics limitation, a physical
non-identifiability, or a V2 association failure.

This module never writes official conditions, never starts C_dx Mode,
never emits a new ``C_dx`` payload, never uses the Schur production
estimator, and never treats residual reduction as alignment success.
"""

from __future__ import annotations

import json
import math
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from alignment.calibration_modes import OPERATING_BAND_UM
from alignment.five_dof_sampling import DEFAULT_SCALES, SURVEY_PARAMETER
from alignment.hierarchical_v1 import STATION_SOLVE_PARAMETERS
from alignment.physical_jacobian import _svd_rank
from alignment.real_data_operating_protocol import (
    ROLE_CALIBRATION,
    ROLE_HELD_OUT_DQ,
    ROLE_HOLDOUT,
    cross_level_contamination_diagnostic,
)
from alignment.real_data_station_mode_fullscale import frozen_a_stability_status

SCHEMA_VERSION = "faser-operating-protocol-v1-real-data-station-mode-failure-audit"
DEFAULT_CONFIG_RELATIVE = (
    "configs/operating_protocol_v1_real_data_station_mode_failure_audit_v1.yaml"
)

CLASS_A = "reconstruction_statistics_limitation"
CLASS_B = "physical_nonidentifiability"
CLASS_C = "v2_association_failure"
FAILURE_CLASSES = (CLASS_A, CLASS_B, CLASS_C)

SURVEY_DZ = SURVEY_PARAMETER
DEFAULT_TARGET_CONDITION = 1.0e4
DEFAULT_NEAR_DEGENERATE = 0.5
DEFAULT_A_ALIGNMENT = 0.90


def load_failure_audit_config(path: str | None = None) -> dict[str, Any]:
    from pathlib import Path

    source = Path(path).expanduser().resolve() if path is not None else (
        Path(__file__).resolve().parents[1] / DEFAULT_CONFIG_RELATIVE
    )
    with source.open(encoding="utf-8") as handle:
        payload = yaml.safe_load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"failure-audit config must be a mapping: {source}")
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"unexpected failure-audit config schema: {source}")
    if payload.get("frozen") is not True:
        raise ValueError("failure-audit config must be frozen")
    if payload.get("geometry_write_allowed") is not False:
        raise ValueError("failure-audit config must keep geometry_write_allowed false")
    if payload.get("official_conditions_db_write") is not False:
        raise ValueError("failure-audit config must forbid official conditions writes")
    if payload.get("do_not_enter_cdx_mode") is not True or payload.get("cdx_mode_blocked") is not True:
        raise ValueError("failure-audit config must block C_dx Mode")
    if payload.get("joint_station_cdx_newton") is not False:
        raise ValueError("failure-audit config must forbid joint Newton")
    if payload.get("new_layer_or_module_dof") is not False:
        raise ValueError("failure-audit config must forbid new DoF")
    if payload.get("schur_projection_production") is not False:
        raise ValueError("failure-audit config must not use the Schur production estimator")
    if payload.get("do_not_retrain_v2") is not True:
        raise ValueError("failure-audit config must freeze V2")
    if payload.get("do_not_emit_cdx_payload") is not True:
        raise ValueError("failure-audit config must not emit a new C_dx payload")
    if payload.get("residual_reduction_is_not_alignment_success") is not True:
        raise ValueError("failure-audit config must treat residual reduction as DQ only")
    if payload.get("do_not_open_sealed_test") is not True:
        raise ValueError("failure-audit config must keep the sealed test closed")
    return {"path": str(source), "schema_version": SCHEMA_VERSION, **dict(payload)}


def campaign_allows_cdx_mode(decision: str) -> bool:
    return False


def campaign_allows_geometry_write(decision: str) -> bool:
    return False


def route_station_count(signature: object) -> int:
    if signature in (None, ""):
        return 0
    payload = json.loads(signature) if isinstance(signature, str) else signature
    if not isinstance(payload, Sequence):
        return 0
    stations = {
        int(item["station_id"])
        for item in payload
        if isinstance(item, Mapping) and item.get("station_id") is not None
    }
    return int(len(stations))


def _as_names(names: Sequence[str] | None) -> tuple[str, ...]:
    resolved = tuple(STATION_SOLVE_PARAMETERS if names is None else (str(name) for name in names))
    if len(set(resolved)) != len(resolved):
        raise ValueError("parameter names must be unique")
    return resolved


def _scales_for(names: Sequence[str], scales: Mapping[str, float] | None = None) -> np.ndarray:
    resolved = DEFAULT_SCALES if scales is None else scales
    return np.asarray([float(resolved[name]) for name in names], dtype=np.float64)


def _inverse_covariances(covariance: np.ndarray) -> np.ndarray:
    blocks = np.asarray(covariance, dtype=np.float64)
    if blocks.ndim != 3 or blocks.shape[1:] != (4, 4):
        raise ValueError("covariance must have shape (n_obs, 4, 4)")
    try:
        return np.linalg.inv(blocks)
    except np.linalg.LinAlgError:
        return np.asarray([np.linalg.pinv(block) for block in blocks], dtype=np.float64)


def scaled_normal_from_js(
    derivative_native: np.ndarray,
    covariance: np.ndarray,
    *,
    parameter_names: Sequence[str] | None = None,
    parameter_scales: Mapping[str, float] | None = None,
    observation_mask: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, tuple[str, ...]]:
    """Rebuild ``S J^T W J S`` from the FD Jacobian ``J_s``."""
    names = _as_names(parameter_names)
    jacobian = np.asarray(derivative_native, dtype=np.float64)
    if jacobian.ndim != 3 or jacobian.shape[1:] != (4, len(names)):
        raise ValueError(
            f"J_s must have shape (n_obs, 4, {len(names)}); got {tuple(jacobian.shape)}"
        )
    inverse = _inverse_covariances(covariance)
    if inverse.shape[0] != jacobian.shape[0]:
        raise ValueError("covariance observation count does not match J_s")
    if observation_mask is None:
        mask = np.ones(jacobian.shape[0], dtype=bool)
    else:
        mask = np.asarray(observation_mask, dtype=bool)
        if mask.shape != (jacobian.shape[0],):
            raise ValueError("observation_mask length must match J_s")
    native = np.zeros((len(names), len(names)), dtype=np.float64)
    for index in np.flatnonzero(mask):
        native += jacobian[index].T @ inverse[index] @ jacobian[index]
    native = 0.5 * (native + native.T)
    scales = _scales_for(names, parameter_scales)
    scale_matrix = np.diag(scales)
    scaled = scale_matrix @ native @ scale_matrix
    scaled = 0.5 * (scaled + scaled.T)
    return native, scaled, names


def _spectrum(normal_scaled: np.ndarray, *, rcond: float) -> dict[str, Any]:
    matrix = np.asarray(normal_scaled, dtype=np.float64)
    singular, rank, _tolerance, full_condition, retained = _svd_rank(matrix, rcond=rcond)
    values, vectors = np.linalg.eigh(matrix)
    order = np.argsort(values)[::-1]
    values = np.clip(values[order], 0.0, None)
    vectors = vectors[:, order]
    raw = None
    if values.size and float(values[-1]) > 0.0:
        raw = float(values[0] / values[-1])
    return {
        "rank": int(rank),
        "condition_number": None if full_condition is None else float(full_condition),
        "raw_condition_number": raw,
        "identifiable_subspace_condition_number": None if retained is None else float(retained),
        "singular_values": tuple(float(value) for value in singular.tolist()),
        "eigenvalues": tuple(float(value) for value in values.tolist()),
        "right_vectors": np.asarray(vectors, dtype=np.float64),
        "jacobian_singular_values": tuple(
            float(math.sqrt(value)) if value > 0.0 else 0.0 for value in values.tolist()
        ),
    }


def near_degenerate_parameters(
    names: Sequence[str],
    right_vectors: np.ndarray,
    *,
    abs_component: float = DEFAULT_NEAR_DEGENERATE,
) -> dict[str, Any]:
    """Flag parameters that dominate the smallest scaled-normal direction."""
    if right_vectors.size == 0:
        return {
            "smallest_direction": {},
            "second_smallest_direction": {},
            "near_degenerate": [],
            "abs_component_threshold": float(abs_component),
        }
    smallest = {
        str(name): float(right_vectors[index, -1]) for index, name in enumerate(names)
    }
    second = {}
    if right_vectors.shape[1] > 1:
        second = {
            str(name): float(right_vectors[index, -2]) for index, name in enumerate(names)
        }
    flagged = [
        str(name)
        for name, value in smallest.items()
        if math.isfinite(value) and abs(value) >= float(abs_component)
    ]
    return {
        "smallest_direction": smallest,
        "second_smallest_direction": second,
        "near_degenerate": flagged,
        "abs_component_threshold": float(abs_component),
    }


def drop_parameters(normal_scaled: np.ndarray, names: Sequence[str], drop: Sequence[str]) -> tuple[np.ndarray, tuple[str, ...]]:
    keep = tuple(name for name in names if name not in set(drop))
    index = [list(names).index(name) for name in keep]
    return np.asarray(normal_scaled, dtype=np.float64)[np.ix_(index, index)], keep


def cosine_with_vector(
    components: Mapping[str, float],
    reference: Mapping[str, float],
    names: Sequence[str],
) -> float | None:
    left = np.asarray([float(components.get(name, 0.0)) for name in names], dtype=np.float64)
    right = np.asarray([float(reference.get(name, 0.0)) for name in names], dtype=np.float64)
    denom = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denom <= 0.0:
        return None
    return float(abs(float(left @ right)) / denom)


def reconstruct_js_identifiability(
    derivative_native: np.ndarray,
    covariance: np.ndarray,
    *,
    parameter_names: Sequence[str] | None = None,
    parameter_scales: Mapping[str, float] | None = None,
    route_lengths: Sequence[int] | None = None,
    rcond: float = 1.0e-10,
    near_degenerate_abs_component: float = DEFAULT_NEAR_DEGENERATE,
    a_native_per_mm: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Identifiability audit of real-data ``J_s`` from the 12 station FD probes.

    Residual improvement is not a closure score and is not computed here.
    """
    native, scaled, names = scaled_normal_from_js(
        derivative_native,
        covariance,
        parameter_names=parameter_names,
        parameter_scales=parameter_scales,
    )
    spectrum = _spectrum(scaled, rcond=rcond)
    directions = near_degenerate_parameters(
        names, spectrum["right_vectors"], abs_component=near_degenerate_abs_component
    )
    five_normal, five_names = drop_parameters(scaled, names, (SURVEY_DZ,))
    five = _spectrum(five_normal, rcond=rcond)
    five_dir = near_degenerate_parameters(
        five_names, five["right_vectors"], abs_component=near_degenerate_abs_component
    )
    a_scaled = None
    if a_native_per_mm is not None:
        scales = _scales_for(names, parameter_scales)
        a_scaled = {
            name: float(a_native_per_mm[name]) / float(scales[index])
            for index, name in enumerate(names)
            if name in a_native_per_mm
        }
    five_a_cosine = (
        None
        if a_scaled is None
        else cosine_with_vector(five_dir["smallest_direction"], a_scaled, five_names)
    )
    partitions: dict[str, Any] = {}
    if route_lengths is not None:
        lengths = np.asarray(list(route_lengths), dtype=np.int64)
        if lengths.shape != (int(np.asarray(derivative_native).shape[0]),):
            raise ValueError("route_lengths length must match J_s")
        for length in sorted(int(value) for value in set(lengths.tolist())):
            mask = lengths == length
            _native_part, scaled_part, _ = scaled_normal_from_js(
                derivative_native,
                covariance,
                parameter_names=names,
                parameter_scales=parameter_scales,
                observation_mask=mask,
            )
            part = _spectrum(scaled_part, rcond=rcond)
            part_dir = near_degenerate_parameters(
                names, part["right_vectors"], abs_component=near_degenerate_abs_component
            )
            five_part, _ = drop_parameters(scaled_part, names, (SURVEY_DZ,))
            five_part_spec = _spectrum(five_part, rcond=rcond)
            partitions[str(length)] = {
                "n_observations": int(np.count_nonzero(mask)),
                "six_dof_condition_number": part["condition_number"],
                "six_dof_raw_condition_number": part["raw_condition_number"],
                "six_dof_singular_values": list(part["singular_values"]),
                "five_dof_condition_number": five_part_spec["condition_number"],
                "five_dof_raw_condition_number": five_part_spec["raw_condition_number"],
                "near_degenerate": part_dir["near_degenerate"],
                "smallest_direction": part_dir["smallest_direction"],
            }
    return {
        "parameter_names": list(names),
        "n_observations": int(np.asarray(derivative_native).shape[0]),
        "n_fd_probes": 12,
        "reconstruction": "central_finite_difference_J_s_on_frozen_selected_edges",
        "rank": spectrum["rank"],
        "full_rank": spectrum["rank"] == len(names),
        "condition_number": spectrum["condition_number"],
        "raw_condition_number": spectrum["raw_condition_number"],
        "identifiable_subspace_condition_number": spectrum["identifiable_subspace_condition_number"],
        "singular_values": list(spectrum["singular_values"]),
        "jacobian_singular_values": list(spectrum["jacobian_singular_values"]),
        "near_degenerate": directions,
        "five_dof_excluding_survey_dz": {
            "parameter_names": list(five_names),
            "rank": five["rank"],
            "condition_number": five["condition_number"],
            "raw_condition_number": five["raw_condition_number"],
            "singular_values": list(five["singular_values"]),
            "near_degenerate": five_dir,
            "weak_direction_cosine_with_A": five_a_cosine,
        },
        "route_length_partitions": partitions,
        "residual_reduction_is_not_alignment_success": True,
        "residual_improvement_is_not_closure": True,
    }


def estimate_statistics_scaling(
    base_normal_scaled: np.ndarray,
    complete_unit_normal_scaled: np.ndarray | None,
    *,
    names: Sequence[str],
    n_complete_observations: int,
    n_complete_routes: int,
    n_events: int,
    target_condition_number: float = DEFAULT_TARGET_CONDITION,
    practical_complete_routes_max: int = 200,
    practical_events_max: int = 500_000,
) -> dict[str, Any]:
    """Analysis-only control: can more complete four-station rows recover covariance?

    Same-topology scaling leaves the condition number invariant.  Complete-track
    scaling adds copies of the observed complete-route normal block.  This is
    not a Schur estimator and does not write geometry.
    """
    base = np.asarray(base_normal_scaled, dtype=np.float64)
    same = _spectrum(base, rcond=1.0e-10)
    same_scaled = _spectrum(1000.0 * base, rcond=1.0e-10)
    same_invariant = (
        same["condition_number"] is not None
        and same_scaled["condition_number"] is not None
        and math.isfinite(same["condition_number"])
        and abs(float(same_scaled["condition_number"]) - float(same["condition_number"]))
        <= 1.0e-6 * max(1.0, abs(float(same["condition_number"])))
    )
    idx5 = [index for index, name in enumerate(names) if name != SURVEY_DZ]

    def _condition(matrix: np.ndarray, drop_dz: bool) -> float | None:
        work = matrix[np.ix_(idx5, idx5)] if drop_dz else matrix
        return _spectrum(work, rcond=1.0e-10)["condition_number"]

    def _min_copies(drop_dz: bool) -> dict[str, Any]:
        if complete_unit_normal_scaled is None or n_complete_observations <= 0:
            current = _condition(base, drop_dz)
            already = current is not None and current <= float(target_condition_number)
            return {
                "recovers": already,
                "already_below_target": already,
                "minimum_copies_of_observed_complete_block": 0.0 if already else None,
                "condition_at_estimate": current,
                "reason": "no_observed_complete_four_station_block" if not already else "already_below_target",
            }
        unit = np.asarray(complete_unit_normal_scaled, dtype=np.float64)
        current = _condition(base, drop_dz)
        if current is not None and current <= float(target_condition_number):
            return {
                "recovers": True,
                "already_below_target": True,
                "minimum_copies_of_observed_complete_block": 0.0,
                "condition_at_estimate": current,
                "reason": "already_below_target",
            }

        def cond_at(copies: float) -> float | None:
            return _condition(base + float(copies) * unit, drop_dz)

        low = 0.0
        high = 1.0
        value = cond_at(high)
        for _ in range(60):
            if value is not None and value <= float(target_condition_number):
                break
            high *= 2.0
            value = cond_at(high)
            if high > 1.0e16:
                return {
                    "recovers": False,
                    "already_below_target": False,
                    "minimum_copies_of_observed_complete_block": None,
                    "condition_at_estimate": value,
                    "reason": "observed_complete_block_does_not_lift_weak_direction",
                }
        if value is None or value > float(target_condition_number):
            return {
                "recovers": False,
                "already_below_target": False,
                "minimum_copies_of_observed_complete_block": None,
                "condition_at_estimate": value,
                "reason": "observed_complete_block_does_not_lift_weak_direction",
            }
        for _ in range(80):
            mid = 0.5 * (low + high)
            mid_value = cond_at(mid)
            if mid_value is not None and mid_value <= float(target_condition_number):
                high = mid
            else:
                low = mid
        copies = float(high)
        implied_routes = None if n_complete_routes <= 0 else float(copies) * float(n_complete_routes)
        rate = None if n_events <= 0 or n_complete_routes <= 0 else float(n_complete_routes) / float(n_events)
        implied_events = None if implied_routes is None or rate in (None, 0.0) else float(implied_routes) / float(rate)
        practical = bool(
            implied_routes is not None
            and implied_routes <= float(practical_complete_routes_max)
            and implied_events is not None
            and implied_events <= float(practical_events_max)
        )
        return {
            "recovers": True,
            "already_below_target": False,
            "minimum_copies_of_observed_complete_block": copies,
            "condition_at_estimate": cond_at(copies),
            "implied_complete_routes": implied_routes,
            "implied_events_at_observed_rate": implied_events,
            "practical": practical,
            "reason": "complete_block_lifts_condition_to_target",
        }

    six = _min_copies(False)
    five = _min_copies(True)
    recovers_six = bool(
        six["recovers"]
        and not six.get("already_below_target")
        and six.get("practical") is True
    )
    return {
        "schur_production_estimator_used": False,
        "geometry_write_allowed": False,
        "target_condition_number": float(target_condition_number),
        "same_topology_scaling": {
            "condition_invariant": bool(same_invariant),
            "base_condition_number": same["condition_number"],
            "scaled_x1000_condition_number": same_scaled["condition_number"],
            "recovers_six_dof": False,
            "note": "Scaling every current selected-route row by the same factor leaves the condition number unchanged.",
        },
        "complete_four_station_scaling": {
            "n_complete_observations": int(n_complete_observations),
            "n_complete_routes": int(n_complete_routes),
            "n_events": int(n_events),
            "six_dof": six,
            "five_dof_excluding_survey_dz": five,
        },
        "station_covariance_can_recover_by_more_complete_tracks": bool(recovers_six),
        "practical_path_to_geometry_write": False,
        "residual_reduction_is_not_alignment_success": True,
    }


def contamination_backprojection(
    contract: Mapping[str, Any],
    *,
    station_dx_by_run: Mapping[int, float | None],
    station_ry_by_run: Mapping[int, float | None],
) -> dict[str, Any]:
    """Frozen-A implied ``|C_dx|``.  Rejection diagnostic only."""
    dx_values = [float(value) for value in station_dx_by_run.values() if value is not None]
    spread = None if len(dx_values) < 2 else float(max(dx_values) - min(dx_values))
    per_run: dict[str, Any] = {}
    for run, dx in station_dx_by_run.items():
        item = cross_level_contamination_diagnostic(
            contract,
            station_dx_mm=dx,
            station_ry_mrad=station_ry_by_run.get(run),
            run_to_run_dx_spread_mm=spread,
            independent_cdx_evidence=False,
        )
        item["rejection_diagnostic_only"] = True
        item["do_not_emit_cdx_payload"] = True
        item["not_a_C_dx_measurement"] = True
        per_run[str(run)] = item
    exceeds = any(bool(item.get("exceeds_operating_band")) for item in per_run.values())
    consistent = [
        bool((item.get("implied_Cdx_consistency") or {}).get("same_sign"))
        and float((item.get("implied_Cdx_consistency") or {}).get("relative_difference") or 1.0) <= 0.25
        for item in per_run.values()
        if item.get("implied_Cdx_consistency") is not None
    ]
    return {
        "A_stability": frozen_a_stability_status(contract),
        "operating_band_um": list(OPERATING_BAND_UM),
        "per_run": per_run,
        "run_to_run_dx_spread_mm": spread,
        "exceeds_operating_band": exceeds,
        "implied_Cdx_internally_consistent": bool(consistent) and all(consistent),
        "rejection_diagnostic_only": True,
        "do_not_emit_cdx_payload": True,
        "new_cdx_payload": None,
        "not_a_C_dx_measurement": True,
        "cdx_mode_started": False,
        "cdx_mode_allowed": False,
        "geometry_write_allowed": False,
        "schur_projection_production": False,
        "residual_reduction_is_not_alignment_success": True,
    }


def classify_failure(
    *,
    selected_routes_calibration: Sequence[int],
    complete_four_station_calibration: Sequence[int],
    all_pairs_nonempty: bool,
    six_dof_recoverable_from_complete_tracks: bool,
    complete_track_recovery_practical: bool,
    same_topology_recovers_six_dof: bool,
    implied_cdx_exceeds_operating_band: bool,
    weak_direction_cosine_with_A: float | None,
    a_alignment_threshold: float = DEFAULT_A_ALIGNMENT,
    residual_used_as_success: bool = False,
) -> dict[str, Any]:
    """Unique A / B / C label.  Residual drop is never success."""
    if residual_used_as_success:
        raise ValueError("residual reduction must not be used as alignment success")
    selected = [int(value) for value in selected_routes_calibration]
    complete = [int(value) for value in complete_four_station_calibration]
    if len(selected) != 2 or len(complete) != 2:
        raise ValueError("classification uses exactly the two calibration runs")
    reasons: list[str] = []
    ruled_out: list[str] = []
    concurrent: str | None = None
    complete_fraction_low = sum(complete) <= 2 or (
        sum(selected) > 0 and float(sum(complete)) / float(sum(selected)) < 0.05
    )
    leakage_aligned = (
        weak_direction_cosine_with_A is not None
        and math.isfinite(float(weak_direction_cosine_with_A))
        and float(weak_direction_cosine_with_A) >= float(a_alignment_threshold)
    )
    v2_failed = (not all_pairs_nonempty) or min(selected) <= 0
    if v2_failed:
        unique = CLASS_C
        reasons.append("selected_graph_empty_or_all_pairs_empty")
    else:
        ruled_out.append(CLASS_C)
        reasons.append("selected_graph_nonempty_frozen_v2_unchanged")
        physical = (
            (not six_dof_recoverable_from_complete_tracks)
            or leakage_aligned
            or not complete_track_recovery_practical
        )
        if physical:
            unique = CLASS_B
            if not six_dof_recoverable_from_complete_tracks:
                reasons.append("six_dof_not_recoverable_from_observed_complete_topology")
            if leakage_aligned:
                reasons.append("weak_station_direction_aligned_with_frozen_A")
            if not complete_track_recovery_practical and six_dof_recoverable_from_complete_tracks:
                reasons.append("complete_track_recovery_not_practical")
            if implied_cdx_exceeds_operating_band:
                reasons.append("implied_C_dx_exceeds_operating_band_rejection_diagnostic")
            if complete_fraction_low:
                concurrent = CLASS_A
                reasons.append("complete_four_station_statistics_also_limited")
        elif complete_fraction_low and not same_topology_recovers_six_dof:
            unique = CLASS_A
            reasons.append("more_complete_four_station_statistics_could_recover_station_covariance")
            if implied_cdx_exceeds_operating_band:
                concurrent = CLASS_B
                reasons.append("frozen_A_isolation_still_fails_as_rejection_diagnostic")
        else:
            unique = CLASS_B if implied_cdx_exceeds_operating_band or leakage_aligned else CLASS_A
            if unique == CLASS_B:
                reasons.append("cross_level_isolation_fails_independently_of_route_count")
            else:
                reasons.append("station_graph_statistics_limited")
    return {
        "unique_class": unique,
        "decision": unique,
        "concurrent_class": concurrent,
        "ruled_out": ruled_out,
        "reasons": list(dict.fromkeys(reasons)),
        "geometry_write_allowed": False,
        "cdx_mode_allowed": False,
        "official_conditions_db_write": False,
        "joint_station_cdx_newton": False,
        "new_layer_or_module_dof": False,
        "schur_projection_production": False,
        "do_not_retrain_v2": True,
        "do_not_open_sealed_test": True,
        "do_not_emit_cdx_payload": True,
        "residual_reduction_is_not_alignment_success": True,
        "verdict_roles": [ROLE_CALIBRATION],
        "excluded_roles": [ROLE_HOLDOUT, ROLE_HELD_OUT_DQ],
    }
