"""Operating Protocol V1 real-data dry-run helpers.

This module never writes the official conditions database.  Residual
reduction is a data-quality observable, never alignment success.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from alignment.calibration_modes import (
    OPERATING_BAND_UM,
    STATION_MODE,
    evaluate_station_mode_validity,
    implied_unmodeled_cdx_um,
)
from alignment.hierarchical_v1 import STATION_SOLVE_PARAMETERS
from alignment.physical_jacobian import station_transforms_with_parameter_values

SCHEMA_VERSION = "faser-operating-protocol-v1-real-data-dryrun"
BLOCK_VALID_FOR_STATION_MODE = "valid_for_station_mode"
BLOCK_VALID_FOR_CDX_MODE = "valid_for_cdx_mode"
BLOCK_CROSS_LEVEL_CONTAMINATED = "cross_level_contaminated"
BLOCK_DQ_FAILED = "dq_failed"
BLOCK_GEOMETRY_WRITE_CANDIDATE = "geometry_write_candidate"
BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY = "insufficient_real_data_occupancy"
BLOCK_CANDIDATE_GRAPH_DQ_PASSED = "candidate_graph_dq_passed"
BLOCK_CANDIDATE_GRAPH_DQ_FAILED = "candidate_graph_dq_failed"
BLOCK_STATUSES = (
    BLOCK_VALID_FOR_STATION_MODE,
    BLOCK_VALID_FOR_CDX_MODE,
    BLOCK_CROSS_LEVEL_CONTAMINATED,
    BLOCK_DQ_FAILED,
    BLOCK_GEOMETRY_WRITE_CANDIDATE,
    BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY,
    BLOCK_CANDIDATE_GRAPH_DQ_PASSED,
    BLOCK_CANDIDATE_GRAPH_DQ_FAILED,
)
ROLE_CALIBRATION = "calibration"
ROLE_HOLDOUT = "holdout"
ROLE_HELD_OUT_DQ = "held_out_dq"
BLOCK_ROLES = (ROLE_CALIBRATION, ROLE_HOLDOUT, ROLE_HELD_OUT_DQ)
WORKBOOK_03_BLOCKED_REASON = (
    "Workbook 03 rejected 2022 data0 IFT xAOD/PHYS re-export: event IDs "
    "1...N after ntuple export did not match the existing PHYS event sequence, "
    "and SegmentFit/Segments/Tracklets were empty.  That pairing is not a V1 "
    "real-data input."
)


def _safe_name(value: str, *, label: str) -> str:
    if not value or not value.replace("_", "").isalnum():
        raise ValueError(f"{label} must be a non-empty alphanumeric identifier")
    return value


def _parameter_specs(scan: Mapping[str, object]) -> list[dict[str, object]]:
    raw = scan.get("alignment_parameter_specs")
    if not isinstance(raw, list) or not raw:
        raise ValueError("station_rigid_multidof template lacks alignment_parameter_specs")
    specs = [dict(item) for item in raw if isinstance(item, Mapping)]
    if len(specs) != len(raw):
        raise ValueError("alignment_parameter_specs entries must be mappings")
    names = [str(item.get("name", "")) for item in specs]
    if not all(names) or len(set(names)) != len(names):
        raise ValueError("alignment_parameter_specs requires unique non-empty names")
    return specs


def _zero_reference_point(scan: Mapping[str, object]) -> dict[str, object]:
    raw_points = scan.get("rigid_points")
    if not isinstance(raw_points, list) or not raw_points:
        raise ValueError("iteration template lacks rigid_points")
    nominal = [
        dict(item)
        for item in raw_points
        if isinstance(item, Mapping) and item.get("point_role") == "nominal"
    ]
    if len(nominal) != 1:
        raise ValueError("iteration template must contain exactly one nominal reference point")
    transforms = nominal[0].get("station_transforms")
    if not isinstance(transforms, Mapping):
        raise ValueError("iteration template nominal point lacks station_transforms")
    return nominal[0]


def _condition_severity(values: Mapping[str, float], specs: Sequence[Mapping[str, object]]) -> float:
    return float(
        math.sqrt(
            sum(
                (float(values[str(spec["name"])]) / float(spec["severity_scale"])) ** 2
                for spec in specs
            )
        )
    )


def _point(
    *,
    name: str,
    values: Mapping[str, float],
    specs: Sequence[Mapping[str, object]],
    base_transforms: Mapping[int | str, Sequence[float]],
    role: str,
    direction_trial: str,
    finite_difference_for: str | None = None,
    probe_sign: str | None = None,
    finite_difference_anchor: str | None = None,
) -> dict[str, object]:
    result: dict[str, object] = {
        "name": _safe_name(name, label="physical point name"),
        "point_role": role,
        "direction_trial": _safe_name(direction_trial, label="direction trial"),
        "alignment_parameter_values": {key: float(value) for key, value in values.items()},
        "station_transforms": station_transforms_with_parameter_values(specs, base_transforms, values),
        "condition_value": _condition_severity(values, specs),
        "condition_magnitude": _condition_severity(values, specs),
    }
    if finite_difference_for is not None:
        result.update(
            {
                "finite_difference_for": finite_difference_for,
                "probe_sign": probe_sign,
                "finite_difference_anchor": finite_difference_anchor,
            }
        )
    return result


def compile_real_data_station_linearization(
    template: Mapping[str, object],
    *,
    current_name: str = "iteration_00_current",
) -> tuple[dict[str, object], dict[str, object]]:
    """Linearize Station Mode at the official additional-payload origin.

    Real data has no independently known nominal target.  The scan is current
    geometry (all-zero extra /Tracker/Align) plus pure axial finite-difference
    probes.  There is no MC misaligned start and no known-reference remaining.
    """
    raw_scan = template.get("physical_refit_capture_scan", template)
    if not isinstance(raw_scan, Mapping):
        raise ValueError("physical_refit_capture_scan must be a mapping")
    scan = dict(raw_scan)
    if scan.get("scan_mode") != "station_rigid_multidof":
        raise ValueError("real-data Station Mode compiler requires scan_mode: station_rigid_multidof")
    if int(scan.get("q_over_p_mode", -1)) != 0:
        raise ValueError("real-data Station Mode compiler requires q_over_p_mode: 0")
    specs = _parameter_specs(scan)
    names = [str(spec["name"]) for spec in specs]
    if set(names) != set(STATION_SOLVE_PARAMETERS):
        raise ValueError("real-data Station Mode floats exactly station dx/dy/dz/rx/ry/rz")
    values = {name: 0.0 for name in names}
    nominal = _zero_reference_point(scan)
    base = nominal["station_transforms"]
    if not isinstance(base, Mapping):
        raise ValueError("nominal station_transforms must be a mapping")
    current = _point(
        name=current_name,
        values=values,
        specs=specs,
        base_transforms=base,
        role="nominal",
        direction_trial=current_name,
    )
    points: list[dict[str, object]] = [current]
    for spec in specs:
        parameter = str(spec["name"])
        step = float(spec["finite_difference_step"])
        for sign, suffix, role in (
            (1.0, "p", "finite_difference_positive"),
            (-1.0, "m", "finite_difference_negative"),
        ):
            probe = dict(values)
            probe[parameter] += sign * step
            points.append(
                _point(
                    name=f"{current_name}_fd_{parameter}_{suffix}",
                    values=probe,
                    specs=specs,
                    base_transforms=base,
                    role=role,
                    direction_trial=f"{current_name}_fd_{parameter}_{suffix}",
                    finite_difference_for=parameter,
                    probe_sign="positive" if sign > 0.0 else "negative",
                    finite_difference_anchor=current_name,
                )
            )
    scan["rigid_points"] = points
    scan["run_alignment_closure"] = False
    scan["is_mc"] = False
    scan["include_truth"] = False
    scan["require_mc_labels"] = False
    contract: dict[str, object] = {
        "method": "real_data_station_mode_self_nulling_linearization",
        "physical_geometry_repropagation": True,
        "coordinate_surrogate": False,
        "q_over_p_mode": 0,
        "current_point": current_name,
        "anchor_point": current_name,
        "self_nulling_update": True,
        "mc_truth_used": False,
        "official_conditions_db_write": False,
        "cdx_fixed_by": "external_geometry",
        "cdx_fixed_declaration": (
            "C_dx is taken as already fixed by the current official geometry / "
            "external alignment.  True C_dx is unknown on real data and is not "
            "measured in this Station Mode stage."
        ),
        "update_semantics": (
            "Real-data DQ dry-run: Gauss-Newton self-nulling of unbiased route-selected "
            "residuals linearized at the official additional-payload origin.  Residual "
            "reduction is not alignment success.  Do not write the official conditions DB."
        ),
        "finite_difference_steps": {str(spec["name"]): float(spec["finite_difference_step"]) for spec in specs},
    }
    return {"alignment_iteration": contract, "physical_refit_capture_scan": scan}, contract


def compile_real_data_current_only(
    template: Mapping[str, object],
    *,
    current_name: str = "iteration_00_current",
) -> tuple[dict[str, object], dict[str, object]]:
    """Holdout / held-out DQ reconstruction at official geometry only."""
    compiled, contract = compile_real_data_station_linearization(template, current_name=current_name)
    scan = dict(compiled["physical_refit_capture_scan"])
    scan["rigid_points"] = [
        point for point in scan["rigid_points"] if str(point.get("name")) == current_name
    ]
    if len(scan["rigid_points"]) != 1:
        raise ValueError("current-only scan must retain exactly the official-geometry point")
    scan["current_geometry_only"] = True
    scan["run_alignment_closure"] = False
    contract = dict(contract)
    contract["finite_difference_probes"] = False
    contract["geometry_estimation_allowed"] = False
    return {"alignment_iteration": contract, "physical_refit_capture_scan": scan}, contract


def compile_real_data_candidate_current_only(
    template: Mapping[str, object],
    parameter_values: Mapping[str, float],
    *,
    current_name: str = "iteration_00_current",
) -> tuple[dict[str, object], dict[str, object]]:
    """Holdout / held-out DQ reconstruction at one frozen candidate geometry.

    Written ``dz`` is forced to 0.  This compiler does not estimate parameters.
    """
    compiled, contract = compile_real_data_current_only(template, current_name=current_name)
    scan = dict(compiled["physical_refit_capture_scan"])
    specs = _parameter_specs(scan)
    names = [str(spec["name"]) for spec in specs]
    missing = [name for name in names if name not in parameter_values]
    if missing:
        raise ValueError("candidate geometry is missing " + ", ".join(missing))
    values = {name: float(parameter_values[name]) for name in names}
    values["ift_dz_mm"] = 0.0
    if len(scan["rigid_points"]) != 1:
        raise ValueError("candidate-eval scan must contain exactly one current point")
    base = scan["rigid_points"][0].get("station_transforms")
    if not isinstance(base, Mapping):
        raise ValueError("candidate-eval current point lacks station_transforms")
    scan["rigid_points"] = [
        _point(
            name=current_name,
            values=values,
            specs=specs,
            base_transforms=base,
            role="nominal",
            direction_trial=current_name,
        )
    ]
    scan["candidate_geometry_eval"] = True
    scan["geometry_estimation_allowed"] = False
    contract = dict(contract)
    contract["candidate_geometry_eval"] = True
    contract["geometry_estimation_allowed"] = False
    contract["written_dz_mm"] = 0.0
    return {"alignment_iteration": contract, "physical_refit_capture_scan": scan}, contract


def cross_level_contamination_diagnostic(
    contract: Mapping[str, Any],
    *,
    station_dx_mm: float | None,
    station_ry_mrad: float | None,
    run_to_run_dx_spread_mm: float | None,
    independent_cdx_evidence: bool = False,
    independent_abs_C_dx_um: float | None = None,
) -> dict[str, Any]:
    """Implied unmodeled C_dx from frozen A and the observed station pattern.

    Real data cannot know true C_dx.  ``cdx_fixed_by=external_geometry`` is a
    provenance declaration only: without independent evidence that
    ``|C_dx|`` is inside the frozen statistical budget, the candidate is
    ``cross_level_contaminated`` even if residuals fall.  A station dx/ry
    whose implied ``|C_dx|`` exceeds the frozen 1.5–1.7 µm band is a second,
    independent reject indicator, not a C_dx measurement.

    ``implied_Cdx_dx = -Δdx / A_dx`` and ``implied_Cdx_ry = -Δry / A_ry``
    project the Newton correction onto leakage direction A.  Absolute
    values match the omitted-variable formula ``C_dx = dx / A_dx``.
    """
    a_map = contract["leakage_operator"]["A_native_per_mm_C_dx"]
    a_dx = float(a_map["ift_dx_mm"])
    a_ry = float(a_map["ift_ry_mrad"])
    implied_from_dx = None if station_dx_mm is None else implied_unmodeled_cdx_um(
        station_dx_mm=float(station_dx_mm), A_dx=a_dx
    )
    implied_cdx_dx = None if implied_from_dx is None else -float(implied_from_dx)
    implied_from_ry = None
    implied_cdx_ry = None
    if station_ry_mrad is not None and a_ry != 0.0:
        implied_from_ry = 1.0e3 * float(station_ry_mrad) / a_ry
        implied_cdx_ry = -float(implied_from_ry)
    abs_implied_dx = None if implied_cdx_dx is None else abs(float(implied_cdx_dx))
    abs_implied_ry = None if implied_cdx_ry is None else abs(float(implied_cdx_ry))
    operating_hi = float(OPERATING_BAND_UM[1])
    exceeds_operating = bool(
        (abs_implied_dx is not None and abs_implied_dx > operating_hi)
        or (abs_implied_ry is not None and abs_implied_ry > operating_hi)
    )
    implied_consistency = None
    if implied_cdx_dx is not None and implied_cdx_ry is not None:
        scale = max(abs(float(implied_cdx_dx)), abs(float(implied_cdx_ry)), 1.0e-12)
        implied_consistency = {
            "difference_um": float(implied_cdx_dx) - float(implied_cdx_ry),
            "relative_difference": abs(float(implied_cdx_dx) - float(implied_cdx_ry)) / scale,
            "same_sign": bool(float(implied_cdx_dx) * float(implied_cdx_ry) >= 0.0),
        }
    statistical = float(contract["station_mode"]["unmodeled_C_dx"]["statistical_max_abs_um"])
    independent_within_budget = bool(
        independent_cdx_evidence
        and independent_abs_C_dx_um is not None
        and math.isfinite(float(independent_abs_C_dx_um))
        and abs(float(independent_abs_C_dx_um)) <= statistical
    )
    validity = evaluate_station_mode_validity(
        contract,
        floated_parameters=STATION_SOLVE_PARAMETERS,
        unmodeled_abs_C_dx_um=0.0 if abs_implied_dx is None else abs_implied_dx,
        cdx_fixed_by="external_geometry",
        same_data_stage_as_other_mode=False,
        implied_C_dx_um_from_station_dx=implied_from_dx,
    )
    validity = dict(validity)
    indicators = list(validity.get("reject_indicators") or [])
    if station_dx_mm is None:
        validity["status"] = "prerequisite_unmet"
        indicators.append("true_C_dx_unknown_implied_pattern_unavailable")
    if not independent_within_budget:
        indicators.append("true_C_dx_not_independently_proven_le_statistical_budget")
        validity["status"] = "prerequisite_unmet"
    if exceeds_operating:
        indicators.append("implied_C_dx_from_station_correction_exceeds_operating_band")
    validity["reject_indicators"] = list(dict.fromkeys(indicators))
    validity["geometry_write_allowed"] = False
    validity["cross_level_contaminated"] = True
    if independent_within_budget and not exceeds_operating and station_dx_mm is not None:
        # Independent evidence inside the frozen budget, and the station
        # correction does not itself imply a larger C_dx, is the only path
        # that can keep the contract's write decision.
        validity["geometry_write_allowed"] = bool(
            evaluate_station_mode_validity(
                contract,
                floated_parameters=STATION_SOLVE_PARAMETERS,
                unmodeled_abs_C_dx_um=abs(float(independent_abs_C_dx_um)),
                cdx_fixed_by="dedicated_calibration",
                same_data_stage_as_other_mode=False,
                implied_C_dx_um_from_station_dx=implied_from_dx,
            )["geometry_write_allowed"]
        )
        validity["cross_level_contaminated"] = not bool(validity["geometry_write_allowed"])
        if validity["geometry_write_allowed"]:
            validity["status"] = "valid"
            validity["reject_indicators"] = [
                item
                for item in validity["reject_indicators"]
                if item not in {
                    "true_C_dx_not_independently_proven_le_statistical_budget",
                    "implied_C_dx_from_station_correction_exceeds_operating_band",
                    "true_C_dx_unknown_implied_pattern_unavailable",
                }
            ]
    return {
        "A_dx": a_dx,
        "A_ry": a_ry,
        "station_dx_mm": None if station_dx_mm is None else float(station_dx_mm),
        "station_ry_mrad": None if station_ry_mrad is None else float(station_ry_mrad),
        "implied_C_dx_um_from_station_dx": implied_from_dx,
        "implied_C_dx_um_from_station_ry": implied_from_ry,
        "implied_Cdx_dx": implied_cdx_dx,
        "implied_Cdx_ry": implied_cdx_ry,
        "implied_Cdx_consistency": implied_consistency,
        "run_to_run_dx_spread_mm": None if run_to_run_dx_spread_mm is None else float(run_to_run_dx_spread_mm),
        "operating_band_um": list(OPERATING_BAND_UM),
        "exceeds_operating_band": exceeds_operating,
        "true_C_dx_unknown": True,
        "independent_cdx_evidence": bool(independent_cdx_evidence),
        "independent_abs_C_dx_um": None if independent_abs_C_dx_um is None else float(independent_abs_C_dx_um),
        "independent_within_statistical_budget": independent_within_budget,
        "cdx_fixed_by_is_provenance_only": True,
        "not_a_C_dx_measurement": True,
        "mode_validity": validity,
    }


def assign_block_status(
    *,
    role: str,
    dq_failed: bool,
    cross_level_contaminated: bool,
    station_mode_valid: bool,
    cdx_mode_valid: bool,
    holdout_dq_worsened: bool,
    anomalous_run_drift: bool,
    official_conditions_write_requested: bool = False,
    insufficient_real_data_occupancy: bool = False,
    candidate_graph_dq_passed: bool = False,
    candidate_graph_dq_failed: bool = False,
) -> str:
    """Assign exactly one Operating Protocol V1 block label."""
    if role not in BLOCK_ROLES:
        raise ValueError(f"unknown block role {role!r}")
    if official_conditions_write_requested:
        raise ValueError("this dry-run must not request an official conditions DB write")
    if insufficient_real_data_occupancy:
        return BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY
    if candidate_graph_dq_failed:
        return BLOCK_CANDIDATE_GRAPH_DQ_FAILED
    if dq_failed or holdout_dq_worsened or anomalous_run_drift:
        return BLOCK_DQ_FAILED
    if cross_level_contaminated:
        return BLOCK_CROSS_LEVEL_CONTAMINATED
    if cdx_mode_valid and station_mode_valid and role == ROLE_CALIBRATION:
        return BLOCK_GEOMETRY_WRITE_CANDIDATE
    if cdx_mode_valid:
        return BLOCK_VALID_FOR_CDX_MODE
    if station_mode_valid:
        return BLOCK_VALID_FOR_STATION_MODE
    if candidate_graph_dq_passed:
        return BLOCK_CANDIDATE_GRAPH_DQ_PASSED
    return BLOCK_DQ_FAILED
