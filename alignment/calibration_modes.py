"""Mutually exclusive FASER calibration modes and the mode-validity contract.

Hierarchical alignment V1 is closed as a joint hierarchy (workbook 45).
Production has two exclusive modes:

* ``station``: float IFT station ``dx/dy/rx/ry/rz`` with a 5 mm survey
  ``dz`` prior.  Written ``dz`` stays 0.  IFT internal ``C_dx`` is not a
  floated parameter and is not a nuisance column.
* ``ift_internal``: float 1-D ``C_dx=(dx_L0-dx_L2)/2`` only, with payload
  ``L0=+C_dx``, ``L1=0``, ``L2=-C_dx``.  The station six-vector is not a
  floated parameter and is not a nuisance column.

The two modes must not iterate on the same data stage, must not treat the
other level as a nuisance, and must not be written as a mixed remaining
chart.  Residual reduction is never alignment success (workbook 44).
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

from alignment.five_dof_sampling import FREE_PARAMETERS as STATION_FREE_PARAMETERS
from alignment.five_dof_sampling import SURVEY_PARAMETER
from alignment.hierarchical_v1 import C_DX, STATION_SOLVE_PARAMETERS
from alignment.hierarchical_v1_leakage import (
    STATION_SIX,
    cdx_systematic_from_station_sigma,
)

SCHEMA_VERSION = "faser-ift-calibration-mode-validity-v1"
STATION_MODE = "station"
IFT_INTERNAL_MODE = "ift_internal"
CALIBRATION_MODES: tuple[str, ...] = (STATION_MODE, IFT_INTERNAL_MODE)
MODE_ALIASES = {
    "station": STATION_MODE,
    "station_mode": STATION_MODE,
    "ift_internal": IFT_INTERNAL_MODE,
    "ift-internal": IFT_INTERNAL_MODE,
    "ift_internal_mode": IFT_INTERNAL_MODE,
    "C_dx": IFT_INTERNAL_MODE,
    "cdx": IFT_INTERNAL_MODE,
    "layer": IFT_INTERNAL_MODE,
}
CDX_FIXED_BY = (
    "external_geometry",
    "dedicated_calibration",
    "isolation_zero",
)
DEFAULT_CONTRACT_RELATIVE = (
    "outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json"
)
OPERATING_BAND_UM = (1.5, 1.7)
A_STABILITY_MAX_REL_DEVIATION = 0.10
STATUS_VALID = "valid"
STATUS_CONTAMINATED = "cross_level_contaminated"
STATUS_PREREQUISITE_UNMET = "prerequisite_unmet"
STATUS_EXCLUSIVE_VIOLATION = "exclusive_mode_violation"


def normalize_mode(mode: str | None) -> str:
    if mode is None:
        raise ValueError("calibration mode is required")
    key = str(mode).strip()
    if key not in MODE_ALIASES:
        raise ValueError(f"unknown calibration mode {mode!r}; expected station or ift_internal")
    return MODE_ALIASES[key]


def floated_parameters_for_mode(mode: str) -> tuple[str, ...]:
    resolved = normalize_mode(mode)
    if resolved == STATION_MODE:
        return STATION_SOLVE_PARAMETERS
    return (C_DX,)


def assert_exclusive_parameters(mode: str, names: Sequence[str]) -> str:
    """Raise if ``names`` are not exactly the floated set of one exclusive mode."""
    resolved = normalize_mode(mode)
    names = tuple(str(name) for name in names)
    if len(set(names)) != len(names):
        raise ValueError("floated parameter names must be unique")
    has_cdx = C_DX in names
    has_station = any(name in STATION_SOLVE_PARAMETERS for name in names)
    if has_cdx and has_station:
        raise ValueError("exclusive calibration modes forbid a joint station+C_dx Newton step")
    expected = floated_parameters_for_mode(resolved)
    if resolved == STATION_MODE:
        missing_free = [name for name in STATION_FREE_PARAMETERS if name not in names]
        extra = [name for name in names if name not in STATION_SOLVE_PARAMETERS]
        if missing_free or extra:
            raise ValueError(
                "Station Mode floats only station dx/dy/rx/ry/rz with optional survey dz; "
                "got " + ", ".join(names)
            )
        if SURVEY_PARAMETER not in names:
            raise ValueError("Station Mode requires survey dz in the solve (5 mm prior; written dz stays 0)")
        return resolved
    if names != expected:
        raise ValueError("IFT-Internal Mode floats only C_dx; got " + ", ".join(names))
    return resolved


def implied_unmodeled_cdx_um(*, station_dx_mm: float, A_dx: float) -> float:
    """``C_dx = dx / A_dx`` from an omitted-variable station dx shift."""
    if not math.isfinite(float(A_dx)) or float(A_dx) == 0.0:
        raise ValueError("A_dx must be finite and non-zero")
    return 1.0e3 * float(station_dx_mm) / float(A_dx)


def _require_mapping(payload: Mapping[str, Any], key: str) -> dict[str, Any]:
    value = payload.get(key)
    if not isinstance(value, Mapping):
        raise ValueError(f"mode-validity contract lacks {key!r}")
    return dict(value)


def validate_mode_validity_contract(payload: Mapping[str, Any], path: Path | None = None) -> dict[str, Any]:
    where = "" if path is None else f" ({path})"
    if payload.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"not a calibration mode-validity contract{where}")
    if payload.get("validation_used_in_registration") is not False:
        raise ValueError("mode-validity contract must declare validation_used_in_registration=false")
    if payload.get("test_data_accessed") is not False:
        raise ValueError("mode-validity contract must not access test data")
    if payload.get("hierarchical_v1_rescue_closed") is not True:
        raise ValueError("mode-validity contract must declare hierarchical_v1_rescue_closed=true")
    if payload.get("schur_projection_production") is not False:
        raise ValueError("Schur projection is not a production estimator")
    modes = payload.get("modes")
    if list(modes) != list(CALIBRATION_MODES):
        raise ValueError("contract modes must be exactly [station, ift_internal]")
    station = _require_mapping(payload, "station_mode")
    internal = _require_mapping(payload, "ift_internal_mode")
    leakage = _require_mapping(payload, "leakage_operator")
    a_map = leakage.get("A_native_per_mm_C_dx")
    if not isinstance(a_map, Mapping) or "ift_dx_mm" not in a_map:
        raise ValueError("contract leakage_operator lacks A_native_per_mm_C_dx.ift_dx_mm")
    if abs(float(a_map["ift_dx_mm"])) < 50.0:
        raise ValueError("frozen A_dx is not the workbook-45 production operator")
    budget = _require_mapping(station, "unmodeled_C_dx")
    for key in ("statistical_max_abs_um", "engineering_max_abs_um", "A_dx"):
        if key not in budget:
            raise ValueError(f"station_mode.unmodeled_C_dx lacks {key}")
    if not isinstance(internal.get("station_to_C_dx_propagation"), Mapping):
        raise ValueError("ift_internal_mode lacks station_to_C_dx_propagation")
    if payload.get("residual_reduction_is_not_alignment_success") is not True:
        raise ValueError("contract must forbid treating residual reduction as alignment success")
    return dict(payload)


def load_mode_validity_contract(path: Path) -> dict[str, Any]:
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"expected a JSON object at {path}")
    return validate_mode_validity_contract(payload, path)


def build_mode_validity_contract(
    *,
    leakage_operator: Mapping[str, Any],
    reverse_B_native: Mapping[str, float],
    station_capture_path: Path,
    cdx_capture_path: Path,
    station_registered_sigma: Mapping[str, float],
    cdx_registered_sigma_mm: float,
    statistical_max_abs_C_dx_um: float,
    engineering_max_abs_C_dx_um: float,
    A_stability: Mapping[str, Any],
    provenance: Mapping[str, Any],
    created_utc: str,
) -> dict[str, Any]:
    """Assemble the frozen train-only mode-validity contract."""
    a_map = {str(name): float(value) for name, value in dict(leakage_operator["A_native_per_mm_C_dx"]).items()}
    b_map = {str(name): float(value) for name, value in reverse_B_native.items()}
    include = list(STATION_FREE_PARAMETERS)
    systematic = cdx_systematic_from_station_sigma(b_map, station_registered_sigma, include=include)
    dz_sigma = station_registered_sigma.get(SURVEY_PARAMETER)
    dz_term = None if dz_sigma is None else abs(b_map[SURVEY_PARAMETER]) * abs(float(dz_sigma))
    payload: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "created_utc": created_utc,
        "validation_used_in_registration": False,
        "test_data_accessed": False,
        "hierarchical_v1_rescue_closed": True,
        "schur_projection_production": False,
        "joint_newton_forbidden": True,
        "same_data_stage_iteration_forbidden": True,
        "other_level_as_nuisance_forbidden": True,
        "new_layer_or_module_dof_forbidden": True,
        "residual_reduction_is_not_alignment_success": True,
        "q_over_p_mode": 0,
        "modes": list(CALIBRATION_MODES),
        "leakage_operator": {
            "source": "workbook_45_train_route_selected_iteration_00_start_production_A6",
            "A_native_per_mm_C_dx": a_map,
            "disguised_um_or_urad_per_um_C_dx": {
                name: float(leakage_operator["disguised_um_or_urad_per_um_C_dx"][name]) for name in a_map
            },
            "subspace_r2": float(leakage_operator["subspace_r2"]),
            "subspace_cosine": float(leakage_operator["subspace_cosine"]),
            "normal_cc_native": float(leakage_operator["normal_cc_native"]),
            "B_native_C_dx_per_station_unit": b_map,
            "used_prior_sigma_native": dict(leakage_operator.get("used_prior_sigma_native") or {}),
            "do_not_retune": True,
        },
        "A_stability": dict(A_stability),
        "station_mode": {
            "floated_parameters": list(STATION_SOLVE_PARAMETERS),
            "survey_prior_sigma": {SURVEY_PARAMETER: 5.0},
            "written_dz": 0.0,
            "capture_criteria": str(station_capture_path),
            "prerequisite": {
                "C_dx_fixed_by": list(CDX_FIXED_BY),
                "dedicated_calibration_must_still_meet_unmodeled_budget": True,
                "registered_C_dx_sigma_um_does_not_satisfy_budget": True,
                "registered_C_dx_sigma_um": 1.0e3 * float(cdx_registered_sigma_mm),
            },
            "unmodeled_C_dx": {
                "A_dx": float(a_map["ift_dx_mm"]),
                "statistical_max_abs_um": float(statistical_max_abs_C_dx_um),
                "engineering_max_abs_um": float(engineering_max_abs_C_dx_um),
                "operating_band_um": list(OPERATING_BAND_UM),
                "binding_gate": "statistical",
                "binding_name": "ift_dx_mm",
                "if_exceeded": STATUS_CONTAMINATED,
            },
            "geometry_write": {
                "station_five_dof": True,
                "written_dz": 0.0,
                "layers_unchanged": True,
                "does_not_write_C_dx": True,
            },
        },
        "ift_internal_mode": {
            "floated_parameters": [C_DX],
            "payload_expansion": "L0=(+C_dx,0), L1=0, L2=(-C_dx,0); C_rx identically 0",
            "capture_criteria": str(cdx_capture_path),
            "capture_row": C_DX,
            "prerequisite": {
                "station_six_vector_closed": True,
                "station_capture_criteria": str(station_capture_path),
                "workbook": 36,
                "same_data_stage_as_station_mode_forbidden": True,
            },
            "station_to_C_dx_propagation": {
                "B_native_C_dx_per_station_unit": b_map,
                "registered_station_sigma": {name: float(station_registered_sigma[name]) for name in STATION_SIX},
                "free_station_rss": systematic,
                "survey_dz_1sigma_mm": dz_term,
                "survey_dz_1sigma_um": None if dz_term is None else 1.0e3 * dz_term,
                "note": (
                    "Record this RSS as a C_dx systematic.  Do not float station as a "
                    "nuisance and do not retune registered sigma(C_dx)."
                ),
            },
            "geometry_write": {
                "layer_transforms": "L0=+C_dx, L1=0, L2=-C_dx",
                "station_six_vector_unchanged": True,
                "does_not_write_station": True,
            },
        },
        "reject_indicators": [
            "unmodeled_C_dx_exceeds_statistical_budget",
            "station_framework_capture_not_passed",
            "same_data_stage_cross_mode_iteration",
            "other_level_as_nuisance",
            "joint_newton",
            "schur_projection_used",
            "residual_reduction_used_as_success",
            "A_unstable",
            "C_dx_not_declared_fixed",
        ],
        "provenance": dict(provenance),
    }
    return validate_mode_validity_contract(payload)


def evaluate_A_stability(
    contract: Mapping[str, Any],
    *,
    measured_A_dx: float,
    measured_A_ry: float | None = None,
) -> dict[str, Any]:
    frozen_a = contract["leakage_operator"]["A_native_per_mm_C_dx"]
    frozen_dx = float(frozen_a["ift_dx_mm"])
    max_rel = float(contract["A_stability"].get("transfer_max_rel_deviation", A_STABILITY_MAX_REL_DEVIATION))
    rel_dx = abs(float(measured_A_dx) - frozen_dx) / abs(frozen_dx)
    rel_ry = None
    if measured_A_ry is not None:
        frozen_ry = float(frozen_a["ift_ry_mrad"])
        rel_ry = abs(float(measured_A_ry) - frozen_ry) / abs(frozen_ry)
    stable = bool(rel_dx <= max_rel and (rel_ry is None or rel_ry <= max_rel))
    return {
        "frozen_A_dx": frozen_dx,
        "measured_A_dx": float(measured_A_dx),
        "rel_deviation_A_dx": rel_dx,
        "frozen_A_ry": float(frozen_a["ift_ry_mrad"]),
        "measured_A_ry": None if measured_A_ry is None else float(measured_A_ry),
        "rel_deviation_A_ry": rel_ry,
        "transfer_max_rel_deviation": max_rel,
        "stable": stable,
        "do_not_retune": True,
        "if_unstable": "do not apply this contract's C_dx budget to the new sample; do not retune A",
    }


def _cdx_budget_flags(contract: Mapping[str, Any], abs_cdx_um: float) -> dict[str, Any]:
    budget = contract["station_mode"]["unmodeled_C_dx"]
    statistical = float(budget["statistical_max_abs_um"])
    engineering = float(budget["engineering_max_abs_um"])
    within_statistical = bool(abs_cdx_um <= statistical)
    within_engineering = bool(abs_cdx_um <= engineering)
    return {
        "abs_unmodeled_C_dx_um": float(abs_cdx_um),
        "statistical_max_abs_um": statistical,
        "engineering_max_abs_um": engineering,
        "operating_band_um": list(budget["operating_band_um"]),
        "within_statistical_budget": within_statistical,
        "within_engineering_budget": within_engineering,
        "within_operating_band": bool(abs_cdx_um <= float(budget["operating_band_um"][1])),
        "cross_level_contaminated": not within_statistical,
    }


def evaluate_station_mode_validity(
    contract: Mapping[str, Any],
    *,
    floated_parameters: Sequence[str],
    unmodeled_abs_C_dx_um: float | None,
    cdx_fixed_by: str | None,
    same_data_stage_as_other_mode: bool = False,
    other_level_as_nuisance: bool = False,
    schur_projection_used: bool = False,
    residual_reduction_used_as_success: bool = False,
    implied_C_dx_um_from_station_dx: float | None = None,
) -> dict[str, Any]:
    """Score a Station Mode run against the frozen leakage budget."""
    indicators: list[str] = []
    exclusive_ok = True
    try:
        assert_exclusive_parameters(STATION_MODE, floated_parameters)
    except ValueError:
        exclusive_ok = False
        indicators.append("exclusive_mode_violation")
    if same_data_stage_as_other_mode:
        indicators.append("same_data_stage_cross_mode_iteration")
    if other_level_as_nuisance:
        indicators.append("other_level_as_nuisance")
    if schur_projection_used:
        indicators.append("schur_projection_used")
    if residual_reduction_used_as_success:
        indicators.append("residual_reduction_used_as_success")
    if cdx_fixed_by is not None and cdx_fixed_by not in CDX_FIXED_BY:
        raise ValueError(f"unknown cdx_fixed_by {cdx_fixed_by!r}")
    if unmodeled_abs_C_dx_um is None or cdx_fixed_by is None:
        indicators.append("C_dx_not_declared_fixed")
        budget_flags = None
        contaminated = True
        status = STATUS_PREREQUISITE_UNMET
    else:
        budget_flags = _cdx_budget_flags(contract, abs(float(unmodeled_abs_C_dx_um)))
        if budget_flags["cross_level_contaminated"]:
            indicators.append("unmodeled_C_dx_exceeds_statistical_budget")
        if not budget_flags["within_engineering_budget"]:
            indicators.append("unmodeled_C_dx_exceeds_engineering_budget")
        contaminated = bool(budget_flags["cross_level_contaminated"])
        status = STATUS_CONTAMINATED if contaminated else STATUS_VALID
    if implied_C_dx_um_from_station_dx is not None:
        implied_flags = _cdx_budget_flags(contract, abs(float(implied_C_dx_um_from_station_dx)))
        if implied_flags["cross_level_contaminated"]:
            indicators.append("implied_C_dx_from_station_dx_exceeds_budget")
            contaminated = True
            if status == STATUS_VALID:
                status = STATUS_CONTAMINATED
    else:
        implied_flags = None
    if not exclusive_ok or same_data_stage_as_other_mode or other_level_as_nuisance or schur_projection_used:
        status = STATUS_EXCLUSIVE_VIOLATION
        contaminated = True
    if residual_reduction_used_as_success:
        contaminated = True
        if status == STATUS_VALID:
            status = STATUS_CONTAMINATED
    unique_indicators = list(dict.fromkeys(indicators))
    valid = status == STATUS_VALID and not unique_indicators
    return {
        "calibration_mode": STATION_MODE,
        "status": status if not valid else STATUS_VALID,
        "cross_level_contaminated": contaminated or not valid,
        "geometry_write_allowed": bool(valid),
        "cdx_fixed_by": cdx_fixed_by,
        "unmodeled_C_dx": budget_flags,
        "implied_C_dx_from_station_dx": implied_flags,
        "reject_indicators": unique_indicators,
        "residual_reduction_is_not_alignment_success": True,
        "A_dx": float(contract["station_mode"]["unmodeled_C_dx"]["A_dx"]),
    }


def evaluate_ift_internal_mode_validity(
    contract: Mapping[str, Any],
    *,
    floated_parameters: Sequence[str],
    station_framework_capture_success: bool | None,
    same_data_stage_as_other_mode: bool = False,
    other_level_as_nuisance: bool = False,
    schur_projection_used: bool = False,
    residual_reduction_used_as_success: bool = False,
) -> dict[str, Any]:
    """Score a dedicated C_dx Mode run.  Station leftover is a recorded systematic, not a nuisance."""
    indicators: list[str] = []
    exclusive_ok = True
    try:
        assert_exclusive_parameters(IFT_INTERNAL_MODE, floated_parameters)
    except ValueError:
        exclusive_ok = False
        indicators.append("exclusive_mode_violation")
    if same_data_stage_as_other_mode:
        indicators.append("same_data_stage_cross_mode_iteration")
    if other_level_as_nuisance:
        indicators.append("other_level_as_nuisance")
    if schur_projection_used:
        indicators.append("schur_projection_used")
    if residual_reduction_used_as_success:
        indicators.append("residual_reduction_used_as_success")
    if station_framework_capture_success is not True:
        indicators.append("station_framework_capture_not_passed")
    propagation = dict(contract["ift_internal_mode"]["station_to_C_dx_propagation"])
    contaminated = bool(indicators)
    if not exclusive_ok or same_data_stage_as_other_mode or other_level_as_nuisance or schur_projection_used:
        status = STATUS_EXCLUSIVE_VIOLATION
    elif station_framework_capture_success is not True:
        status = STATUS_PREREQUISITE_UNMET
    elif residual_reduction_used_as_success:
        status = STATUS_CONTAMINATED
    else:
        status = STATUS_VALID
    valid = status == STATUS_VALID and not indicators
    return {
        "calibration_mode": IFT_INTERNAL_MODE,
        "status": status if not valid else STATUS_VALID,
        "cross_level_contaminated": contaminated or not valid,
        "geometry_write_allowed": bool(valid),
        "station_framework_capture_success": station_framework_capture_success,
        "station_to_C_dx_propagation": {
            "rss_1sigma_um": propagation["free_station_rss"]["rss_1sigma_um"],
            "rss_3sigma_um": propagation["free_station_rss"]["rss_3sigma_um"],
            "survey_dz_1sigma_um": propagation.get("survey_dz_1sigma_um"),
            "note": propagation.get("note"),
        },
        "reject_indicators": list(dict.fromkeys(indicators)),
        "residual_reduction_is_not_alignment_success": True,
    }


def read_station_framework_capture_success(path: Path) -> bool:
    """Read workbook-36 framework capture from a station-mode artifact."""
    with path.open(encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, Mapping):
        raise ValueError(f"station capture artifact is not a JSON object: {path}")
    aggregate = payload.get("capture_aggregate")
    if isinstance(aggregate, Mapping) and "framework_capture_success" in aggregate:
        return bool(aggregate["framework_capture_success"])
    if isinstance(aggregate, Mapping) and "capture_success" in aggregate:
        return bool(aggregate["capture_success"])
    if "framework_capture_success" in payload:
        return bool(payload["framework_capture_success"])
    if "capture_success" in payload:
        return bool(payload["capture_success"])
    raise ValueError(f"station capture artifact has no framework_capture_success: {path}")


def evaluate_mode_validity(
    contract: Mapping[str, Any],
    *,
    mode: str,
    floated_parameters: Sequence[str],
    unmodeled_abs_C_dx_um: float | None = None,
    cdx_fixed_by: str | None = None,
    station_framework_capture_success: bool | None = None,
    same_data_stage_as_other_mode: bool = False,
    other_level_as_nuisance: bool = False,
    schur_projection_used: bool = False,
    residual_reduction_used_as_success: bool = False,
    implied_C_dx_um_from_station_dx: float | None = None,
) -> dict[str, Any]:
    resolved = normalize_mode(mode)
    if resolved == STATION_MODE:
        return evaluate_station_mode_validity(
            contract,
            floated_parameters=floated_parameters,
            unmodeled_abs_C_dx_um=unmodeled_abs_C_dx_um,
            cdx_fixed_by=cdx_fixed_by,
            same_data_stage_as_other_mode=same_data_stage_as_other_mode,
            other_level_as_nuisance=other_level_as_nuisance,
            schur_projection_used=schur_projection_used,
            residual_reduction_used_as_success=residual_reduction_used_as_success,
            implied_C_dx_um_from_station_dx=implied_C_dx_um_from_station_dx,
        )
    return evaluate_ift_internal_mode_validity(
        contract,
        floated_parameters=floated_parameters,
        station_framework_capture_success=station_framework_capture_success,
        same_data_stage_as_other_mode=same_data_stage_as_other_mode,
        other_level_as_nuisance=other_level_as_nuisance,
        schur_projection_used=schur_projection_used,
        residual_reduction_used_as_success=residual_reduction_used_as_success,
    )
