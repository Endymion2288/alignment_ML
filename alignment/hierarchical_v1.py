"""FASER exclusive calibration-mode remaining bookkeeping.

Hierarchical alignment V1 is closed as a joint hierarchy (workbook 45).
These helpers still describe leftover after a *single* exclusive mode step
for historical artifacts.  They are not a production sequential hierarchy:
Station Mode and IFT-Internal Mode must not iterate on the same data stage
and must not treat the other level as a nuisance column.

Production floated sets:

* Station Mode: ``dx/dy/rx/ry/rz`` with survey-constrained ``dz`` (5 mm prior;
  written ``dz`` stays 0)
* IFT-Internal Mode: 1-D ``C_dx=(dx_L0-dx_L2)/2`` with ``L0=+C_dx``, ``L1=0``,
  ``L2=-C_dx``
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

from alignment.five_dof_sampling import FREE_PARAMETERS as STATION_FREE_PARAMETERS
from alignment.five_dof_sampling import SURVEY_PARAMETER, five_dof_severity
from alignment.layer_hierarchy import contrast_layer_six_vectors


C_DX = "C_dx"
C_DX_ENVELOPE_MM = 0.12
STATION_SURVEY_PARAMETER = SURVEY_PARAMETER
STATION_SOLVE_PARAMETERS: tuple[str, ...] = STATION_FREE_PARAMETERS + (STATION_SURVEY_PARAMETER,)
HIERARCHICAL_V1_PARAMETERS: tuple[str, ...] = STATION_SOLVE_PARAMETERS + (C_DX,)
STATION_LEVEL = "station"
LAYER_LEVEL = "C_dx"
LEVELS: tuple[str, ...] = (STATION_LEVEL, LAYER_LEVEL)


def complete_hierarchical_values(values: Mapping[str, float]) -> dict[str, float]:
    missing = [name for name in HIERARCHICAL_V1_PARAMETERS if name not in values]
    if missing:
        raise ValueError("hierarchical V1 values missing " + ", ".join(missing))
    completed = {name: float(values[name]) for name in HIERARCHICAL_V1_PARAMETERS}
    extra = [name for name in values if name not in completed]
    if extra:
        raise ValueError("hierarchical V1 values have unknown parameter(s): " + ", ".join(sorted(extra)))
    if any(not math.isfinite(value) for value in completed.values()):
        raise ValueError("hierarchical V1 values must be finite")
    return completed


def remaining_after_station_step(
    *,
    injected: Mapping[str, float],
    recovered_station: Mapping[str, float],
) -> dict[str, float]:
    """Keep true ``C_dx``; subtract only the floated station 5-DoF.

    Written ``dz`` is identically 0 even if the solve returned a noisy
    survey posterior.  ``C_dx`` is copied from the current geometry.
    """
    current = complete_hierarchical_values(injected)
    missing = [name for name in STATION_FREE_PARAMETERS if name not in recovered_station]
    if missing:
        raise ValueError("recovered station is missing " + ", ".join(missing))
    remaining = dict(current)
    for name in STATION_FREE_PARAMETERS:
        remaining[name] = float(current[name]) - float(recovered_station[name])
    remaining[STATION_SURVEY_PARAMETER] = 0.0
    remaining[C_DX] = float(current[C_DX])
    return remaining


def remaining_after_cdx_step(
    *,
    injected: Mapping[str, float],
    recovered_cdx: float,
) -> dict[str, float]:
    """Keep the already-written station six-vector; subtract only ``C_dx``."""
    current = complete_hierarchical_values(injected)
    remaining = dict(current)
    remaining[C_DX] = float(current[C_DX]) - float(recovered_cdx)
    remaining[STATION_SURVEY_PARAMETER] = 0.0
    return remaining


def remaining_after_level_step(
    *,
    injected: Mapping[str, float],
    recovered: Mapping[str, float],
    floated_level: str,
) -> dict[str, float]:
    if floated_level == STATION_LEVEL:
        return remaining_after_station_step(injected=injected, recovered_station=recovered)
    if floated_level == LAYER_LEVEL:
        if C_DX not in recovered:
            raise ValueError("recovered C_dx step lacks C_dx")
        return remaining_after_cdx_step(injected=injected, recovered_cdx=float(recovered[C_DX]))
    raise ValueError(f"floated_level must be one of {LEVELS}, got {floated_level!r}")


def naive_proposed_next_zeros_unfloated(
    *,
    injected: Mapping[str, float],
    recovered: Mapping[str, float],
    floated_level: str,
) -> dict[str, float]:
    """Reference-linearized ``proposed_next``: unfloated coordinates become 0."""
    current = complete_hierarchical_values(injected)
    proposed = {name: 0.0 for name in HIERARCHICAL_V1_PARAMETERS}
    if floated_level == STATION_LEVEL:
        for name in STATION_FREE_PARAMETERS:
            proposed[name] = float(recovered[name])
        if STATION_SURVEY_PARAMETER in recovered:
            proposed[STATION_SURVEY_PARAMETER] = float(recovered[STATION_SURVEY_PARAMETER])
        proposed[C_DX] = 0.0
        return proposed
    if floated_level == LAYER_LEVEL:
        proposed[C_DX] = float(recovered[C_DX])
        for name in STATION_SOLVE_PARAMETERS:
            proposed[name] = 0.0
        return proposed
    del current
    raise ValueError(f"floated_level must be one of {LEVELS}, got {floated_level!r}")


def station_absorption_of_cdx(
    *,
    injected: Mapping[str, float],
    recovered_station: Mapping[str, float],
) -> dict[str, float | None]:
    """Leakage ①: station step absorbing true ``C_dx`` into rigid DoF."""
    current = complete_hierarchical_values(injected)
    errors = {
        name: float(recovered_station[name]) - float(current[name])
        for name in STATION_FREE_PARAMETERS
        if name in recovered_station
    }
    if STATION_SURVEY_PARAMETER in recovered_station:
        errors[STATION_SURVEY_PARAMETER] = (
            float(recovered_station[STATION_SURVEY_PARAMETER]) - float(current[STATION_SURVEY_PARAMETER])
        )
    c_dx = float(current[C_DX])
    dx_error = errors.get("ift_dx_mm")
    ry_error = errors.get("ift_ry_mrad")
    return {
        "injected_C_dx_mm": c_dx,
        "remaining_C_dx_kept_at_injected": c_dx,
        "ift_dx_mm_error": dx_error,
        "ift_ry_mrad_error": ry_error,
        "dx_error_over_injected_C_dx": None if c_dx == 0.0 or dx_error is None else dx_error / c_dx,
        "ry_error_over_injected_C_dx": None if c_dx == 0.0 or ry_error is None else ry_error / c_dx,
        **{f"{name}_error": value for name, value in errors.items()},
    }


def station_payload_stability(
    before: Mapping[str, float],
    after: Mapping[str, float],
) -> dict[str, float]:
    """Leakage ②: layer step must not rewrite the closed station six-vector."""
    first = complete_hierarchical_values(before)
    second = complete_hierarchical_values(after)
    deltas = {
        name: float(second[name]) - float(first[name]) for name in STATION_SOLVE_PARAMETERS
    }
    return {
        **{f"{name}_delta": value for name, value in deltas.items()},
        "max_abs_station_delta": max(abs(value) for value in deltas.values()),
        "station_five_dof_severity_before": five_dof_severity(first),
        "station_five_dof_severity_after": five_dof_severity(second),
    }


def layer_transforms_for_cdx(c_dx_mm: float) -> dict[str, list[float]]:
    """Canonical V1 layer expansion: ``C_rx`` identically 0."""
    return contrast_layer_six_vectors(float(c_dx_mm), 0.0)


def inside_cdx_envelope(c_dx_mm: float, *, envelope: float = C_DX_ENVELOPE_MM) -> bool:
    return math.isfinite(c_dx_mm) and abs(float(c_dx_mm)) <= float(envelope) + 1.0e-12


def is_hierarchical_v1_specs(specs: Sequence[Mapping[str, object]]) -> bool:
    names = [str(spec.get("name", "")) for spec in specs]
    scopes = {str(spec.get("scope", "")) for spec in specs}
    return (
        len(names) == len(HIERARCHICAL_V1_PARAMETERS)
        and set(names) == set(HIERARCHICAL_V1_PARAMETERS)
        and scopes == {"station", "contrast"}
    )
