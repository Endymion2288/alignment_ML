"""Joint station 5-DoF + IFT ``C_dx`` sampling for hierarchical alignment V1.

Station ``dx/dy/rx/ry/rz`` are drawn on the already-verified 5-DoF severity
sphere.  ``C_dx`` is drawn independently inside ``|C_dx|<=0.12`` mm.
Station ``dz`` is identically 0.  ``C_rx``, relative ry, layer dy/rz,
layer 1, and module parameters stay out.
"""

from __future__ import annotations

import math
from typing import Mapping

import numpy as np

from alignment.five_dof_sampling import (
    LINEAR_SEVERITY_MAX,
    SURVEY_PARAMETER,
    draw_joint_five_dof,
    five_dof_severity,
)
from alignment.hierarchical_v1 import (
    C_DX,
    C_DX_ENVELOPE_MM,
    HIERARCHICAL_V1_PARAMETERS,
    STATION_FREE_PARAMETERS,
)


DEFAULT_SEED = 20260821
DEFAULT_START_SEVERITY = 0.12
DEFAULT_HELD_OUT_SEVERITY = 0.14
DEFAULT_MIN_CDX_FRACTION = 0.40


def draw_independent_cdx(
    rng: np.random.Generator,
    *,
    envelope: float = C_DX_ENVELOPE_MM,
    min_fraction: float = DEFAULT_MIN_CDX_FRACTION,
    sign: float | None = None,
) -> float:
    """Draw one ``C_dx`` with both levels excited, inside the frozen envelope."""
    if not math.isfinite(envelope) or envelope <= 0.0:
        raise ValueError("C_dx envelope must be positive")
    if not math.isfinite(min_fraction) or not 0.0 < min_fraction < 1.0:
        raise ValueError("min_fraction must lie in (0, 1)")
    low = float(min_fraction) * float(envelope)
    magnitude = float(rng.uniform(low, float(envelope)))
    if sign is None:
        sign = 1.0 if float(rng.random()) < 0.5 else -1.0
    resolved_sign = 1.0 if float(sign) >= 0.0 else -1.0
    return resolved_sign * magnitude


def merge_station_and_cdx(station: Mapping[str, float], c_dx: float) -> dict[str, float]:
    values = {name: float(station[name]) for name in STATION_FREE_PARAMETERS}
    values[SURVEY_PARAMETER] = 0.0
    values[C_DX] = float(c_dx)
    if set(values) != set(HIERARCHICAL_V1_PARAMETERS):
        raise RuntimeError("hierarchical V1 sampler produced an incomplete parameter set")
    return values


def sample_hierarchical_v1_points(
    *,
    seed: int = DEFAULT_SEED,
    start_severity: float = DEFAULT_START_SEVERITY,
    held_out_severity: float = DEFAULT_HELD_OUT_SEVERITY,
    envelope: float = C_DX_ENVELOPE_MM,
    min_cdx_fraction: float = DEFAULT_MIN_CDX_FRACTION,
) -> dict[str, object]:
    """One shared-geometry start and one fully held-out joint injection."""
    if start_severity <= 0.0 or start_severity > LINEAR_SEVERITY_MAX:
        raise ValueError(f"start severity must lie in (0, {LINEAR_SEVERITY_MAX}]")
    if held_out_severity <= 0.0 or held_out_severity > LINEAR_SEVERITY_MAX:
        raise ValueError(f"held-out severity must lie in (0, {LINEAR_SEVERITY_MAX}]")
    rng = np.random.default_rng(int(seed))
    start_station = draw_joint_five_dof(rng, severity=float(start_severity))
    start_cdx = draw_independent_cdx(rng, envelope=envelope, min_fraction=min_cdx_fraction)
    start = merge_station_and_cdx(start_station, start_cdx)
    held_out_station = draw_joint_five_dof(rng, severity=float(held_out_severity))
    held_out_cdx = draw_independent_cdx(
        rng,
        envelope=envelope,
        min_fraction=min_cdx_fraction,
        sign=-math.copysign(1.0, start_cdx),
    )
    held_out = merge_station_and_cdx(held_out_station, held_out_cdx)
    if math.isclose(start_cdx, held_out_cdx, rel_tol=0.0, abs_tol=1.0e-12):
        raise RuntimeError("held-out C_dx collided with the start injection")
    return {
        "seed": int(seed),
        "envelope_C_dx_mm": float(envelope),
        "min_cdx_fraction": float(min_cdx_fraction),
        "start_severity": float(start_severity),
        "held_out_severity": float(held_out_severity),
        "start": start,
        "held_out": held_out,
        "start_station_five_dof_severity": five_dof_severity(start),
        "held_out_station_five_dof_severity": five_dof_severity(held_out),
        "start_C_dx_mm": float(start[C_DX]),
        "held_out_C_dx_mm": float(held_out[C_DX]),
        "station_dz_forced_zero": True,
        "C_rx_forced_zero": True,
        "validation_used_in_registration": False,
        "test_data_accessed": False,
    }
