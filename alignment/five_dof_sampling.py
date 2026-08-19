"""Deterministic joint 5-DoF sampling in the validated linear regime.

Station-0 ``dx/dy/rx/ry/rz`` are drawn jointly on the severity sphere
``sqrt(sum((theta_i / scale_i)^2))``.  ``dz`` is identically zero: it is a
survey-constrained coordinate, not a free alignment parameter, and must not
enter a random initial misalignment.  Severity 0.5–1.0 is rejected because
the 6-DoF pilot showed single-step Newton closure fails there.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np

FREE_PARAMETERS: tuple[str, ...] = (
    "ift_dx_mm",
    "ift_dy_mm",
    "ift_rx_mrad",
    "ift_ry_mrad",
    "ift_rz_mrad",
)
SURVEY_PARAMETER = "ift_dz_mm"
DEFAULT_SCALES: dict[str, float] = {
    "ift_dx_mm": 5.0,
    "ift_dy_mm": 5.0,
    "ift_dz_mm": 5.0,
    "ift_rx_mrad": 60.0,
    "ift_ry_mrad": 60.0,
    "ift_rz_mrad": 60.0,
}
LINEAR_SEVERITY_MAX = 0.15
STRESS_SEVERITY_MAX = 0.21
FORBIDDEN_SINGLE_STEP_SEVERITY = 0.5


def five_dof_severity(values: Mapping[str, float], scales: Mapping[str, float] | None = None) -> float:
    """Normalized L2 severity over the five track-constrained parameters only."""
    resolved = DEFAULT_SCALES if scales is None else scales
    return float(
        math.sqrt(
            sum((float(values[name]) / float(resolved[name])) ** 2 for name in FREE_PARAMETERS)
        )
    )


def draw_joint_five_dof(
    rng: np.random.Generator,
    *,
    severity: float,
    scales: Mapping[str, float] | None = None,
) -> dict[str, float]:
    """Draw one joint 5-DoF misalignment at a fixed normalized severity."""
    if not math.isfinite(severity) or severity < 0.0:
        raise ValueError("severity must be finite and non-negative")
    if severity >= FORBIDDEN_SINGLE_STEP_SEVERITY:
        raise ValueError(
            f"refusing severity {severity}: single-step closure is forbidden at "
            f">={FORBIDDEN_SINGLE_STEP_SEVERITY}"
        )
    resolved = DEFAULT_SCALES if scales is None else scales
    direction = rng.normal(size=len(FREE_PARAMETERS))
    norm = float(np.linalg.norm(direction))
    if not math.isfinite(norm) or norm <= 0.0:
        raise RuntimeError("failed to draw a non-zero 5-DoF direction")
    direction = direction / norm
    values = {
        name: float(direction[index] * severity * float(resolved[name]))
        for index, name in enumerate(FREE_PARAMETERS)
    }
    values[SURVEY_PARAMETER] = 0.0
    return values


def sample_linear_regime_points(
    *,
    seed: int,
    start_severity: float,
    linear_held_out: Sequence[float],
    stress_held_out: Sequence[float],
    scales: Mapping[str, float] | None = None,
) -> dict[str, object]:
    """Frozen, seed-complete set of shared-geometry 5-DoF points.

    All sources in both splits receive this same payload set so a pooled
    alignment update remains physically meaningful.  Train/validation
    disjointness is by original xAOD source, not by geometry.
    """
    if start_severity <= 0.0 or start_severity > LINEAR_SEVERITY_MAX:
        raise ValueError(f"start severity must lie in (0, {LINEAR_SEVERITY_MAX}]")
    for severity in linear_held_out:
        if severity <= 0.0 or severity > LINEAR_SEVERITY_MAX:
            raise ValueError(f"linear held-out severity {severity} exceeds {LINEAR_SEVERITY_MAX}")
    for severity in stress_held_out:
        if severity <= LINEAR_SEVERITY_MAX or severity > STRESS_SEVERITY_MAX:
            raise ValueError(
                f"stress held-out severity {severity} must lie in "
                f"({LINEAR_SEVERITY_MAX}, {STRESS_SEVERITY_MAX}]"
            )
    rng = np.random.default_rng(int(seed))
    start = draw_joint_five_dof(rng, severity=float(start_severity), scales=scales)
    linear = [
        {
            "name": f"linear_{index:02d}",
            "severity": float(severity),
            "alignment_parameter_values": draw_joint_five_dof(rng, severity=float(severity), scales=scales),
        }
        for index, severity in enumerate(linear_held_out)
    ]
    stress = [
        {
            "name": f"stress_{index:02d}",
            "severity": float(severity),
            "alignment_parameter_values": draw_joint_five_dof(rng, severity=float(severity), scales=scales),
        }
        for index, severity in enumerate(stress_held_out)
    ]
    return {
        "seed": int(seed),
        "start_severity": float(start_severity),
        "start": start,
        "linear_held_out": linear,
        "stress_held_out": stress,
        "survey_parameter": SURVEY_PARAMETER,
        "free_parameters": list(FREE_PARAMETERS),
    }
