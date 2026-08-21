"""Joint 2-D IFT outer-contrast sampling inside the verified linear envelope.

Canonical coordinates are ``C_dx=(dx_L0-dx_L2)/2`` and
``C_rx=(rx_L0-rx_L2)/2``.  The physical payload is always
``L0=(+C_dx,+C_rx)``, ``L1=0``, ``L2=(-C_dx,-C_rx)`` with the station
six-vector identically zero.  Draws are joint on the ellipse
``hypot(C_dx/0.12, C_rx/0.70)``, not two independent axial scans.
Relative ry, layer dy/rz, layer-1, and module parameters stay out.
"""

from __future__ import annotations

import math
from typing import Mapping, Sequence

import numpy as np


CONTRAST_PARAMETERS: tuple[str, ...] = ("C_dx", "C_rx")
CONTRAST_ENVELOPE: dict[str, float] = {
    "C_dx": 0.12,
    "C_rx": 0.70,
}
CONTRAST_UNITS: dict[str, str] = {
    "C_dx": "mm",
    "C_rx": "mrad",
}
DEFAULT_MIN_AXIS_FRACTION = 0.25
DEFAULT_DIRECTION_COSINE_MAX = 0.85


def contrast_radius(
    values: Mapping[str, float],
    envelope: Mapping[str, float] | None = None,
) -> float:
    """Return the ellipse radius ``hypot(C_dx/E_dx, C_rx/E_rx)``."""
    resolved = CONTRAST_ENVELOPE if envelope is None else envelope
    return float(
        math.hypot(
            float(values["C_dx"]) / float(resolved["C_dx"]),
            float(values["C_rx"]) / float(resolved["C_rx"]),
        )
    )


def inside_contrast_envelope(
    values: Mapping[str, float],
    envelope: Mapping[str, float] | None = None,
    *,
    atol: float = 1.0e-12,
) -> bool:
    resolved = CONTRAST_ENVELOPE if envelope is None else envelope
    return bool(
        abs(float(values["C_dx"])) <= float(resolved["C_dx"]) + atol
        and abs(float(values["C_rx"])) <= float(resolved["C_rx"]) + atol
    )


def draw_joint_contrast(
    rng: np.random.Generator,
    *,
    radius: float,
    envelope: Mapping[str, float] | None = None,
    min_axis_fraction: float = DEFAULT_MIN_AXIS_FRACTION,
) -> dict[str, float]:
    """Draw one joint ``(C_dx, C_rx)`` point at a fixed ellipse radius.

    ``min_axis_fraction`` rejects near-axial directions so both already-closed
    internal modes are excited together.  The returned point always lies
    inside the axis-aligned box ``|C| <= envelope``.
    """
    if not math.isfinite(radius) or radius < 0.0:
        raise ValueError("contrast radius must be finite and non-negative")
    if radius > 1.0 + 1.0e-12:
        raise ValueError(f"contrast radius {radius} exceeds the verified linear envelope")
    if not math.isfinite(min_axis_fraction) or not 0.0 < min_axis_fraction < 0.5:
        raise ValueError("min_axis_fraction must lie in (0, 0.5)")
    resolved = CONTRAST_ENVELOPE if envelope is None else dict(envelope)
    for name in CONTRAST_PARAMETERS:
        if name not in resolved or float(resolved[name]) <= 0.0:
            raise ValueError(f"contrast envelope lacks a positive scale for '{name}'")
    for _ in range(10000):
        direction = rng.normal(size=len(CONTRAST_PARAMETERS))
        norm = float(np.linalg.norm(direction))
        if not math.isfinite(norm) or norm <= 0.0:
            continue
        direction = direction / norm
        if min(abs(float(direction[0])), abs(float(direction[1]))) < min_axis_fraction:
            continue
        values = {
            name: float(direction[index] * radius * float(resolved[name]))
            for index, name in enumerate(CONTRAST_PARAMETERS)
        }
        if not inside_contrast_envelope(values, resolved):
            continue
        return values
    raise RuntimeError("failed to draw a joint contrast direction inside the envelope")


def _direction(values: Mapping[str, float], envelope: Mapping[str, float]) -> np.ndarray:
    vector = np.asarray(
        [float(values[name]) / float(envelope[name]) for name in CONTRAST_PARAMETERS],
        dtype=np.float64,
    )
    norm = float(np.linalg.norm(vector))
    if not math.isfinite(norm) or norm <= 0.0:
        raise ValueError("contrast direction is identically zero")
    return vector / norm


def sample_joint_contrast_points(
    *,
    seed: int,
    start_radius: float,
    linear_radii: Sequence[float],
    stress_radii: Sequence[float],
    held_out_radii: Sequence[float],
    envelope: Mapping[str, float] | None = None,
    min_axis_fraction: float = DEFAULT_MIN_AXIS_FRACTION,
    max_direction_cosine: float = DEFAULT_DIRECTION_COSINE_MAX,
) -> dict[str, object]:
    """Frozen, seed-complete set of shared-geometry 2-D contrast points.

    Finite-difference probes stay axial by construction and are compiled
    separately around the all-zero payload.  Train/validation disjointness is
    by original xAOD source; every source receives this same payload set.
    """
    resolved = CONTRAST_ENVELOPE if envelope is None else dict(envelope)
    if start_radius <= 0.0 or start_radius > 1.0:
        raise ValueError("start radius must lie in (0, 1]")
    for radius in linear_radii:
        if radius <= 0.0 or radius > 1.0:
            raise ValueError(f"linear contrast radius {radius} must lie in (0, 1]")
    for radius in stress_radii:
        if radius <= 0.85 or radius > 1.0:
            raise ValueError(f"stress contrast radius {radius} must lie in (0.85, 1]")
    for radius in held_out_radii:
        if radius <= 0.0 or radius > 1.0:
            raise ValueError(f"held-out contrast radius {radius} must lie in (0, 1]")
    rng = np.random.default_rng(int(seed))
    start = draw_joint_contrast(
        rng, radius=float(start_radius), envelope=resolved, min_axis_fraction=min_axis_fraction
    )
    start_direction = _direction(start, resolved)

    def _draw(radius: float) -> dict[str, float]:
        return draw_joint_contrast(
            rng, radius=float(radius), envelope=resolved, min_axis_fraction=min_axis_fraction
        )

    def _draw_held_out(radius: float) -> dict[str, float]:
        for _ in range(10000):
            candidate = _draw(radius)
            direction = _direction(candidate, resolved)
            if abs(float(np.dot(direction, start_direction))) > max_direction_cosine:
                continue
            return candidate
        raise RuntimeError(f"failed to draw a held-out joint contrast distinct from start at radius {radius}")

    linear = [
        {
            "name": f"linear_{index:02d}",
            "radius": float(radius),
            "role": "linear",
            "alignment_parameter_values": _draw(float(radius)),
        }
        for index, radius in enumerate(linear_radii)
    ]
    stress = [
        {
            "name": f"stress_{index:02d}",
            "radius": float(radius),
            "role": "stress",
            "alignment_parameter_values": _draw(float(radius)),
        }
        for index, radius in enumerate(stress_radii)
    ]
    held_out = [
        {
            "name": f"heldout_{index:02d}",
            "radius": float(radius),
            "role": "held_out",
            "reserved_from_fd_and_operating_point": True,
            "alignment_parameter_values": _draw_held_out(float(radius)),
        }
        for index, radius in enumerate(held_out_radii)
    ]
    return {
        "seed": int(seed),
        "envelope": {name: float(resolved[name]) for name in CONTRAST_PARAMETERS},
        "min_axis_fraction": float(min_axis_fraction),
        "max_direction_cosine": float(max_direction_cosine),
        "start_radius": float(start_radius),
        "start": start,
        "linear": linear,
        "stress": stress,
        "held_out": held_out,
        "free_parameters": list(CONTRAST_PARAMETERS),
        "forbidden": [
            "relative_ry",
            "layer_dy",
            "layer_rz",
            "layer1",
            "module",
            "station_six_vector",
            "reference_layer_as_gauge",
        ],
    }
