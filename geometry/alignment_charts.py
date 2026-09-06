"""Explicit SE(3) alignment charts.  Charts are not implicitly converted.

Stations payload chart:
    active left-multiply, G = T * Rz * Ry * Rx, pivot = FASER origin,
    native units mm / rad.

Legacy cluster-local Ry is a different chart: opposite Ry sign, pivot at
``(0, 0, station_z_mm)``.  It must not be copied as Stations ``ry``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence

import numpy as np


class ChartError(ValueError):
    """Raised when a chart mix or Euler singularity is requested."""


@dataclass(frozen=True)
class AlignmentChart:
    name: str
    active: bool
    composition: str
    pivot: str
    translation_unit: str
    rotation_unit: str
    ry_sign: float


STATIONS_GLOBAL_ORIGIN = AlignmentChart(
    name="stations_global_origin_TRzRyRx",
    active=True,
    composition="T*Rz*Ry*Rx",
    pivot="global_origin",
    translation_unit="mm",
    rotation_unit="rad",
    ry_sign=1.0,
)

LEGACY_CLUSTER_LOCAL_STATION_Z_RY = AlignmentChart(
    name="legacy_cluster_local_station_z_ry_opposite_sign",
    active=True,
    composition="Ry_about_station_z",
    pivot="station_z",
    translation_unit="mm",
    rotation_unit="rad",
    ry_sign=-1.0,
)

MM_PER_M = 1000.0
MRAD_PER_RAD = 1000.0
ROUND_TRIP_TOLERANCE = 1.0e-10
SINGULARITY_CY_MIN = 1.0e-8


def _as_six(values: Sequence[float]) -> tuple[float, float, float, float, float, float]:
    if len(values) != 6:
        raise ChartError("six-vector must be [dx, dy, dz, rx, ry, rz]")
    numbers = tuple(float(item) for item in values)
    if not all(math.isfinite(item) for item in numbers):
        raise ChartError("six-vector must be finite")
    return numbers  # type: ignore[return-value]


def _rx(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]],
        dtype=np.float64,
    )


def _ry(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, 0.0, s], [0.0, 1.0, 0.0], [-s, 0.0, c]],
        dtype=np.float64,
    )


def _rz(angle: float) -> np.ndarray:
    c, s = math.cos(angle), math.sin(angle)
    return np.array(
        [[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]],
        dtype=np.float64,
    )


def _hom(rotation: np.ndarray, translation: Sequence[float]) -> np.ndarray:
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = np.asarray(translation, dtype=np.float64)
    return matrix


def se3_from_sixvector(
    values: Sequence[float],
    chart: AlignmentChart,
    *,
    pivot_mm: Sequence[float] | None = None,
) -> np.ndarray:
    """Build a 4×4 transform.  Charts are not silently converted."""
    dx, dy, dz, rx, ry, rz = _as_six(values)
    if chart.translation_unit != "mm" or chart.rotation_unit != "rad":
        raise ChartError("native chart units are mm/rad; convert at the boundary")
    if not bool(chart.active):
        raise ChartError("passive charts are not implemented; invert an active G explicitly")
    if chart is STATIONS_GLOBAL_ORIGIN or chart.name == STATIONS_GLOBAL_ORIGIN.name:
        if chart.composition != "T*Rz*Ry*Rx" or chart.pivot != "global_origin":
            raise ChartError("stations chart contract mismatch")
        rotation = _rz(rz) @ _ry(chart.ry_sign * ry) @ _rx(rx)
        return _hom(rotation, (dx, dy, dz))
    if chart is LEGACY_CLUSTER_LOCAL_STATION_Z_RY or chart.name == LEGACY_CLUSTER_LOCAL_STATION_Z_RY.name:
        if pivot_mm is None:
            raise ChartError("cluster-local Ry requires an explicit station-z pivot")
        if abs(rx) > 0.0 or abs(rz) > 0.0 or abs(dy) > 0.0 or abs(dz) > 0.0:
            raise ChartError("legacy cluster-local chart only carries dx and ry")
        pivot = np.asarray(pivot_mm, dtype=np.float64).reshape(3)
        rotation = _ry(chart.ry_sign * ry)
        translation = np.asarray([dx, 0.0, 0.0], dtype=np.float64) + (np.eye(3) - rotation) @ pivot
        return _hom(rotation, translation)
    raise ChartError(f"unknown chart: {chart.name}")


def sixvector_from_se3(transform: np.ndarray, chart: AlignmentChart) -> np.ndarray:
    """Inverse of ``se3_from_sixvector`` for the stations origin chart only."""
    if chart.name != STATIONS_GLOBAL_ORIGIN.name:
        raise ChartError("only the stations origin chart has a general six-vector inverse")
    matrix = np.asarray(transform, dtype=np.float64)
    if matrix.shape != (4, 4):
        raise ChartError("transform must be 4x4")
    rotation = matrix[:3, :3]
    cosine_y = math.hypot(float(rotation[0, 0]), float(rotation[1, 0]))
    if cosine_y < SINGULARITY_CY_MIN:
        raise ChartError("declared Euler singularity: |Ry| near pi/2")
    rx = math.atan2(float(rotation[2, 1]), float(rotation[2, 2]))
    ry = math.atan2(-float(rotation[2, 0]), cosine_y)
    rz = math.atan2(float(rotation[1, 0]), float(rotation[0, 0]))
    return np.array(
        [matrix[0, 3], matrix[1, 3], matrix[2, 3], rx, ry, rz],
        dtype=np.float64,
    )


def apply_se3(points_mm: np.ndarray, transform: np.ndarray) -> np.ndarray:
    array = np.asarray(points_mm, dtype=np.float64)
    matrix = np.asarray(transform, dtype=np.float64)
    if array.ndim != 2 or array.shape[1] != 3:
        raise ChartError("points must have shape (N, 3)")
    homogeneous = np.concatenate(
        [array, np.ones((array.shape[0], 1), dtype=np.float64)], axis=1
    )
    return (homogeneous @ matrix.T)[:, :3]


def origin_translation_from_center(
    translation_center_mm: Sequence[float],
    rotation: np.ndarray,
    center_mm: Sequence[float],
) -> np.ndarray:
    """t_origin = t_center + (I − R) c."""
    t_center = np.asarray(translation_center_mm, dtype=np.float64).reshape(3)
    center = np.asarray(center_mm, dtype=np.float64).reshape(3)
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return t_center + (np.eye(3, dtype=np.float64) - rot) @ center


def center_translation_from_origin(
    translation_origin_mm: Sequence[float],
    rotation: np.ndarray,
    center_mm: Sequence[float],
) -> np.ndarray:
    t_origin = np.asarray(translation_origin_mm, dtype=np.float64).reshape(3)
    center = np.asarray(center_mm, dtype=np.float64).reshape(3)
    rot = np.asarray(rotation, dtype=np.float64).reshape(3, 3)
    return t_origin - (np.eye(3, dtype=np.float64) - rot) @ center


def compose_left_increment(base: np.ndarray, increment: np.ndarray) -> np.ndarray:
    """Left-composed increment: G_new = G_inc @ G_base.  Not Euler-angle addition."""
    return np.asarray(increment, dtype=np.float64) @ np.asarray(base, dtype=np.float64)


def cross_product_matrix(vector: Sequence[float]) -> np.ndarray:
    x, y, z = (float(item) for item in vector)
    return np.array([[0.0, -z, y], [z, 0.0, -x], [-y, x, 0.0]], dtype=np.float64)


def center_to_origin_jacobian_at_identity(center_mm: Sequence[float]) -> np.ndarray:
    """6×6 map (t_center, ω) → (t_origin, ω) at the identity.

    At identity, (I−R)c ≈ [c]× ω, so J = [[I, [c]×], [0, I]].
    """
    jacobian = np.eye(6, dtype=np.float64)
    jacobian[:3, 3:] = cross_product_matrix(center_mm)
    return jacobian


def transport_center_covariance_to_origin(
    covariance: np.ndarray,
    center_mm: Sequence[float],
) -> np.ndarray:
    jacobian = center_to_origin_jacobian_at_identity(center_mm)
    cov = np.asarray(covariance, dtype=np.float64)
    if cov.shape != (6, 6):
        raise ChartError("center-pivot covariance must be 6x6")
    return jacobian @ cov @ jacobian.T


def native_to_report_mm_mrad(values: Sequence[float]) -> np.ndarray:
    dx, dy, dz, rx, ry, rz = _as_six(values)
    return np.array(
        [dx, dy, dz, rx * MRAD_PER_RAD, ry * MRAD_PER_RAD, rz * MRAD_PER_RAD],
        dtype=np.float64,
    )


def report_mm_mrad_to_native(values: Sequence[float]) -> np.ndarray:
    dx, dy, dz, rx_mrad, ry_mrad, rz_mrad = _as_six(values)
    return np.array(
        [dx, dy, dz, rx_mrad / MRAD_PER_RAD, ry_mrad / MRAD_PER_RAD, rz_mrad / MRAD_PER_RAD],
        dtype=np.float64,
    )


def charts_are_compatible(left: AlignmentChart, right: AlignmentChart) -> bool:
    return (
        left.name == right.name
        and left.active == right.active
        and left.composition == right.composition
        and left.pivot == right.pivot
        and left.ry_sign == right.ry_sign
    )
