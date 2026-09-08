"""SE(3) charts, Lie-log pose error, and field/surface co-transformation.

Workbook 82.  Payload six-vectors stay ``T * Rz * Ry * Rx`` in mm / rad,
left-applied as in GeoModel: ``T_new = g * T_nominal``.  Finite pose error
is the se(3) logarithm, never Euler subtraction.  Algebraic
``g_i^{-1} g_j`` invariance is a relative-chart identity, not an observable
gauge of a fixed lab field.
"""

from __future__ import annotations

import math
from typing import Sequence

import numpy as np

from alignment.four_station import (
    IDENTITY_SIX,
    STATION_IDS,
    invert_six_vector,
    matrix_to_six_vector,
    six_vector_to_matrix,
)
from alignment.gauge_equivalence import calypso_alignment_matrix, transform_point


SOLVER_CONTRACT = "common_track_alignment_v1"
UPDATE_LEFT_SE3 = "left_se3"
UPDATE_RIGHT_SE3 = "right_se3"
PAYLOAD_UNITS = ("mm", "mm", "mm", "rad", "rad", "rad")
_SE3_ATOL = 1.0e-12


def _hat3(omega: Sequence[float]) -> np.ndarray:
    wx, wy, wz = (float(omega[0]), float(omega[1]), float(omega[2]))
    return np.asarray([[0.0, -wz, wy], [wz, 0.0, -wx], [-wy, wx, 0.0]], dtype=np.float64)


def se3_exp(twist: Sequence[float]) -> np.ndarray:
    """Exponential of an se(3) twist ``(vx, vy, vz, wx, wy, wz)`` in mm / rad."""
    values = np.asarray(twist, dtype=np.float64)
    if values.shape != (6,) or not np.isfinite(values).all():
        raise ValueError("se(3) twist must be a finite 6-vector")
    rho = values[:3]
    omega = values[3:]
    angle = float(np.linalg.norm(omega))
    if angle < 1.0e-12:
        rotation = np.eye(3) + _hat3(omega)
        linear = np.eye(3) + 0.5 * _hat3(omega)
    else:
        axis = omega / angle
        skew = _hat3(axis)
        rotation = np.eye(3) + math.sin(angle) * skew + (1.0 - math.cos(angle)) * (skew @ skew)
        linear = (
            np.eye(3)
            + (1.0 - math.cos(angle)) / angle * skew
            + (angle - math.sin(angle)) / angle * (skew @ skew)
        )
    matrix = np.eye(4, dtype=np.float64)
    matrix[:3, :3] = rotation
    matrix[:3, 3] = linear @ rho
    return matrix


def se3_log(matrix: np.ndarray) -> np.ndarray:
    """Inverse of :func:`se3_exp`.  Fail-closed on a non-SE(3) matrix."""
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (4, 4) or not np.isfinite(values).all():
        raise ValueError("SE(3) matrix must be a finite 4x4 array")
    if not np.allclose(values[3], [0.0, 0.0, 0.0, 1.0], rtol=0.0, atol=_SE3_ATOL):
        raise ValueError("SE(3) matrix has a non-homogeneous last row")
    rotation = values[:3, :3]
    if abs(float(np.linalg.det(rotation)) - 1.0) > 1.0e-6:
        raise ValueError("SE(3) rotation has determinant away from 1")
    cosine = float(np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0))
    angle = math.acos(cosine)
    if angle < 1.0e-12:
        omega = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        ) * 0.5
        linear_inv = np.eye(3) - 0.5 * _hat3(omega)
    else:
        axis = np.asarray(
            [
                rotation[2, 1] - rotation[1, 2],
                rotation[0, 2] - rotation[2, 0],
                rotation[1, 0] - rotation[0, 1],
            ],
            dtype=np.float64,
        ) / (2.0 * math.sin(angle))
        omega = axis * angle
        skew = _hat3(axis)
        linear = (
            np.eye(3)
            + (1.0 - math.cos(angle)) / angle * skew
            + (angle - math.sin(angle)) / angle * (skew @ skew)
        )
        linear_inv = np.linalg.inv(linear)
    twist = np.zeros(6, dtype=np.float64)
    twist[:3] = linear_inv @ values[:3, 3]
    twist[3:] = omega
    return twist


def left_update(matrix: np.ndarray, twist: Sequence[float]) -> np.ndarray:
    """Calypso / GeoModel convention: ``T ← Exp(ξ) T``."""
    return se3_exp(twist) @ np.asarray(matrix, dtype=np.float64)


def right_update(matrix: np.ndarray, twist: Sequence[float]) -> np.ndarray:
    """Right trivialization: ``T ← T Exp(ξ)``.  Not the Calypso default."""
    return np.asarray(matrix, dtype=np.float64) @ se3_exp(twist)


def left_update_payload(payload: Sequence[float], twist: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in matrix_to_six_vector(left_update(six_vector_to_matrix(payload), twist)))


def right_update_payload(payload: Sequence[float], twist: Sequence[float]) -> tuple[float, ...]:
    return tuple(float(value) for value in matrix_to_six_vector(right_update(six_vector_to_matrix(payload), twist)))


def pose_error_lie(predicted: Sequence[float], truth: Sequence[float]) -> np.ndarray:
    """``log(T_pred^{-1} T_truth)`` in mm / rad.  Not Euler subtraction."""
    inverse = invert_six_vector(predicted)
    composed = six_vector_to_matrix(inverse) @ six_vector_to_matrix(truth)
    return se3_log(composed)


def euler_subtraction(predicted: Sequence[float], truth: Sequence[float]) -> np.ndarray:
    """Diagnostic only.  Illegal as a finite rotation residual."""
    return np.asarray(truth, dtype=np.float64) - np.asarray(predicted, dtype=np.float64)


def rotate_vector(matrix: np.ndarray, vector: Sequence[float]) -> np.ndarray:
    return np.asarray(matrix, dtype=np.float64)[:3, :3] @ np.asarray(vector, dtype=np.float64)


def transform_field_vector(common: Sequence[float], field: Sequence[float]) -> np.ndarray:
    """Rotate a lab field together with a common left SE(3) coordinate change."""
    return rotate_vector(six_vector_to_matrix(common), field)


def transform_surface_z(common: Sequence[float], z_mm: float) -> float:
    """Move a z=const lab surface with the same common left SE(3) change."""
    return float(transform_point(six_vector_to_matrix(common), (0.0, 0.0, float(z_mm)))[2])


def identity_station_map() -> dict[int, tuple[float, ...]]:
    return {int(station): IDENTITY_SIX for station in STATION_IDS}


def composed_station_matrix(
    payload: Sequence[float],
    nominal_z_mm: float,
) -> np.ndarray:
    """``G = g * T(0,0,z)``.  Measurement frame is this station's local frame."""
    nominal = calypso_alignment_matrix(0.0, 0.0, float(nominal_z_mm), 0.0, 0.0, 0.0)
    return six_vector_to_matrix(payload) @ nominal


def require_update_convention(convention: str) -> str:
    text = str(convention)
    if text not in (UPDATE_LEFT_SE3, UPDATE_RIGHT_SE3):
        raise ValueError(f"unsupported SE(3) update convention: {convention}")
    return text
