"""Straight-line state propagation used by the V1 chi-square baseline."""

from __future__ import annotations

import numpy as np


def transition_matrix(delta_z_mm: float) -> np.ndarray:
    """Return F for [x, y, tx, ty] propagation over delta_z_mm."""
    return np.array(
        [
            [1.0, 0.0, delta_z_mm, 0.0],
            [0.0, 1.0, 0.0, delta_z_mm],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def propagate_line(
    state: np.ndarray,
    covariance: np.ndarray,
    delta_z_mm: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Propagate a local tracklet state and covariance in a straight line."""
    matrix = transition_matrix(delta_z_mm)
    propagated_state = matrix @ np.asarray(state, dtype=np.float64)
    propagated_covariance = matrix @ np.asarray(covariance, dtype=np.float64) @ matrix.T
    return propagated_state, propagated_covariance


def mahalanobis_chi2(residual: np.ndarray, covariance: np.ndarray) -> float:
    """Calculate r^T S^-1 r without explicitly inverting S."""
    residual = np.asarray(residual, dtype=np.float64)
    covariance = np.asarray(covariance, dtype=np.float64)
    covariance = 0.5 * (covariance + covariance.T)
    try:
        solution = np.linalg.solve(covariance, residual)
    except np.linalg.LinAlgError as error:
        raise ValueError("candidate covariance is singular") from error
    value = float(residual @ solution)
    if not np.isfinite(value):
        raise ValueError("candidate chi2 is non-finite")
    return value

