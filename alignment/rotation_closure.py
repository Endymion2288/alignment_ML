"""Finite-difference physical closure for one station-level IFT ``R_y``."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class RotationClosureFit:
    """One-parameter WLS result in mrad using physical residual responses."""

    recovered_ry_mrad: float
    variance_mrad2: float
    normal_matrix: np.ndarray
    right_hand_side: np.ndarray
    response_chi2: float
    response_ndof: int
    used_pairs: int
    finite_difference_step_mrad: float


@dataclass(frozen=True)
class RotationScanClosureFit:
    """Non-linear one-parameter fit calibrated from multiple true payloads."""

    recovered_ry_mrad: float
    variance_mrad2: float | None
    response_chi2: float
    response_ndof: int
    used_pairs: int
    polynomial_degree: int
    calibration_ry_mrad: np.ndarray
    search_interval_mrad: tuple[float, float]


def _inverse_covariances(covariance: np.ndarray, *, label: str) -> list[np.ndarray]:
    weights = np.asarray(covariance, dtype=np.float64)
    if weights.ndim != 3 or weights.shape[1:] != (4, 4):
        raise ValueError(f"{label} covariance must have shape [pairs, 4, 4]")
    result: list[np.ndarray] = []
    for matrix in weights:
        if not np.allclose(matrix, matrix.T, rtol=1.0e-7, atol=1.0e-12):
            raise ValueError(f"{label} covariance is not symmetric")
        try:
            inverse = np.linalg.inv(matrix)
        except np.linalg.LinAlgError as error:
            raise ValueError(f"{label} covariance is singular") from error
        if not np.isfinite(inverse).all():
            raise ValueError(f"{label} inverse covariance is non-finite")
        result.append(inverse)
    return result


def solve_ift_ry_finite_difference(
    nominal_residual: np.ndarray,
    positive_residual: np.ndarray,
    negative_residual: np.ndarray,
    observed_residual: np.ndarray,
    covariance: np.ndarray,
    *,
    positive_ry_mrad: float,
    negative_ry_mrad: float,
    prior_sigma_mrad: float | None = None,
) -> RotationClosureFit:
    """Fit IFT ``R_y`` from true refit response at two physical probe points.

    ``positive_residual`` and ``negative_residual`` are separately refitted
    mode-0 Acts outputs, not coordinate shifts.  The nominal covariance is a
    deterministic WLS weight for the paired response comparison; it is not
    treated as an independent covariance of the difference between refits of
    the same clusters.
    """
    arrays = [
        np.asarray(values, dtype=np.float64)
        for values in (nominal_residual, positive_residual, negative_residual, observed_residual)
    ]
    if any(values.ndim != 2 or values.shape[1] != 4 for values in arrays):
        raise ValueError("all residual inputs must have shape [pairs, 4]")
    size = arrays[0].shape[0]
    if any(values.shape[0] != size for values in arrays[1:]):
        raise ValueError("residual response arrays have inconsistent pair counts")
    weights = np.asarray(covariance, dtype=np.float64)
    if weights.shape != (size, 4, 4):
        raise ValueError("covariance must have shape [pairs, 4, 4]")
    if size < 1 or not all(np.isfinite(values).all() for values in arrays) or not np.isfinite(weights).all():
        raise ValueError("physical rotation closure requires finite non-empty inputs")
    if not np.isfinite(positive_ry_mrad) or not np.isfinite(negative_ry_mrad):
        raise ValueError("finite-difference R_y probes must be finite")
    denominator = float(positive_ry_mrad - negative_ry_mrad)
    if denominator == 0.0:
        raise ValueError("positive and negative R_y probes must differ")
    if prior_sigma_mrad is not None and (
        not np.isfinite(prior_sigma_mrad) or prior_sigma_mrad <= 0.0
    ):
        raise ValueError("prior_sigma_mrad must be finite and positive when supplied")

    nominal, positive, negative, observed = arrays
    derivative = (positive - negative) / denominator
    response = observed - nominal
    normal = 0.0
    rhs = 0.0
    inverse_covariances = _inverse_covariances(weights, label="physical rotation closure")
    for row, inverse in enumerate(inverse_covariances):
        normal += float(derivative[row] @ inverse @ derivative[row])
        rhs += float(derivative[row] @ inverse @ response[row])
    if prior_sigma_mrad is not None:
        normal += 1.0 / float(prior_sigma_mrad) ** 2
    if not np.isfinite(normal) or normal <= 0.0:
        raise ValueError("IFT R_y normal matrix is rank deficient")
    recovered = rhs / normal
    response_chi2 = 0.0
    for row, inverse in enumerate(inverse_covariances):
        difference = response[row] - derivative[row] * recovered
        response_chi2 += float(difference @ inverse @ difference)
    return RotationClosureFit(
        recovered_ry_mrad=float(recovered),
        variance_mrad2=float(1.0 / normal),
        normal_matrix=np.asarray([[normal]], dtype=np.float64),
        right_hand_side=np.asarray([rhs], dtype=np.float64),
        response_chi2=float(response_chi2),
        response_ndof=int(4 * size - 1),
        used_pairs=size,
        finite_difference_step_mrad=float(abs(denominator) / 2.0),
    )


def _golden_section_minimum(
    objective,
    left: float,
    right: float,
    *,
    tolerance_mrad: float,
) -> tuple[float, float]:
    """Deterministic bounded scalar minimization without a SciPy dependency."""
    ratio = (np.sqrt(5.0) - 1.0) / 2.0
    a = float(left)
    b = float(right)
    c = b - ratio * (b - a)
    d = a + ratio * (b - a)
    fc = float(objective(c))
    fd = float(objective(d))
    while b - a > tolerance_mrad:
        if fc <= fd:
            b, d, fd = d, c, fc
            c = b - ratio * (b - a)
            fc = float(objective(c))
        else:
            a, c, fc = c, d, fd
            d = a + ratio * (b - a)
            fd = float(objective(d))
    value = (a + b) / 2.0
    return value, float(objective(value))


def solve_ift_ry_polynomial_response(
    calibration_ry_mrad: np.ndarray,
    calibration_residual: np.ndarray,
    observed_residual: np.ndarray,
    covariance: np.ndarray,
    *,
    polynomial_degree: int,
    search_interval_mrad: tuple[float, float],
) -> RotationScanClosureFit:
    """Recover IFT ``R_y`` from a held-out physical response curve.

    Every calibration row is a real SCT-cluster -> segment-refit -> Acts
    result at its own alignment payload.  The caller must withhold the
    observed payload from ``calibration_ry_mrad``.  A low-order polynomial is
    only an interpolation/extrapolation of this physical response bank; no
    tracklet coordinate or residual-level injection is constructed here.
    """
    angles = np.asarray(calibration_ry_mrad, dtype=np.float64)
    responses = np.asarray(calibration_residual, dtype=np.float64)
    observed = np.asarray(observed_residual, dtype=np.float64)
    weights = np.asarray(covariance, dtype=np.float64)
    if angles.ndim != 1 or angles.size < 3 or not np.isfinite(angles).all():
        raise ValueError("physical R_y response calibration needs at least three finite angles")
    if len(np.unique(angles)) != angles.size:
        raise ValueError("physical R_y response calibration angles must be unique")
    if responses.ndim != 3 or responses.shape[0] != angles.size or responses.shape[2] != 4:
        raise ValueError("calibration residual must have shape [points, pairs, 4]")
    if observed.shape != responses.shape[1:]:
        raise ValueError("observed residual shape disagrees with calibration pairs")
    if weights.shape != (responses.shape[1], 4, 4):
        raise ValueError("physical R_y response covariance has an invalid shape")
    if not np.isfinite(responses).all() or not np.isfinite(observed).all() or not np.isfinite(weights).all():
        raise ValueError("physical R_y response closure requires finite arrays")
    if not np.any(np.isclose(angles, 0.0, rtol=0.0, atol=1.0e-12)):
        raise ValueError("physical R_y response calibration requires a nominal zero payload")
    if polynomial_degree < 1 or polynomial_degree >= angles.size:
        raise ValueError("polynomial degree must be in [1, calibration_points - 1]")
    left, right = (float(search_interval_mrad[0]), float(search_interval_mrad[1]))
    if not np.isfinite(left) or not np.isfinite(right) or left >= right:
        raise ValueError("physical R_y response search interval is invalid")

    nominal_index = int(np.flatnonzero(np.isclose(angles, 0.0, rtol=0.0, atol=1.0e-12))[0])
    response = responses - responses[nominal_index : nominal_index + 1]
    observed_response = observed - responses[nominal_index]
    pair_count = response.shape[1]
    inverse = _inverse_covariances(weights, label="physical R_y response")
    coefficients = np.polynomial.polynomial.polyfit(
        angles,
        response.reshape(angles.size, pair_count * 4),
        deg=polynomial_degree,
    )

    def objective(angle: float) -> float:
        predicted = np.polynomial.polynomial.polyval(float(angle), coefficients).reshape(pair_count, 4)
        delta = observed_response - predicted
        return float(sum(row @ matrix @ row for row, matrix in zip(delta, inverse)))

    # A coarse grid protects the bounded optimizer from a non-convex polynomial
    # extrapolation.  The final golden-section refinement stays inside the
    # best grid cell and is deterministic across hosts.
    grid = np.linspace(left, right, 4001, dtype=np.float64)
    values = np.asarray([objective(float(angle)) for angle in grid], dtype=np.float64)
    best = int(np.argmin(values))
    if best == 0 or best == grid.size - 1:
        recovered = float(grid[best])
        chi2 = float(values[best])
    else:
        recovered, chi2 = _golden_section_minimum(
            objective,
            float(grid[best - 1]),
            float(grid[best + 1]),
            tolerance_mrad=1.0e-5,
        )
    step = max(1.0e-3, (right - left) * 1.0e-5)
    curvature = (objective(recovered + step) - 2.0 * chi2 + objective(recovered - step)) / step**2
    variance = None if not np.isfinite(curvature) or curvature <= 0.0 else float(2.0 / curvature)
    return RotationScanClosureFit(
        recovered_ry_mrad=float(recovered),
        variance_mrad2=variance,
        response_chi2=float(chi2),
        response_ndof=int(pair_count * 4 - 1),
        used_pairs=int(pair_count),
        polynomial_degree=int(polynomial_degree),
        calibration_ry_mrad=np.asarray(sorted(float(value) for value in angles), dtype=np.float64),
        search_interval_mrad=(left, right),
    )
