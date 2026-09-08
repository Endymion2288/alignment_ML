"""Official projected-z and 95% coverage construction for WB85 / future WB86.

Does not change WB85 physical thresholds.  Coverage is the symmetric
Gaussian interval of the same projected estimator that produces ``z``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import stats

from alignment.wb85_physical_qualification_protocol import unit_direction


GAUSSIAN_95_QUANTILE = float(stats.norm.ppf(0.975))
COVERAGE_CONSTRUCTION = "symmetric_gaussian_interval_|z|<=norm.ppf(0.975)"


class ProjectedUnitError(ValueError):
    """Fail-closed projected statistical unit."""


def direction_vector(parameter_names: Sequence[str], direction: Mapping[str, float]) -> np.ndarray:
    unit = unit_direction(direction)
    vector = np.zeros(len(parameter_names), dtype=np.float64)
    for index, name in enumerate(parameter_names):
        if name in unit:
            vector[index] = unit[name]
    norm = float(np.linalg.norm(vector))
    if norm <= 0.0:
        raise ProjectedUnitError("projection has no support on the parameter chart")
    return vector / norm


def official_projected_estimate(
    theta_hat: Sequence[float],
    theta_true: Sequence[float],
    covariance: np.ndarray,
    parameter_names: Sequence[str],
    direction: Mapping[str, float],
) -> dict[str, Any]:
    """``a_hat = u^T theta_hat``, ``sigma_a^2 = u^T Cov u``, ``z = (a_hat-a_true)/sigma_a``."""
    hat = np.asarray(theta_hat, dtype=np.float64)
    true = np.asarray(theta_true, dtype=np.float64)
    cov = np.asarray(covariance, dtype=np.float64)
    if hat.shape != true.shape or hat.size != len(parameter_names):
        raise ProjectedUnitError("projected unit requires aligned theta and names")
    if cov.shape != (hat.size, hat.size):
        raise ProjectedUnitError("covariance does not match the parameter chart")
    u = direction_vector(parameter_names, direction)
    a_hat = float(u @ hat)
    a_true = float(u @ true)
    sigma2 = float(u @ cov @ u)
    if (not math.isfinite(sigma2)) or sigma2 <= 0.0:
        raise ProjectedUnitError("projected variance is not a positive finite value")
    sigma = math.sqrt(sigma2)
    z = (a_hat - a_true) / sigma
    return {
        "a_hat": a_hat,
        "a_true": a_true,
        "sigma_a": sigma,
        "sigma_a2": sigma2,
        "z": z,
        "covered_95": official_covered_95(z),
        "coverage_construction": COVERAGE_CONSTRUCTION,
        "n_independent_z": 1,
    }


def official_z(a_hat: float, a_true: float, sigma_a: float) -> float:
    if (not math.isfinite(float(sigma_a))) or float(sigma_a) <= 0.0:
        raise ProjectedUnitError("sigma_a must be a finite positive value")
    return (float(a_hat) - float(a_true)) / float(sigma_a)


def official_covered_95(z: float) -> bool:
    """Same construction as WB83 ``abs(residual) <= 1.95996 * sigma``."""
    if not math.isfinite(float(z)):
        raise ProjectedUnitError("z must be finite")
    return abs(float(z)) <= GAUSSIAN_95_QUANTILE


def official_z_and_covered_95(a_hat: float, a_true: float, sigma_a: float) -> tuple[float, bool]:
    value = official_z(a_hat, a_true, sigma_a)
    return value, official_covered_95(value)
