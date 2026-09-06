"""Numerical contracts: complete SVD null basis and SPD covariance.

Illegal covariances are rejected.  Eigenvalues are never clipped.
"""

from __future__ import annotations

import numpy as np


class CovarianceNotSPDError(ValueError):
    """Raised when a covariance used for chi2/probability is not SPD."""


def require_spd(covariance: object, *, name: str = "covariance") -> np.ndarray:
    """Finite, symmetric, Cholesky-factorable.  No eigenvalue clipping."""
    matrix = np.asarray(covariance, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] != matrix.shape[1] or matrix.shape[0] < 1:
        raise CovarianceNotSPDError(f"{name} must be a square matrix")
    if not np.isfinite(matrix).all():
        raise CovarianceNotSPDError(f"{name} contains non-finite values")
    if not np.allclose(matrix, matrix.T, rtol=1.0e-7, atol=1.0e-12):
        raise CovarianceNotSPDError(f"{name} is not symmetric")
    symmetric = 0.5 * (matrix + matrix.T)
    try:
        np.linalg.cholesky(symmetric)
    except np.linalg.LinAlgError as error:
        raise CovarianceNotSPDError(f"{name} is not positive definite") from error
    return symmetric


def is_spd(covariance: object) -> bool:
    try:
        require_spd(covariance)
    except CovarianceNotSPDError:
        return False
    return True


def svd_complete_right(matrix: object) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Economy SVD for tall/square A; complete right basis when m < n."""
    array = np.asarray(matrix, dtype=np.float64)
    if array.ndim != 2:
        raise ValueError("matrix must be two-dimensional")
    n_rows, n_cols = array.shape
    if n_rows < n_cols:
        left, singular, right_t = np.linalg.svd(array, full_matrices=True)
    else:
        left, singular, right_t = np.linalg.svd(array, full_matrices=False)
    if singular.size < n_cols:
        singular = np.concatenate(
            [singular, np.zeros(n_cols - singular.size, dtype=np.float64)]
        )
    return (
        np.asarray(left, dtype=np.float64),
        np.asarray(singular, dtype=np.float64),
        np.asarray(right_t.T, dtype=np.float64),
    )
