"""T02 numerical contract: complete null basis and SPD rejection."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE, identifiable_svd
from alignment.numerical_contract import CovarianceNotSPDError, is_spd, require_spd
from baselines.field_chi2_matching import _valid_covariance
from geometry.propagation import mahalanobis_chi2


def _svd(matrix, names):
    return identifiable_svd(
        matrix,
        parameter_names=names,
        parameter_units=("mm",) * len(names),
        parameter_scales=(5.0,) * len(names),
        rank_tolerance=FROZEN_RANK_TOLERANCE,
    )


def test_tall_4x2_keeps_economy_right_basis():
    matrix = np.array(
        [[1.0, 0.0], [0.0, 2.0], [0.0, 0.0], [0.0, 0.0]],
        dtype=np.float64,
    )
    subspace = _svd(matrix, ("ift_dx_mm", "ift_dy_mm"))
    assert subspace.n_observations == 4
    assert subspace.n_parameters == 2
    assert subspace.identifiable_rank == 2
    assert subspace.null_dimension == 0
    assert subspace.v_null.shape == (2, 0)
    assert subspace.v.shape[0] == 2


def test_wide_2x3_has_complete_right_null():
    matrix = np.array([[1.0, 0.0, 0.0], [0.0, 1.0, 0.0]], dtype=np.float64)
    subspace = _svd(matrix, ("ift_dx_mm", "ift_dy_mm", "ift_dz_mm"))
    assert subspace.identifiable_rank == 2
    assert subspace.null_dimension == 1
    assert subspace.v_null.shape == (3, 1)
    assert np.allclose(matrix @ subspace.v_null, 0.0, atol=1.0e-12)
    assert np.allclose(subspace.v_id.T @ subspace.v_null, 0.0, atol=1.0e-12)
    assert np.allclose(subspace.v.T @ subspace.v, np.eye(3), atol=1.0e-12)


def test_zero_matrix_null_is_full_space():
    matrix = np.zeros((2, 3), dtype=np.float64)
    subspace = _svd(matrix, ("ift_dx_mm", "ift_dy_mm", "ift_dz_mm"))
    assert subspace.identifiable_rank == 0
    assert subspace.null_dimension == 3
    assert subspace.v_null.shape == (3, 3)
    assert np.allclose(matrix @ subspace.v_null, 0.0, atol=1.0e-12)
    assert np.allclose(subspace.v_null.T @ subspace.v_null, np.eye(3), atol=1.0e-12)


def test_non_spd_xy_block_is_rejected():
    bad = np.array([[1.0, 2.0], [2.0, 1.0]], dtype=np.float64)
    assert is_spd(bad) is False
    with pytest.raises(CovarianceNotSPDError, match="not positive definite"):
        require_spd(bad)
    with pytest.raises(CovarianceNotSPDError):
        mahalanobis_chi2(np.array([1.0, 0.0]), bad)
    assert _valid_covariance(np.eye(4)) is True
    illegal = np.eye(4)
    illegal[:2, :2] = bad
    assert _valid_covariance(illegal) is False


def test_rank_tolerance_not_retuned():
    assert FROZEN_RANK_TOLERANCE == pytest.approx(0.01)
