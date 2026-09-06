"""T12 Schur / profiled-information contract on synthetic blocks."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.profiled_information import (
    ProfiledInformationError,
    assert_no_information_inflation,
    solve_joint,
    solve_profiled,
    weak_direction_status,
    weight_matrix,
)


def _toy(seed: int = 3):
    rng = np.random.default_rng(seed)
    g = rng.normal(size=(8, 2))
    h = rng.normal(size=(8, 3))
    covariance = np.eye(8) + 0.05 * rng.normal(size=(8, 8))
    covariance = 0.5 * (covariance + covariance.T)
    covariance += 2.0 * np.eye(8)
    true_a = np.array([0.04, -0.02])
    true_q = np.array([0.1, -0.05, 0.02])
    residual = g @ true_a + h @ true_q + 0.01 * rng.normal(size=8)
    return g, h, residual, covariance, true_a


def test_schur_matches_joint_solve_and_covariance():
    g, h, residual, covariance, _true = _toy()
    joint_a, _joint_q, joint_normal = solve_joint(g, h, residual, covariance)
    profiled = solve_profiled(g, h, residual, covariance)
    n_align = g.shape[1]
    joint_cov = np.linalg.inv(joint_normal)[:n_align, :n_align]
    assert np.allclose(profiled.delta_alignment, joint_a, atol=1.0e-8)
    assert np.allclose(profiled.covariance_alignment, joint_cov, atol=1.0e-8)


def test_known_gauge_is_unconstrained():
    g = np.zeros((8, 2), dtype=np.float64)
    g[:, 0] = 1.0
    rng = np.random.default_rng(1)
    h = rng.normal(size=(8, 5))
    residual = rng.normal(size=8)
    covariance = np.eye(8)
    weight = weight_matrix(covariance)
    from alignment.profiled_information import schur_normal

    normal, _p = schur_normal(g, h, weight)
    status = weak_direction_status(normal)
    assert status["unconstrained"] is True
    assert status["do_not_force_zero"] is True
    assert status["n_weak"] >= 1


def test_profiling_does_not_inflate_information():
    g, h, residual, covariance, _true = _toy()
    weight = np.linalg.inv(covariance)
    fixed = g.T @ weight @ g
    profiled = solve_profiled(g, h, residual, covariance).normal_alignment
    assert_no_information_inflation(fixed, profiled)
    inflated = fixed + 10.0 * np.eye(2)
    with pytest.raises(ProfiledInformationError):
        assert_no_information_inflation(fixed, inflated)


def test_strengthening_one_mode_does_not_kill_the_other():
    g, h, residual, covariance, _true = _toy()
    ns = np.diag([1.0e6, 0.0])
    profiled = solve_profiled(g, h, residual, covariance, ns=ns)
    cov = profiled.covariance_alignment
    assert cov[0, 0] < 1.0e-5
    assert cov[1, 1] > cov[0, 0]
    assert np.isfinite(cov[1, 1])
    assert cov[1, 1] > 0.0
