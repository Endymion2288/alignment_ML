from __future__ import annotations

import numpy as np

from alignment.rotation_closure import (
    solve_ift_ry_finite_difference,
    solve_ift_ry_polynomial_response,
)


def test_physical_finite_difference_rotation_closure_recovers_one_parameter_response():
    derivative = np.asarray(
        [
            [0.3, 0.0, 0.01, 0.0],
            [-0.2, 0.1, -0.02, 0.0],
        ]
    )
    nominal = np.asarray([[1.0, -1.0, 0.0, 0.0], [0.5, 0.2, 0.0, 0.0]])
    positive = nominal + derivative * 10.0
    negative = nominal - derivative * 10.0
    observed = nominal + derivative * 40.0
    covariance = np.tile(np.eye(4), (2, 1, 1))

    fit = solve_ift_ry_finite_difference(
        nominal,
        positive,
        negative,
        observed,
        covariance,
        positive_ry_mrad=10.0,
        negative_ry_mrad=-10.0,
    )

    assert np.isclose(fit.recovered_ry_mrad, 40.0)
    assert fit.used_pairs == 2
    assert fit.response_ndof == 7
    assert np.isclose(fit.response_chi2, 0.0)


def test_physical_scan_rotation_closure_recovers_held_out_nonlinear_response():
    angles = np.asarray([-40.0, -25.0, -10.0, 0.0, 10.0, 25.0, 40.0])
    linear = np.asarray([[0.3, 0.0, 0.01, 0.0], [-0.2, 0.1, -0.02, 0.0]])
    quadratic = np.asarray([[0.002, 0.0, 0.0001, 0.0], [0.001, -0.001, 0.0, 0.0]])
    nominal = np.asarray([[1.0, -1.0, 0.0, 0.0], [0.5, 0.2, 0.0, 0.0]])
    calibration = np.asarray(
        [nominal + linear * angle + quadratic * angle**2 for angle in angles]
    )
    observed = nominal + linear * 60.0 + quadratic * 60.0**2
    covariance = np.tile(np.eye(4), (2, 1, 1))

    fit = solve_ift_ry_polynomial_response(
        angles,
        calibration,
        observed,
        covariance,
        polynomial_degree=2,
        search_interval_mrad=(-80.0, 80.0),
    )

    assert np.isclose(fit.recovered_ry_mrad, 60.0, atol=1.0e-3)
    assert fit.used_pairs == 2
    assert fit.response_ndof == 7
    assert np.isclose(fit.response_chi2, 0.0, atol=1.0e-7)
