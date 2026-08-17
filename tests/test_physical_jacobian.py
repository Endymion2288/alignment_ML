from __future__ import annotations

import numpy as np

from alignment.physical_jacobian import (
    parameter_values_from_station_transforms,
    solve_physical_finite_difference,
    station_transforms_with_parameter_values,
)


def _linear_problem(derivative: np.ndarray, truth: np.ndarray):
    nominal = np.asarray([[0.2, -0.1, 0.0, 0.0], [-0.3, 0.5, 0.0, 0.0]])
    positive = np.asarray([nominal + derivative[:, :, parameter] for parameter in range(derivative.shape[2])])
    negative = np.asarray([nominal - derivative[:, :, parameter] for parameter in range(derivative.shape[2])])
    observed = nominal + np.einsum("nrp,p->nr", derivative, truth)
    covariance = np.tile(np.eye(4), (nominal.shape[0], 1, 1))
    return nominal, positive, negative, observed, covariance


def test_multidof_physical_jacobian_recovers_scaled_native_parameters():
    derivative = np.asarray(
        [
            [[1.0, 0.1, 0.0], [0.0, 2.0, 0.3], [0.1, 0.0, 1.0], [0.0, 0.2, 0.4]],
            [[2.0, -0.2, 0.2], [0.0, 1.0, -0.1], [0.3, 0.1, 0.8], [0.1, 0.0, 0.5]],
        ]
    )
    truth = np.asarray([1.5, -2.0, 35.0])
    nominal, positive, negative, observed, covariance = _linear_problem(derivative, truth)

    fit = solve_physical_finite_difference(
        nominal,
        positive,
        negative,
        observed,
        covariance,
        parameter_names=("ift_dx_mm", "ift_dy_mm", "ift_ry_mrad"),
        positive_values=(1.0, 1.0, 1.0),
        negative_values=(-1.0, -1.0, -1.0),
        parameter_scales=(5.0, 5.0, 60.0),
    )

    assert np.allclose(fit.recovered_parameters, truth)
    assert fit.normal_matrix_rank == 3
    assert fit.full_rank
    assert fit.normal_matrix_condition_number is not None
    assert fit.response_ndof == 5
    assert np.isclose(fit.response_chi2, 0.0)
    assert np.allclose(np.diag(fit.correlation_native), 1.0)


def test_multidof_physical_jacobian_reports_rank_deficiency_without_prior():
    derivative = np.asarray(
        [
            [[1.0, 1.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
            [[2.0, 2.0], [0.0, 0.0], [0.0, 0.0], [0.0, 0.0]],
        ]
    )
    nominal, positive, negative, observed, covariance = _linear_problem(derivative, np.asarray([2.0, -1.0]))

    fit = solve_physical_finite_difference(
        nominal,
        positive,
        negative,
        observed,
        covariance,
        parameter_names=("dx", "dy"),
        positive_values=(1.0, 1.0),
        negative_values=(-1.0, -1.0),
        parameter_scales=(1.0, 1.0),
    )

    assert fit.normal_matrix_rank == 1
    assert not fit.full_rank
    assert fit.normal_matrix_condition_number is None
    assert fit.identifiable_subspace_condition_number == 1.0
    assert np.allclose(fit.parameter_observability_fraction, [0.5, 0.5])


def test_parameter_values_convert_rotation_payload_radians_to_reported_mrad():
    values = parameter_values_from_station_transforms(
        (
            {"name": "ift_dx_mm", "station_id": 0, "component": "dx_mm"},
            {"name": "ift_ry_mrad", "station_id": 0, "component": "ry_mrad"},
        ),
        {0: (1.25, 0.0, 0.0, 0.0, 0.045, 0.0)},
    )
    assert values == {"ift_dx_mm": 1.25, "ift_ry_mrad": 45.0}


def test_parameter_update_preserves_other_payload_components_and_converts_mrad():
    transforms = station_transforms_with_parameter_values(
        (
            {"name": "ift_dx_mm", "station_id": 0, "component": "dx_mm"},
            {"name": "ift_ry_mrad", "station_id": 0, "component": "ry_mrad"},
        ),
        {
            0: (1.0, -2.0, 3.0, 0.01, 0.02, -0.03),
            1: (4.0, 5.0, 6.0, 0.04, 0.05, 0.06),
        },
        {"ift_dx_mm": -1.5, "ift_ry_mrad": 60.0},
    )

    assert transforms["0"] == [-1.5, -2.0, 3.0, 0.01, 0.06, -0.03]
    assert transforms["1"] == [4.0, 5.0, 6.0, 0.04, 0.05, 0.06]
