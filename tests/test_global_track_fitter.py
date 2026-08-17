from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from models.track_fitter import (
    fit_event_route,
    fit_global_straight_track,
    global_track_fit_summary,
    leave_one_out_event_route_residuals,
)


def test_global_straight_track_fit_recovers_state_and_quality():
    z = np.asarray([0.0, 100.0, 200.0, 300.0])
    parameters = np.asarray([1.0, -2.0, 0.01, -0.02])
    state = np.asarray(
        [[parameters[0] + parameters[2] * value, parameters[1] + parameters[3] * value, parameters[2], parameters[3]] for value in z]
    )
    covariance = np.tile(np.diag([0.01, 0.01, 1.0e-6, 1.0e-6]), (4, 1, 1))

    fit = fit_global_straight_track(state, covariance, z, station_ids=(0, 1, 2, 3))

    assert np.allclose(
        fit.parameters,
        [parameters[0] + parameters[2] * 150.0, parameters[1] + parameters[3] * 150.0, parameters[2], parameters[3]],
    )
    assert np.allclose(fit.residual, 0.0)
    assert np.isclose(fit.chi2, 0.0, atol=1.0e-20)
    assert fit.ndof == 12
    assert fit.normal_matrix_rank == 4
    assert np.isclose(global_track_fit_summary([fit])["reduced_chi2"], 0.0, atol=1.0e-20)


def test_event_route_fitter_uses_selected_event_endpoint_order():
    z = np.asarray([100.0, 0.0, 300.0, 200.0])
    state = np.asarray(
        [[1.0 + 0.01 * value, -2.0 - 0.02 * value, 0.01, -0.02] for value in z]
    )
    event = SimpleNamespace(
        state=state,
        covariance=np.tile(np.diag([0.01, 0.01, 1.0e-6, 1.0e-6]), (4, 1, 1)),
        z_mm=z,
        station_id=np.asarray([1, 0, 3, 2]),
    )

    fit = fit_event_route(event, (1, 0, 3, 2))

    assert fit.endpoint_indices == (1, 0, 3, 2)
    assert fit.station_ids == (0, 1, 2, 3)
    assert np.allclose(fit.residual, 0.0)


def test_leave_one_out_residuals_are_zero_for_consistent_route_and_have_spd_covariance():
    z = np.asarray([0.0, 100.0, 200.0, 300.0])
    state = np.asarray(
        [[1.0 + 0.01 * value, -2.0 - 0.02 * value, 0.01, -0.02] for value in z]
    )
    event = SimpleNamespace(
        state=state,
        covariance=np.tile(np.diag([0.01, 0.01, 1.0e-6, 1.0e-6]), (4, 1, 1)),
        z_mm=z,
        station_id=np.asarray([0, 1, 2, 3]),
    )

    residuals = leave_one_out_event_route_residuals(event, (0, 1, 2, 3))

    assert [item.station_id for item in residuals] == [0, 1, 2, 3]
    assert all(np.allclose(item.residual, 0.0) for item in residuals)
    assert all(np.isclose(item.chi2, 0.0, atol=1.0e-20) for item in residuals)
    assert all(np.all(np.linalg.eigvalsh(item.combined_covariance) > 0.0) for item in residuals)


def test_global_fit_keeps_slope_rank_at_faser_scale_z_separations():
    z = np.asarray([-3000.0, -1800.0, -600.0, 600.0])
    parameters = np.asarray([15.0, -4.0, 0.025, -0.012])
    state = np.asarray(
        [[parameters[0] + parameters[2] * value, parameters[1] + parameters[3] * value, parameters[2], parameters[3]] for value in z]
    )
    covariance = np.tile(np.diag([0.01, 0.01, 1.0e-6, 1.0e-6]), (4, 1, 1))

    fit = fit_global_straight_track(state, covariance, z, station_ids=(0, 1, 2, 3))

    assert fit.normal_matrix_rank == 4
    assert fit.z_scale_mm > 1.0
    assert np.allclose(fit.parameters, [parameters[0] - 30.0, parameters[1] + 14.4, 0.025, -0.012])
