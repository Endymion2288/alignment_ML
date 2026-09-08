"""WB83-v2 hermetic solver-fix regressions.  Run before treating DAG 1109806 as official."""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from alignment.common_track_geometry import se3_log
from alignment.common_track_solver import (
    SURVEY_FINITE_PRIOR,
    SURVEY_FIXED_DZ,
    AlignmentSolverError,
    StationHit,
    absolute_survey_prior_terms,
    assemble_track_block,
    inverse_spd_after_symmetrize,
    iterate_common_track,
    schur_reduce,
    solve_common_track,
    solver_implementation_fixed,
    symmetrize_normal,
)
from alignment.four_station import IDENTITY_SIX, STATION_IDS, six_vector_to_matrix
from alignment.wb83_qualification import qualify


def _hit(track_id: int, station: int, measurement_id: str, observed, scale: float = 1.0) -> StationHit:
    return StationHit(
        track_id=track_id,
        station=station,
        measurement_id=measurement_id,
        observed=np.asarray(observed, dtype=np.float64),
        covariance=np.eye(4) * float(scale),
    )


def test_spd_plus_1e8_antisymmetric_perturbation_is_accepted_after_symmetrize():
    spd = np.asarray([[4.0, 1.0], [1.0, 3.0]], dtype=np.float64)
    perturbation = np.asarray([[0.0, 1.0e-8], [-1.0e-8, 0.0]], dtype=np.float64)
    raw = spd + perturbation
    symmetric, diagnostics = symmetrize_normal(raw, "spd+antisym")
    inverse, factored, recorded = inverse_spd_after_symmetrize(raw, "spd+antisym")
    assert np.allclose(symmetric, spd)
    assert np.allclose(factored, spd)
    assert diagnostics["antisymmetry_norm"] == pytest.approx(1.0e-8 * np.sqrt(2.0))
    assert recorded["antisymmetry_norm"] == pytest.approx(1.0e-8 * np.sqrt(2.0))
    assert recorded["relative_antisymmetry"] > 0.0
    assert recorded["minimum_eigenvalue"] > 0.0
    assert np.isfinite(recorded["condition_number"])
    assert np.allclose(symmetric @ inverse, np.eye(2), atol=1.0e-12)


def test_true_indefinite_is_rejected_without_ridge():
    indefinite = np.asarray([[1.0, 0.0], [0.0, -1.0]], dtype=np.float64)
    perturbation = np.asarray([[0.0, 1.0e-8], [-1.0e-8, 0.0]], dtype=np.float64)
    raw = indefinite + perturbation
    _symmetric, diagnostics = symmetrize_normal(raw, "indefinite")
    assert diagnostics["minimum_eigenvalue"] < 0.0
    with pytest.raises(AlignmentSolverError, match="not SPD"):
        inverse_spd_after_symmetrize(raw, "indefinite")
    # A 1e-12 ridge cannot hide this; the contract forbids using one anyway.
    ridge = 0.5 * (raw + raw.T) + 1.0e-12 * np.eye(2)
    assert float(np.linalg.eigvalsh(ridge)[0]) < 0.0
    source = inspect.getsource(schur_reduce) + inspect.getsource(inverse_spd_after_symmetrize)
    assert "1.0e-12 * np.eye" not in source
    assert "1e-12 * np.eye" not in source


def test_scalar_analytic_survey_prior_map():
    a_current = 1.2
    a_survey = 0.0
    y = 2.0
    weight_data = 4.0
    sigma = 5.0
    weight_prior = 1.0 / (sigma ** 2)
    theta_star = (
        weight_data * (y - a_current) + weight_prior * (a_survey - a_current)
    ) / (weight_data + weight_prior)
    terms = absolute_survey_prior_terms(a_current, a_survey, sigma=sigma)
    assert terms["hessian"] == pytest.approx(weight_prior)
    assert terms["gradient_at_linearization"] == pytest.approx(weight_prior * (a_current - a_survey))
    assert terms["rhs"] == pytest.approx(-weight_prior * (a_current - a_survey))
    rng = np.random.default_rng(20260907)
    h_xi = rng.normal(scale=0.8, size=(8, 5))
    h_th = rng.normal(scale=0.5, size=(8, 1))
    h_th = h_th - h_xi @ np.linalg.lstsq(h_xi, h_th, rcond=None)[0]
    h_th = h_th / np.linalg.norm(h_th) * np.sqrt(weight_data)
    alpha = y - a_current
    residual = (h_th * alpha).ravel()
    hits = (_hit(1, 1, "a", residual[:4]), _hit(1, 3, "b", residual[4:]))
    block = assemble_track_block(hits, h_xi, h_th, residual)
    assert float(np.asarray(h_th.T @ block.weight @ h_th).reshape(-1)[0]) == pytest.approx(weight_data)
    payloads = {station: IDENTITY_SIX for station in STATION_IDS}
    payloads[1] = (0.0, 0.0, a_current, 0.0, 0.0, 0.0)
    assert se3_log(six_vector_to_matrix(payloads[1]))[2] == pytest.approx(a_current)
    solution = solve_common_track(
        (block,),
        ("s1_dz_mm",),
        survey_mode=SURVEY_FINITE_PRIOR,
        survey_sigma_mm=sigma,
        current_payloads=payloads,
        damping=0.0,
    )
    assert solution.ok
    assert float(solution.update[0]) == pytest.approx(theta_star, rel=1.0e-9, abs=1.0e-9)


def test_zero_prior_residual_gives_zero_update():
    terms = absolute_survey_prior_terms(0.0, 0.0, sigma=5.0)
    assert terms["prior_residual"] == 0.0
    assert terms["gradient_at_linearization"] == 0.0
    assert terms["rhs"] == 0.0
    rng = np.random.default_rng(11)
    h_xi = rng.normal(scale=0.7, size=(8, 5))
    h_th = rng.normal(scale=0.4, size=(8, 1))
    residual = np.zeros(8, dtype=np.float64)
    hits = (_hit(1, 1, "z0", residual[:4]), _hit(1, 3, "z1", residual[4:]))
    block = assemble_track_block(hits, h_xi, h_th, residual)
    payloads = {station: IDENTITY_SIX for station in STATION_IDS}
    solution = solve_common_track(
        (block,),
        ("s1_dz_mm",),
        survey_mode=SURVEY_FINITE_PRIOR,
        survey_sigma_mm=5.0,
        current_payloads=payloads,
        damping=0.0,
    )
    assert solution.ok
    assert float(solution.update[0]) == pytest.approx(0.0, abs=1.0e-12)


def test_iterative_prior_converges_without_constant_dz():
    from alignment.common_track_solver import predict_local_measurement
    from alignment.wb83_qualification import Z_MM

    start = {station: IDENTITY_SIX for station in STATION_IDS}
    start[3] = (0.0, 0.0, 2.0, 0.0, 0.0, 0.0)
    truth = {station: IDENTITY_SIX for station in STATION_IDS}
    tracks = []
    states = {}
    for track_id in range(8):
        state = np.asarray([0.15 * track_id, -0.07 * track_id, 0.001, 0.0, 0.1], dtype=np.float64)
        states[track_id] = state
        hits = []
        for station in STATION_IDS:
            predicted = predict_local_measurement(
                state, truth[station], z_ref_mm=0.0, z_station_mm=Z_MM[station], field_y=0.35
            )
            hits.append(_hit(track_id, station, f"{track_id}-{station}", predicted, 0.02**2))
        tracks.append(hits)
    result = iterate_common_track(
        tracks,
        states,
        start,
        z_mm=Z_MM,
        field_y=0.35,
        survey_mode=SURVEY_FINITE_PRIOR,
        validation_track_ids=(6, 7),
        consecutive_required=2,
        components=("dz_mm",),
        stations=(3,),
    )
    assert result["converged"] is True
    assert result["iterations"] < 10
    dz_steps = [float(row["scaled_update_norm"]) for row in result["history"]]
    assert len(dz_steps) >= 2
    assert not np.allclose(dz_steps, dz_steps[0], atol=1.0e-6)
    final_dz = float(se3_log(six_vector_to_matrix(result["payloads"][3]))[2])
    assert abs(final_dz) < 0.2


def test_fixed_dz_regression_parity():
    rng = np.random.default_rng(20260906)
    n_xi = 3
    n_th = 2
    blocks = []
    true_th = rng.normal(scale=0.4, size=n_th)
    for track_id in (1, 2, 3):
        xi = rng.normal(scale=0.3, size=n_xi)
        h_xi = rng.normal(scale=0.8, size=(8, n_xi))
        h_th = rng.normal(scale=0.5, size=(8, n_th))
        residual = h_xi @ xi + h_th @ true_th
        hits = (
            _hit(track_id, 0, f"t{track_id}-s0", residual[:4]),
            _hit(track_id, 1, f"t{track_id}-s1", residual[4:]),
        )
        blocks.append(assemble_track_block(hits, h_xi, h_th, residual))
    names = ("p0", "p1")
    payloads = {station: IDENTITY_SIX for station in STATION_IDS}
    first = solve_common_track(blocks, names, survey_mode=SURVEY_FIXED_DZ)
    second = solve_common_track(
        blocks, names, survey_mode=SURVEY_FIXED_DZ, current_payloads=payloads
    )
    assert first.ok and second.ok
    assert np.allclose(first.update, second.update, atol=1.0e-12)
    assert np.allclose(first.update, true_th, atol=1.0e-8)
    assert np.allclose(first.covariance, second.covariance, atol=1.0e-12)


def test_three_layer_output_keeps_physical_oracle_false():
    assert solver_implementation_fixed() is True
    gate = qualify([], fd_ok=True, n_independent=100, forbidden_accessed=True)
    assert gate["solver_implementation_fixed"] is True
    assert gate["common_track_solver_qualified_under_toy_model"] is False
    assert gate["alignment_oracle_qualified_for_physical_FASER"] is False
    assert gate["alignment_oracle_qualified"] is False
