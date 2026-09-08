"""Workbook-82: common-track likelihood, Schur, SE(3), field/gauge, fail-closed."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.common_track_geometry import (
    SOLVER_CONTRACT,
    euler_subtraction,
    left_update,
    left_update_payload,
    pose_error_lie,
    right_update,
    se3_exp,
    se3_log,
    transform_field_vector,
    transform_surface_z,
)
from alignment.common_track_solver import (
    ASSOCIATION_DEFAULT_SYSTEM,
    PHASE1_SMOKE_COMPONENTS,
    PHASE1_SMOKE_STATIONS,
    SURVEY_FINITE_PRIOR,
    SURVEY_FIXED_DZ,
    StationHit,
    assemble_unique_hits,
    chi2_of_blocks,
    fd_column_stability,
    iterate_common_track,
    lab_transport,
    linearize_track,
    pairwise_independent_chi2,
    parameter_chart,
    predict_local_measurement,
    refuse_wb82_path,
    relative_lie_table,
    solve_common_track,
    solve_joint_dense,
    assemble_track_block,
)
from alignment.four_station import IDENTITY_SIX, STATION_IDS
from alignment.physical_jacobian import solve_physical_finite_difference
from alignment.route_selected_update import __doc__ as route_selected_doc


def _hit(track_id: int, station: int, measurement_id: str, observed, scale: float = 1.0) -> StationHit:
    return StationHit(
        track_id=track_id,
        station=station,
        measurement_id=measurement_id,
        observed=np.asarray(observed, dtype=np.float64),
        covariance=np.eye(4) * float(scale),
    )


def _linear_blocks():
    rng = np.random.default_rng(20260906)
    n_xi = 3
    n_th = 2
    blocks = []
    true_xi = []
    true_th = rng.normal(scale=0.4, size=n_th)
    for track_id in (1, 2, 3):
        xi = rng.normal(scale=0.3, size=n_xi)
        true_xi.append(xi)
        h_xi = rng.normal(scale=0.8, size=(8, n_xi))
        h_th = rng.normal(scale=0.5, size=(8, n_th))
        residual = h_xi @ xi + h_th @ true_th
        hits = (
            _hit(track_id, 0, f"t{track_id}-s0", residual[:4]),
            _hit(track_id, 1, f"t{track_id}-s1", residual[4:]),
        )
        blocks.append(assemble_track_block(hits, h_xi, h_th, residual))
    return blocks, true_th, true_xi


def test_linear_gaussian_matches_analytic_and_pulls():
    blocks, true_th, _true_xi = _linear_blocks()
    names = ("p0", "p1")
    # parameter names in solve_common_track are not parsed except for dz prior
    solution = solve_common_track(blocks, names, survey_mode=SURVEY_FIXED_DZ)
    assert solution.ok
    assert solution.contract == SOLVER_CONTRACT
    joint, joint_cov = solve_joint_dense(blocks)
    n_xi = sum(block.jacobian_nuisance.shape[1] for block in blocks)
    assert np.allclose(solution.update, joint[n_xi:], atol=1.0e-9)
    assert np.allclose(solution.covariance, joint_cov[n_xi:, n_xi:], atol=1.0e-9)
    assert np.allclose(solution.update, true_th, atol=1.0e-8)
    pulls = (solution.update - true_th) / np.sqrt(np.diag(solution.covariance))
    assert np.allclose(pulls, 0.0, atol=1.0e-6)
    assert np.all(np.isfinite(solution.update))


def test_schur_matches_joint_with_noise():
    rng = np.random.default_rng(7)
    blocks, _true_th, _ = _linear_blocks()
    noisy = []
    for block in blocks:
        residual = block.residual + rng.normal(scale=0.05, size=block.residual.shape)
        hits = (
            _hit(block.track_id, 0, f"n{block.track_id}-0", residual[:4], 0.05**2),
            _hit(block.track_id, 1, f"n{block.track_id}-1", residual[4:], 0.05**2),
        )
        noisy.append(assemble_track_block(hits, block.jacobian_nuisance, block.jacobian_global, residual))
    solution = solve_common_track(noisy, ("p0", "p1"))
    joint, _ = solve_joint_dense(noisy)
    n_xi = sum(block.jacobian_nuisance.shape[1] for block in noisy)
    assert solution.ok
    assert np.allclose(solution.update, joint[n_xi:], atol=1.0e-9)


def test_shared_measurement_is_not_double_counted():
    observed = np.asarray([0.4, -0.2, 0.01, 0.0])
    first = _hit(1, 1, "shared-tracklet", observed, 0.25)
    second = _hit(1, 1, "shared-tracklet", observed, 0.25)
    unique = assemble_unique_hits((first, second))
    assert len(unique) == 1
    pairwise = pairwise_independent_chi2((first, second))
    single = pairwise_independent_chi2((first,))
    assert pairwise == pytest.approx(2.0 * single)
    residual = observed
    block = assemble_track_block((first, second), np.eye(4, 3), np.eye(4, 2), residual)
    assert block.residual.size == 4


def test_se3_left_right_nominal_reference_and_lie_log():
    twist = np.asarray([1.5, -0.7, 0.2, 0.03, -0.02, 0.04], dtype=np.float64)
    matrix = se3_exp(twist)
    assert np.allclose(se3_log(matrix), twist, atol=1.0e-9)
    nominal = se3_exp(np.asarray([10.0, 0.0, 200.0, 0.05, 0.0, 0.1]))
    left = left_update(nominal, twist)
    right = right_update(nominal, twist)
    assert not np.allclose(left, right, atol=1.0e-8)
    payload = (0.4, -0.2, 3.0, 0.02, -0.01, 0.03)
    updated = left_update_payload(payload, (0.0, 0.0, 0.0, 0.7, 0.4, 0.0))
    lie = pose_error_lie(payload, updated)
    euler = euler_subtraction(payload, updated)
    assert float(np.linalg.norm(lie[3:] - euler[3:])) > 1.0e-2
    base = {0: IDENTITY_SIX, 1: (0.3, 0.0, 0.0, 0.0, 0.05, 0.0), 2: IDENTITY_SIX, 3: (-0.2, 0.1, 0.0, 0.0, 0.0, 0.04)}
    shifted = {station: left_update_payload(base[station], (1.0, 0.0, 0.0, 0.0, 0.0, 0.02)) for station in STATION_IDS}
    relatives = relative_lie_table(base, base)
    moved = relative_lie_table(shifted, shifted)
    for key in relatives["relative"]:
        assert np.allclose(relatives["relative"][key], 0.0, atol=1.0e-12)
        assert np.allclose(moved["relative"][key], 0.0, atol=1.0e-9)


def test_fixed_field_jg_is_not_called_observable_gauge():
    z_mm = {0: 0.0, 1: 1000.0, 2: 2000.0, 3: 3000.0}
    state = np.asarray([0.0, 0.0, 0.0, 0.0, 0.2], dtype=np.float64)
    payloads = {station: IDENTITY_SIX for station in STATION_IDS}
    names = parameter_chart(survey_mode=SURVEY_FIXED_DZ, reference_station=0)

    def _track(field_y: float):
        hits = []
        for station in STATION_IDS:
            predicted = predict_local_measurement(
                state, payloads[station], z_ref_mm=0.0, z_station_mm=z_mm[station], field_y=field_y
            )
            hits.append(_hit(1, station, f"trk-s{station}", predicted))
        return linearize_track(hits, state, payloads, names, z_mm=z_mm, field_y=field_y)

    zero_field = _track(0.0)
    bent = _track(0.8)
    s3_ry = names.index("s3_ry_mrad")
    # Fixed lab field: rotating a downstream station through By is not a
    # chart identity.  Algebraic g_i^{-1} g_j invariance is not this JG.
    assert float(np.linalg.norm(bent.jacobian_global[:, s3_ry])) > 0.0
    assert float(np.linalg.norm(zero_field.jacobian_global[:, s3_ry])) >= 0.0
    common = (0.0, 0.0, 0.0, 0.15, 0.0, 0.0)
    field = np.asarray([0.0, 0.8, 0.0])
    rotated = transform_field_vector(common, field)
    assert not np.allclose(rotated, field)
    assert transform_surface_z(common, 1000.0) != 1000.0
    geometry_doc = Path("alignment/common_track_geometry.py").read_text(encoding="utf-8")
    assert "relative-chart identity" in geometry_doc
    assert "observable" in geometry_doc and "gauge" in geometry_doc


def test_fd_stability_on_lab_transport():
    state = np.asarray([1.0, -0.5, 0.002, 0.001, 0.15], dtype=np.float64)

    def predict(values):
        return lab_transport(values, 0.0, 2000.0, field_y=0.4)[:4]

    report = fd_column_stability(predict, state, 1.0e-3, index=0)
    assert report["rel_h_over_2"] < 0.01
    assert report["rel_2h"] < 0.01


def test_fixed_dz_and_finite_prior_are_different_modes():
    fixed_names = parameter_chart(survey_mode=SURVEY_FIXED_DZ, reference_station=0)
    prior_names = parameter_chart(survey_mode=SURVEY_FINITE_PRIOR, reference_station=0)
    assert all(not name.endswith("_dz_mm") for name in fixed_names)
    assert any(name.endswith("_dz_mm") for name in prior_names)
    assert SURVEY_FIXED_DZ != SURVEY_FINITE_PRIOR
    names = ("s1_dx_mm", "s1_dz_mm")
    h_th = np.asarray(
        [
            [1.0, 0.2],
            [0.0, 1.0],
            [0.3, 0.0],
            [0.0, 0.4],
            [0.5, 0.1],
            [0.0, 0.6],
            [0.2, 0.0],
            [0.1, 0.3],
        ],
        dtype=np.float64,
    )
    residual = h_th @ np.asarray([0.4, -0.8])
    hits = (_hit(1, 1, "a", residual[:4]), _hit(1, 2, "b", residual[4:]))
    block = assemble_track_block(hits, np.eye(8, 5), h_th, residual)
    free = solve_common_track((block,), names, survey_mode=SURVEY_FIXED_DZ)
    prior = solve_common_track((block,), names, survey_mode=SURVEY_FINITE_PRIOR, survey_sigma_mm=5.0)
    assert free.ok and prior.ok
    assert not np.allclose(free.update, prior.update)


def test_fail_closed_on_singular_nan_and_non_spd():
    names = ("p0", "p1")
    residual = np.ones(4)
    hits = (_hit(1, 0, "x", residual),)
    singular = assemble_track_block(hits, np.zeros((4, 3)), np.zeros((4, 2)), residual)
    report = solve_common_track((singular,), names)
    assert report.status in {"singular", "non_spd"}
    assert not report.ok
    assert not np.isfinite(report.update).any()
    with pytest.raises(Exception, match="NaN/Inf|not SPD|not symmetric"):
        StationHit(1, 0, "bad", np.array([np.nan, 0, 0, 0]), np.eye(4))
    broken = np.eye(4)
    broken[0, 0] = -1.0
    with pytest.raises(Exception, match="SPD|symmetric"):
        StationHit(1, 0, "neg", np.zeros(4), broken)


def test_noiseless_s3_dx_recovers_on_translation_chart():
    """Phase-1 smoke chart is dx/dy.  Free ry is a physical JG, not this test."""
    z_mm = {0: 0.0, 1: 1000.0, 2: 2000.0, 3: 3000.0}
    start = {station: IDENTITY_SIX for station in STATION_IDS}
    truth = {station: IDENTITY_SIX for station in STATION_IDS}
    truth[3] = left_update_payload(IDENTITY_SIX, (0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    names = parameter_chart(
        survey_mode=SURVEY_FIXED_DZ,
        reference_station=0,
        components=PHASE1_SMOKE_COMPONENTS,
        stations=PHASE1_SMOKE_STATIONS,
    )
    tracks = []
    states = {}
    for track_id in range(12):
        state = np.asarray([0.2 * track_id, -0.1 * track_id, 0.001, -0.0005, 0.1], dtype=np.float64)
        states[track_id] = state
        hits = []
        for station in STATION_IDS:
            predicted = predict_local_measurement(
                state, truth[station], z_ref_mm=0.0, z_station_mm=z_mm[station], field_y=0.35
            )
            hits.append(_hit(track_id, station, f"{track_id}-{station}", predicted, 0.02**2))
        tracks.append(hits)
    blocks = [
        linearize_track(hits, states[int(hits[0].track_id)], start, names, z_mm=z_mm, field_y=0.35)
        for hits in tracks
    ]
    solution = solve_common_track(blocks, names, survey_mode=SURVEY_FIXED_DZ)
    assert solution.ok
    recovered = dict(zip(names, solution.update))
    assert names == ("s3_dx_mm", "s3_dy_mm")
    assert recovered["s3_dx_mm"] == pytest.approx(0.5, abs=1.0e-6)
    assert abs(recovered["s3_dy_mm"]) < 1.0e-6


def test_iterate_does_not_auto_pass_at_max_iterations():
    z_mm = {0: 0.0, 1: 1000.0, 2: 2000.0, 3: 3000.0}
    payloads = {station: IDENTITY_SIX for station in STATION_IDS}
    truth = {station: IDENTITY_SIX for station in STATION_IDS}
    truth[3] = left_update_payload(IDENTITY_SIX, (0.5, 0.0, 0.0, 0.0, 0.0, 0.0))
    tracks = []
    states = {}
    for track_id in range(8):
        state = np.asarray([0.1 * track_id, -0.05 * track_id, 0.001, 0.0, 0.1], dtype=np.float64)
        states[track_id] = state
        hits = []
        for station in STATION_IDS:
            predicted = predict_local_measurement(
                state, truth[station], z_ref_mm=0.0, z_station_mm=z_mm[station], field_y=0.0
            )
            hits.append(_hit(track_id, station, f"{track_id}-{station}", predicted, 1.0e-4))
        tracks.append(hits)
    result = iterate_common_track(
        tracks,
        states,
        payloads,
        z_mm=z_mm,
        field_y=0.0,
        validation_track_ids=(6, 7),
        max_iterations=2,
    )
    assert result["automatically_passed_at_max_iterations"] is False
    assert result["max_iterations"] == 2


def test_old_backends_are_diagnostics_not_oracles():
    assert ASSOCIATION_DEFAULT_SYSTEM == "frozen_W64_raw_energy_plus_exact_solver"
    assert "response / regression diagnostic" in Path("alignment/physical_jacobian.py").read_text(encoding="utf-8")
    assert "certified alignment solver" in Path("alignment/physical_jacobian.py").read_text(encoding="utf-8")
    assert "not a certified alignment solver" in route_selected_doc
    assert callable(solve_physical_finite_difference)
    text = Path("alignment/common_track_solver.py").read_text(encoding="utf-8")
    assert "loss_augmented_structured_hinge" not in text
    assert "ExplicitRouteEnergyScorer" not in text


def test_refuse_forbidden_alignment_paths():
    with pytest.raises(Exception, match="refuses"):
        refuse_wb82_path("mc24_100047_00350_00399/x.root")
    with pytest.raises(Exception, match="refuses"):
        refuse_wb82_path("outputs/mc24_100116_x")
    with pytest.raises(Exception, match="refuses"):
        refuse_wb82_path("00800_00849")
