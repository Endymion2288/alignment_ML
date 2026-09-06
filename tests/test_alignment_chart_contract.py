"""T03 SE(3) chart contract.  No SQLite/POOL write, no payload ingest."""

from __future__ import annotations

import math

import numpy as np
import pytest

from alignment.nov22_metrology_provenance_station_ry import (
    extract_alpha_beta_gamma,
    station_alignment_transform,
)
from alignment.true_cluster_local_residual import rotate_about_y
from geometry.alignment_charts import (
    LEGACY_CLUSTER_LOCAL_STATION_Z_RY,
    MRAD_PER_RAD,
    ROUND_TRIP_TOLERANCE,
    STATIONS_GLOBAL_ORIGIN,
    ChartError,
    apply_se3,
    center_to_origin_jacobian_at_identity,
    center_translation_from_origin,
    compose_left_increment,
    native_to_report_mm_mrad,
    origin_translation_from_center,
    report_mm_mrad_to_native,
    se3_from_sixvector,
    sixvector_from_se3,
    transport_center_covariance_to_origin,
)


def test_stations_matches_calypso_writer():
    values = (1.5, -2.0, 0.25, 0.01, -0.02, 0.03)
    built = se3_from_sixvector(values, STATIONS_GLOBAL_ORIGIN)
    assert np.allclose(built, station_alignment_transform(values), atol=1.0e-15)


def test_single_axis_and_mixed_round_trip():
    cases = [
        (0.4, 0.0, 0.0, 0.02, 0.0, 0.0),
        (0.0, -0.3, 0.0, 0.0, -0.015, 0.0),
        (0.0, 0.0, 1.2, 0.0, 0.0, 0.04),
        (1.0, -2.0, 0.5, 0.03, -0.04, 0.05),
    ]
    for values in cases:
        built = se3_from_sixvector(values, STATIONS_GLOBAL_ORIGIN)
        extracted = sixvector_from_se3(built, STATIONS_GLOBAL_ORIGIN)
        rebuilt = se3_from_sixvector(extracted, STATIONS_GLOBAL_ORIGIN)
        assert np.linalg.norm(rebuilt - built) < ROUND_TRIP_TOLERANCE
        assert np.allclose(extracted, values, atol=1.0e-12)


def test_declared_singularity_is_refused():
    values = (0.0, 0.0, 0.0, 0.0, 0.5 * math.pi, 0.0)
    built = se3_from_sixvector(values, STATIONS_GLOBAL_ORIGIN)
    with pytest.raises(ChartError, match="singularity"):
        sixvector_from_se3(built, STATIONS_GLOBAL_ORIGIN)


def test_origin_and_center_charts_predict_the_same_points():
    center = np.array([10.0, -4.0, -1860.0], dtype=np.float64)
    values_center = np.array([0.2, -0.1, 0.05, 0.01, -0.02, 0.015], dtype=np.float64)
    rotation = se3_from_sixvector(
        (0.0, 0.0, 0.0, *values_center[3:]), STATIONS_GLOBAL_ORIGIN
    )[:3, :3]
    t_origin = origin_translation_from_center(values_center[:3], rotation, center)
    origin_values = np.concatenate([t_origin, values_center[3:]])
    points = np.array(
        [[center[0] + 30.0, center[1] - 8.0, center[2] + 2.0],
         [center[0] - 12.0, center[1] + 5.0, center[2] - 1.5]],
        dtype=np.float64,
    )
    g_origin = se3_from_sixvector(origin_values, STATIONS_GLOBAL_ORIGIN)
    moved_origin = apply_se3(points, g_origin)
    moved_center = center + (rotation @ (points - center).T).T + values_center[:3]
    assert np.allclose(moved_origin, moved_center, atol=1.0e-12)
    assert np.allclose(
        center_translation_from_origin(t_origin, rotation, center),
        values_center[:3],
        atol=1.0e-12,
    )


def test_different_pivot_is_a_counterexample():
    values = (0.0, 0.0, 0.0, 0.0, 0.02, 0.0)
    g_origin = se3_from_sixvector(values, STATIONS_GLOBAL_ORIGIN)
    pivot = (0.0, 0.0, -1860.0)
    g_station = se3_from_sixvector(values, LEGACY_CLUSTER_LOCAL_STATION_Z_RY, pivot_mm=pivot)
    point = np.array([[100.0, 0.0, 0.0]], dtype=np.float64)
    moved_origin = apply_se3(point, g_origin)
    moved_station = apply_se3(point, g_station)
    assert np.linalg.norm(moved_origin - moved_station) > 1.0


def test_cluster_local_ry_has_opposite_sign():
    ry = 0.02
    pivot = np.array([0.0, 0.0, -1860.0], dtype=np.float64)
    point = np.array([80.0, 10.0, -1850.0], dtype=np.float64)
    g_legacy = se3_from_sixvector(
        (0.0, 0.0, 0.0, 0.0, ry, 0.0),
        LEGACY_CLUSTER_LOCAL_STATION_Z_RY,
        pivot_mm=pivot,
    )
    moved = apply_se3(point[None, :], g_legacy)[0]
    expected = rotate_about_y(point, ry, pivot)
    assert np.allclose(moved, expected, atol=1.0e-12)
    g_stations = se3_from_sixvector((0.0, 0.0, 0.0, 0.0, ry, 0.0), STATIONS_GLOBAL_ORIGIN)
    assert not np.allclose(apply_se3(point[None, :], g_stations)[0], expected, atol=1.0e-3)


def test_legacy_survey_extract_is_not_a_general_inverse():
    values = (0.0, 0.0, 0.0, 0.03, -0.04, 0.05)
    transform = station_alignment_transform(values)
    extracted = extract_alpha_beta_gamma(transform)
    rebuilt = station_alignment_transform((0.0, 0.0, 0.0, *extracted))
    frobenius = float(np.linalg.norm(rebuilt[:3, :3] - transform[:3, :3]))
    assert frobenius > 1.0e-4
    correct = sixvector_from_se3(transform, STATIONS_GLOBAL_ORIGIN)
    assert np.allclose(correct[3:], values[3:], atol=1.0e-12)


def test_mm_mrad_boundary_units():
    native = np.array([1.0, 0.0, 0.0, 0.001, -0.002, 0.0], dtype=np.float64)
    report = native_to_report_mm_mrad(native)
    assert report[3] == pytest.approx(1.0)
    assert report[4] == pytest.approx(-2.0)
    assert np.allclose(report_mm_mrad_to_native(report), native)
    assert MRAD_PER_RAD == 1000.0


def test_center_to_origin_jacobian_matches_fd():
    center = np.array([12.0, -7.0, -1860.0], dtype=np.float64)
    analytic = center_to_origin_jacobian_at_identity(center)
    step = 1.0e-6
    numeric = np.zeros((6, 6), dtype=np.float64)
    numeric[:3, :3] = np.eye(3)
    numeric[3:, 3:] = np.eye(3)
    for axis in range(3):
        plus = np.zeros(3)
        minus = np.zeros(3)
        plus[axis] = step
        minus[axis] = -step
        r_plus = se3_from_sixvector((0, 0, 0, *plus), STATIONS_GLOBAL_ORIGIN)[:3, :3]
        r_minus = se3_from_sixvector((0, 0, 0, *minus), STATIONS_GLOBAL_ORIGIN)[:3, :3]
        t_plus = origin_translation_from_center((0, 0, 0), r_plus, center)
        t_minus = origin_translation_from_center((0, 0, 0), r_minus, center)
        numeric[:3, 3 + axis] = (t_plus - t_minus) / (2.0 * step)
    rel = np.linalg.norm(numeric - analytic) / np.linalg.norm(analytic)
    assert rel < 1.0e-6


def test_left_increment_is_not_euler_addition():
    first = se3_from_sixvector((0.0, 0.0, 0.0, 0.04, -0.03, 0.02), STATIONS_GLOBAL_ORIGIN)
    second = se3_from_sixvector((0.0, 0.0, 0.0, 0.01, 0.05, -0.04), STATIONS_GLOBAL_ORIGIN)
    composed = compose_left_increment(first, second)
    added = se3_from_sixvector((0.0, 0.0, 0.0, 0.05, 0.02, -0.02), STATIONS_GLOBAL_ORIGIN)
    assert not np.allclose(composed, added, atol=1.0e-8)


def test_covariance_cross_terms_from_pivot_map():
    center = np.array([10.0, 0.0, 0.0], dtype=np.float64)
    cov = np.diag([0.0, 0.0, 0.0, 0.0, 1.0e-6, 0.0])
    transported = transport_center_covariance_to_origin(cov, center)
    # ry variance maps into dx through z-component of [c]×, here c=(10,0,0)
    # [c]× ω_y = (0, 0, -10) * ω_y → dz, not dx. Use rz: [c]× ω_z = (0, 10, 0) ω_z
    cov_z = np.diag([0.0, 0.0, 0.0, 0.0, 0.0, 1.0e-6])
    moved = transport_center_covariance_to_origin(cov_z, center)
    assert moved[1, 1] == pytest.approx(1.0e-4)
    assert transported[2, 2] == pytest.approx(1.0e-4)


def test_cannot_invert_cluster_local_as_stations():
    with pytest.raises(ChartError, match="stations origin chart"):
        sixvector_from_se3(np.eye(4), LEGACY_CLUSTER_LOCAL_STATION_Z_RY)
