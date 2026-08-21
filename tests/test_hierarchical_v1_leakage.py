from __future__ import annotations

import math

import numpy as np
import pytest

from alignment.hierarchical_v1 import C_DX
from alignment.hierarchical_v1_leakage import (
    MICRON_MM,
    cdx_budget_from_operator,
    cdx_systematic_from_station_sigma,
    compare_station_estimators,
    estimator_diagnostics,
    leakage_operator_from_normal,
    predicted_cdx_bias,
    predicted_station_bias,
    profiled_right_hand_side,
    realizability_from_budget,
    reverse_leakage_from_A_and_covariance,
    schur_profiled_station_normal,
    sigma_inflation,
    slice_parameter_bank,
)
from alignment.physical_jacobian import solve_physical_finite_difference


def test_leakage_operator_matches_explicit_normal_blocks():
    names = ("ift_dx_mm", "ift_ry_mrad", C_DX)
    normal = np.array(
        [
            [4.0, 0.5, 2.0],
            [0.5, 9.0, -3.0],
            [2.0, -3.0, 10.0],
        ],
        dtype=np.float64,
    )
    operator = leakage_operator_from_normal(
        normal,
        names,
        station_names=("ift_dx_mm", "ift_ry_mrad"),
    )
    n_ss = normal[:2, :2]
    n_sc = normal[:2, 2]
    expected = np.linalg.solve(n_ss, n_sc)
    assert operator.A_native_per_mm["ift_dx_mm"] == pytest.approx(expected[0])
    assert operator.A_native_per_mm["ift_ry_mrad"] == pytest.approx(expected[1])
    assert operator.A_per_um["ift_dx_mm"] == pytest.approx(expected[0] * MICRON_MM)
    assert operator.disguised_same_unit_per_um["ift_dx_mm"] == pytest.approx(expected[0])
    assert operator.full_rank is True
    r2 = float(n_sc @ np.linalg.solve(n_ss, n_sc) / normal[2, 2])
    assert operator.subspace_r2 == pytest.approx(r2)
    assert operator.subspace_cosine == pytest.approx(math.sqrt(r2))


def test_leakage_operator_with_dz_prior_matches_augmented_normal():
    names = ("ift_dx_mm", "ift_dz_mm", C_DX)
    normal = np.array(
        [
            [4.0, 0.2, 8.0],
            [0.2, 0.01, 0.4],
            [8.0, 0.4, 20.0],
        ],
        dtype=np.float64,
    )
    prior = {"ift_dz_mm": 5.0}
    operator = leakage_operator_from_normal(
        normal,
        names,
        station_names=("ift_dx_mm", "ift_dz_mm"),
        prior_sigma_native=prior,
        parameter_scales=(5.0, 5.0, 0.12),
    )
    n_ss = np.array([[4.0, 0.2], [0.2, 0.01 + 1.0 / 25.0]])
    expected = np.linalg.solve(n_ss, np.array([8.0, 0.4]))
    assert operator.A_native_per_mm["ift_dx_mm"] == pytest.approx(expected[0])
    assert operator.A_native_per_mm["ift_dz_mm"] == pytest.approx(expected[1])
    assert operator.used_prior_sigma_native == prior


def test_omitted_variable_bias_is_A_times_C_dx():
    rng = np.random.default_rng(20260821)
    pairs = 24
    jacobian = rng.normal(size=(pairs, 4, 3))
    jacobian[:, :, 2] = -59.0 * jacobian[:, :, 0] - 32.0 * jacobian[:, :, 1] + 0.05 * rng.normal(
        size=(pairs, 4)
    )
    station = np.array([0.43, -0.34])
    c_dx = -0.0917
    theta = np.array([station[0], station[1], c_dx])
    observed = np.einsum("nrp,p->nr", jacobian, theta)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = 0.1
    positive = np.stack([nominal + step * jacobian[:, :, index] for index in range(3)])
    negative = np.stack([nominal - step * jacobian[:, :, index] for index in range(3)])
    covariance = np.repeat(np.eye(4)[None, :, :], pairs, axis=0)
    names = ("ift_dx_mm", "ift_ry_mrad", C_DX)
    joint = solve_physical_finite_difference(
        nominal,
        positive,
        negative,
        observed,
        covariance,
        parameter_names=names,
        positive_values=np.full(3, step),
        negative_values=np.full(3, -step),
        parameter_scales=np.array([5.0, 60.0, 0.12]),
    )
    ordinary = solve_physical_finite_difference(
        nominal,
        positive[:2],
        negative[:2],
        observed,
        covariance,
        parameter_names=names[:2],
        positive_values=np.full(2, step),
        negative_values=np.full(2, -step),
        parameter_scales=np.array([5.0, 60.0]),
    )
    operator = leakage_operator_from_normal(
        joint.normal_matrix_native,
        names,
        station_names=names[:2],
        parameter_scales=np.array([5.0, 60.0, 0.12]),
    )
    predicted = predicted_station_bias(operator, c_dx)
    for index, name in enumerate(names[:2]):
        error = float(ordinary.recovered_parameters[index]) - float(station[index])
        assert error == pytest.approx(predicted[name], rel=1.0e-6, abs=1.0e-8)
    assert operator.A_native_per_mm["ift_dx_mm"] == pytest.approx(-59.0, rel=0.05)
    assert operator.A_native_per_mm["ift_ry_mrad"] == pytest.approx(-32.0, rel=0.05)
    assert operator.subspace_r2 > 0.99
    reverse = operator.reverse_A_native()
    n_sc = operator.normal_sc_native
    assert reverse["ift_dx_mm"] == pytest.approx(n_sc[0] / operator.normal_cc_native)
    recovered_b = reverse_leakage_from_A_and_covariance(
        A_native_per_mm=operator.A_native_per_mm,
        station_covariance_native={
            names[i]: {names[j]: float(ordinary.covariance_native[i, j]) for j in range(2)}
            for i in range(2)
        },
        normal_cc_native=operator.normal_cc_native,
        station_names=names[:2],
    )
    assert recovered_b["ift_dx_mm"] == pytest.approx(reverse["ift_dx_mm"], rel=1.0e-6, abs=1.0e-10)
    leftover = {"ift_dx_mm": 0.1, "ift_ry_mrad": 0.0}
    assert predicted_cdx_bias(reverse, leftover) == pytest.approx(reverse["ift_dx_mm"] * 0.1)
    systematic = cdx_systematic_from_station_sigma(
        reverse, {"ift_dx_mm": 0.03, "ift_ry_mrad": 0.1}, include=("ift_dx_mm",)
    )
    assert systematic["rss_1sigma_mm"] == pytest.approx(abs(reverse["ift_dx_mm"]) * 0.03)


def test_schur_profiled_station_matches_joint_nuisance_column():
    rng = np.random.default_rng(7)
    pairs = 16
    jacobian = rng.normal(size=(pairs, 4, 3))
    theta = np.array([0.2, 1.5, 0.04])
    observed = np.einsum("nrp,p->nr", jacobian, theta)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = 1.0
    positive = np.stack([nominal + step * jacobian[:, :, index] for index in range(3)])
    negative = np.stack([nominal - step * jacobian[:, :, index] for index in range(3)])
    covariance = np.repeat(np.eye(4)[None, :, :], pairs, axis=0)
    names = ("ift_dx_mm", "ift_ry_mrad", C_DX)
    joint = solve_physical_finite_difference(
        nominal,
        positive,
        negative,
        observed,
        covariance,
        parameter_names=names,
        positive_values=np.full(3, step),
        negative_values=np.full(3, -step),
        parameter_scales=np.ones(3),
    )
    profiled_normal, _n_sc, n_cc = schur_profiled_station_normal(
        joint.normal_matrix_native,
        names,
        station_names=names[:2],
    )
    rhs = profiled_right_hand_side(
        joint.right_hand_side_native,
        names,
        station_names=names[:2],
        normal_native=joint.normal_matrix_native,
    )
    recovered = np.linalg.solve(profiled_normal, rhs)
    assert recovered[0] == pytest.approx(joint.recovered_parameters[0])
    assert recovered[1] == pytest.approx(joint.recovered_parameters[1])
    assert n_cc == pytest.approx(joint.normal_matrix_native[2, 2])


def test_near_collinear_contrast_inflates_profiled_sigma_and_condition():
    rng = np.random.default_rng(11)
    pairs = 30
    jacobian = rng.normal(size=(pairs, 4, 3))
    jacobian[:, :, 2] = -59.0 * jacobian[:, :, 0] - 32.0 * jacobian[:, :, 1] + 1.0e-4 * rng.normal(
        size=(pairs, 4)
    )
    observed = np.einsum("nrp,p->nr", jacobian, np.array([0.1, 0.2, 0.03]))
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = 0.1
    positive = np.stack([nominal + step * jacobian[:, :, index] for index in range(3)])
    negative = np.stack([nominal - step * jacobian[:, :, index] for index in range(3)])
    covariance = np.repeat(np.eye(4)[None, :, :], pairs, axis=0)
    names = ("ift_dx_mm", "ift_ry_mrad", C_DX)
    joint = solve_physical_finite_difference(
        nominal,
        positive,
        negative,
        observed,
        covariance,
        parameter_names=names,
        positive_values=np.full(3, step),
        negative_values=np.full(3, -step),
        parameter_scales=np.array([5.0, 60.0, 0.12]),
    )
    ordinary_normal = joint.normal_matrix_native[:2, :2]
    profiled_normal, _, _ = schur_profiled_station_normal(
        joint.normal_matrix_native,
        names,
        station_names=names[:2],
    )
    ordinary = estimator_diagnostics(
        ordinary_normal,
        names[:2],
        parameter_scales=(5.0, 60.0),
    )
    profiled = estimator_diagnostics(
        profiled_normal,
        names[:2],
        parameter_scales=(5.0, 60.0),
    )
    comparison = compare_station_estimators(ordinary=ordinary, profiled=profiled, five_names=names[:2])
    raw_ratio = comparison["raw_condition_ratio_profiled_over_ordinary"]
    assert comparison["profiled_rank"] < comparison["ordinary_rank"] or (
        raw_ratio is not None and raw_ratio > 10.0
    )
    assert comparison["five_dof_profiled_looks_full"] is False or (
        raw_ratio is not None and raw_ratio > 10.0
    )


def test_cdx_budget_inverts_frozen_dx_tolerance():
    names = ("ift_dx_mm", "ift_ry_mrad", C_DX)
    normal = np.array(
        [
            [1.0, 0.0, -59.0],
            [0.0, 1.0, -32.0],
            [-59.0, -32.0, 59.0**2 + 32.0**2],
        ],
        dtype=np.float64,
    )
    operator = leakage_operator_from_normal(
        normal,
        names,
        station_names=("ift_dx_mm", "ift_ry_mrad"),
    )
    budget = cdx_budget_from_operator(
        operator,
        engineering_tolerance={"ift_dx_mm": 0.1, "ift_ry_mrad": 1.0},
        statistical_limit={"ift_dx_mm": 3.0 * 0.02886, "ift_ry_mrad": 3.0 * 0.1114},
    )
    assert budget["binding_engineering"]["name"] == "ift_dx_mm"
    assert budget["binding_engineering"]["max_abs_C_dx_um"] == pytest.approx(1.0e3 * 0.1 / 59.0, rel=1.0e-12)
    verdict = realizability_from_budget(
        binding_engineering_um=budget["binding_engineering"]["max_abs_C_dx_um"],
        binding_statistical_um=budget["binding_statistical"]["max_abs_C_dx_um"],
        registered_cdx_sigma_um=6.91,
        leftover_cdx_um=(6.2, 8.4),
    )
    assert verdict["sequential_hierarchy_statistically_realizable"] is False
    assert verdict["registered_sigma_below_engineering_budget"] is False


def test_slice_parameter_bank_keeps_aligned_rows():
    bank = {
        "names": ("ift_dx_mm", "ift_ry_mrad", C_DX),
        "scales": np.array([5.0, 60.0, 0.12]),
        "specs": [{"name": "ift_dx_mm"}, {"name": "ift_ry_mrad"}, {"name": C_DX}],
        "positive_residual": np.zeros((3, 2, 4)),
        "negative_residual": np.ones((3, 2, 4)),
        "positive_values": np.array([0.5, 10.0, 0.1]),
        "negative_values": np.array([-0.5, -10.0, -0.1]),
        "anchor_values": np.zeros(3),
        "reference_values": np.array([0.4, -0.3, -0.09]),
        "target_values": {"iteration_00_start": np.array([0.4, -0.3, -0.09])},
        "anchor_residual": np.zeros((2, 4)),
    }
    sliced = slice_parameter_bank(bank, ("ift_dx_mm", "ift_ry_mrad"))
    assert sliced["names"] == ("ift_dx_mm", "ift_ry_mrad")
    assert sliced["positive_residual"].shape == (2, 2, 4)
    assert sliced["reference_values"][0] == pytest.approx(0.4)
    assert C_DX not in sliced["names"]
    inflation = sigma_inflation({"ift_dx_mm": 0.03}, {"ift_dx_mm": 0.12})
    assert inflation["ift_dx_mm"] == pytest.approx(4.0)
