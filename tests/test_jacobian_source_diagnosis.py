from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import pytest

from scripts.audit_multisource_jacobian_sources import (
    _axis_quadratic_coefficients,
    _correlation_summary,
    _production_group,
    _quadratic_corrected_solve,
)


def _synthetic_quadratic_bank(*, curvature=(0.0, 0.0, 0.0), seed=7):
    """Bank whose per-pair responses are exactly axis-quadratic in each parameter."""
    rng = np.random.default_rng(seed)
    pairs = 40
    names = ("ift_dx_mm", "ift_dy_mm", "ift_ry_mrad")
    anchor_values = np.array([2.0, -1.5, 35.0])
    target_values = np.array([0.0, 0.0, 0.0])
    steps = np.array([0.5, 0.5, 10.0])
    positive_values = anchor_values + steps
    negative_values = anchor_values - steps
    slopes = rng.normal(size=(len(names), pairs, 4))
    intercept = rng.normal(size=(pairs, 4))
    curvature = np.asarray(curvature, dtype=np.float64)[:, None, None]

    def probe_response(values):  # values: (3,) absolute probe values -> (3, pairs, 4)
        u = (np.asarray(values) - anchor_values)[:, None, None]
        return intercept[None, :, :] + slopes * u + curvature * u**2

    def joint_response(values):  # all parameters move together -> (pairs, 4)
        u = (np.asarray(values) - anchor_values)[:, None, None]
        return intercept + (slopes * u + curvature * u**2).sum(axis=0)

    covariance = np.repeat(np.eye(4)[None, :, :], pairs, axis=0)
    bank = SimpleNamespace(
        parameter_names=names,
        parameter_scales=np.array([5.0, 5.0, 60.0]),
        anchor_values=anchor_values,
        target_values=target_values,
        positive_values=positive_values,
        negative_values=negative_values,
        anchor_residual=joint_response(anchor_values),
        positive_residual=probe_response(positive_values),
        negative_residual=probe_response(negative_values),
        target_residual=joint_response(target_values),
        covariance=covariance,
    )
    bank.true_slopes = slopes
    bank.true_curvature = curvature
    return bank


def test_axis_quadratic_recovers_known_coefficients():
    bank = _synthetic_quadratic_bank(curvature=(0.3, -0.2, 0.05))
    c1, c2 = _axis_quadratic_coefficients(
        bank.anchor_residual,
        bank.positive_residual,
        bank.negative_residual,
        bank.positive_values,
        bank.negative_values,
        bank.anchor_values,
    )
    assert np.allclose(c1, bank.true_slopes, rtol=1e-10, atol=1e-12)
    assert np.allclose(c2, bank.true_curvature, rtol=1e-10, atol=1e-12)


def test_quadratic_corrected_solve_recovers_truth_under_pure_axis_curvature():
    bank = _synthetic_quadratic_bank(curvature=(0.05, 0.03, 0.001))
    mask = np.ones(bank.anchor_residual.shape[0], dtype=bool)
    result = _quadratic_corrected_solve(bank, mask, rcond=1.0e-12)
    true_delta = bank.target_values - bank.anchor_values
    linear = np.array([result["linear_delta"][name] for name in bank.parameter_names])
    corrected = np.array(
        [result["quadratic_corrected_delta"][name] for name in bank.parameter_names]
    )
    predicted_bias = np.array(
        [result["axis_quadratic_predicted_bias"][name] for name in bank.parameter_names]
    )
    linear_bias = linear - true_delta
    corrected_bias = corrected - true_delta
    # The linear solve is biased by the curvature; the corrected solve nearly removes it.
    assert np.all(np.abs(corrected_bias) < np.maximum(np.abs(linear_bias) * 0.2, 1e-6))
    # The deterministic prediction tracks the actual linear bias.
    assert np.allclose(predicted_bias, linear_bias, rtol=0.2, atol=1e-3)


def test_linear_bank_has_zero_predicted_bias():
    bank = _synthetic_quadratic_bank(curvature=(0.0, 0.0, 0.0))
    mask = np.ones(bank.anchor_residual.shape[0], dtype=bool)
    result = _quadratic_corrected_solve(bank, mask, rcond=1.0e-12)
    for name in bank.parameter_names:
        assert abs(result["axis_quadratic_predicted_bias"][name]) < 1e-8
        assert abs(result["linear_bias"][name]) < 1e-8


def test_correlation_summary_handles_constant_input():
    assert _correlation_summary([(1.0, 2.0), (1.0, 3.0)])["pearson"] is None
    summary = _correlation_summary([(float(i), float(i) ** 2) for i in range(10)])
    assert summary["n"] == 10
    assert summary["spearman"] == pytest.approx(1.0)


def test_production_group_parsing():
    assert _production_group("mc24_100043_00200_00299") == "100043"
    assert _production_group("mc24_100048_00150_00199") == "100048"
    assert _production_group("something_else") == "other"
