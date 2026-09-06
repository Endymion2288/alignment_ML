"""T10: zero-target WLS must not see the reference residual."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.operating_protocol_v1_final_closure import sha256_file
from alignment.reference_oracle_gap import (
    DX_DY_GATE_MM,
    RY_GATE_MRAD,
    ReferenceOracleError,
    estimate_paired_target,
    estimate_zero_target,
    evaluate_split,
    load_config,
    load_frozen_arrays,
    load_frozen_report,
    saved_paired_delta,
)


@pytest.fixture(scope="module")
def validation_arrays():
    config = load_config()
    spec = config["splits"]["validation"]
    arrays = load_frozen_arrays(spec["arrays"])
    assert arrays["sha256"] == spec["arrays_sha256"]
    return arrays


@pytest.fixture(scope="module")
def validation_report():
    config = load_config()
    spec = config["splits"]["validation"]
    report = load_frozen_report(spec["report"])
    assert sha256_file(spec["report"]) == spec["report_sha256"]
    return report


def test_e02_paired_delta_matches_saved(validation_arrays, validation_report):
    names = [str(name) for name in validation_arrays["parameter_names"].tolist()]
    paired = estimate_paired_target(
        validation_arrays["derivative_native"],
        validation_arrays["covariance"],
        validation_arrays["anchor_residual"],
        validation_arrays["target_residual"],
    )
    saved = saved_paired_delta(validation_report, names)
    assert np.allclose(paired, saved, rtol=1.0e-8, atol=1.0e-8)


def test_zero_target_runs_without_reference(validation_arrays):
    zero = estimate_zero_target(
        validation_arrays["derivative_native"],
        validation_arrays["covariance"],
        validation_arrays["anchor_residual"],
    )
    assert zero.shape == (3,)
    assert np.isfinite(zero).all()


def test_zero_target_refuses_oracle_kwargs(validation_arrays):
    with pytest.raises(ReferenceOracleError, match="target_residual"):
        estimate_zero_target(
            validation_arrays["derivative_native"],
            validation_arrays["covariance"],
            validation_arrays["anchor_residual"],
            target_residual=validation_arrays["target_residual"],
        )
    with pytest.raises(ReferenceOracleError, match="truth_displacement"):
        estimate_zero_target(
            validation_arrays["derivative_native"],
            validation_arrays["covariance"],
            validation_arrays["anchor_residual"],
            truth_displacement=np.ones(3),
        )


def test_poisoned_reference_does_not_change_zero_target(validation_arrays):
    jacobian = validation_arrays["derivative_native"]
    covariance = validation_arrays["covariance"]
    anchor = validation_arrays["anchor_residual"]
    baseline = estimate_zero_target(jacobian, covariance, anchor)
    poisoned = np.asarray(validation_arrays["target_residual"], dtype=np.float64) + 10.0
    again = estimate_zero_target(jacobian, covariance, anchor)
    paired_poison = estimate_paired_target(jacobian, covariance, anchor, poisoned)
    assert np.allclose(again, baseline)
    assert not np.allclose(paired_poison, baseline)


def test_shuffled_truth_labels_do_not_change_fixed_edges(validation_arrays):
    jacobian = validation_arrays["derivative_native"]
    covariance = validation_arrays["covariance"]
    anchor = validation_arrays["anchor_residual"]
    fake_truth = np.arange(anchor.shape[0])
    shuffled = fake_truth.copy()
    rng = np.random.default_rng(0)
    rng.shuffle(shuffled)
    left = estimate_zero_target(jacobian, covariance, anchor)
    right = estimate_zero_target(jacobian, covariance, anchor)
    del shuffled
    assert np.allclose(left, right)


def test_gates_are_frozen():
    assert DX_DY_GATE_MM == 0.1
    assert RY_GATE_MRAD == 1.0


def test_evaluate_split_marks_diagnostic(validation_arrays, validation_report):
    result = evaluate_split(validation_arrays, validation_report, split="validation")
    assert result["paired_matches_saved_1e-8"] is True
    assert result["is_correct_likelihood"] is False
    assert result["n_edges"] == 467
    assert result["zero_remaining_gate"]["all_pass"] is False
