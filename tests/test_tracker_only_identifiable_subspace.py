from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.identifiable_subspace import (
    FROZEN_RANK_TOLERANCE,
    MixedUnitNakedJacobianError,
    RankThresholdRetuneError,
    ScaleRetuneError,
    apply_mode_sign_convention,
    closure_metrics,
    flatten_physical_jacobian,
    frozen_scales_for,
    frozen_units_for,
    identifiable_svd,
    inject_identifiable,
    inject_null,
    native_to_scaled,
    principal_angles_deg,
    project_native,
    refuse_naked_mixed_unit_svd,
    refuse_rank_threshold_from_spectrum,
    refuse_scale_or_threshold_retune,
    solve_identifiable_amplitudes,
    subspace_distance,
    svd_naked_jacobian,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.tracker_only_identifiable_subspace import (
    DECISION_ARM_FAIL,
    DECISION_BASIS_UNSTABLE,
    DECISION_PASS,
    SCHEMA_VERSION,
    build_all_reports,
    decide_campaign,
    evaluate_three_arms,
    load_campaign_config,
    subspace_from_physical_bank,
    summarize_stability,
    three_arm_payloads,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_campaign_config(
        Path("configs/tracker_only_identifiable_subspace_three_arm_closure_v1.yaml")
    )


def _synthetic_derivative():
    # Three physical parameters with mixed units.  Column 2 is exactly
    # parallel to column 0, so the identifiable subspace is 2-D.
    derivative = np.zeros((6, 4, 3), dtype=np.float64)
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 0.5
    derivative[:, 0, 2] = 1.0
    return derivative


def _synthetic_bank(derivative, *, source_id="s0", split="train", scales=(5.0, 60.0, 0.12)):
    pairs, _dim, parameters = derivative.shape
    covariance = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], pairs, axis=0)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = 1.0
    positive = np.zeros((parameters, pairs, 4), dtype=np.float64)
    negative = np.zeros((parameters, pairs, 4), dtype=np.float64)
    for index in range(parameters):
        positive[index] = nominal + step * derivative[:, :, index]
        negative[index] = nominal - step * derivative[:, :, index]
    names = ("ift_dx_mm", "ift_ry_mrad", "C_dx")[:parameters]
    run_offset = 1000 * (sum(ord(char) for char in source_id) + 1)
    return {
        "source_id": source_id,
        "split": split,
        "specs": (),
        "names": names,
        "scales": np.asarray(scales[:parameters], dtype=np.float64),
        "anchor_values": np.zeros(parameters, dtype=np.float64),
        "reference_values": np.zeros(parameters, dtype=np.float64),
        "target_names": (),
        "target_values": {},
        "target_residuals": {},
        "positive_values": np.full(parameters, step, dtype=np.float64),
        "negative_values": np.full(parameters, -step, dtype=np.float64),
        "anchor_residual": nominal,
        "positive_residual": positive,
        "negative_residual": negative,
        "reference_residual": np.array(nominal, copy=True),
        "covariance": covariance,
        "run_id": np.arange(pairs, dtype=np.int64) + run_offset,
        "event_id": np.arange(pairs, dtype=np.int64),
        "truth_particle_id": np.ones(pairs, dtype=np.int64),
        "source_station_id": np.zeros(pairs, dtype=np.int64),
        "target_station_id": np.ones(pairs, dtype=np.int64),
        "layer_weights": np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        "overlap": {"pairs": int(pairs)},
    }


def test_config_freezes_protocol_and_forbids_geometry_write():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["geometry_write_allowed"] is False
    assert config["survey_is_alignment_input"] is False
    assert config["survey_is_external_cross_check_only"] is True
    assert config["do_not_run_full_parameter_newton"] is True
    assert config["do_not_svd_naked_mixed_unit_jacobian"] is True
    assert config["do_not_retune_scales_from_singular_values"] is True
    assert config["do_not_retune_rank_threshold_from_spectrum"] is True
    assert config["solver_restricted_to_identifiable_subspace"] is True
    assert config["truth_selected_association_is_solver_control"] is True
    assert config["unknown_association_frozen_v2_closure_opened"] is False
    assert config["rank_tolerance"] == pytest.approx(FROZEN_RANK_TOLERANCE)
    assert config["scale_matrix_S"]["ift_dx_mm"] == pytest.approx(5.0)
    assert config["scale_matrix_S"]["ift_ry_mrad"] == pytest.approx(60.0)
    assert config["scale_matrix_S"]["C_dx"] == pytest.approx(0.12)
    assert config["injections"]["identifiable_scaled_amplitude"] < 0.15


def test_refuses_naked_mixed_unit_svd_and_spectrum_retune():
    with pytest.raises(MixedUnitNakedJacobianError):
        refuse_naked_mixed_unit_svd(("mm", "mrad", "mm"))
    with pytest.raises(MixedUnitNakedJacobianError):
        svd_naked_jacobian(np.eye(3), parameter_units=("mm", "mrad", "mm"))
    with pytest.raises(RankThresholdRetuneError):
        refuse_rank_threshold_from_spectrum()
    with pytest.raises(ScaleRetuneError):
        refuse_scale_or_threshold_retune(from_singular_values=True)
    with pytest.raises(RankThresholdRetuneError):
        identifiable_svd(
            np.eye(3),
            parameter_names=("ift_dx_mm", "ift_dy_mm", "ift_dz_mm"),
            parameter_units=("mm", "mm", "mm"),
            parameter_scales=(5.0, 5.0, 5.0),
            rank_tolerance=1.0e-4,
        )


def test_weighted_scaled_svd_recovers_rank_two_and_null_of_parallel_column():
    derivative = _synthetic_derivative()
    covariance = np.repeat(np.eye(4)[None], derivative.shape[0], axis=0)
    names = ("ift_dx_mm", "ift_ry_mrad", "C_dx")
    scales = frozen_scales_for(names)
    units = frozen_units_for(names)
    weighted, _flat, _blocks = flatten_physical_jacobian(derivative, covariance, scales)
    subspace = identifiable_svd(
        weighted,
        parameter_names=names,
        parameter_units=units,
        parameter_scales=scales,
    )
    assert subspace.identifiable_rank == 2
    assert subspace.null_dimension == 1
    null = subspace.v_null[:, 0]
    # A = J S, so the native degeneracy J[:,0] || J[:,2] becomes
    # s0 v0 + s2 v2 = 0 in scaled coordinates: v ∝ (s2, 0, -s0).
    expected = np.array([scales[2], 0.0, -scales[0]], dtype=np.float64)
    expected = expected / np.linalg.norm(expected)
    assert abs(float(null @ expected)) == pytest.approx(1.0, abs=1.0e-8)
    assert subspace.mode_signs
    assert all(int(sign) in (-1, 1) for sign in subspace.mode_signs)
    leading = int(np.argmax(np.abs(subspace.v[:, 0])))
    assert subspace.v[leading, 0] > 0.0


def test_sign_convention_is_deterministic_and_principal_angles_ignore_sign():
    matrix = np.array([[1.0, 0.0], [0.0, 2.0], [0.0, 0.0]], dtype=np.float64)
    left, singular, right_t = np.linalg.svd(matrix, full_matrices=False)
    flipped_left = np.array(left, copy=True)
    flipped_right = np.array(right_t.T, copy=True)
    flipped_left[:, 0] *= -1.0
    flipped_right[:, 0] *= -1.0
    a_u, a_v, a_signs = apply_mode_sign_convention(left, right_t.T)
    b_u, b_v, b_signs = apply_mode_sign_convention(flipped_left, flipped_right)
    assert np.allclose(a_v, b_v)
    assert np.allclose(a_u, b_u)
    assert a_signs[0] * b_signs[0] in (-1, 1)
    angles = principal_angles_deg(a_v, -a_v)
    assert float(np.max(angles)) == pytest.approx(0.0, abs=1.0e-12)


def test_solver_stays_in_identifiable_subspace_and_null_is_gauge_not_measurement():
    derivative = _synthetic_derivative()
    bank = _synthetic_bank(derivative)
    subspace, extras = subspace_from_physical_bank(bank)
    payloads = three_arm_payloads(
        subspace,
        identifiable_scaled_amplitude=0.10,
        null_scaled_amplitude=0.10,
        seed=7,
        n_replicates=3,
    )
    result = evaluate_three_arms(
        subspace,
        extras["weighted_matrix"],
        payloads,
        gates={
            "arm_a_max_projected_error_relative": 1.0e-8,
            "arm_a_max_amplitude_error": 1.0e-8,
            "arm_b_max_identifiable_fake_relative": 1.0e-8,
            "arm_c_max_projector_closure_relative": 1.0e-8,
        },
    )
    assert result["pass"] is True
    assert result["null_injection_leakage_gate"] is True
    assert result["mixed_injection_projected_closure"] is True
    mixed_truth = payloads[0]["theta_c_native"]
    mixed_hat = np.asarray(result["arm_c_mixed_injection"]["replicates"][0]["q_hat_native"])
    projected = project_native(subspace, mixed_truth)
    assert np.allclose(mixed_hat, projected, atol=1.0e-10)
    assert not np.allclose(mixed_hat, mixed_truth, atol=1.0e-6)
    null_hat = native_to_scaled(mixed_hat, subspace.parameter_scales)
    assert float(np.linalg.norm(subspace.v_null.T @ null_hat)) == pytest.approx(0.0, abs=1.0e-12)


def test_null_injection_does_not_create_identifiable_fake_on_perturbed_generating_matrix():
    derivative = _synthetic_derivative()
    bank = _synthetic_bank(derivative)
    subspace, extras = subspace_from_physical_bank(bank)
    generating = np.array(extras["weighted_matrix"], copy=True)
    rng = np.random.default_rng(11)
    generating = generating + 1.0e-4 * rng.normal(size=generating.shape)
    payloads = [
        {
            "replicate": 0,
            "arm_a_identifiable_amplitudes": np.array([0.08, -0.04]),
            "arm_b_null_amplitudes": np.array([0.10]),
            "theta_a_native": inject_identifiable(subspace, [0.08, -0.04]),
            "theta_b_native": inject_null(subspace, [0.10]),
            "theta_c_native": inject_identifiable(subspace, [0.08, -0.04])
            + inject_null(subspace, [0.10]),
        }
    ]
    result = evaluate_three_arms(
        subspace,
        generating,
        payloads,
        gates={
            "arm_a_max_projected_error_relative": 0.05,
            "arm_a_max_amplitude_error": 0.05,
            "arm_b_max_identifiable_fake_relative": 0.05,
            "arm_c_max_projector_closure_relative": 0.05,
        },
    )
    assert result["arm_b_null_injection"]["pass"] is True
    assert result["arm_c_mixed_injection"]["pass"] is True


def test_source_stability_uses_principal_angles_not_signed_vector_elements():
    derivative = _synthetic_derivative()
    left, extras = subspace_from_physical_bank(_synthetic_bank(derivative, source_id="a"))
    right, _ = subspace_from_physical_bank(_synthetic_bank(derivative, source_id="b"))
    right_flipped = identifiable_svd(
        extras["weighted_matrix"],
        parameter_names=left.parameter_names,
        parameter_units=left.parameter_units,
        parameter_scales=left.parameter_scales,
    )
    # Compare a sign-flipped identifiable basis against the original projector.
    flipped_vid = -right_flipped.v_id
    angles = principal_angles_deg(left.v_id, flipped_vid)
    assert float(np.max(angles)) == pytest.approx(0.0, abs=1.0e-8)
    distance = subspace_distance(left, right)
    assert distance["compares_subspace_not_signed_vector_elements"] is True
    assert distance["same_rank"] is True
    summary = summarize_stability(
        left,
        [("b", right)],
        max_principal_angle_deg=15.0,
        max_projector_frobenius=1.0,
        require_same_rank=True,
    )
    assert summary["stable"] is True


def test_unstable_basis_stops_solve_and_does_not_authorize_v2():
    decision = decide_campaign(identifiable_rank=5, basis_stable=False, three_arm=None)
    assert decision["decision"] == DECISION_BASIS_UNSTABLE
    assert decision["identifiable_basis_stable"] is False
    assert decision["authorize_frozen_v2_unknown_association_closure"] is False
    assert decision["geometry_write_allowed"] is False
    assert decision["if_failed_do_not_chase_by_retuning"] is True
    failed_arms = decide_campaign(
        identifiable_rank=5,
        basis_stable=True,
        three_arm={"pass": False, "null_injection_leakage_gate": False, "mixed_injection_projected_closure": False},
    )
    assert failed_arms["decision"] == DECISION_ARM_FAIL
    passed = decide_campaign(
        identifiable_rank=5,
        basis_stable=True,
        three_arm={"pass": True, "null_injection_leakage_gate": True, "mixed_injection_projected_closure": True},
    )
    assert passed["decision"] == DECISION_PASS
    assert passed["authorize_frozen_v2_unknown_association_closure"] is True
    assert passed["real_data_alignment_correction_authorized"] is False


def test_build_all_reports_on_synthetic_source_disjoint_banks():
    derivative = _synthetic_derivative()
    train_a = _synthetic_bank(derivative, source_id="train_a", split="train")
    train_b = _synthetic_bank(derivative * 1.02, source_id="train_b", split="train")
    val = _synthetic_bank(derivative * 0.98, source_id="val_a", split="validation")
    config = _config()
    config["bootstrap"]["n_event_replicates"] = 4
    config["bootstrap"]["n_half_splits"] = 2
    config["bootstrap"]["min_pairs"] = 2
    config["injections"]["n_replicates"] = 3
    config["cluster_local"]["enabled"] = False
    reports = build_all_reports(config, banks=[train_a, train_b, val])
    assert_no_alignment_payload(reports["next_stage_decision"])
    assert reports["identifiable_basis"]["identifiable_rank"] == 2
    assert reports["identifiable_basis"]["not_mechanical_ry_or_C_dx"] is True
    assert reports["subspace_stability"]["identifiable_basis_stable"] is True
    assert reports["three_arm_closure"]["pass"] is True
    assert reports["three_arm_closure"]["athena_not_rerun"] is True
    assert reports["three_arm_closure"]["association_control"] == "truth_selected_physical_edge"
    assert reports["next_stage_decision"]["decision"] == DECISION_PASS
    assert reports["next_stage_decision"]["geometry_write_allowed"] is False
    assert reports["parameter_definition"]["operating_state"]["survey_is_alignment_input"] is False


def test_diagonal_mixed_unit_jacobian_is_scaled_before_svd():
    jacobian = np.column_stack(
        [
            np.ones(8),
            np.linspace(-1.0, 1.0, 8),
            np.array([1.0, 0.0, -1.0, 0.0] * 2),
        ]
    )
    names = ("station_dx_mm", "station_ry_mrad", "C_dx_mm")
    from alignment.identifiable_subspace import CLUSTER_LOCAL_SCALE_MAP, flatten_diagonal_jacobian

    scales = frozen_scales_for(names, CLUSTER_LOCAL_SCALE_MAP)
    variance = np.full(8, 0.01, dtype=np.float64)
    weighted, _flat, _w = flatten_diagonal_jacobian(jacobian, variance, scales)
    subspace = identifiable_svd(
        weighted,
        parameter_names=names,
        parameter_units=frozen_units_for(names),
        parameter_scales=scales,
    )
    assert subspace.n_parameters == 3
    assert subspace.identifiable_rank >= 2
    assert subspace.parameter_units == ("mm", "mrad", "mm")


def test_cluster_local_empty_jacobian_is_skipped_not_a_gate():
    from alignment.tracker_only_identifiable_subspace import cluster_local_subspace

    with pytest.raises(ValueError, match="shape"):
        cluster_local_subspace(
            [],
            station_id=0,
            steps={"translation_step_mm": 0.01, "rotation_step_rad": 5.0e-5, "c_dx_step_mm": 0.01},
            rank_tolerance=FROZEN_RANK_TOLERANCE,
        )


def test_closure_metrics_do_not_use_full_physical_error_as_gate():
    names = ("ift_dx_mm", "ift_ry_mrad", "C_dx")
    scales = frozen_scales_for(names)
    weighted = np.array(
        [
            [1.0, 0.0, 1.0],
            [0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    subspace = identifiable_svd(
        weighted,
        parameter_names=names,
        parameter_units=frozen_units_for(names),
        parameter_scales=scales,
    )
    truth = inject_identifiable(subspace, [0.1, 0.0]) + inject_null(subspace, [0.2])
    solved = solve_identifiable_amplitudes(
        subspace,
        (weighted @ native_to_scaled(truth, scales)),
    )
    metrics = closure_metrics(
        subspace,
        q_hat_native=solved["native"],
        q_truth_native=truth,
        weighted_residual=weighted @ native_to_scaled(truth, scales),
        predicted_weighted_residual=solved["predicted_weighted_residual"],
    )
    assert metrics["per_physical_parameter_truth_error_is_not_the_gate"] is True
    assert metrics["null_component_is_not_a_measurement_of_zero"] is True
    assert metrics["projected_identifiable_error_norm"] == pytest.approx(0.0, abs=1.0e-12)
    assert metrics["native_full_parameter_error_norm_not_a_gate"] > 0.0
