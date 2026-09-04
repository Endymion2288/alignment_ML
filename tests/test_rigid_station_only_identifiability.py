from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.five_dof_sampling import FREE_PARAMETERS
from alignment.identifiable_subspace import (
    FROZEN_RANK_TOLERANCE,
    MixedUnitNakedJacobianError,
    RankThresholdRetuneError,
    frozen_scales_for,
    refuse_rank_threshold_from_spectrum,
    svd_naked_jacobian,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.rigid_station_only_identifiability import (
    DECISION_CLOSURE_PASS,
    DECISION_IDENTIFIABILITY_FAIL,
    DECISION_JACOBIAN_INVALID,
    DECISION_PASS,
    DECISION_THREE_ARM_FAIL,
    FORBIDDEN_TRACKER_PARAMETERS,
    INHERITED_ENTRY_68,
    INHERITED_ENTRY_69,
    INHERITED_ENTRY_70,
    INHERITED_ENTRY_71,
    INHERITED_ENTRY_72,
    SCHEMA_VERSION,
    STATION_FIVE_NAMES,
    SevenDColumnDeletionError,
    build_all_reports,
    decide_campaign,
    diagnose_rank_loss,
    drop_seven_d_columns,
    jacobian_validity_audit,
    load_campaign_config,
    refuse_seven_d_column_deletion,
    slope_tertile_audit,
)
from alignment.tracker_only_identifiable_subspace import subspace_from_physical_bank
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_campaign_config(
        Path("configs/rigid_station_only_tracker_alignment_identifiability_v1.yaml")
    )


def _five_dof_derivative(kind="full", n=36):
    derivative = np.zeros((n, 4, 5), dtype=np.float64)
    if kind == "full":
        rng = np.random.default_rng(11)
        q, _ = np.linalg.qr(rng.normal(size=(n * 4, 5)))
        scales = frozen_scales_for(STATION_FIVE_NAMES)
        weighted = q * 3.0
        native = weighted / scales[None, :]
        return native.reshape(n, 4, 5)
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 0.4
    if kind != "rank2":
        derivative[:, 2, 2] = 0.5
        derivative[:, 3, 3] = 0.6
        derivative[:, 0, 4] = 0.3
    return derivative


def _synthetic_bank(
    derivative,
    *,
    source_id="s0",
    split="train",
    slopes=None,
    source_stations=None,
    target_stations=None,
    truth_ids=None,
):
    pairs, _dim, parameters = derivative.shape
    covariance = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], pairs, axis=0)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = np.asarray([0.5, 0.5, 10.0, 10.0, 10.0], dtype=np.float64)[:parameters]
    positive = np.zeros((parameters, pairs, 4), dtype=np.float64)
    negative = np.zeros((parameters, pairs, 4), dtype=np.float64)
    for index in range(parameters):
        positive[index] = nominal + step[index] * derivative[:, :, index]
        negative[index] = nominal - step[index] * derivative[:, :, index]
    names = STATION_FIVE_NAMES[:parameters]
    run_offset = 1000 * (sum(ord(char) for char in source_id) + 1)
    if slopes is None:
        slopes = np.linspace(1.0e-4, 3.0e-3, pairs)
    slope_arr = np.asarray(slopes, dtype=np.float64)
    if source_stations is None:
        source_stations = np.zeros(pairs, dtype=np.int64)
    if target_stations is None:
        target_stations = np.ones(pairs, dtype=np.int64)
        if pairs >= 3:
            target_stations[1::3] = 2
            target_stations[2::3] = 3
    if truth_ids is None:
        truth_ids = np.ones(pairs, dtype=np.int64)
        for index in range(pairs):
            truth_ids[index] = 1 + index // 3
    return {
        "source_id": source_id,
        "split": split,
        "specs": (),
        "names": names,
        "scales": frozen_scales_for(names),
        "anchor_values": np.zeros(parameters, dtype=np.float64),
        "reference_values": np.zeros(parameters, dtype=np.float64),
        "target_names": (),
        "target_values": {},
        "target_residuals": {},
        "positive_values": np.asarray(step, dtype=np.float64),
        "negative_values": -np.asarray(step, dtype=np.float64),
        "anchor_residual": nominal,
        "positive_residual": positive,
        "negative_residual": negative,
        "reference_residual": np.array(nominal, copy=True),
        "covariance": covariance,
        "run_id": np.full(pairs, run_offset, dtype=np.int64),
        "event_id": np.arange(pairs, dtype=np.int64),
        "source_tracklet_id": np.arange(pairs, dtype=np.int32),
        "target_tracklet_id": np.arange(pairs, dtype=np.int32) + 100,
        "truth_particle_id": np.asarray(truth_ids, dtype=np.int64),
        "source_station_id": np.asarray(source_stations, dtype=np.int64),
        "target_station_id": np.asarray(target_stations, dtype=np.int64),
        "layer_weights": np.asarray([1.0 / 3.0, 1.0 / 3.0, 1.0 / 3.0]),
        "overlap": {"pairs": int(pairs)},
        "source_tx": np.asarray(slope_arr, dtype=np.float64),
        "source_ty": np.zeros(pairs, dtype=np.float64),
        "source_slope": np.asarray(slope_arr, dtype=np.float64),
        "source_slope_missing_pairs": 0,
        "held_out_physical_points_loaded": False,
        "test_data_accessed": False,
    }


def test_config_freezes_five_dof_model_before_svd():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert tuple(config["parameter_names"]) == FREE_PARAMETERS
    assert tuple(config["parameter_names"]) == STATION_FIVE_NAMES
    assert "ift_dz_mm" not in config["parameter_names"]
    assert "C_dx" not in config["parameter_names"]
    assert tuple(FORBIDDEN_TRACKER_PARAMETERS) == ("ift_dz_mm", "C_dx")
    assert config["rank_tolerance"] == pytest.approx(FROZEN_RANK_TOLERANCE)
    assert config["scale_matrix_S"]["ift_dx_mm"] == pytest.approx(5.0)
    assert config["scale_matrix_S"]["ift_ry_mrad"] == pytest.approx(60.0)
    assert "ift_dz_mm" not in config["scale_matrix_S"]
    assert "C_dx" not in config["scale_matrix_S"]
    assert config["geometry_write_allowed"] is False
    assert config["three_arm_authorized"] is False
    assert config["frozen_v2_alignment_loop_authorized"] is False
    assert config["real_data_correction_authorized"] is False
    assert config["cdx_fixed_is_not_a_measurement_of_zero"] is True
    assert config["dz_is_not_a_tracker_free_parameter"] is True
    assert config["this_is_not_seven_d_column_deletion"] is True
    assert config["do_not_delete_dz_or_cdx_columns_from_seven_d_svd"] is True
    assert config["do_not_rescue_seven_d_observable"] is True
    assert config["do_not_rescue_cluster_local_observable"] is True
    assert config["do_not_drop_failed_sources_to_recover_rank_five"] is True
    assert config["inherited_entry_68_decision"] == INHERITED_ENTRY_68
    assert config["inherited_entry_69_decision"] == INHERITED_ENTRY_69
    assert config["inherited_entry_70_decision"] == INHERITED_ENTRY_70
    assert config["inherited_entry_71_decision"] == INHERITED_ENTRY_71
    assert config["inherited_entry_72_decision"] == INHERITED_ENTRY_72
    train = set(config["jacobian_corpus"]["train_source_ids"])
    validation = set(config["jacobian_corpus"]["validation_source_ids"])
    assert not train & validation
    assert "mc24_100043_00400_00499" in train
    assert "mc24_100043_00500_00599" in train
    assert "mc24_100044_00200_00299" in train
    assert config["jacobian_corpus"]["reconstruction_method"] == "native_five_column_only_parameters"
    assert config["coverage"]["require_no_slope_tertile_rank_drop"] is True
    assert config["gates"]["pooled_rank_is_not_portability"] is True


def test_refuses_naked_mixed_unit_svd_spectrum_retune_and_seven_d_deletion():
    with pytest.raises(MixedUnitNakedJacobianError):
        svd_naked_jacobian(np.eye(5), parameter_units=("mm", "mm", "mrad", "mrad", "mrad"))
    with pytest.raises(RankThresholdRetuneError):
        refuse_rank_threshold_from_spectrum()
    with pytest.raises(SevenDColumnDeletionError):
        refuse_seven_d_column_deletion()
    with pytest.raises(SevenDColumnDeletionError):
        drop_seven_d_columns(object())


def test_native_five_column_svd_is_not_seven_d_truncation():
    bank = _synthetic_bank(_five_dof_derivative("full"))
    subspace, extras = subspace_from_physical_bank(bank)
    assert tuple(subspace.parameter_names) == STATION_FIVE_NAMES
    assert subspace.n_parameters == 5
    assert "ift_dz_mm" not in subspace.parameter_names
    assert "C_dx" not in subspace.parameter_names
    assert extras["fit"].derivative_native.shape[2] == 5
    assert subspace.identifiable_rank == 5


def test_jacobian_validity_flags_near_zero_column():
    bank = _synthetic_bank(_five_dof_derivative("rank2"))
    subspace, extras = subspace_from_physical_bank(bank)
    audit = jacobian_validity_audit(bank, config=_config(), subspace=subspace, extras=extras)
    assert audit["valid"] is False
    assert "near_zero_jacobian_column" in audit["failure_reasons"]
    assert audit["native_five_parameters"] is True
    assert audit["not_seven_d_column_deletion"] is True


def test_slope_tertile_detects_coverage_conditioned_rank_drop():
    n = 36
    derivative = _five_dof_derivative("full", n=n)
    # Collapse the last two native columns on the lowest-slope third.
    low = slice(0, 12)
    derivative[low, :, 3:] = 0.0
    slopes = np.concatenate(
        [
            np.full(12, 1.0e-4),
            np.full(12, 1.0e-3),
            np.full(12, 3.0e-3),
        ]
    )
    bank = _synthetic_bank(derivative, slopes=slopes)
    tertiles = slope_tertile_audit(
        bank,
        rank_tolerance=FROZEN_RANK_TOLERANCE,
        rcond=1.0e-10,
        n_bins=3,
        min_events=8,
    )
    assert tertiles["too_small"] is False
    assert tertiles["rank_drop"] is True
    ranks = [row["identifiable_rank"] for row in tertiles["bins"] if not row.get("too_small")]
    assert min(ranks) < max(ranks)


def test_diagnose_rank_loss_does_not_drop_sources():
    validity = {"valid": True, "failure_reasons": []}
    diagnosis = diagnose_rank_loss(
        validity=validity,
        rank=3,
        required_rank=5,
        tertiles={"rank_drop": True},
        topology={"too_small": False, "same_rank": False},
    )
    assert diagnosis["classification"] == "normal_physics_or_coverage_loss"
    assert diagnosis["do_not_drop_source"] is True
    artifact = diagnose_rank_loss(
        validity={"valid": False, "failure_reasons": ["missing_fd_probes"]},
        rank=3,
        required_rank=5,
    )
    assert artifact["classification"] == "explicit_artifact"


def test_decide_campaign_never_authorizes_geometry_or_frozen_v2():
    invalid = decide_campaign(
        jacobian_valid=False,
        identifiability_pass=False,
        three_arm=None,
        pooled_rank=5,
        reasons=["jacobian_validity_failed"],
    )
    assert invalid["decision"] == DECISION_JACOBIAN_INVALID
    assert invalid["three_arm_authorized"] is False
    assert invalid["geometry_write_allowed"] is False
    failed = decide_campaign(
        jacobian_valid=True,
        identifiability_pass=False,
        three_arm=None,
        pooled_rank=5,
        reasons=["slope_tertile_rank_drop_or_not_five"],
    )
    assert failed["decision"] == DECISION_IDENTIFIABILITY_FAIL
    assert failed["pooled_rank_is_not_portability"] is True
    assert failed["frozen_v2_alignment_loop_authorized"] is False
    assert failed["real_data_correction_authorized"] is False
    assert failed["if_failed_do_not_chase_by_retuning_or_dropping_sources"] is True
    passed = decide_campaign(
        jacobian_valid=True,
        identifiability_pass=True,
        three_arm=None,
        pooled_rank=5,
        reasons=[],
    )
    assert passed["decision"] == DECISION_PASS
    assert passed["three_arm_authorized"] is False
    assert passed["authorize_frozen_v2_unknown_association_closure"] is False
    closed = decide_campaign(
        jacobian_valid=True,
        identifiability_pass=True,
        three_arm={"pass": True, "null_injection_leakage_gate": True, "mixed_injection_projected_closure": True},
        pooled_rank=5,
        reasons=[],
    )
    assert closed["decision"] == DECISION_CLOSURE_PASS
    assert closed["truth_selected_joint_closure_opened"] is True
    assert closed["frozen_v2_alignment_loop_authorized"] is False
    arm_fail = decide_campaign(
        jacobian_valid=True,
        identifiability_pass=True,
        three_arm={"pass": False, "null_injection_leakage_gate": False, "mixed_injection_projected_closure": False},
        pooled_rank=5,
        reasons=["truth_selected_joint_closure_failed"],
    )
    assert arm_fail["decision"] == DECISION_THREE_ARM_FAIL


def test_build_all_reports_fails_closed_on_synthetic_rank_loss():
    full = _five_dof_derivative("full", n=24)
    weak = _five_dof_derivative("rank2", n=24)
    banks = [
        _synthetic_bank(full, source_id="train_a", split="train"),
        _synthetic_bank(weak, source_id="train_b", split="train"),
        _synthetic_bank(full, source_id="val_a", split="validation"),
        _synthetic_bank(full, source_id="val_b", split="validation"),
    ]
    config = _config()
    reports = build_all_reports(config, banks=banks)
    assert_no_alignment_payload(reports["next_stage_decision"])
    decision = reports["next_stage_decision"]
    assert decision["five_dof_identifiable_and_portable"] is False
    assert decision["geometry_write_allowed"] is False
    assert decision["three_arm_authorized"] is False
    assert reports["three_arm_closure"] is None
    assert reports["rank_loss_diagnosis"]["sources_dropped"] is False
    ranks = [row["identifiable_rank"] for row in reports["identifiable_basis"]["per_source"]]
    assert min(ranks) < 5
    assert "predeclared_source_rank_is_not_five" in decision["reasons"]
