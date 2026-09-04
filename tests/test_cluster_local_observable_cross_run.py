from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.cluster_local_jacobian_transfer_repair import EXACT_JOIN_KEY, INHERITED_ENTRY_58
from alignment.cluster_local_observable_cross_run import (
    DECISION_JACOBIAN_INVALID,
    DECISION_NOT_PORTABLE,
    DECISION_TRANSFERABLE,
    INHERITED_ENTRY_68,
    INHERITED_ENTRY_69,
    INHERITED_ENTRY_70,
    SCHEMA_VERSION,
    build_all_reports,
    classify_failure_cause,
    compare_official_subspaces,
    decide_campaign,
    jacobian_validity_audit,
    load_campaign_config,
    official_subspace_from_native,
)
from alignment.identifiable_subspace import (
    CLUSTER_LOCAL_NATIVE_NAMES,
    FROZEN_RANK_TOLERANCE,
    FuzzyJoinError,
    MixedUnitNakedJacobianError,
    frozen_scales_for,
    refuse_fuzzy_or_nearest_neighbour_join,
    svd_naked_jacobian,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload
from alignment.true_cluster_local_stability_transfer import DECISION_NOT_TRANSFERABLE


def _config():
    return load_campaign_config(
        Path("configs/cluster_local_observable_cross_run_identifiability_v1.yaml")
    )


def _full_rank_native(n=36, seed=7):
    rng = np.random.default_rng(seed)
    # Construct A-space orthonormal columns, then undo frozen S so the
    # official SVD of W^{1/2} J S is rank 3 at the frozen cut.
    q, _ = np.linalg.qr(rng.normal(size=(n, 3)))
    scales = frozen_scales_for(CLUSTER_LOCAL_NATIVE_NAMES)
    return q / scales[None, :]


def _rank_drop_native(n=36):
    jacobian = np.zeros((n, 3), dtype=np.float64)
    jacobian[:, 0] = 1.0
    jacobian[:, 2] = 1.0
    return jacobian


class _FakeResidual:
    def __init__(self, run_id, event_id, route_index, slope, module_id="s0_l0_e1_p3"):
        self.run_id = run_id
        self.event_id = event_id
        self.route_index = route_index
        self.module_id = module_id
        self.local_u_var_mm2 = 0.01
        self.unbiased_residual_u_mm = 0.0
        self.track_direction = np.asarray([slope, 0.0, 1.0], dtype=np.float64)


def _measurements(n=36, slope_pattern="spread"):
    rows = []
    for index in range(n):
        if slope_pattern == "spread":
            slope = 1.0e-4 * (1 + index)
        else:
            slope = 1.0e-3
        rows.append(_FakeResidual(14973, index, 0, slope))
    return rows


def test_config_is_phase1_cluster_local_not_tracklet_rescue():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["geometry_write_allowed"] is False
    assert config["rank_tolerance"] == pytest.approx(FROZEN_RANK_TOLERANCE)
    assert tuple(config["parameter_names"]) == CLUSTER_LOCAL_NATIVE_NAMES
    assert config["scale_matrix_S"]["station_dx_mm"] == pytest.approx(5.0)
    assert config["scale_matrix_S"]["station_ry_mrad"] == pytest.approx(60.0)
    assert config["scale_matrix_S"]["C_dx_mm"] == pytest.approx(0.12)
    assert config["inherited_entry_58_decision"] == INHERITED_ENTRY_58
    assert config["inherited_entry_68_decision"] == INHERITED_ENTRY_68
    assert config["inherited_entry_69_decision"] == INHERITED_ENTRY_69
    assert config["inherited_entry_70_decision"] == INHERITED_ENTRY_70
    assert config["exact_join_key"] == EXACT_JOIN_KEY
    assert config["do_not_rescue_tracklet_level_observable"] is True
    assert config["do_not_redefine_tracklet_stable_core_criterion"] is True
    assert config["do_not_svd_naked_mixed_unit_jacobian"] is True
    assert config["do_not_run_three_arm_this_stage"] is True
    assert config["do_not_expand_beyond_r14973_r14974_this_stage"] is True
    assert config["entry_70_join_repair_is_not_identifiability_success"] is True
    assert config["runs"]["reference"]["run"] == 14973
    assert config["runs"]["transfer"]["run"] == 14974
    assert "cluster_local_r14974.root" in config["runs"]["transfer"]["cluster_dump"]


def test_refuses_naked_mixed_unit_svd_and_fuzzy_join():
    with pytest.raises(MixedUnitNakedJacobianError):
        svd_naked_jacobian(np.eye(3), parameter_units=("mm", "mrad", "mm"))
    with pytest.raises(FuzzyJoinError):
        refuse_fuzzy_or_nearest_neighbour_join()


def test_official_svd_uses_frozen_cluster_local_scales():
    native = _full_rank_native()
    variance = np.full(native.shape[0], 0.01)
    subspace, extras = official_subspace_from_native(native, variance)
    assert list(subspace.parameter_names) == list(CLUSTER_LOCAL_NATIVE_NAMES)
    assert np.allclose(subspace.parameter_scales, frozen_scales_for(CLUSTER_LOCAL_NATIVE_NAMES))
    assert subspace.identifiable_rank == 3
    assert extras["not_relabeled_as_mechanical_ry_or_C_dx"] is True
    assert extras["column_cosines_native"]["station_dx_vs_C_dx"] < 0.5


def test_native_jacobian_converts_ry_from_radian_to_mrad():
    class _Item:
        def __init__(self):
            self.station_id = 0
            self.layer_id = 0
            self.module_id = "s0_l0_e1_p3"
            self.unbiased_residual_u_mm = 0.0
            self.local_u_var_mm2 = 0.01
            self.global_x_mm = 0.0
            self.global_y_mm = 0.0
            self.global_z_mm = 0.0

    from alignment.true_cluster_local_residual import parameter_columns as real_parameter_columns

    # Guard: the conversion factor is the documented rad→mrad map, independent
    # of live reconstruction.  A 1/rad column must become 1e-3 / mrad.
    mixed = np.asarray([[1.0, 1000.0, 0.5]], dtype=np.float64)
    converted = np.array(mixed, copy=True)
    converted[:, 1] *= 1.0e-3
    assert converted[0, 1] == pytest.approx(1.0)
    del real_parameter_columns
    del _Item
    steps = {"translation_step_mm": 0.010, "rotation_step_rad": 5.0e-5, "c_dx_step_mm": 0.010}
    assert steps["rotation_step_rad"] == pytest.approx(0.05 / 1000.0)


def test_jacobian_validity_fails_incomplete_join_and_near_zero_column():
    join = {
        "join_complete": False,
        "n_wanted_cluster_keys": 10,
        "n_missing_cluster_keys": 10,
        "fuzzy_join_used": False,
        "nearest_neighbour_join_used": False,
    }
    native = np.zeros((8, 3), dtype=np.float64)
    native[:, 0] = 1.0
    native[:, 1] = 0.5
    audit = jacobian_validity_audit(
        join=join,
        measurements=[],
        native_jacobian=native,
        fd_smoke={"passed": False},
        validity_cfg=_config()["jacobian_validity"],
    )
    assert audit["valid"] is False
    assert "exact_join_incomplete" in audit["failure_reasons"]
    assert "near_zero_jacobian_column" in audit["failure_reasons"]


def test_compare_official_subspaces_uses_angles_not_signed_elements():
    native = _full_rank_native()
    variance = np.full(native.shape[0], 0.01)
    left, _ = official_subspace_from_native(native, variance)
    flipped = np.array(native, copy=True)
    flipped[:, 1] *= -1.0
    right, _ = official_subspace_from_native(flipped, variance)
    result = compare_official_subspaces(left, right, gates=_config()["gates"])
    assert result["same_identifiable_rank"] is True
    assert result["pass"] is True
    assert result["compares_subspace_not_signed_vector_elements"] is True
    assert result["max_principal_angle_deg"] < 1.0
    collapsed, _ = official_subspace_from_native(_rank_drop_native(), variance)
    failed = compare_official_subspaces(left, collapsed, gates=_config()["gates"])
    assert failed["pass"] is False
    assert "identifiable_rank_inconsistent" in failed["failure_reasons"]


def test_decide_campaign_never_authorizes_three_arm_or_geometry():
    failure = {
        "labels": ["true_physics_or_coverage_non_transferability"],
        "old_entry_58_metric_reproduced": True,
    }
    invalid = decide_campaign(
        reference_valid=False,
        transfer_valid=True,
        rank_consistent=True,
        subspace_transferable=True,
        entry_58_decision=DECISION_NOT_TRANSFERABLE,
        failure_cause=failure,
    )
    assert invalid["decision"] == DECISION_JACOBIAN_INVALID
    assert invalid["three_arm_authorized"] is False
    assert invalid["geometry_write_allowed"] is False
    not_portable = decide_campaign(
        reference_valid=True,
        transfer_valid=True,
        rank_consistent=True,
        subspace_transferable=False,
        entry_58_decision=DECISION_NOT_TRANSFERABLE,
        failure_cause=failure,
    )
    assert not_portable["decision"] == DECISION_NOT_PORTABLE
    assert not_portable["authorize_more_runs"] is False
    assert not_portable["if_failed_do_not_chase_by_retuning"] is True
    passed = decide_campaign(
        reference_valid=True,
        transfer_valid=True,
        rank_consistent=True,
        subspace_transferable=True,
        entry_58_decision=DECISION_NOT_TRANSFERABLE,
        failure_cause=failure,
    )
    assert passed["decision"] == DECISION_TRANSFERABLE
    assert passed["all_four_gates_passed"] is True
    assert passed["three_arm_authorized"] is False
    assert passed["frozen_v2_alignment_loop_authorized"] is False
    assert passed["real_data_correction_authorized"] is False
    assert passed["entry_58_not_silently_overwritten"] is True


def test_classify_failure_keeps_entry_58_as_negative_control():
    cause = classify_failure_cause(
        official_compare={"pass": False, "left_rank": 3, "right_rank": 3},
        reference_validity={"valid": True},
        transfer_validity={"valid": True},
        reference_tertiles={"rank_drop": True},
        transfer_tertiles={"rank_drop": False},
        entry_58={"decision": {"decision": DECISION_NOT_TRANSFERABLE}, "reference_point": {"rank": 3}, "transfer_point": {"rank": 3}},
        wrong_dump={
            "exact_join_nonzero": False,
            "mismatch_reason": "run_id_disjoint_dump_does_not_contain_route_run",
        },
    )
    assert cause["old_entry_58_metric_reproduced"] is True
    assert cause["workbook_70_wrong_dump_still_zero"] is True
    assert cause["entry_70_does_not_automatically_overturn_entry_58"] is True
    assert "true_physics_or_coverage_non_transferability" in cause["labels"]
    assert "source_or_topology_coverage_difference_slope_tertile_rank_drop" in cause["labels"]


def test_build_all_reports_on_synthetic_packs_does_not_open_three_arm():
    native = _full_rank_native()
    variance = np.full(native.shape[0], 0.01)
    subspace, extras = official_subspace_from_native(native, variance)
    measurements = _measurements()
    join = {
        "join_complete": True,
        "n_wanted_cluster_keys": 36,
        "n_missing_cluster_keys": 0,
        "n_measurements": 36,
        "fuzzy_join_used": False,
        "nearest_neighbour_join_used": False,
        "exact_join_nonzero": True,
    }
    validity = {
        "valid": True,
        "failure_reasons": [],
        "join_complete": True,
        "n_measurements": 36,
        "n_routes": 36,
    }
    pack = {
        "join": join,
        "bundle": {"measurements": measurements},
        "measurements": measurements,
        "native_jacobian": native,
        "variance": variance,
        "validity": validity,
        "subspace": subspace,
        "extras": extras,
        "tertiles": {"too_small": False, "bins": [], "rank_drop": False},
        "event_bootstrap": {"too_small": True},
        "steps": {"translation_step_mm": 0.010, "rotation_step_rad": 5.0e-5, "c_dx_step_mm": 0.010},
    }
    reports = build_all_reports(
        _config(),
        packs={"reference": pack, "transfer": pack},
        wrong_dump={
            "join_complete": False,
            "n_measurements": 0,
            "exact_join_nonzero": False,
            "mismatch_reason": "run_id_disjoint_dump_does_not_contain_route_run",
        },
        entry_58_override={
            "reference_point": {"rank": 3, "station_ry_vs_C_dx": 0.53},
            "transfer_point": {"rank": 3, "station_ry_vs_C_dx": 0.57},
            "decision": {"decision": DECISION_NOT_TRANSFERABLE},
        },
    )
    assert_no_alignment_payload(reports["next_stage_decision"])
    assert reports["next_stage_decision"]["three_arm_authorized"] is False
    assert reports["next_stage_decision"]["geometry_write_allowed"] is False
    assert reports["next_stage_decision"]["inherited_entry_58_decision"] == INHERITED_ENTRY_58
    assert reports["cross_run_subspace"]["same_identifiable_rank"] is True
    assert reports["parameter_definition"]["weighted_matrix"] == "A = W^{1/2} J S"
    assert reports["wrong_dump_control"]["exact_join_nonzero"] is False
