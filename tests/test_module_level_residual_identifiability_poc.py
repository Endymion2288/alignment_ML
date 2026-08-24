from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.module_level_residual_poc import (
    DECISION_MIXED,
    DECISION_RESTORES,
    DECISION_TOPOLOGY,
    GO_NO_GO_QUESTION,
    MEASUREMENT_SOURCE,
    MIXED_REASON,
    ModuleMeasurement,
    RESIDUAL_DECREASE_LABEL,
    RESIDUAL_KIND,
    SCHEMA_VERSION,
    UNBIASED_METHOD,
    assert_no_alignment_payload,
    c_dx_layer_weight,
    cosine,
    decide_next_stage,
    decode_hit_pattern,
    intercept_shift_xy,
    load_poc_config,
    module_key,
    parse_module_key,
    se3_delta_xyz,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)


def _config():
    return load_poc_config(Path("configs/module_level_residual_identifiability_poc_v1.yaml"))


def _measurement(**overrides) -> ModuleMeasurement:
    values = dict(
        event_id=1,
        route_index=0,
        run_id=14973,
        tracklet_id=0,
        station_id=0,
        layer_id=0,
        module_id="s0_l0_e1_p3",
        cluster_id=11,
        module_identifier=22,
        phi_module=3,
        eta_module=1,
        side=0,
        global_x_mm=20.0,
        global_y_mm=40.0,
        global_z_mm=-1891.65,
        local_u_mm=0.0,
        local_v_mm=0.0,
        sigma_x_mm=0.02,
        sigma_y_mm=0.02,
        tx=0.01,
        ty=0.0,
        predicted_x_mm=20.0,
        predicted_y_mm=40.0,
        unbiased_residual_x_mm=0.0,
        unbiased_residual_y_mm=0.0,
    )
    values.update(overrides)
    return ModuleMeasurement(**values)


def test_json_ready_keeps_booleans():
    from alignment.module_level_residual_poc import json_ready

    assert json_ready(True) is True
    assert json_ready(False) is False
    assert json_ready({"geometry_write_allowed": False})["geometry_write_allowed"] is False


def test_config_stays_monitoring_only_and_keeps_frozen_v2():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["geometry_write_allowed"] is False
    assert config["station_calibration_mode_available"] is False
    assert config["cdx_mode_allowed"] is False
    assert config["do_not_retrain_v2"] is True
    assert config["do_not_run_newton"] is True
    assert config["do_not_solve_alignment_correction"] is True
    assert config["software_fd_sensitivity_only"] is True
    assert config["do_not_claim_acts_cluster_residual"] is True
    assert config["residual_kind"] == RESIDUAL_KIND
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["residual_decrease_label"] == RESIDUAL_DECREASE_LABEL
    assert config["representative"]["station_id"] == 0
    assert 2 <= int(config["representative"]["n_modules"]) <= 4


def test_hit_pattern_and_module_keys_and_cdx_weights():
    assert decode_hit_pattern(0b111111) == (
        (0, 0),
        (0, 1),
        (1, 0),
        (1, 1),
        (2, 0),
        (2, 1),
    )
    assert module_key(0, 0, 1, 3) == "s0_l0_e1_p3"
    assert parse_module_key("s0_l0_e1_p3") == (0, 0, 1, 3)
    assert c_dx_layer_weight(0) == 1.0
    assert c_dx_layer_weight(1) == 0.0
    assert c_dx_layer_weight(2) == -1.0


def test_dx_and_cdx_are_orthogonal_ry_is_not():
    dx = np.array([1.0, 1.0, 1.0])
    cdx = np.array([1.0, 0.0, -1.0])
    ry = np.array([-1.0, 0.0, 1.0])
    assert abs(cosine(dx, cdx)) < 1.0e-12
    assert abs(cosine(ry, cdx)) == pytest.approx(1.0)


def test_payload_shift_sign_and_target_isolation():
    delta = se3_delta_xyz([10.0, 0.0, 0.0], center_mm=[0.0, 0.0, 0.0], dx_mm=0.01)
    assert delta[0] == pytest.approx(0.01)
    shift = intercept_shift_xy([0.01, 0.0, 0.0], tx=0.0, ty=0.0)
    assert shift[0] > 0.0
    from alignment.module_level_residual_poc import (
        SoftwarePayload,
        measurement_shift_xy,
        zero_six,
    )

    target = _measurement(module_id="s0_l0_e1_p3", layer_id=0)
    other = _measurement(module_id="s0_l0_e1_p0", layer_id=0, phi_module=0, global_y_mm=-40.0)
    six = zero_six()
    six[0] = 0.01
    payload = SoftwarePayload(module_six={"s0_l0_e1_p3": six})
    assert measurement_shift_xy(target, payload)[0] == pytest.approx(0.01)
    assert abs(measurement_shift_xy(other, payload)[0]) < 1.0e-12


def test_decision_requires_both_pairs_and_forbids_payload():
    yes = decide_next_stage(
        {
            "dx_separated": True,
            "ry_separated": True,
            "dx_cosine_dropped": True,
            "ry_cosine_dropped": True,
            "dx_still_collinear": False,
            "ry_still_collinear": False,
            "leakage_subspace_resolvable": True,
            "worse_null_than_two_physical_directions": False,
        },
        {"condition_number": 1.0e6, "rank": 10, "n_parameters": 27},
        {"passed": True},
        worse_null_condition_ratio=1.0e3,
    )
    assert yes["answer"] == "Yes"
    assert yes["go_to_full_module_identifiability_map"] is True
    assert yes["decision"] == DECISION_RESTORES
    assert yes["go_no_go_question"] == GO_NO_GO_QUESTION

    mixed = decide_next_stage(
        {
            "dx_separated": True,
            "ry_separated": False,
            "dx_cosine_dropped": True,
            "ry_cosine_dropped": False,
            "dx_still_collinear": False,
            "ry_still_collinear": True,
            "leakage_subspace_resolvable": False,
            "worse_null_than_two_physical_directions": False,
        },
        {"condition_number": 10.0, "rank": 3, "n_parameters": 3},
        {"passed": True},
        worse_null_condition_ratio=1.0e3,
    )
    assert mixed["answer"] == "No"
    assert mixed["go_to_full_module_identifiability_map"] is False
    assert mixed["decision"] == DECISION_MIXED
    assert mixed["reason"] == MIXED_REASON
    assert mixed["go_to_full_detector_module_level_alignment_basis_study"] is False
    assert mixed["still_no_new_network"] is True
    assert mixed["geometry_write_allowed"] is False

    no = decide_next_stage(
        {
            "dx_separated": False,
            "ry_separated": False,
            "dx_still_collinear": True,
            "ry_still_collinear": True,
            "worse_null_than_two_physical_directions": True,
        },
        {},
        {"passed": True},
        worse_null_condition_ratio=1.0e3,
    )
    assert no["answer"] == "No"
    assert no["decision"] == DECISION_TOPOLOGY
    assert no["next_allowed_step"] == "stop_ml_descent_prefer_survey_or_new_topology"

    assert_no_alignment_payload(mixed)
    with pytest.raises(ValueError, match="geometry write"):
        assert_no_alignment_payload({**mixed, "geometry_write_allowed": True})
    with pytest.raises(ValueError, match="alignment payload keys"):
        assert_no_alignment_payload({"alignment_payload": {"dx": 1.0}})
    with pytest.raises(ValueError, match="alignment correction"):
        assert_no_alignment_payload({"alignment_correction": [0.1]})


def test_software_jacobian_separates_dx_not_ry():
    from alignment.module_level_residual_poc import (
        compare_leakage,
        parameter_columns,
        run_fd_smoke,
        validate_residuals,
    )

    measurements = []
    for route, x0 in ((0, 15.0), (1, -10.0)):
        for layer, z in ((0, -1891.65), (1, -1860.15), (2, -1828.65)):
            for phi, y, mid in ((3, 40.0, "s0_l0_e1_p3"), (0, -40.0, "s0_l0_e1_p0")):
                module = f"s0_l{layer}_e1_p{phi}"
                measurements.append(
                    _measurement(
                        event_id=100 + route,
                        route_index=route,
                        layer_id=layer,
                        module_id=module,
                        phi_module=phi,
                        global_x_mm=x0,
                        global_y_mm=y,
                        global_z_mm=z,
                        predicted_x_mm=x0,
                        predicted_y_mm=y,
                    )
                )
    residual = validate_residuals(measurements)
    assert residual["biased_residual_used_as_alignment_observable"] is False
    assert residual["leave_one_station_out"] is True
    assert residual["residual_kind"] == RESIDUAL_KIND
    assert residual["not_a_complete_acts_cluster_residual"] is True
    fd = run_fd_smoke(
        measurements,
        ["s0_l0_e1_p3", "s0_l0_e1_p0"],
        station_id=0,
        translation_steps_mm=[0.010, 0.020],
        rotation_steps_mrad=[0.05, 0.10],
        c_dx_steps_mm=[0.010, 0.020],
        linearity_max_relative_deviation=0.15,
        targeting_max_offmodule_fraction=1.0e-6,
    )
    assert fd["passed"] is True
    assert fd["c_dx_layer0_positive"] is True
    assert fd["c_dx_layer2_negative"] is True
    names, jacobian = parameter_columns(
        measurements,
        ["s0_l0_e1_p3", "s0_l0_e1_p0"],
        station_id=0,
        translation_step_mm=0.010,
        rotation_step_rad=5.0e-5,
        c_dx_step_mm=0.010,
    )
    leakage = compare_leakage(
        measurements,
        names,
        jacobian,
        station_id=0,
        translation_step_mm=0.010,
        rotation_step_rad=5.0e-5,
        c_dx_step_mm=0.010,
        thresholds={"high_cosine": 0.85, "restored_cosine": 0.50, "minimum_drop": 0.25},
    )
    assert leakage["dx_separated"] is True
    assert leakage["ry_separated"] is False
    assert leakage["dx_cosine_dropped"] is True
    assert leakage["ry_cosine_dropped"] is False
    assert leakage["module_level_abs_cosine"]["station_dx_vs_C_dx"] < 0.2
    assert leakage["module_level_abs_cosine"]["station_ry_vs_C_dx"] > 0.85
    assert leakage["station_level_abs_cosine"]["station_dx_vs_C_dx"] > 0.85
    assert leakage["leakage_subspace_resolvable"] is False


def test_unbiased_contract_labels():
    assert MEASUREMENT_SOURCE == "tracklet_intercept_at_nominal_layer_z"
    assert UNBIASED_METHOD == "leave_one_station_out_projected_to_layer"
    assert RESIDUAL_KIND == "module-proxy unbiased residual"
    assert "biased" not in UNBIASED_METHOD
    assert "Acts cluster residual" not in RESIDUAL_KIND
