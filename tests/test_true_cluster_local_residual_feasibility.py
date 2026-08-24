from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.true_cluster_local_residual import (
    DECISION_RESTORES,
    DECISION_TOPOLOGY,
    GO_NO_GO_QUESTION,
    MEASUREMENT_SOURCE,
    RESIDUAL_KIND,
    SCHEMA_VERSION,
    UNBIASED_METHOD,
    ClusterResidual,
    DetectorSurface,
    SoftwarePayload,
    assert_no_alignment_payload,
    compare_leakage,
    decide_next_stage,
    intersect_line_plane,
    load_stage_config,
    local_uv,
    parameter_columns,
    predicted_local_u,
    residual_u,
    rotate_about_y,
    run_fd_smoke,
    validate_residuals,
)


STATION_Z = -1860.15
LAYER_Z = {0: -1891.65, 1: -1860.15, 2: -1828.65}


def _config():
    return load_stage_config(Path("configs/true_cluster_local_residual_feasibility_v1.yaml"))


def _surface(layer_id: int, *, stereo: float = 0.020, y0: float = 40.0) -> DetectorSurface:
    phi = np.asarray([np.sin(stereo), np.cos(stereo), 0.0], dtype=np.float64)
    eta = np.asarray([np.cos(stereo), -np.sin(stereo), 0.0], dtype=np.float64)
    normal = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    return DetectorSurface(
        center_mm=[30.0, y0, LAYER_Z[layer_id]],
        phi_axis=phi,
        eta_axis=eta,
        normal=normal,
        sin_stereo=float(np.sin(stereo)),
    )


def _measurement(layer_id: int, *, event_id: int = 1, x0: float = 10.0, y0: float = 40.0) -> ClusterResidual:
    surface = _surface(layer_id, y0=y0)
    origin = np.asarray([x0, y0, 47.4], dtype=np.float64)
    direction = np.asarray([0.0, 0.0, 1.0], dtype=np.float64)
    predicted = predicted_local_u(origin, direction, surface)
    point = intersect_line_plane(origin, direction, surface)
    assert point is not None
    local_u, _local_v = local_uv(point, surface)
    return ClusterResidual(
        event_id=event_id,
        route_index=0,
        run_id=14973,
        tracklet_id=0,
        station_id=0,
        layer_id=layer_id,
        module_id=f"s0_l{layer_id}_e1_p3",
        cluster_id=100 + layer_id,
        module_identifier=200 + layer_id,
        phi_module=3,
        eta_module=1,
        side=0,
        local_u_mm=float(local_u),
        local_u_var_mm2=1.0e-4,
        sigma_u_mm=0.01,
        predicted_u_mm=float(predicted),
        unbiased_residual_u_mm=float(local_u - predicted),
        global_x_mm=float(point[0]),
        global_y_mm=float(point[1]),
        global_z_mm=float(point[2]),
        surface=surface,
        track_origin_mm=origin,
        track_direction=direction,
        unbiased_method=UNBIASED_METHOD,
    )


def test_config_stays_monitoring_only_and_forbids_intercept_residual():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["geometry_write_allowed"] is False
    assert config["station_calibration_mode_available"] is False
    assert config["cdx_mode_allowed"] is False
    assert config["do_not_retrain_v2"] is True
    assert config["do_not_use_tracklet_intercept_as_cluster_residual"] is True
    assert config["residual_kind"] == RESIDUAL_KIND
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["representative"]["station_id"] == 0


def test_surface_local_u_matches_phi_axis_projection():
    surface = _surface(0)
    point = surface.center_mm + 1.5 * surface.phi_axis - 0.3 * surface.eta_axis
    local_u, local_v = local_uv(point, surface)
    assert local_u == pytest.approx(1.5)
    assert local_v == pytest.approx(-0.3)


def test_se3_ry_convention_moves_downstream_x_negative():
    downstream = rotate_about_y([0.0, 0.0, 10.0], 0.01, [0.0, 0.0, 0.0])
    assert downstream[0] == pytest.approx(-0.01 * 10.0, rel=1.0e-3)


def test_true_local_residual_is_not_nominal_z_intercept():
    assert "tracklet_intercept" not in MEASUREMENT_SOURCE
    assert "detector_surface" in UNBIASED_METHOD
    assert RESIDUAL_KIND == "true cluster-local unbiased residual"
    measurement = _measurement(0)
    assert measurement.uses_nominal_layer_z_tracklet_intercept_as_residual is False
    report = validate_residuals([_measurement(0), _measurement(1), _measurement(2)])
    assert report["uses_nominal_layer_z_tracklet_intercept_as_residual"] is False
    assert report["projected_to_real_detector_surface"] is True
    assert report["biased_residual_used_as_alignment_observable"] is False


def test_parallel_planes_keep_ry_collinear_with_cdx():
    measurements = []
    for event_id, x0 in ((1, 12.0), (2, -8.0), (3, 4.0), (4, -15.0)):
        for layer in (0, 1, 2):
            measurements.append(_measurement(layer, event_id=event_id, x0=x0))
    fd = run_fd_smoke(
        measurements,
        ["s0_l0_e1_p3"],
        station_id=0,
        translation_steps_mm=[0.010, 0.020],
        rotation_steps_mrad=[0.05, 0.10],
        c_dx_steps_mm=[0.010, 0.020],
        linearity_max_relative_deviation=0.15,
        targeting_max_offmodule_fraction=1.0e-6,
        l1_locality_max_fraction=0.15,
    )
    assert fd["passed"] is True
    assert fd["c_dx_layer0_positive_after_stereo"] is True
    assert fd["c_dx_layer2_negative_after_stereo"] is True
    _names, jacobian = parameter_columns(
        measurements,
        station_id=0,
        translation_step_mm=0.010,
        rotation_step_rad=5.0e-5,
        c_dx_step_mm=0.010,
    )
    leakage = compare_leakage(
        measurements,
        jacobian,
        station_id=0,
        fd_report=fd,
        previous_poc={
            "station_level_abs_cosine": {
                "station_dx_vs_C_dx": 0.973,
                "station_ry_vs_C_dx": 1.000,
            },
            "module_level_abs_cosine": {
                "station_dx_vs_C_dx": 0.022,
                "station_ry_vs_C_dx": 0.995,
            },
            "station_leakage_subspace_rank": 1,
            "module_leakage_subspace_rank": 2,
        },
        thresholds={
            "high_cosine": 0.85,
            "topology_cosine": 0.90,
            "restored_cosine": 0.50,
            "minimum_drop": 0.25,
            "stability_max_abs_delta": 0.05,
        },
        steps={
            "translation_step_mm": 0.010,
            "rotation_step_rad": 5.0e-5,
            "c_dx_step_mm": 0.010,
        },
    )
    assert leakage["true_cluster_local_abs_cosine"]["station_dx_vs_C_dx"] < 0.2
    assert leakage["true_cluster_local_abs_cosine"]["station_ry_vs_C_dx"] > 0.9
    assert leakage["dx_separated"] is True
    assert leakage["ry_separated"] is False
    decision = decide_next_stage(leakage, fd)
    assert decision["answer"] == "No"
    assert decision["decision"] == DECISION_TOPOLOGY
    assert decision["go_to_full_module_identifiability_map"] is False
    assert decision["go_no_go_question"] == GO_NO_GO_QUESTION
    assert decision["still_no_new_network"] is True
    assert_no_alignment_payload(decision)
    with pytest.raises(ValueError, match="geometry write"):
        assert_no_alignment_payload({**decision, "geometry_write_allowed": True})


def test_restore_decision_requires_independent_ry():
    yes = decide_next_stage(
        {
            "true_cluster_local_abs_cosine": {"station_ry_vs_C_dx": 0.21},
            "ry_cosine_dropped": True,
            "ry_separated": True,
            "ry_topology_collinear": False,
            "leakage_subspace_resolvable": True,
            "stable_across_fd_and_subsets": True,
        },
        {"passed": True},
    )
    assert yes["decision"] == DECISION_RESTORES
    assert yes["answer"] == "Yes"
    assert yes["go_to_full_module_identifiability_map"] is True
    assert yes["still_no_new_network"] is True


def test_module_dx_shifts_only_target_residual():
    target = _measurement(0)
    other = _measurement(2, event_id=2, x0=-4.0)
    payload = SoftwarePayload(module_dx_mm={"s0_l0_e1_p3": 0.02})
    shifted = residual_u(target, payload) - residual_u(target)
    untouched = residual_u(other, payload) - residual_u(other)
    assert abs(shifted) > 1.0e-6
    assert abs(untouched) < 1.0e-12
