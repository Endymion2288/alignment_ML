from __future__ import annotations

from pathlib import Path

import numpy as np

from alignment.five_dof_sampling import DEFAULT_SCALES
from alignment.hierarchical_v1 import STATION_SOLVE_PARAMETERS
from alignment.real_data_station_mode_failure_audit import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    campaign_allows_cdx_mode,
    campaign_allows_geometry_write,
    classify_failure,
    contamination_backprojection,
    estimate_statistics_scaling,
    load_failure_audit_config,
    near_degenerate_parameters,
    reconstruct_js_identifiability,
    route_station_count,
    scaled_normal_from_js,
)


def _config():
    return load_failure_audit_config(
        Path("configs/operating_protocol_v1_real_data_station_mode_failure_audit_v1.yaml")
    )


def _identity_covariance(n_obs: int) -> np.ndarray:
    return np.repeat(np.eye(4)[None, :, :], n_obs, axis=0)


def _full_rank_js(n_obs: int = 30, dz_scale: float = 1.0) -> np.ndarray:
    rng = np.random.default_rng(0)
    jacobian = rng.normal(size=(n_obs, 4, 6))
    jacobian[:, :, 2] *= dz_scale
    return jacobian


def test_config_freezes_write_cdx_v2_and_schur():
    config = _config()
    assert config["geometry_write_allowed"] is False
    assert config["official_conditions_db_write"] is False
    assert config["cdx_mode_blocked"] is True
    assert config["do_not_enter_cdx_mode"] is True
    assert config["joint_station_cdx_newton"] is False
    assert config["new_layer_or_module_dof"] is False
    assert config["schur_projection_production"] is False
    assert config["do_not_retrain_v2"] is True
    assert config["do_not_emit_cdx_payload"] is True
    assert config["do_not_open_sealed_test"] is True
    assert config["residual_reduction_is_not_alignment_success"] is True
    assert campaign_allows_cdx_mode(CLASS_A) is False
    assert campaign_allows_geometry_write(CLASS_B) is False


def test_route_station_count_from_signature():
    signature = (
        '[{"station_id":0,"origin_tracklet_id":0},'
        '{"station_id":1,"origin_tracklet_id":1},'
        '{"station_id":2,"origin_tracklet_id":2}]'
    )
    assert route_station_count(signature) == 3
    assert route_station_count("[]") == 0


def test_reconstruct_js_reports_rank_condition_and_spectrum():
    jacobian = _full_rank_js(n_obs=40, dz_scale=1.0)
    report = reconstruct_js_identifiability(jacobian, _identity_covariance(40))
    assert report["parameter_names"] == list(STATION_SOLVE_PARAMETERS)
    assert report["n_fd_probes"] == 12
    assert report["full_rank"] is True
    assert report["rank"] == 6
    assert report["condition_number"] is not None
    assert len(report["singular_values"]) == 6
    assert report["residual_improvement_is_not_closure"] is True
    assert "purity" not in report


def test_near_degenerate_flags_dominant_smallest_direction():
    names = ("dx", "dy", "dz")
    vectors = np.array(
        [
            [1.0, 0.0, 0.01],
            [0.0, 1.0, 0.02],
            [0.0, 0.0, 0.999],
        ],
        dtype=np.float64,
    )
    flagged = near_degenerate_parameters(names, vectors, abs_component=0.5)
    assert flagged["near_degenerate"] == ["dz"]


def test_weak_dz_column_is_near_degenerate():
    jacobian = np.zeros((12, 4, 6), dtype=np.float64)
    for parameter, name in enumerate(STATION_SOLVE_PARAMETERS):
        jacobian[2 * parameter, 0, parameter] = 1.0e-4 if name == "ift_dz_mm" else 1.0
        jacobian[2 * parameter + 1, 1, parameter] = 1.0e-4 if name == "ift_dz_mm" else 1.0
    report = reconstruct_js_identifiability(
        jacobian, _identity_covariance(12), near_degenerate_abs_component=0.5
    )
    assert "ift_dz_mm" in report["near_degenerate"]["near_degenerate"]
    five = report["five_dof_excluding_survey_dz"]
    assert report["raw_condition_number"] is not None
    assert five["raw_condition_number"] is not None
    assert five["raw_condition_number"] < report["raw_condition_number"]


def test_same_topology_scaling_is_invariant():
    jacobian = _full_rank_js(n_obs=20)
    _native, scaled, names = scaled_normal_from_js(jacobian, _identity_covariance(20))
    estimate = estimate_statistics_scaling(
        scaled,
        None,
        names=names,
        n_complete_observations=0,
        n_complete_routes=0,
        n_events=1000,
    )
    assert estimate["same_topology_scaling"]["condition_invariant"] is True
    assert estimate["same_topology_scaling"]["recovers_six_dof"] is False
    assert estimate["schur_production_estimator_used"] is False
    assert estimate["geometry_write_allowed"] is False
    assert estimate["station_covariance_can_recover_by_more_complete_tracks"] is False


def test_complete_track_scaling_finds_copies_when_unit_lifts_weak_mode():
    names = list(STATION_SOLVE_PARAMETERS)
    n = len(names)
    base = np.diag(np.array([1.0e6, 1.0e5, 1.0, 1.0e4, 1.0e4, 1.0e4], dtype=np.float64))
    unit = np.zeros((n, n), dtype=np.float64)
    unit[2, 2] = 100.0
    estimate = estimate_statistics_scaling(
        base,
        unit,
        names=names,
        n_complete_observations=2,
        n_complete_routes=2,
        n_events=1000,
        target_condition_number=1.0e4,
        practical_complete_routes_max=50,
        practical_events_max=20_000,
    )
    six = estimate["complete_four_station_scaling"]["six_dof"]
    assert six["recovers"] is True
    assert six["minimum_copies_of_observed_complete_block"] is not None
    assert six["minimum_copies_of_observed_complete_block"] > 0.5
    assert six["implied_complete_routes"] is not None


def test_complete_track_scaling_reports_none_when_unit_is_more_degenerate():
    names = list(STATION_SOLVE_PARAMETERS)
    base = np.diag(np.array([1.0e6, 1.0e5, 10.0, 1.0e4, 1.0e4, 1.0e4], dtype=np.float64))
    unit = np.diag(np.array([1.0e2, 1.0e2, 1.0e-12, 1.0, 1.0, 1.0], dtype=np.float64))
    estimate = estimate_statistics_scaling(
        base,
        unit,
        names=names,
        n_complete_observations=2,
        n_complete_routes=2,
        n_events=60000,
        target_condition_number=1.0e4,
    )
    six = estimate["complete_four_station_scaling"]["six_dof"]
    assert six["recovers"] is False
    assert six["minimum_copies_of_observed_complete_block"] is None
    assert estimate["station_covariance_can_recover_by_more_complete_tracks"] is False


def test_classify_c_when_selected_graph_empty():
    result = classify_failure(
        selected_routes_calibration=(0, 0),
        complete_four_station_calibration=(0, 0),
        all_pairs_nonempty=True,
        six_dof_recoverable_from_complete_tracks=False,
        complete_track_recovery_practical=False,
        same_topology_recovers_six_dof=False,
        implied_cdx_exceeds_operating_band=True,
        weak_direction_cosine_with_A=0.99,
    )
    assert result["unique_class"] == CLASS_C
    assert result["geometry_write_allowed"] is False


def test_classify_a_when_complete_tracks_can_recover():
    result = classify_failure(
        selected_routes_calibration=(100, 100),
        complete_four_station_calibration=(0, 2),
        all_pairs_nonempty=True,
        six_dof_recoverable_from_complete_tracks=True,
        complete_track_recovery_practical=True,
        same_topology_recovers_six_dof=False,
        implied_cdx_exceeds_operating_band=False,
        weak_direction_cosine_with_A=0.1,
    )
    assert result["unique_class"] == CLASS_A
    assert result["do_not_retrain_v2"] is True


def test_classify_b_when_six_dof_cannot_recover():
    result = classify_failure(
        selected_routes_calibration=(121, 109),
        complete_four_station_calibration=(0, 2),
        all_pairs_nonempty=True,
        six_dof_recoverable_from_complete_tracks=False,
        complete_track_recovery_practical=False,
        same_topology_recovers_six_dof=False,
        implied_cdx_exceeds_operating_band=True,
        weak_direction_cosine_with_A=0.99,
    )
    assert result["unique_class"] == CLASS_B
    assert result["concurrent_class"] == CLASS_A
    assert CLASS_C in result["ruled_out"]
    assert result["cdx_mode_allowed"] is False
    assert result["do_not_emit_cdx_payload"] is True


def test_classify_rejects_residual_as_success():
    try:
        classify_failure(
            selected_routes_calibration=(10, 10),
            complete_four_station_calibration=(0, 0),
            all_pairs_nonempty=True,
            six_dof_recoverable_from_complete_tracks=False,
            complete_track_recovery_practical=False,
            same_topology_recovers_six_dof=False,
            implied_cdx_exceeds_operating_band=False,
            weak_direction_cosine_with_A=None,
            residual_used_as_success=True,
        )
    except ValueError as exc:
        assert "residual" in str(exc)
    else:
        raise AssertionError("expected residual-success rejection")


def test_backprojection_is_rejection_only():
    from alignment.calibration_modes import load_mode_validity_contract

    contract = load_mode_validity_contract(
        Path("outputs/mc24_ift_calibration_mode_validity_contract_v1/mode_validity_contract.json")
    )
    report = contamination_backprojection(
        contract,
        station_dx_by_run={14973: -1.3, 14974: 6.6},
        station_ry_by_run={14973: 20.0, 14974: 33.5},
    )
    assert report["rejection_diagnostic_only"] is True
    assert report["do_not_emit_cdx_payload"] is True
    assert report["new_cdx_payload"] is None
    assert report["exceeds_operating_band"] is True
    assert report["geometry_write_allowed"] is False
    assert "recommended_C_dx" not in report
    assert "purity" not in report
    assert DEFAULT_SCALES["ift_dx_mm"] == 5.0
