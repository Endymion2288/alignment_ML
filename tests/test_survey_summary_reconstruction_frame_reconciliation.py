from __future__ import annotations

from pathlib import Path

import pytest

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.survey_summary_reconstruction import (
    DECISION,
    SCHEMA_VERSION,
    build_all_reports,
    c_dx_from_layer_x,
    cad_normal_to_provisional_rx_ry,
    fit_layer_x_trend,
    load_reconstruction_config,
    station_layer_average_check,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_reconstruction_config(
        Path("configs/survey_summary_reconstruction_frame_reconciliation_v1.yaml")
    )


def test_config_freezes_protocol_and_forbids_ppt_promotion():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["do_not_treat_ppt_as_alignment_prior"] is True
    assert config["do_not_treat_ppt_as_geometry"] is True
    assert config["do_not_use_station_sd_as_measurement_sigma"] is True
    assert config["do_not_equate_layer_trend_to_calypso_station_ry"] is True
    assert config["geometry_write_allowed"] is False
    assert config["geometry_candidate_from_slide_summary"] is False
    assert config["physics_scales"]["ift_layer_pitch_is_survey"] is False
    assert config["slide_2022_recovered"]["station_offset_width_label"] == (
        "population_spread_not_measurement_sigma"
    )


def test_2021_normal_gives_provisional_cad_rx_ry_not_calypso_ry():
    angles = cad_normal_to_provisional_rx_ry(
        [0.0, 0.0, -1.0],
        [-0.00306, -0.003954, -0.999987],
    )
    assert angles["provisional_rx_cad_mrad"] == pytest.approx(-3.954, abs=1.0e-3)
    assert angles["provisional_ry_cad_mrad"] == pytest.approx(3.06, abs=1.0e-3)
    assert angles["total_tilt_from_transverse_normal_mrad"] == pytest.approx(5.0, abs=0.02)
    assert angles["label"] == "frame_not_yet_reconciled"
    assert angles["not_calypso_station_ry"] is True


def test_2022_layers_reproduce_station_means_and_cdx():
    config = _config()
    recovered = config["slide_2022_recovered"]
    layers = {int(key): value for key, value in recovered["layer_mean_shift_side0_mm"].items()}
    stations = {int(key): value for key, value in recovered["station_mean_offset_mm"].items()}
    check = station_layer_average_check(layers, stations)
    assert check["all_station_means_reproduced"] is True
    assert check["layers_0_1_2_are_ift"] is True
    c_dx = c_dx_from_layer_x(layers[0][0], layers[2][0])
    assert c_dx == pytest.approx(0.2541169, abs=5.0e-8)
    assert (layers[0][0] - layers[2][0]) == pytest.approx(0.5082339, abs=5.0e-8)
    trend = fit_layer_x_trend({0: layers[0][0], 1: layers[1][0], 2: layers[2][0]}, pitch_mm=31.5)
    assert trend["layer_mean_coherent_tilt_candidate_mrad"] == pytest.approx(-8.07, abs=0.05)
    assert trend["not_calypso_station_ry"] is True
    assert trend["name"] == "layer_mean_coherent_tilt_candidate"


def test_reports_answer_the_one_question_without_geometry():
    reports = build_all_reports(_config())
    contrast = reports["ift_layer_contrast_reconstruction"]
    frames = reports["survey_calypso_frame_reconciliation"]
    feasibility = reports["slide_summary_prior_feasibility"]
    decision = reports["next_stage_decision"]
    assert contrast["layers_0_1_2_confirmed_as_ift"] is True
    assert contrast["C_dx_slide_mm"] == pytest.approx(0.2541169, abs=5.0e-8)
    assert contrast["not_calypso_station_ry"] is True
    assert contrast["same_l0_l2_contrast_expressed_two_ways"]["independent_constraints"] is False
    assert reports["survey_2021_summary"]["frame_status"] == "frame_not_yet_reconciled"
    assert reports["survey_2022_summary"]["station_offset_width_mm"]["0"]["may_be_used_as_gaussian_prior_uncertainty"] is False
    assert frames["explicit_transform_completed"] is False
    assert frames["may_compare_2021_with_2022_as_same_number"] is False
    assert frames["mapped_2021_under_default_signs_only"]["maps_to_calypso_station_ry"] is False
    comparison = frames["existing_conditions_numeric_comparison"]
    if comparison is not None:
        assert comparison["origin_upgrade"] is None
        assert comparison["conditions_label"] == "existing_conditions_state"
    assert feasibility["mode"] == "slide_summary_feasibility_only"
    assert feasibility["station_sd_used_as_sigma"] is False
    assert feasibility["geometry_candidate"] is False
    assert feasibility["coarsest_unlocking_sigma_ry_mrad"] == pytest.approx(20.0)
    assert feasibility["coarsest_unlocking_sigma_cdx_mm"] == pytest.approx(0.34514281813225994)
    assert feasibility["degeneracy_breaking_frozen"]["sigma_cdx_mm"] == pytest.approx(0.315)
    assert any(
        row["family"] == "cdx_only"
        and row["sigma_cdx_mm"] is not None
        and abs(float(row["sigma_cdx_mm"]) - 0.315) < 1.0e-12
        and row["unlocked"]
        for row in feasibility["points"]
    )
    assert decision["decision"] == DECISION
    assert decision["answer"]["internal_l0_l2_contrast_defined"] is True
    assert decision["answer"]["mutually_coordinate_consistent"] is False
    assert decision["answer"]["remaining_uncertainty"] == "missing_measurement_covariance"
    assert decision["answer"]["track_degeneracy_is_unknown_source"] is False
    assert decision["enough_to_write_geometry"] is False
    assert decision["survey_derived"] is False
    for payload in reports.values():
        assert payload["geometry_candidate"] is False
        assert_no_alignment_payload(payload)
