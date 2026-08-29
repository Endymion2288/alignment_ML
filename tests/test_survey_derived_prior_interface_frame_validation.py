from __future__ import annotations

from pathlib import Path

import pytest

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.survey_derived_prior_interface import (
    DECISION,
    SCHEMA_VERSION,
    UnconfirmedRyMappingError,
    attach_candidate_to_scan_parameter,
    build_all_reports,
    load_interface_config,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_interface_config(
        Path("configs/survey_derived_prior_interface_frame_validation_v1.yaml")
    )


def test_config_freezes_protocol_and_forbids_ry_mapping():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["do_not_map_unconfirmed_tilt_to_ry"] is True
    assert config["do_not_expand_ml_or_module_map"] is True
    assert config["do_not_expand_collision_track_statistics"] is True
    assert config["geometry_candidate_from_survey_candidate"] is False
    assert config["geometry_write_allowed"] is False


def test_unconfirmed_tilts_cannot_map_to_ry_and_cdx_can_attach():
    reports = build_all_reports(_config())
    database = reports["survey_candidate_database"]
    by_id = database["by_id"]
    assert database["any_candidate_maps_to_ry"] is False
    assert database["n_may_map_to_station_ry"] == 0
    for candidate_id in (
        "2021_ift_if_provisional_ry_cad_mrad",
        "2021_ift_if_total_tilt_mrad",
        "2022_layer_mean_coherent_tilt_candidate_mrad",
        "2021_ift_if_inplane_yaw_cad_rz",
    ):
        with pytest.raises(UnconfirmedRyMappingError):
            attach_candidate_to_scan_parameter(by_id[candidate_id], "station_ry")
    attached = attach_candidate_to_scan_parameter(by_id["2022_ift_C_dx_slide_mm"], "C_dx")
    assert attached["scan_parameter"] == "C_dx"
    assert attached["sigma"] is None
    assert attached["availability"] == "feasibility_only"
    assert attached["may_enter_geometry_candidate"] is False
    assert by_id["2022_ift_C_dx_slide_mm"]["value"] == pytest.approx(0.2541169, abs=5.0e-8)
    assert by_id["2022_station0_offset_width_mm"]["candidate_status"] == (
        "population_spread_not_sigma"
    )


def test_interface_does_not_possess_information_requirement():
    reports = build_all_reports(_config())
    frames = reports["survey_frame_reconciliation_validation"]
    feasibility = reports["track_fisher_external_prior_feasibility"]
    decision = reports["next_stage_decision"]
    slots = {row["constraint_id"]: row for row in reports["survey_candidate_database"]["official_constraint_slots"]}
    assert slots["ift_station0_ry"]["availability"] == "unavailable"
    assert slots["ift_station0_ry"]["value"] is None
    assert slots["ift_C_dx"]["availability"] == "feasibility_only"
    assert slots["ift_C_dx"]["sigma"] is None
    assert slots["ift_C_dx"]["may_enter_geometry_candidate"] is False
    assert frames["explicit_transform_completed"] is False
    assert frames["unconfirmed_tilt_mapped_to_ry"] is False
    assert feasibility["station_sd_used_as_sigma"] is False
    assert feasibility["unconfirmed_tilt_mapped_to_ry"] is False
    assert feasibility["coarsest_unlocking_sigma_ry_mrad"] == pytest.approx(20.0)
    assert feasibility["coarsest_unlocking_sigma_cdx_mm"] == pytest.approx(0.34514281813225994)
    assert feasibility["geometry_candidate"] is False
    assert decision["decision"] == DECISION
    assert decision["survey_candidates_possess_degeneracy_breaking_information_requirement"] is False
    assert decision["have_validated_ry_mapping"] is False
    assert decision["have_measurement_covariance"] is False
    assert decision["enough_to_write_geometry"] is False
    request = reports["metrology_covariance_request"]
    assert request["candidates_do_not_yet_satisfy_information_requirement"] is True
    texts = " ".join(item["what"] for item in request["requests"])
    assert "C_dx" in texts
    assert "station ry" in texts
    for payload in reports.values():
        assert payload["geometry_candidate"] is False
        assert_no_alignment_payload(payload)
