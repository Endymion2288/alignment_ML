from __future__ import annotations

from pathlib import Path

import pytest

from alignment.cad_survey_nov22 import (
    DECISION,
    SCHEMA_VERSION,
    IndexTupleGuessError,
    build_all_reports,
    load_audit_config,
    parse_cad_survey_nov22_file,
    refuse_index_tuple_guess,
    refuse_population_sigma_as_prior,
    refuse_unconfirmed_ry_mapping,
    reproduce_quoted_summaries,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
    sha256_file,
)
from alignment.survey_derived_prior_interface import UnconfirmedRyMappingError
from alignment.survey_summary_reconstruction import c_dx_from_layer_x
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_audit_config(Path("configs/cad_survey_nov22_frame_covariance_audit_v1.yaml"))


def test_config_freezes_protocol_and_forbids_geometry_write():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["geometry_write_allowed"] is False
    assert config["geometry_candidate_from_cad_survey"] is False
    assert config["do_not_run_newton"] is True
    assert config["do_not_map_unconfirmed_tilt_to_ry"] is True
    assert config["do_not_use_station_sd_as_measurement_sigma"] is True
    assert config["do_not_guess_index_tuple_meanings"] is True
    assert config["do_not_modify_original_survey_file"] is True
    assert config["do_not_expand_ml_or_module_map"] is True
    assert config["do_not_expand_collision_track_statistics"] is True
    frozen = config["entry_61_frozen"]
    assert frozen["degeneracy_breaking_sigma_ry_mrad"] == pytest.approx(20.0)
    assert frozen["degeneracy_breaking_sigma_cdx_mm"] == pytest.approx(0.315)
    assert config["entry_64_frozen"]["C_dx_slide_mm"] == pytest.approx(0.2541169, abs=5.0e-8)


def test_original_file_is_immutable_and_parser_reproduces_quoted_summaries():
    config = _config()
    source = Path(config["source"]["relative"])
    assert source.is_file()
    assert sha256_file(source) == config["source"]["expected_sha256"]
    assert source.stat().st_size == config["source"]["expected_size_bytes"]
    parsed = parse_cad_survey_nov22_file(source)
    assert parsed["n_sensors"] == 192
    assert len({tuple(row["index_tuple"]) for row in parsed["sensors"]}) == 192
    assert parsed["delta_faser_mm"][0] == pytest.approx(0.006016124530863248)
    assert parsed["delta_faser_mm"][1] == pytest.approx(268.96422105806727)
    assert parsed["delta_faser_mm"][2] == pytest.approx(1228.6941899444103)
    reproduced = reproduce_quoted_summaries(parsed)
    assert reproduced["quoted_summaries_parsed_exactly"] is True
    assert reproduced["recomputed_means_match_quoted_within_print_rounding"] is True
    assert reproduced["sigma_matches_population_std_of_printed_sensors_better_than_sample_std"] is True
    assert reproduced["sigma_is_population_std_ddof0"] is True
    assert reproduced["sigma_is_sample_std_ddof_1"] is False
    assert reproduced["sigma_is_population_spread_not_measurement_sigma"] is True
    assert reproduced["sigma_may_be_used_as_gaussian_prior_uncertainty"] is False
    assert reproduced["all_quoted_layer_triples_average_to_quoted_stations"] is True
    assert reproduced["quoted_layers_0_1_2_are_ift"] is True
    assert reproduced["quoted_C_dx_mm"] == pytest.approx(0.2541169, abs=5.0e-8)
    assert reproduced["quoted_delta_x_L0_minus_L2_mm"] == pytest.approx(0.5082339, abs=5.0e-8)
    quoted_l0 = parsed["quoted_layer_summaries"][0]["delta_mm"][0]
    quoted_l2 = parsed["quoted_layer_summaries"][2]["delta_mm"][0]
    assert c_dx_from_layer_x(quoted_l0, quoted_l2) == pytest.approx(reproduced["quoted_C_dx_mm"])
    station0 = parsed["quoted_station_summaries"][0]["delta_mm"]
    assert station0[0] == pytest.approx(0.727, abs=5.0e-4)
    assert station0[1] == pytest.approx(-0.982, abs=5.0e-4)
    assert station0[2] == pytest.approx(-27.772, abs=5.0e-4)
    assert reproduced["recomputed_C_dx_mm"] != pytest.approx(reproduced["quoted_C_dx_mm"], abs=1.0e-8)


def test_index_tuple_is_source_backed_not_guessed():
    reports = build_all_reports(_config())
    fields = reports["cad_survey_nov22_parsed"]["index_tuple"]["fields"]
    by_name = {row["name"]: row for row in fields}
    assert reports["cad_survey_nov22_parsed"]["index_tuple"]["guessed"] is False
    assert by_name["station_id"]["physical_meaning"].startswith("SCT station")
    assert by_name["layer_id"]["status"] == "confirmed"
    assert by_name["phi_module"]["physical_meaning"].startswith("precision")
    assert "Starboard" in by_name["eta_module"]["physical_meaning"]
    assert "pigtail" in by_name["side"]["physical_meaning"]
    for row in fields:
        assert row["status"] == "confirmed"
        assert row["evidence"]
    with pytest.raises(IndexTupleGuessError):
        refuse_index_tuple_guess("phi_module")


def test_geometry_reproduces_slide_cdx_but_refuses_ry_and_population_sigma():
    reports = build_all_reports(_config())
    geometry = reports["cad_survey_nov22_geometry"]
    covariance = reports["cad_survey_nov22_covariance_audit"]
    frames = reports["cad_survey_nov22_frame_reconciliation"]
    decision = reports["next_stage_decision"]
    assert geometry["reproduces_entry64_C_dx_slide"] is True
    assert geometry["C_dx_mm"] == pytest.approx(0.2541169, abs=5.0e-8)
    assert geometry["delta_x_L0_minus_L2_mm"] == pytest.approx(0.5082339, abs=5.0e-8)
    assert geometry["station0_quoted_mean_matches_entry64_slide"] is True
    assert geometry["may_map_any_fitted_ry_to_calypso_station_ry"] is False
    assert geometry["layer_x_slope_not_used_as_station_ry"] is True
    assert geometry["stereo_not_used_as_station_ry"] is True
    assert geometry["layerpitch_not_used_as_station_ry"] is True
    assert geometry["planes_rotations_not_used_as_station_ry"] is True
    station0 = next(row for row in geometry["stations"] if row["station_id"] == 0)
    assert station0["rigid_kabsch"]["not_calypso_station_ry"] is True
    assert station0["ry_from_dz_vs_x"]["not_calypso_station_ry"] is True
    assert station0["may_map_kabsch_ry_to_calypso_station_ry"] is False
    assert covariance["may_be_used_as_gaussian_prior_sigma"] is False
    assert covariance["constructed_measurement_covariance"] is None
    assert covariance["availability"] == "feasibility_only"
    assert frames["parameter_mapping"]["station_ry"]["validated"] is False
    assert frames["explicit_transform_completed"] is False
    assert frames["unconfirmed_tilt_mapped_to_ry"] is False
    assert decision["decision"] == DECISION
    assert decision["cad_survey_nov22_provides_independent_frame_correct_uncertain_measurement"] is False
    assert decision["have_validated_ry_mapping"] is False
    assert decision["have_measurement_covariance"] is False
    assert decision["have_matching_year_conditions_iov"] is False
    assert decision["enough_to_write_alignment_prior"] is False
    assert decision["enough_to_write_geometry"] is False
    assert decision["did_run_newton"] is False
    assert decision["did_fill_measured_constraint_slot"] is False
    with pytest.raises(UnconfirmedRyMappingError):
        refuse_unconfirmed_ry_mapping("ift_kabsch_ry")
    with pytest.raises(ValueError, match="population spread"):
        refuse_population_sigma_as_prior()


def test_official_slots_stay_feasibility_only_and_payload_is_forbidden():
    reports = build_all_reports(_config())
    slots = {row["constraint_id"]: row for row in reports["official_constraint_slots"]["slots"]}
    assert slots["ift_station0_ry"]["availability"] == "unavailable"
    assert slots["ift_station0_ry"]["value"] is None
    assert slots["ift_C_dx"]["availability"] == "feasibility_only"
    assert slots["ift_C_dx"]["sigma"] is None
    assert slots["ift_C_dx"]["value"][0] == pytest.approx(0.2541169, abs=5.0e-8)
    assert slots["ift_l0_minus_l2_dx"]["availability"] == "feasibility_only"
    assert slots["ift_l0_minus_l2_dx"]["sigma"] is None
    assert slots["ift_C_dx"]["year"] == 2022
    assert slots["ift_C_dx"]["may_enter_geometry_candidate"] is False
    decision = reports["next_stage_decision"]
    blocker_ids = {row["id"] for row in decision["blockers"]}
    assert "cad_to_calypso_station_ry_mapping" in blocker_ids
    assert "measurement_covariance" in blocker_ids
    assert "iov_provenance" in blocker_ids
    request_text = " ".join(item["what"] for item in decision["minimum_request_to_hardware_alignment_survey"])
    assert "covariance" in request_text.lower()
    assert "station-ry" in request_text or "station ry" in request_text.lower()
    assert "IOV" in request_text
    for payload in reports.values():
        assert payload["geometry_candidate"] is False
        assert_no_alignment_payload(payload)
        assert payload["geometry_write_allowed"] is False
        assert payload["did_train_model"] is False
        assert payload["did_run_alignment_fit"] is False
