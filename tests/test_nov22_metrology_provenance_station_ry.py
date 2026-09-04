from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.cad_survey_nov22 import parse_cad_survey_nov22_file
from alignment.nov22_metrology_provenance_station_ry import (
    DECISION,
    SCHEMA_VERSION,
    LiveCalypsoSensorDumpMissingError,
    apply_station_alignment,
    build_all_reports,
    extract_alpha_beta_gamma,
    jacobian_global_ry,
    jacobian_ry_about_pivot,
    load_contract_config,
    refuse_kabsch_as_station_ry,
    refuse_live_geomodeltest_as_aligned_dump,
    rotation_about_point,
    software_fd_station_ry,
    station_alignment_transform,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.survey_derived_prior_interface import UnconfirmedRyMappingError
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_contract_config(
        Path("configs/nov22_metrology_provenance_station_ry_contract_v1.yaml")
    )


def test_config_freezes_protocol_and_forbids_geometry_write():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["geometry_write_allowed"] is False
    assert config["geometry_candidate_from_software_fd"] is False
    assert config["do_not_run_newton"] is True
    assert config["do_not_map_kabsch_or_dz_vs_x_to_station_ry"] is True
    assert config["do_not_construct_covariance_from_population_scatter"] is True
    assert config["software_fd_sensitivity_only"] is True
    assert config["entry_61_frozen"]["degeneracy_breaking_sigma_ry_mrad"] == pytest.approx(20.0)
    assert config["entry_61_frozen"]["degeneracy_breaking_sigma_cdx_mm"] == pytest.approx(0.315)


def test_station_alignment_is_left_multiply_about_the_faser_origin():
    transform = station_alignment_transform((0.0, 0.0, 0.0, 0.0, 0.0001, 0.0))
    rx, ry, rz = extract_alpha_beta_gamma(transform)
    assert ry == pytest.approx(0.0001, abs=1.0e-12)
    assert rx == pytest.approx(0.0, abs=1.0e-12)
    assert rz == pytest.approx(0.0, abs=1.0e-12)
    identity = station_alignment_transform((0.0, 0.0, 0.0, 0.0, 0.0, 0.0))
    assert np.allclose(identity, np.eye(4), atol=1.0e-15)
    point = np.array([[10.0, 20.0, 30.0]])
    moved = apply_station_alignment(point, (0.0, 0.0, 0.0, 0.0, 0.0001, 0.0))
    analytic = jacobian_global_ry(point) * 0.0001
    assert moved[0, 0] == pytest.approx(point[0, 0] + analytic[0, 0], rel=1.0e-6)
    assert moved[0, 2] == pytest.approx(point[0, 2] + analytic[0, 2], rel=1.0e-6)
    about_station = jacobian_ry_about_pivot(point, (0.0, 0.0, -1860.15))
    assert about_station[0, 0] != pytest.approx(analytic[0, 0])


def test_software_fd_selects_faser_origin_not_centroid_or_station_origin():
    config = _config()
    parsed = parse_cad_survey_nov22_file(Path(config["source"]["relative"]))
    result = software_fd_station_ry(parsed, config)
    rms = result["rms_numeric_minus_analytic_mm_per_rad"]
    assert result["closest_pivot"] == "faser_global_origin"
    assert rms["about_faser_origin"] < rms["about_sensor_centroid"]
    assert rms["about_faser_origin"] < rms["about_station_geomodel_origin"]
    assert rms["about_faser_origin"] < rms["about_casper_support_beam_delta_faser"]
    assert result["analytic_dx_dry_equals_mean_z"] is True
    assert result["numeric_matches_analytic_sinc_factor"] is True
    assert result["live_calypso_sensor_dump_ran"] is False
    assert result["not_an_alignment_solve"] is True
    assert result["not_a_geometry_candidate"] is True
    assert result["may_fill_measured_ry_slot"] is False
    sensors = [
        row
        for row in parsed["sensors"]
        if int(row["station_id"]) == 0 and int(row["side"]) == 0
    ]
    nominal = np.asarray([row["implied_nominal_mm"] for row in sensors], dtype=np.float64)
    centroid = np.mean(nominal, axis=0)
    about_centroid = rotation_about_point(nominal, ry_rad=0.0001, pivot_mm=centroid)
    about_origin = apply_station_alignment(nominal, (0.0, 0.0, 0.0, 0.0, 0.0001, 0.0))
    assert not np.allclose(about_centroid, about_origin, atol=1.0e-6)


def test_inventory_and_gates_stay_closed():
    reports = build_all_reports(_config())
    provenance = reports["nov22_raw_metrology_provenance"]
    iov = reports["nov22_iov_provenance"]
    decision = reports["next_stage_decision"]
    slots = {row["constraint_id"]: row for row in reports["official_constraint_slots"]["slots"]}
    assert provenance["producing_script"]["identified"] is False
    assert provenance["ingestion_layer_built"] is False
    assert provenance["raw_measurement_objects"]["fit_covariance_for_this_dump"] is False
    assert provenance["raw_measurement_objects"]["literature_unige_cmm_nima_1034"][
        "usable_as_nov22_covariance"
    ] is False
    assert provenance["author_homes_readable"] is False
    assert iov["may_be_used_as_2024_or_2025_station_rigid_body"] is False
    assert iov["C_dx_upgraded_to_common_static"] is False
    assert iov["may_constrain_year"] == [2022]
    assert slots["ift_station0_ry"]["availability"] == "unavailable"
    assert slots["ift_station0_ry"]["value"] is None
    assert slots["ift_C_dx"]["availability"] == "feasibility_only"
    assert slots["ift_C_dx"]["sigma"] is None
    assert slots["ift_C_dx"]["value"][0] == pytest.approx(0.2541169, abs=5.0e-8)
    assert decision["decision"] == DECISION
    assert decision["enough_to_enter_fisher"] is False
    assert decision["enough_to_fill_measured"] is False
    assert decision["did_enter_fisher"] is False
    assert decision["blocker_status"]["station_ry_software_contract"] == "resolved"
    assert decision["blocker_status"]["station_ry_survey_mapping"] == "unresolved"
    assert decision["blocker_status"]["measurement_covariance"] == "unresolved"
    assert decision["blocker_status"]["iov_provenance"] == "unresolved_2022_measurement_only"
    assert decision["closest_pivot_of_stations_ry"] == "faser_global_origin"
    contract = reports["calypso_station_ry_transform_contract"]
    assert contract["pivot"]["not_station_geomodel_origin"] is True
    assert contract["small_angle_derivatives"]["ry"]["dx"] == "+z"
    for payload in reports.values():
        assert payload["geometry_candidate"] is False
        assert_no_alignment_payload(payload)
        assert payload["did_train_model"] is False
        assert payload["did_run_alignment_fit"] is False
    with pytest.raises(UnconfirmedRyMappingError):
        refuse_kabsch_as_station_ry()
    with pytest.raises(LiveCalypsoSensorDumpMissingError):
        refuse_live_geomodeltest_as_aligned_dump()
