from __future__ import annotations

from pathlib import Path

import pytest

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.survey_metrology_iov_infrastructure import (
    DECISION_INGESTION_READY,
    SCHEMA_VERSION,
    combine_track_and_external_fisher,
    conditions_provenance_matrix,
    decide_next_stage,
    empty_constraint_catalog,
    fisher_validation,
    iov_records,
    load_infrastructure_config,
    metrology_requirement_table,
    synthetic_feasibility_constraint,
    validate_constraint,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_infrastructure_config(
        Path("configs/survey_metrology_iov_alignment_infrastructure_v1.yaml")
    )


def test_config_freezes_protocol_and_entry61_requirement():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["do_not_rebuild_2024_r0022_jacobian"] is True
    assert config["do_not_use_design_as_survey"] is True
    assert config["do_not_average_across_iovs"] is True
    assert config["synthetic_prior_feasibility_only"] is True
    assert config["geometry_candidate_from_synthetic"] is False
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    frozen = config["entry_61_frozen"]
    assert frozen["degeneracy_breaking_sigma_ry_mrad"] == pytest.approx(20.0)
    assert frozen["degeneracy_breaking_sigma_cdx_mm"] == pytest.approx(0.315)
    assert config["useful_precision"]["cdx_mm"] == pytest.approx(0.080)
    assert config["useful_precision"]["ry_mrad"] == pytest.approx(0.5)
    assert config["useful_precision"]["ry_mrad"] < frozen["degeneracy_breaking_sigma_ry_mrad"]
    assert config["useful_precision"]["cdx_mm"] < frozen["degeneracy_breaking_sigma_cdx_mm"]


def test_empty_catalog_is_unavailable_and_rejects_design_as_measurement():
    config = _config()
    catalog = empty_constraint_catalog(config)
    assert catalog
    assert all(row["availability"] == "unavailable" for row in catalog)
    assert all(row["value"] is None and row["sigma"] is None for row in catalog)
    assert all(row["may_enter_geometry_candidate"] is False for row in catalog)
    assert all(row["independent_of_track_residual"] is True for row in catalog)
    kinds = {row["kind"] for row in catalog}
    assert kinds == {"station_rigid_transform", "layer_relative_transform", "C_dx", "linear_equality"}
    design = dict(catalog[0])
    design["availability"] = "measured"
    design["value"] = [31.5]
    design["sigma"] = [0.0]
    design["provenance"] = "geomdb_layerpitch"
    design["source_file"] = "geomDB.sql"
    with pytest.raises(ValueError, match="design or software gauge"):
        validate_constraint(design)


def test_synthetic_prior_restores_rank_but_cannot_be_geometry_candidate():
    config = _config()
    validation = fisher_validation(config)
    assert validation["did_rebuild_2024_r0022_jacobian"] is False
    assert validation["synthetic_may_enter_geometry_candidate"] is False
    assert validation["perfect_toy_20mrad_does_not_unlock"] is True
    assert validation["leaky_toy_20mrad_unlocks"] is True
    assert validation["leaky_toy_0p315mm_cdx_unlocks"] is True
    leaky = validation["cases"]["leaky_toy_plus_20mrad_ry"]
    assert leaky["rank"] == 3
    assert abs(float(leaky["ry_cdx_correlation"])) <= 0.90
    assert leaky["geometry_candidate"] is False
    assert leaky["n_external_rows"] == 1
    synthetic = synthetic_feasibility_constraint(
        constraint_id="synthetic_ry_degeneracy_break",
        kind="station_rigid_transform",
        parameter_name="ry",
        sigma=20.0,
        unit="mrad",
        scan_column=1,
        frozen_requirement="test",
    )
    with pytest.raises(ValueError, match="cannot enter a geometry candidate"):
        bad = dict(synthetic)
        bad["may_enter_geometry_candidate"] = True
        validate_constraint(bad)
    assert_no_alignment_payload(validation)


def test_2024_conditions_05_and_06_are_distinct_iovs():
    config = _config()
    records = iov_records(config)
    ids = [row["iov_id"] for row in records]
    assert "2024_r0022_production_OFLCOND-FASER-05" in ids
    assert "2024_protocol_OFLCOND-FASER-06" in ids
    matrix = conditions_provenance_matrix(records)
    pair = matrix["production_2024_05_versus_protocol_2024_06"]
    assert pair["same_year"] is True
    assert pair["same_conditions_tag"] is False
    assert pair["may_average_residuals"] is False
    assert pair["may_average_corrections"] is False
    assert all(not item["may_share_alignment_constants"] for item in matrix["pairs"])
    years = {row["year"] for row in records}
    assert {2022, 2023, 2024, 2025}.issubset(years)
    ift_off = next(row for row in records if row["iov_id"] == "2022_r0021_OFLCOND-FASER-04")
    assert ift_off["c_dx_parameter"] is False
    assert ift_off["parameter_blocks"]["C_dx"]["present"] is False
    dz = ift_off["parameter_blocks"]["survey_or_gauge_constrained"][0]
    assert dz["block"] == "station_dz"
    assert dz["float_from_tracks"] is False


def test_metrology_table_does_not_promote_degeneracy_threshold_and_decision_is_ready():
    config = _config()
    table = metrology_requirement_table(config)
    assert table["do_not_quote_20_mrad_or_0_315_mm_as_the_final_target"] is True
    ry = table["requests"][0]
    assert ry["degeneracy_breaking"]["not_the_final_measurement_target"] is True
    assert ry["physically_useful"]["sigma"] == pytest.approx(0.5)
    assert ry["physically_useful"]["sigma"] < ry["degeneracy_breaking"]["sigma"]
    cdx = table["requests"][1]
    assert cdx["physically_useful"]["sigma_C_dx_mm"] == pytest.approx(0.080)
    catalog = empty_constraint_catalog(config)
    validation = fisher_validation(config)
    decision = decide_next_stage(
        catalog=catalog,
        validation=validation,
        matrix=conditions_provenance_matrix(iov_records(config)),
    )
    assert decision["decision"] == DECISION_INGESTION_READY
    assert decision["emits_alignment_payload"] is False
    assert decision["as_built_survey_available"] is False
    assert decision["did_rebuild_2024_r0022_jacobian"] is False
    assert decision["combiner_ready"] is True
    assert_no_alignment_payload(decision)
    assert_no_alignment_payload(table)
