from __future__ import annotations

from pathlib import Path

import pytest

from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.tracker_align_conditions_survey_provenance import (
    DECISION,
    SCHEMA_VERSION,
    build_all_reports,
    c_dx_from_layer_dx,
    classify_origin,
    decode_compact_identifier,
    load_audit_config,
    load_raw_pool_dump,
    query_cool_align_metadata,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_audit_config(
        Path("configs/tracker_align_conditions_survey_provenance_audit_v1.yaml")
    )


def test_config_freezes_protocol_and_forbids_survey_promotion():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    assert config["do_not_treat_conditions_as_independent_survey"] is True
    assert config["do_not_upgrade_numeric_agreement_to_survey_derived"] is True
    assert config["do_not_invent_survey_numbers"] is True
    assert config["geometry_write_allowed"] is False
    assert config["published_survey_metrology_readonly"]["named_2021_ppt_found"] is False
    assert config["published_survey_metrology_readonly"]["do_not_invent_ppt_measured_points"] is True


def test_compact_identifier_decode_and_cdx_formula():
    ift_l0 = decode_compact_identifier(0x80000000)
    ift_l2 = decode_compact_identifier(0x84000000)
    upstream_l0 = decode_compact_identifier(0x88000000)
    assert ift_l0["station_id"] == 0
    assert ift_l0["layer_id"] == 0
    assert ift_l0["is_ift"] is True
    assert ift_l2["station_id"] == 0
    assert ift_l2["layer_id"] == 2
    assert upstream_l0["station_id"] == 1
    assert c_dx_from_layer_dx(143.31650009479299, 136.16341628438252) == pytest.approx(3.576541905205235)


def test_origin_labels_do_not_promote_close_numbers():
    assert classify_origin(value=0.0, published_scale=0.067, explicit_provenance=False, compatible=False) == (
        "survey_not_encoded_in_current_conditions"
    )
    assert classify_origin(value=74.6, published_scale=0.067, explicit_provenance=False, compatible=False) == (
        "alignment_origin_unknown"
    )
    assert classify_origin(value=0.08, published_scale=0.10, explicit_provenance=False, compatible=True) == (
        "survey_origin_candidate"
    )
    assert classify_origin(value=0.08, published_scale=0.10, explicit_provenance=True, compatible=True) == (
        "survey_derived"
    )


def test_cool_and_official_dumps_answer_the_three_questions():
    sqlite = Path("/cvmfs/faser.cern.ch/repo/sw/database/DBRelease/current/sqlite200/ALLP200.db")
    raw_path = Path("outputs/tracker_align_conditions_survey_provenance_audit_v1/_raw_pool_dump.json")
    if not sqlite.is_file() or not raw_path.is_file():
        pytest.skip("CVMFS DBRelease or raw POOL dump is not available")
    config = _config()
    cool = query_cool_align_metadata(sqlite)
    raw = load_raw_pool_dump(raw_path)
    reports = build_all_reports(config, raw, cool)
    data06 = cool["CONDBR3"]["resolved_global_tags"]["OFLCOND-FASER-06"]
    assert data06["tracker_align_tag"] == "TRACKER-ALIGN-06"
    assert data06["tracker_align_tag_description"] is None
    assert "FASER-06_2024_Align.pool.root" in data06["unique_pool_files"]
    assert cool["OFLP200"]["resolved_global_tags"]["OFLCOND-FASER-06"]["unique_pool_files"] == [
        "FASER-02_Align.pool.root"
    ]
    summary = reports["summary"]
    assert summary["label"] == "existing_conditions_state"
    assert summary["not_survey_measurement"] is True
    assert summary["station0_Stations_ry_cond_mrad"] == pytest.approx(0.0)
    assert summary["ift_plane_ry_used_by_reconstruction_mrad"] == pytest.approx(74.6408958977653)
    assert summary["C_dx_cond_mm"] == pytest.approx(3.576541905205235)
    assert reports["diff"]["oflcond_05_2024_equals_06_2024"] is True
    assert reports["diff"]["oflcond_04_has_no_ift_planes"] is True
    assert reports["diff"]["stations_channel_is_identity_in_all_dumped_files"] is True
    assert reports["provenance"]["found_explicit_survey_or_metrology_provenance"] is False
    assert reports["provenance"]["found_cool_tag_description"] is False
    decision = reports["decision"]
    assert decision["decision"] == DECISION
    assert decision["enough_provenance_for_independent_survey_constraint"] is False
    assert decision["emits_alignment_payload"] is False
    assert_no_alignment_payload(decision)
    assert_no_alignment_payload(summary)
    assert_no_alignment_payload(reports["comparison"])
    origins = {row["origin"] for row in reports["comparison"]["comparisons"]}
    assert "survey_derived" not in origins
    assert "survey_not_encoded_in_current_conditions" in origins
    assert "alignment_origin_unknown" in origins
