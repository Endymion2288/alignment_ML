"""Yasu Stage 0 WithoutIFT definition.  No sealed test, no LTO substitute."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_without_ift_definition import (
    DECISION_ESTABLISHED,
    DECISION_NOT_ESTABLISHED,
    IFT_STATION_ID,
    MECHANISM_COLLECTION_ABSENT,
    MECHANISM_DUMP_MISSING,
    MECHANISM_IFT_LEAK,
    MECHANISM_LTO_SUBSTITUTE,
    SOURCE_COLLECTION,
    WithoutIftDefinitionError,
    calypso_source_contract,
    decide,
    evaluate_tracks,
    ift_layer_codes,
    inherit_frozen_stage,
    layer_code,
    load_config,
    refuse_dummy_segmentfit,
    refuse_four_station_substitute,
    refuse_lto_substitute,
    refuse_truth_qoverp,
    state_definition,
)


def test_config_inherits_frozen_and_refuses_substitutes():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_87"]["decision"] == "faseracts_transport_covariance_not_validated"
    assert inherited["workbook_95"]["decision"] == "physical_qoverp_semantics_not_established"
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert inherited["workbook_109"]["decision"] == "leave_target_out_state_materialization_established"
    assert inherited["workbook_109"]["lto_is_not_without_ift"] is True
    assert inherited["workbook_109"]["frozen_artifact_root"].endswith("alignment_ML")
    assert config["source_collection"] == SOURCE_COLLECTION
    assert config["three_st_to_ift_chain_authorized"] is False
    assert tuple(config["masked_layers"]) == (0, 1, 2, 3, 4, 5)


def test_calypso_station_map_and_masked_layers():
    contract = calypso_source_contract()
    assert contract["si_detector_element"]["isInterface"] == "station == 0"
    assert contract["si_detector_element"]["isUpstream"] == "station == 1"
    assert contract["si_detector_element"]["isCentral"] == "station == 2"
    assert contract["si_detector_element"]["isDownstream"] == "station == 3"
    assert ift_layer_codes() == (0, 1, 2, 3, 4, 5)
    assert layer_code(0, 2, 1) == 5
    assert layer_code(1, 0, 0) == 6
    assert IFT_STATION_ID == 0
    definition = state_definition()
    assert definition["q_over_p_native_unit"] == "per_MeV"
    assert definition["q_over_p_signed"] is True
    assert definition["covariance_dimension"] == 5


def test_forbidden_repairs_are_hard_errors():
    with pytest.raises(WithoutIftDefinitionError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(WithoutIftDefinitionError, match="dummy"):
        refuse_dummy_segmentfit()
    with pytest.raises(WithoutIftDefinitionError, match="LTO"):
        refuse_lto_substitute()
    with pytest.raises(WithoutIftDefinitionError, match="4-station"):
        refuse_four_station_substitute()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def _physical_track(*, ift: bool = False, decoded: bool = True) -> dict:
    cov = np.eye(5) * 1.0e-4
    cov[4, 4] = 3.0e-10
    cov[0, 4] = cov[4, 0] = 1.0e-8
    stations = [0, 1, 2, 3] if ift else [1, 2, 3]
    return {
        "collection": SOURCE_COLLECTION,
        "native_state": [0.1, -0.2, 0.01, 0.02, -2.0e-6],
        "native_covariance": cov.tolist(),
        "has_covariance": True,
        "is_truth": False,
        "station_decoded_by_fasersct_id": decoded,
        "measurements_on_track_stations": stations,
        "tsos_stations": stations,
    }


def _split(tracks: list[dict], *, ift_clusters: int = 2) -> dict:
    config = load_config()
    return {
        "root_has_without_ift": True,
        "dumps_present": True,
        "events": {
            "n_events": 4,
            "n_events_with_ift_clusters": ift_clusters,
            "n_events_cluster_unavailable": 0,
            "n_events_station_decoded": 4,
        },
        "without_ift": evaluate_tracks(tracks, config, collection=SOURCE_COLLECTION),
        "four_station": evaluate_tracks([], config, collection="CKFTrackCollection"),
    }


def test_clean_three_station_tracks_would_pass():
    tracks = [_physical_track() for _ in range(4)]
    construction = _split(tracks)
    validation = _split(tracks)
    decision = decide(
        construction, validation, {"all_match": True}, dumps_materialized=True
    )
    assert decision["verdict"] == "PASS"
    assert decision["decision"] == DECISION_ESTABLISHED
    assert decision["three_st_to_ift_chain_authorized"] is True
    assert decision["lto_is_not_this_collection"] is True


def test_missing_dump_and_ift_leak_fail():
    empty = _split([])
    empty["events"]["n_events_with_ift_clusters"] = 2
    missing = decide(empty, empty, {"all_match": True}, dumps_materialized=False)
    assert missing["verdict"] == "FAIL"
    assert missing["decision"] == DECISION_NOT_ESTABLISHED
    assert missing["mechanism"] == MECHANISM_DUMP_MISSING
    assert missing["three_st_to_ift_chain_authorized"] is False

    leaked = _split([_physical_track(ift=True) for _ in range(4)])
    failed = decide(leaked, leaked, {"all_match": True}, dumps_materialized=True)
    assert failed["mechanism"] == MECHANISM_IFT_LEAK

    absent = _split([_physical_track() for _ in range(4)])
    absent["root_has_without_ift"] = False
    failed_absent = decide(absent, absent, {"all_match": True}, dumps_materialized=True)
    assert failed_absent["mechanism"] == MECHANISM_COLLECTION_ABSENT


def test_lto_substitute_token_is_not_a_pass_path():
    assert MECHANISM_LTO_SUBSTITUTE == "lto_used_as_without_ift_substitute"
    with pytest.raises(WithoutIftDefinitionError):
        refuse_lto_substitute()
