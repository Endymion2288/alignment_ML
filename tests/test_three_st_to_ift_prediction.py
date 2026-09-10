"""Yasu Stage 1 3ST→IFT prediction contract.  No sealed test, no LTO substitute."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.three_st_to_ift_prediction import (
    DECISION_ESTABLISHED,
    DECISION_NOT_ESTABLISHED,
    IFT_STATION_ID,
    MECHANISM_DUMP_MISSING,
    MECHANISM_IFT_LEAK,
    MECHANISM_LTO_SUBSTITUTE,
    MECHANISM_PROP,
    SOURCE_COLLECTION_NAME,
    STATION_Z_MM,
    ThreeStToIftPredictionError,
    decide,
    evaluate_tracks,
    inherit_frozen_stage,
    load_config,
    refuse_dummy_segmentfit,
    refuse_four_station_substitute,
    refuse_lto_substitute,
    refuse_truth_qoverp,
    residual_definition,
    state_definition,
)


def test_config_inherits_stage0_and_refuses_substitutes():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_87"]["decision"] == "faseracts_transport_covariance_not_validated"
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert inherited["workbook_109"]["decision"] == "leave_target_out_state_materialization_established"
    assert inherited["workbook_109"]["lto_is_not_without_ift"] is True
    assert inherited["workbook_117"]["decision"] == "ckf_without_ift_definition_established"
    assert config["source_collection"] == SOURCE_COLLECTION_NAME
    assert config["mc_qp_calibration_authorized"] is False
    assert config["residual_conditional_authorized"] is False
    assert config["propagation_direction"] == "backward"
    assert config["target_station"] == IFT_STATION_ID
    assert tuple(float(config["station_z_mm"][i]) for i in range(4)) == STATION_Z_MM
    assert residual_definition()["prediction_uses_ift_measurement"] is False
    assert state_definition()["q_over_p_signed"] is True


def test_forbidden_repairs_are_hard_errors():
    with pytest.raises(ThreeStToIftPredictionError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(ThreeStToIftPredictionError, match="dummy"):
        refuse_dummy_segmentfit()
    with pytest.raises(ThreeStToIftPredictionError, match="LTO"):
        refuse_lto_substitute()
    with pytest.raises(ThreeStToIftPredictionError, match="4-station"):
        refuse_four_station_substitute()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)
    assert MECHANISM_LTO_SUBSTITUTE == "lto_used_as_without_ift_substitute"


def _physical_track(
    *,
    leak: bool = False,
    plane: bool = True,
    residual: bool = True,
    decoded: bool = True,
) -> dict:
    cov = np.eye(5) * 1.0e-4
    cov[4, 4] = 3.0e-10
    cov[0, 4] = cov[4, 0] = 1.0e-8
    residual_row = {
        "identifier": "1",
        "station": 0,
        "layer": 0,
        "side": 0,
        "loc0_cluster_mm": 0.12,
        "predicted_loc0_mm": 0.10,
        "residual_loc0_mm": 0.02,
        "residual_available": residual,
        "propagate_success": residual,
        "prediction_uses_this_measurement": False,
        "reconstruction_associated": True,
        "failure_class": None if residual else "propagate_surface_failed",
    }
    return {
        "kind": "track",
        "collection": SOURCE_COLLECTION_NAME,
        "native_state": [0.1, -0.2, 0.01, 0.02, -2.0e-6],
        "native_covariance": cov.tolist(),
        "has_covariance": True,
        "is_truth": False,
        "station_decoded_by_fasersct_id": decoded,
        "n_ift_mot": 1 if leak else 0,
        "n_ift_tsos": 1 if leak else 0,
        "ift_leak": leak,
        "prediction_uses_ift_measurement": False,
        "primary_failure_class": "ift_measurement_leak" if leak else None,
        "model0_no_process_noise": {"success": plane},
        "model1_acts_process_noise": {"success": plane},
        "residuals": [residual_row],
    }


def _inherited() -> dict:
    return {
        "workbook_117": {"decision": "ckf_without_ift_definition_established"},
    }


def _split(tracks: list[dict], *, ift_clusters: int = 2) -> dict:
    config = load_config()
    return {
        "dumps_present": True,
        "events": {
            "n_events": 4,
            "n_events_with_ift_clusters": ift_clusters,
            "n_events_cluster_unavailable": 0,
            "n_events_selection_loss": 0,
            "n_events_station_decoded": 4,
        },
        "tracks": evaluate_tracks(tracks, config),
    }


def test_clean_independent_predictions_would_pass():
    tracks = [_physical_track() for _ in range(4)]
    construction = _split(tracks)
    validation = _split(tracks)
    decision = decide(
        construction,
        validation,
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
    )
    assert decision["verdict"] == "PASS"
    assert decision["decision"] == DECISION_ESTABLISHED
    assert decision["mc_qp_calibration_authorized"] is True
    assert decision["residual_conditional_authorized"] is True
    assert decision["prediction_uses_ift_measurement"] is False
    assert decision["warning_count_is_not_reconstruction_loss"] is True


def test_missing_dump_leak_and_propagation_fail():
    empty = _split([])
    empty["events"]["n_events_with_ift_clusters"] = 2
    missing = decide(
        empty, empty, {"all_match": True}, _inherited(), dumps_materialized=False
    )
    assert missing["verdict"] == "FAIL"
    assert missing["decision"] == DECISION_NOT_ESTABLISHED
    assert missing["mechanism"] == MECHANISM_DUMP_MISSING
    assert missing["mc_qp_calibration_authorized"] is False

    leaked = _split([_physical_track(leak=True) for _ in range(4)])
    failed = decide(
        leaked, leaked, {"all_match": True}, _inherited(), dumps_materialized=True
    )
    assert failed["mechanism"] == MECHANISM_IFT_LEAK

    no_plane = _split([_physical_track(plane=False, residual=False) for _ in range(4)])
    failed_prop = decide(
        no_plane, no_plane, {"all_match": True}, _inherited(), dumps_materialized=True
    )
    assert failed_prop["mechanism"] == MECHANISM_PROP
    assert failed_prop["mc_qp_calibration_authorized"] is False


def test_plane_success_without_residual_does_not_open_conditionals():
    tracks = [_physical_track(residual=False) for _ in range(4)]
    construction = _split(tracks)
    validation = _split(tracks)
    decision = decide(
        construction,
        validation,
        {"all_match": True},
        _inherited(),
        dumps_materialized=True,
    )
    assert decision["verdict"] == "PASS"
    assert decision["mc_qp_calibration_authorized"] is True
    assert decision["residual_conditional_authorized"] is False
