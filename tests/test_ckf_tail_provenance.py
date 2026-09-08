"""Task B5 CKF tail provenance.  No sealed test, no C/Q tuning, no cuts."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_tail_provenance import (
    CLASS_A,
    CLASS_B,
    CLASS_C,
    DECISION_AUDITED,
    CkfProvenanceError,
    classify_provenance_flags,
    decide_case,
    dump_path_for_source,
    inherit_frozen_stage,
    load_config,
    load_frozen_classified_tails,
    refuse_covariance_rescale,
    refuse_measurement_model_v2,
    refuse_outlier_rejection,
    refuse_process_noise_tuning,
    refuse_quality_cuts,
    refuse_tail_rescreen,
    refuse_truth_qoverp,
)


def test_config_inherits_frozen_wb100():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert inherited["workbook_98"]["situation"] == "acts_q_materialized_closure_failed"
    assert inherited["workbook_99"]["primary_case"] == "high_chi2_tail_dominated"
    assert inherited["workbook_100"]["decision"] == "acts_transport_tail_uncertainty_analyzed"
    assert inherited["workbook_100"]["primary_case"] == "wrong_track_state_dominated"
    assert inherited["workbook_100"]["classification_sha256"].startswith("4b87eeb9")
    assert config["do_not_rescreen_wb99_tails"] is True
    assert config["do_not_design_quality_cuts"] is True
    assert "ckf_tail_provenance_analysis_v1" in config["output_root"]


def test_frozen_lists_match_wb99_and_wb100():
    frozen = load_frozen_classified_tails(load_config())
    assert frozen["rescreened"] is False
    assert frozen["catalogs"]["0.99"]["n"] == 24
    assert frozen["catalogs"]["0.999"]["n"] == 3


def test_dump_path_stays_on_wb98():
    path = dump_path_for_source(load_config(), "mc24_100043_00200_00299")
    assert path.as_posix().endswith(
        "outputs/acts_transport_dump_v1/dumps/mc24_100043_00200_00299/ckf_acts_transport.jsonl"
    )


def test_forbidden_repairs():
    with pytest.raises(CkfProvenanceError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(CkfProvenanceError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(CkfProvenanceError, match="process noise"):
        refuse_process_noise_tuning()
    with pytest.raises(CkfProvenanceError, match="outlier"):
        refuse_outlier_rejection()
    with pytest.raises(CkfProvenanceError, match="rescreened"):
        refuse_tail_rescreen()
    with pytest.raises(CkfProvenanceError, match="quality cuts"):
        refuse_quality_cuts()
    with pytest.raises(CkfProvenanceError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def _event(**overrides):
    payload = {
        "p_mev": 4.0e5,
        "p_truth_mev": 4.1e5,
        "pull": {"q_over_p": 0.2},
        "flags": {},
    }
    payload.update(overrides)
    return payload


def test_association_beats_reconstruction_and_physics():
    config = load_config()
    reco = {
        "n_long_tracks": 0,
        "n_segments": 2,
        "n_stations_present": 2,
        "ckf_chi2_per_ndof": 1.0,
        "tracklet_truth_ids": [10001, 200014],
        "tracklet_truth_match": [0.5, 1.0],
    }
    result = classify_provenance_flags(
        _event(), reco, config=config, n_ckf_tracks=1, n_pairs_for_identity=2
    )
    assert result["primary_class"] == CLASS_A
    assert result["flags"]["conflicting_tracklet_truth_id"] is True
    assert result["flags"]["same_state_multi_pair"] is True


def test_not_a_long_track_is_reconstruction():
    config = load_config()
    reco = {
        "n_long_tracks": 0,
        "n_segments": 4,
        "n_stations_present": None,
        "ckf_chi2_per_ndof": None,
        "tracklet_truth_ids": [10001],
        "tracklet_truth_match": [1.0],
    }
    result = classify_provenance_flags(
        _event(), reco, config=config, n_ckf_tracks=1, n_pairs_for_identity=1
    )
    assert result["primary_class"] == CLASS_B
    assert result["flags"]["not_a_long_track"] is True


def test_decide_case_majority_reconstruction():
    config = load_config()
    rows = []
    for _ in range(8):
        rows.append(
            {
                "classes": {"A": False, "B": True, "C": False},
                "primary_class": CLASS_B,
                "flags": {"not_a_long_track": True, "same_state_multi_pair": False},
            }
        )
    rows.append(
        {
            "classes": {"A": True, "B": True, "C": False},
            "primary_class": CLASS_A,
            "flags": {"not_a_long_track": True, "same_state_multi_pair": True},
        }
    )
    result = decide_case({"catalogs": {"0.99": {"events": rows}}}, config)
    assert result["primary_case"] == CLASS_B
    assert result["next_step"] == "tracking_reconstruction_audit"
    assert result["measurement_model_v2_entered"] is False
    assert result["quality_cuts_designed"] is False
    assert DECISION_AUDITED == "ckf_tail_provenance_audited"


def test_decide_case_association_majority():
    config = load_config()
    rows = [
        {
            "classes": {"A": True, "B": False, "C": False},
            "primary_class": CLASS_A,
            "flags": {"not_a_long_track": False, "same_state_multi_pair": True},
        }
        for _ in range(6)
    ]
    result = decide_case({"catalogs": {"0.99": {"events": rows}}}, config)
    assert result["primary_case"] == CLASS_A
    assert result["next_step"] == "association_quality_control"
    assert CLASS_C  # keep import live
