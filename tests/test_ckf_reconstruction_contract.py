"""Task B6 CKF reconstruction input contract.  No filter, no C/Q, no sealed test."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_reconstruction_contract import (
    CASE_A,
    CASE_B,
    CASE_C,
    DECISION_AUDITED,
    CkfContractError,
    classify_rejection_reason,
    decide_case,
    define_reconstruction_contract,
    dump_path_for_source,
    inherit_frozen_stage,
    is_long_track_accepted,
    load_config,
    load_frozen_wb101_provenance,
    refuse_closure_pass_after_subset,
    refuse_covariance_rescale,
    refuse_input_filter,
    refuse_measurement_model_v2,
    refuse_outlier_rejection,
    refuse_process_noise_tuning,
    refuse_quality_cuts,
    refuse_tail_rescreen,
    refuse_truth_qoverp,
)


def test_config_inherits_frozen_wb101():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert inherited["workbook_98"]["situation"] == "acts_q_materialized_closure_failed"
    assert inherited["workbook_99"]["primary_case"] == "high_chi2_tail_dominated"
    assert inherited["workbook_100"]["primary_case"] == "wrong_track_state_dominated"
    assert inherited["workbook_101"]["decision"] == "ckf_tail_provenance_audited"
    assert inherited["workbook_101"]["primary_case"] == "ckf_reconstruction_failure"
    assert inherited["workbook_101"]["provenance_sha256"].startswith("37479a16")
    assert config["do_not_implement_input_filter"] is True
    assert config["do_not_declare_closure_pass_after_subset"] is True
    assert config["do_not_design_quality_cuts"] is True
    assert "ckf_reconstruction_contract_audit_v1" in config["output_root"]


def test_frozen_wb101_lists_are_not_rescreened():
    frozen = load_frozen_wb101_provenance(load_config())
    assert frozen["rescreened"] is False
    assert frozen["catalogs"]["0.99"]["n"] == 24
    assert frozen["catalogs"]["0.999"]["n"] == 3


def test_dump_path_stays_on_wb98():
    path = dump_path_for_source(load_config(), "mc24_100043_00200_00299")
    assert path.as_posix().endswith(
        "outputs/acts_transport_dump_v1/dumps/mc24_100043_00200_00299/ckf_acts_transport.jsonl"
    )


def test_forbidden_repairs():
    with pytest.raises(CkfContractError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(CkfContractError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(CkfContractError, match="process noise"):
        refuse_process_noise_tuning()
    with pytest.raises(CkfContractError, match="outlier"):
        refuse_outlier_rejection()
    with pytest.raises(CkfContractError, match="rescreened"):
        refuse_tail_rescreen()
    with pytest.raises(CkfContractError, match="quality cuts"):
        refuse_quality_cuts()
    with pytest.raises(CkfContractError, match="filter"):
        refuse_input_filter()
    with pytest.raises(CkfContractError, match="closure PASS"):
        refuse_closure_pass_after_subset()
    with pytest.raises(CkfContractError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_contract_is_definition_not_filter():
    contract = define_reconstruction_contract(load_config())
    assert contract["implemented_as_filter"] is False
    assert contract["production_filter_exists"] is False
    assert contract["current_transport_input"] == "raw_CKF_collection"
    assert contract["eligible_for_transport_validation"]["implemented_as_filter"] is False
    assert contract["eligible_for_transport_validation"]["defined_in_production"] is False
    assert contract["raw_CKF_collection"]["long_track_selection_applied"] is False


def test_missing_station_is_first_rejection_reason():
    config = load_config()
    result = classify_rejection_reason(
        {
            "n_long_tracks": 0,
            "n_segments": 2,
            "tracklet_stations": [0, 1],
            "tracklet_n_hit": [3, 3],
            "tracklet_truth_ids": [-1],
            "tracklet_truth_match": [0.0],
            "tracklet_chi2_per_ndof": [8.0],
            "ckf_chi2_per_ndof": None,
        },
        config=config,
    )
    assert result["primary_reason"] == "missing_station"
    assert result["flags"]["short_segment"] is True
    assert result["flags"]["no_truth_match"] is True
    assert result["cuts_designed"] is False


def test_none_truth_match_is_skipped():
    config = load_config()
    result = classify_rejection_reason(
        {
            "n_long_tracks": 0,
            "n_segments": 2,
            "tracklet_stations": [0, None],
            "tracklet_n_hit": [3, None],
            "tracklet_truth_ids": [None, -1],
            "tracklet_truth_match": [None, 0.2],
            "tracklet_chi2_per_ndof": [],
            "ckf_chi2_per_ndof": None,
        },
        config=config,
    )
    assert result["primary_reason"] == "missing_station"
    assert result["flags"]["no_truth_match"] is True


def test_other_when_descriptive_floors_are_met():
    config = load_config()
    result = classify_rejection_reason(
        {
            "n_long_tracks": 0,
            "n_segments": 4,
            "tracklet_stations": [0, 1, 2, 3],
            "tracklet_n_hit": [5, 5, 5, 5],
            "tracklet_truth_ids": [10001, 10001, 10001, 10001],
            "tracklet_truth_match": [1.0, 1.0, 1.0, 1.0],
            "tracklet_chi2_per_ndof": [1.0],
            "ckf_chi2_per_ndof": None,
        },
        config=config,
    )
    assert result["primary_reason"] == "other"
    assert is_long_track_accepted({"n_long_tracks": 0}) is False
    assert is_long_track_accepted({"n_long_tracks": 1}) is True


def _chi2(mean: float, median: float) -> dict[str, float]:
    return {"mean": mean, "median": median, "count": 10, "finite_count": 10}


def _reproduction(*, remaining_frozen: int, remaining_qop: int, remaining_wrong_p: int = 0):
    return {
        "closure_pass": False,
        "filter_implemented": False,
        "raw_official": {"chi2_per_ndof": _chi2(33.0, 0.15)},
        "long_track_equivalent_diagnostic": {
            "chi2_per_ndof": _chi2(4.0, 0.12),
            "n_remaining_frozen_1pct": remaining_frozen,
            "n_remaining_qoverp_pull_anomaly": remaining_qop,
            "n_remaining_wrong_momentum": remaining_wrong_p,
        },
    }


def test_decide_case_input_scope_mismatch():
    config = load_config()
    result = decide_case(
        {
            "filter_implemented": False,
            "ckf_not_long_fraction": 0.13,
            "n_unique_ckf_tracks": 100,
            "n_accepted_long_tracks": 87,
            "n_rejected_ckf_tracks": 13,
        },
        _reproduction(remaining_frozen=9, remaining_qop=3),
        {"n_frozen_1pct": 24, "n_not_a_long_track": 15},
        config,
    )
    assert result["primary_case"] == CASE_A
    assert CASE_B in result["secondary_cases"]
    assert result["next_step"] == "transport_covariance_v3_after_input_scope"
    assert result["closure_pass"] is False
    assert result["closure_claimed_after_subset"] is False
    assert result["input_filter_implemented"] is False
    assert result["measurement_model_v2_entered"] is False
    assert DECISION_AUDITED == "ckf_reconstruction_contract_audited"


def test_decide_case_fitting_quality_without_scope_mismatch():
    config = load_config()
    result = decide_case(
        {
            "filter_implemented": False,
            "ckf_not_long_fraction": 0.01,
            "n_unique_ckf_tracks": 100,
            "n_accepted_long_tracks": 99,
            "n_rejected_ckf_tracks": 1,
        },
        _reproduction(remaining_frozen=8, remaining_qop=4),
        {"n_frozen_1pct": 24, "n_not_a_long_track": 2},
        config,
    )
    assert result["primary_case"] == CASE_B
    assert result["next_step"] == "ckf_fitting_quality_audit"


def test_decide_case_refuses_subset_pass():
    config = load_config()
    with pytest.raises(CkfContractError, match="closure PASS"):
        decide_case(
            {"filter_implemented": False, "ckf_not_long_fraction": 0.2},
            {**_reproduction(remaining_frozen=0, remaining_qop=0), "closure_pass": True},
            {"n_frozen_1pct": 24, "n_not_a_long_track": 15},
            config,
        )
    assert CASE_C  # keep import live
