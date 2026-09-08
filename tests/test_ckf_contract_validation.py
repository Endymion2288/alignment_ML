"""Task B7 reconstruction contract validation.  No C/Q tune, no filter-PASS claim."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import (
    CASE_A,
    CASE_B,
    CASE_C,
    DECISION_VALIDATED,
    SAMPLE_A,
    SAMPLE_B,
    CkfValidationError,
    decide_case,
    dump_path_for_source,
    evaluate_eligibility,
    inherit_frozen_stage,
    load_config,
    refuse_closure_designed_selection,
    refuse_covariance_model_fixed,
    refuse_covariance_rescale,
    refuse_filter_pass_claim,
    refuse_measurement_model_v2,
    refuse_outlier_rejection,
    refuse_process_noise_tuning,
    refuse_truth_qoverp,
)


def _row(**overrides):
    payload = {
        "source_id": "mc24_100043_00200_00299",
        "run_id": 100043,
        "event_id": 1,
        "track_index": 0,
        "source_station": 0,
        "target_station": 1,
        "q_over_p_per_mev": -2.0e-6,
        "p_mev": 5.0e5,
        "input_covariance": [[1.0 if i == j else 0.0 for j in range(5)] for i in range(5)],
        "derived_state": [0.0, 0.0, 0.001, 0.001, -2.0e-6],
        "surface": {"type": "plane", "z_mm": 47.4, "origin_mm": [0.0, 0.0, 47.4]},
    }
    payload.update(overrides)
    return payload


def _reco(**overrides):
    payload = {
        "n_long_tracks": 1,
        "n_stations_present": 4,
        "n_segments": 4,
        "ckf_chi2_per_ndof": 12.0,
        "tracklet_truth_ids": [-1],
        "tracklet_truth_match": [0.1],
        "tracklet_stations": [0, 1, 2, 3],
        "in_station": [0, 1, 1, 1],
        "n_stations_present": 3,
    }
    payload.update(overrides)
    return payload


def test_config_inherits_frozen_wb102():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert inherited["workbook_98"]["situation"] == "acts_q_materialized_closure_failed"
    assert inherited["workbook_101"]["decision"] == "ckf_tail_provenance_audited"
    assert inherited["workbook_102"]["decision"] == "ckf_reconstruction_contract_audited"
    assert inherited["workbook_102"]["primary_case"] == "reconstruction_input_contract_missing"
    assert inherited["workbook_102"]["decision_sha256"].startswith("5c593660")
    assert config["eligibility_independent_of_closure"] is True
    assert config["do_not_use_chi2_in_eligibility"] is True
    assert config["do_not_use_truth_in_eligibility"] is True
    assert config["do_not_claim_filter_pass"] is True
    assert config["closure_gates"]["whitened_chi2_per_ndof_max"] == 4.0
    assert "ckf_contract_validation_v1" in config["output_root"]


def test_dump_path_stays_on_wb98():
    path = dump_path_for_source(load_config(), "mc24_100043_00200_00299")
    assert path.as_posix().endswith(
        "outputs/acts_transport_dump_v1/dumps/mc24_100043_00200_00299/ckf_acts_transport.jsonl"
    )


def test_forbidden_repairs():
    with pytest.raises(CkfValidationError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(CkfValidationError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(CkfValidationError, match="process noise"):
        refuse_process_noise_tuning()
    with pytest.raises(CkfValidationError, match="outlier"):
        refuse_outlier_rejection()
    with pytest.raises(CkfValidationError, match="designed from closure"):
        refuse_closure_designed_selection()
    with pytest.raises(CkfValidationError, match="filter-then-PASS"):
        refuse_filter_pass_claim()
    with pytest.raises(CkfValidationError, match="covariance model fixed"):
        refuse_covariance_model_fixed()
    with pytest.raises(CkfValidationError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_eligibility_ignores_chi2_pull_and_truth():
    config = load_config()
    result = evaluate_eligibility(_row(), _reco(), config)
    assert result["eligible"] is True
    assert result["used_chi2"] is False
    assert result["used_pull"] is False
    assert result["used_residual"] is False
    assert result["used_truth"] is False
    assert result["ineligible_reason"] is None


def test_eligibility_requires_long_track_and_stations():
    config = load_config()
    missing_long = evaluate_eligibility(_row(), _reco(n_long_tracks=0, n_stations_present=None), config)
    assert missing_long["eligible"] is False
    assert missing_long["ineligible_reason"] == "not_long_track_equivalent"
    missing_station = evaluate_eligibility(
        _row(),
        _reco(n_stations_present=2, tracklet_stations=[0, 1], in_station=[1, 1, 0, 0]),
        config,
    )
    assert missing_station["ineligible_reason"] == "incomplete_stations"
    instation0_off = evaluate_eligibility(
        _row(),
        _reco(n_stations_present=3, tracklet_stations=[0, 1, 2, 3], in_station=[0, 1, 1, 1]),
        config,
    )
    assert instation0_off["eligible"] is True
    dummy = evaluate_eligibility(_row(q_over_p_per_mev=1.0e-5), _reco(), config)
    assert dummy["ineligible_reason"] == "invalid_q_over_p"


def _chi2(mean: float, median: float) -> dict[str, float]:
    return {"mean": mean, "median": median, "count": 10, "finite_count": 10}


def _comparison(raw_mean: float, contract_mean: float, contract_median: float) -> dict:
    return {
        "samples": {
            SAMPLE_A: {"matched": {"chi2_per_ndof": _chi2(raw_mean, 0.15)}},
            SAMPLE_B: {"matched": {"chi2_per_ndof": _chi2(contract_mean, contract_median)}},
        }
    }


def test_decide_case_scope_confirmed_with_remaining_fit_tail():
    config = load_config()
    result = decide_case(
        _comparison(33.0, 24.0, 0.12),
        {"all_calibrated": False},
        {
            "all_calibrated": False,
            "filter_then_pass_claimed": False,
            "covariance_model_fixed": False,
        },
        {"n_contract_qoverp_pull_anomaly": 3, "n_frozen_1pct_remaining": 9},
        config,
    )
    assert result["primary_case"] == CASE_A
    assert CASE_B in result["secondary_cases"]
    assert result["closure_pass"] is False
    assert result["filter_then_pass_claimed"] is False
    assert result["covariance_model_fixed"] is False
    assert result["closure_under_contracted_input_scope"] is False
    assert result["next_step"] == "transport_covariance_v3"
    assert DECISION_VALIDATED == "ckf_reconstruction_contract_validated"


def test_decide_case_bulk_failure_is_transport_model():
    config = load_config()
    result = decide_case(
        _comparison(40.0, 38.0, 5.0),
        {"all_calibrated": False},
        {
            "all_calibrated": False,
            "filter_then_pass_claimed": False,
            "covariance_model_fixed": False,
        },
        {"n_contract_qoverp_pull_anomaly": 0, "n_frozen_1pct_remaining": 0},
        config,
    )
    assert result["primary_case"] == CASE_C
    assert CASE_A not in result["case_flags"] or result["case_flags"]["A"] is False


def test_decide_case_refuses_filter_pass_claim():
    config = load_config()
    with pytest.raises(CkfValidationError, match="filter-then-PASS"):
        decide_case(
            _comparison(33.0, 1.0, 0.2),
            {"all_calibrated": False},
            {
                "all_calibrated": True,
                "filter_then_pass_claimed": True,
                "covariance_model_fixed": False,
            },
            {"n_contract_qoverp_pull_anomaly": 0, "n_frozen_1pct_remaining": 0},
            config,
        )
    assert CASE_C  # keep import live
