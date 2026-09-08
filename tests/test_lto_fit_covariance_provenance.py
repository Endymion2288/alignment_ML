"""Task B14R provenance audit.  No V2, no B15, no seed-scale shopping."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.lto_fit_covariance_provenance import (
    CASE_A,
    CASE_B,
    CASE_BLOCKED,
    CASE_E,
    _bound_gev_to_mev,
    LtoProvenanceError,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_b15,
    refuse_covariance_rescale,
    refuse_focus_drop,
    refuse_measurement_model_v2,
    refuse_raw_ckf,
    refuse_seed_scale_choice,
    refuse_tighten_contract,
    refuse_truth_qoverp,
)


def test_bound_gev_to_mev_matches_helper_qoverp_conversion():
    import numpy as np

    seed = np.diag([1.0e4, 1.0e4, 2.5e-3, 2.5e-3, 1.0])
    converted = _bound_gev_to_mev(seed)
    assert converted is not None
    assert converted[0, 0] == 1.0e4
    assert abs(converted[4, 4] - 1.0e-6) < 1.0e-18
    assert abs(float(np.sqrt(converted[4, 4])) - 1.0e-3) < 1.0e-12


def test_config_inherits_wb110_fail_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_choose_seed_scale_from_closure"] is True
    assert config["seed_sensitivity"]["scales"] == [0.1, 1.0, 10.0]
    assert inherited["workbook_110"]["decision"] == (
        "lto_input_covariance_shape_not_validated"
    )
    assert inherited["workbook_110"]["b15_authorized"] is False
    assert inherited["workbook_109"]["decision"] == (
        "leave_target_out_state_materialization_established"
    )
    assert "lto_fit_covariance_provenance_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(LtoProvenanceError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(LtoProvenanceError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(LtoProvenanceError, match="seed scale"):
        refuse_seed_scale_choice()
    with pytest.raises(LtoProvenanceError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(LtoProvenanceError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(LtoProvenanceError, match="B15"):
        refuse_b15()
    with pytest.raises(LtoProvenanceError, match="raw CKF"):
        refuse_raw_ckf()
    with pytest.raises(LtoProvenanceError, match="tightened"):
        refuse_tighten_contract()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_blocked_when_smoke_absent():
    result = decide_case(
        {
            "seed_sensitivity": {"present": False},
            "provenance": {},
            "information": {},
            "pulls": {},
            "focus": {"retained": True},
            "denominator": {"frozen_denominator_holds": True},
        }
    )
    assert result["decision"] == CASE_BLOCKED
    assert result["verdict"] == "BLOCKED"
    assert result["b15_authorized"] is False


def test_case_a_seed_or_predicted():
    result = decide_case(
        {
            "seed_sensitivity": {
                "present": True,
                "n_compared": 6,
                "typical_qoverp_seed_sensitive": False,
            },
            "provenance": {"is_seed_or_predicted": True},
            "information": {"median_n_fit_over_used": 0.8},
            "pulls": {"all": {}, "top1_chi2_share": {}},
            "focus": {"retained": True},
            "denominator": {"frozen_denominator_holds": True},
            "thresholds": {"n_fit_over_used_insufficient_max": 0.4},
        }
    )
    assert result["decision"] == CASE_A
    assert result["b15_authorized"] is False


def test_case_b_information_insufficient():
    result = decide_case(
        {
            "seed_sensitivity": {
                "present": True,
                "n_compared": 6,
                "typical_qoverp_seed_sensitive": True,
            },
            "provenance": {"is_seed_or_predicted": False},
            "information": {"median_n_fit_over_used": 0.8},
            "pulls": {
                "all": {"y": {"p68": 0.5}},
                "top1_chi2_share": {"y": 0.2},
            },
            "focus": {"retained": True},
            "denominator": {"frozen_denominator_holds": True},
            "thresholds": {"n_fit_over_used_insufficient_max": 0.4},
        }
    )
    assert result["decision"] == CASE_B


def test_mixed_when_several_flags():
    result = decide_case(
        {
            "seed_sensitivity": {
                "present": True,
                "n_compared": 6,
                "typical_qoverp_seed_sensitive": True,
            },
            "provenance": {"is_seed_or_predicted": False},
            "information": {"median_n_fit_over_used": 0.8},
            "pulls": {
                "all": {
                    "x": {"p68": 3.0},
                    "y": {"p68": 0.4},
                    "tx": {"p68": 3.0},
                    "ty": {"p68": 3.0},
                    "q_over_p": {"p68": 3.0},
                },
                "top1_chi2_share": {"y": 0.8},
            },
            "focus": {"retained": True},
            "denominator": {"frozen_denominator_holds": True},
            "thresholds": {"n_fit_over_used_insufficient_max": 0.4},
        }
    )
    assert result["decision"] == CASE_E
    assert result["active_mechanisms"] == ["B", "C", "D"]
    assert result["b15_authorized"] is False
    assert result["transport_covariance_validated"] is False
