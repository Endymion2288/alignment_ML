"""Task B13 independent LTO materialization.  No V2, no KalmanFitterTool.fit."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.leave_target_out_state_materialization import (
    DECISION_DUMPS_ABSENT,
    DECISION_MATERIALIZED,
    DECISION_NOT_ESTABLISHED,
    LeaveTargetOutMaterializationError,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_closure_selection,
    refuse_dummy_qoverp,
    refuse_empirical_cross_covariance,
    refuse_focus_drop,
    refuse_kalmanfitter_as_lto,
    refuse_measurement_model_v2,
    refuse_raw_ckf,
    refuse_tighten_contract,
    refuse_truth_qoverp,
)


def test_config_inherits_wb108_and_wb103_eligibility():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["official_input_scope"] == "contract_eligible"
    assert config["do_not_use_kalmanfittertool_fit_as_leave_target_out"] is True
    assert config["do_not_select_states_from_closure"] is True
    assert inherited["workbook_108"]["primary_case"] == (
        "leave_target_out_arms_unavailable"
    )
    assert inherited["workbook_107"]["primary_case"] == (
        "official_cin_is_global_kf_refit_front_state"
    )
    assert inherited["workbook_106"]["decision"] == (
        "leave_target_out_prediction_contract_not_established"
    )
    assert inherited["workbook_105"]["decision"] == "mixed_or_inconclusive"
    assert "leave_target_out_state_materialization_v1" in config["output_root"]
    assert config["lto_dump_root"] != config["dump_root"]


def test_forbidden_repairs():
    with pytest.raises(LeaveTargetOutMaterializationError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(LeaveTargetOutMaterializationError, match="dummy"):
        refuse_dummy_qoverp()
    with pytest.raises(LeaveTargetOutMaterializationError, match="Cov"):
        refuse_empirical_cross_covariance()
    with pytest.raises(LeaveTargetOutMaterializationError, match="KalmanFitterTool"):
        refuse_kalmanfitter_as_lto()
    with pytest.raises(LeaveTargetOutMaterializationError, match="raw CKF"):
        refuse_raw_ckf()
    with pytest.raises(LeaveTargetOutMaterializationError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(LeaveTargetOutMaterializationError, match="tightened"):
        refuse_tighten_contract()
    with pytest.raises(LeaveTargetOutMaterializationError, match="closure"):
        refuse_closure_selection()
    with pytest.raises(LeaveTargetOutMaterializationError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_blocked_when_dumps_absent():
    result = decide_case(
        {
            "dumps": {"dumps_present": False},
            "measurement_removal": {"target_exclusion_proven": False},
            "states": {"n_success": 0},
            "denominator": {"denominator_complete": False, "focus_identity_retained": True},
            "covariance": {"not_wb107_front_covariance": False},
        }
    )
    assert result["decision"] == DECISION_DUMPS_ABSENT
    assert result["verdict"] == "BLOCKED"
    assert result["b14_authorized"] is False
    assert result["measurement_model_v2_authorized"] is False
    assert result["independence_proven"] is False
    assert result["lto_states_materialized"] is False
    assert result["transport_covariance_validated"] is False


def test_pass_only_when_six_contracts_hold():
    result = decide_case(
        {
            "dumps": {"dumps_present": True},
            "measurement_removal": {"target_exclusion_proven": True},
            "states": {
                "n_success": 10,
                "successful_fits_propagatable": True,
                "qoverp_is_fitted": True,
                "covariance_origin_independent": True,
                "seed_covariance_used_as_output_covariance": False,
                "large_scale_fit_failure": False,
            },
            "denominator": {
                "denominator_complete": True,
                "frozen_denominator_holds": True,
                "focus_identity_retained": True,
                "focus_identity_in_full_dump": True,
            },
            "covariance": {
                "not_wb107_front_covariance": True,
                "official_WB107_Cin_reused": False,
            },
        }
    )
    assert result["decision"] == DECISION_MATERIALIZED
    assert result["decision"] == "leave_target_out_state_materialization_established"
    assert result["verdict"] == "PASS"
    assert result["independence_proven"] is True
    assert result["lto_states_materialized"] is True
    assert result["b14_authorized"] is True
    assert result["b15_authorized"] is False
    assert result["measurement_model_v2_authorized"] is False
    assert result["transport_covariance_validated"] is False


def test_exclusion_failure_is_not_pass():
    result = decide_case(
        {
            "dumps": {"dumps_present": True},
            "measurement_removal": {"target_exclusion_proven": False},
            "states": {
                "n_success": 10,
                "successful_fits_propagatable": True,
                "qoverp_is_fitted": True,
                "covariance_origin_independent": True,
            },
            "denominator": {
                "denominator_complete": True,
                "frozen_denominator_holds": True,
                "focus_identity_retained": True,
                "focus_identity_in_full_dump": True,
            },
            "covariance": {
                "not_wb107_front_covariance": True,
                "official_WB107_Cin_reused": False,
            },
        }
    )
    assert result["decision"] == DECISION_NOT_ESTABLISHED
    assert result["verdict"] == "FAIL"
    assert result["primary_case"] == "target_exclusion_not_proven"
    assert result["b14_authorized"] is False
    assert result["independence_proven"] is False
