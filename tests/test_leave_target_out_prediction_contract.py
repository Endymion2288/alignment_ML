"""Task B10 leave-target-out contract.  No invented cross-covariance, no V2."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.leave_target_out_prediction_contract import (
    DECISION_ESTABLISHED,
    DECISION_NOT_ESTABLISHED,
    LeaveTargetOutError,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_empirical_cross_covariance,
    refuse_focus_drop,
    refuse_kalmanfitter_as_lto,
    refuse_measurement_model_v2,
    refuse_raw_ckf,
    refuse_tighten_contract,
    refuse_truth_qoverp,
)


def test_config_inherits_wb105_and_wb103_eligibility():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["official_input_scope"] == "contract_eligible"
    assert config["do_not_invent_empirical_cross_covariance"] is True
    assert config["do_not_use_kalmanfittertool_fit_as_leave_target_out"] is True
    assert inherited["workbook_105"]["decision"] == "mixed_or_inconclusive"
    assert inherited["workbook_104"]["decision"] == "transport_covariance_shape_not_validated"
    assert inherited["workbook_105"]["decision_sha256"].startswith("e29fd93b")
    assert "leave_target_out_prediction_contract_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(LeaveTargetOutError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(LeaveTargetOutError, match="Cov"):
        refuse_empirical_cross_covariance()
    with pytest.raises(LeaveTargetOutError, match="KalmanFitterTool"):
        refuse_kalmanfitter_as_lto()
    with pytest.raises(LeaveTargetOutError, match="raw CKF"):
        refuse_raw_ckf()
    with pytest.raises(LeaveTargetOutError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(LeaveTargetOutError, match="tightened"):
        refuse_tighten_contract()
    with pytest.raises(LeaveTargetOutError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_decide_case_not_established_without_lto_states():
    result = decide_case(
        {"station_level_leave_target_out_export_exists": False},
        {"lto_states_materialized": False},
        {"independence_proven": False, "empirical_cross_covariance_invented": False},
        {"n_lto_states": 0},
    )
    assert result["decision"] == DECISION_NOT_ESTABLISHED
    assert result["verdict"] == "NOT_ESTABLISHED"


def test_decide_case_established_only_when_independence_proven():
    result = decide_case(
        {"station_level_leave_target_out_export_exists": True},
        {"lto_states_materialized": True},
        {"independence_proven": True, "empirical_cross_covariance_invented": False},
        {"n_lto_states": 10},
    )
    assert result["decision"] == DECISION_ESTABLISHED
