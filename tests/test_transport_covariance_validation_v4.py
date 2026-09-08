"""Task B12 Transport Covariance V4.  Four arms, no V2, no gate retune."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.transport_covariance_validation_v4 import (
    CASE_CKF_REPAIR,
    CASE_LTO_BLOCKED,
    CASE_VALIDATED,
    TransportV4Error,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_empirical_cross_covariance,
    refuse_focus_drop,
    refuse_gate_retune,
    refuse_measurement_model_v2,
)


def test_config_inherits_b10_b11_and_wb103_eligibility():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert [item["id"] for item in config["pre_registered_arms"]] == [
        "A_full_c0",
        "B_full_c1",
        "C_lto_c0",
        "D_lto_c1",
    ]
    assert inherited["workbook_106"]["decision"] == (
        "leave_target_out_prediction_contract_not_established"
    )
    assert inherited["workbook_107"]["primary_case"] == (
        "official_cin_is_global_kf_refit_front_state"
    )
    assert inherited["workbook_105"]["decision"] == "mixed_or_inconclusive"
    assert "transport_covariance_validation_v4" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(TransportV4Error, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(TransportV4Error, match="frozen gates"):
        refuse_gate_retune()
    with pytest.raises(TransportV4Error, match="Cov"):
        refuse_empirical_cross_covariance()
    with pytest.raises(TransportV4Error, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_lto_blocked_when_arms_c_d_unavailable():
    inherited = {
        "workbook_106": {"decision": CASE_LTO_BLOCKED},
        "workbook_107": {"primary_case": "official_cin_is_global_kf_refit_front_state"},
    }
    arms = {
        "C_lto_c0": {"status": "blocked"},
        "D_lto_c1": {"status": "blocked"},
        "pre_propagation_cin": {"source_cin_shape_holds": False},
    }
    result = decide_case(arms, inherited)
    assert result["decision"] == CASE_LTO_BLOCKED
    assert result["measurement_model_v2_authorized"] is False
    assert result["lto_arms_evaluated"] is False


def test_validated_requires_lto_and_cin():
    inherited = {
        "workbook_106": {"decision": "leave_target_out_prediction_contract_established"},
        "workbook_107": {"primary_case": "official_cin_is_global_kf_refit_front_state"},
    }
    arms = {
        "C_lto_c0": {"status": "evaluated", "shape_holds": True},
        "D_lto_c1": {"status": "evaluated", "shape_holds": True},
        "pre_propagation_cin": {"source_cin_shape_holds": True},
    }
    result = decide_case(arms, inherited)
    assert result["decision"] == CASE_VALIDATED
    assert result["measurement_model_v2_authorized"] is True


def test_lto_pass_cin_fail_is_ckf_repair():
    inherited = {
        "workbook_106": {"decision": "leave_target_out_prediction_contract_established"},
        "workbook_107": {"primary_case": "official_cin_is_global_kf_refit_front_state"},
    }
    arms = {
        "C_lto_c0": {"status": "evaluated", "shape_holds": True},
        "D_lto_c1": {"status": "evaluated", "shape_holds": True},
        "pre_propagation_cin": {"source_cin_shape_holds": False},
    }
    result = decide_case(arms, inherited)
    assert result["decision"] == CASE_CKF_REPAIR
    assert result["measurement_model_v2_authorized"] is False
