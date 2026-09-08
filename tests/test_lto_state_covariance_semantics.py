"""Task B14 LTO Cin semantics.  No V2, no B15, no Cin repair."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.lto_state_covariance_semantics import (
    DECISION_BLOCKED,
    DECISION_ESTABLISHED,
    DECISION_MIXED,
    DECISION_NOT_VALIDATED,
    LtoSemanticsError,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_b15,
    refuse_covariance_rescale,
    refuse_empirical_cross_covariance,
    refuse_focus_drop,
    refuse_measurement_model_v2,
    refuse_q_psd_projection,
    refuse_raw_ckf,
    refuse_tighten_contract,
    refuse_truth_qoverp,
)


def test_config_inherits_wb109_and_wb103_eligibility():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["official_input_scope"] == "contract_eligible"
    assert config["do_not_enter_b15_without_lto_cin_contract"] is True
    assert config["do_not_claim_transport_covariance_validated"] is True
    assert config["do_not_rescale_or_clip_lto_cin"] is True
    assert inherited["workbook_109"]["decision"] == (
        "leave_target_out_state_materialization_established"
    )
    assert inherited["workbook_109"]["b14_authorized"] is True
    assert inherited["workbook_108"]["primary_case"] == "leave_target_out_arms_unavailable"
    assert inherited["workbook_107"]["primary_case"] == (
        "official_cin_is_global_kf_refit_front_state"
    )
    assert inherited["workbook_105"]["decision"] == "mixed_or_inconclusive"
    assert "lto_state_covariance_semantics_v1" in config["output_root"]
    assert config["lto_dump_root"] != config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(LtoSemanticsError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(LtoSemanticsError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(LtoSemanticsError, match="PSD-projected"):
        refuse_q_psd_projection()
    with pytest.raises(LtoSemanticsError, match="Cov"):
        refuse_empirical_cross_covariance()
    with pytest.raises(LtoSemanticsError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(LtoSemanticsError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(LtoSemanticsError, match="B15"):
        refuse_b15()
    with pytest.raises(LtoSemanticsError, match="raw CKF"):
        refuse_raw_ckf()
    with pytest.raises(LtoSemanticsError, match="tightened"):
        refuse_tighten_contract()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_blocked_when_dumps_absent():
    result = decide_case(
        {
            "dumps": {"dumps_present": False},
            "lto_shape": {},
            "comparison": {},
            "focus": {"retained": True},
            "denominator": {"frozen_denominator_holds": True},
        }
    )
    assert result["decision"] == DECISION_BLOCKED
    assert result["verdict"] == "BLOCKED"
    assert result["b15_authorized"] is False
    assert result["transport_covariance_validated"] is False
    assert result["measurement_model_v2_authorized"] is False


def test_pass_only_when_both_splits_hold():
    result = decide_case(
        {
            "dumps": {"dumps_present": True},
            "lto_shape": {
                "all": {"shape_holds": True, "overwide": False, "undercovered": False},
                "construction": {"shape_holds": True},
                "validation": {"shape_holds": True},
            },
            "comparison": {
                "official_WB107_Cin_reused": False,
                "overcoverage_disappeared": True,
                "overcoverage_materially_reduced": True,
                "lto_still_overwide": False,
                "lto_still_undercovered": False,
            },
            "focus": {"retained": True, "momentum_anomaly_remains": True},
            "denominator": {"frozen_denominator_holds": True},
        }
    )
    assert result["decision"] == DECISION_ESTABLISHED
    assert result["verdict"] == "PASS"
    assert result["b15_authorized"] is True
    assert result["transport_covariance_validated"] is False
    assert result["measurement_model_v2_authorized"] is False
    assert result["focus_momentum_anomaly_remains"] is True


def test_overwide_lto_does_not_authorize_b15():
    result = decide_case(
        {
            "dumps": {"dumps_present": True},
            "lto_shape": {
                "all": {"shape_holds": False, "overwide": True, "undercovered": False},
                "construction": {"shape_holds": False},
                "validation": {"shape_holds": False},
            },
            "comparison": {
                "official_WB107_Cin_reused": False,
                "overcoverage_disappeared": False,
                "overcoverage_materially_reduced": True,
                "lto_still_overwide": True,
                "lto_still_undercovered": False,
            },
            "focus": {"retained": True, "momentum_anomaly_remains": True},
            "denominator": {"frozen_denominator_holds": True},
        }
    )
    assert result["decision"] == DECISION_NOT_VALIDATED
    assert result["verdict"] == "FAIL"
    assert result["b15_authorized"] is False
    assert result["lto_still_overwide"] is True


def test_one_split_only_is_mixed():
    result = decide_case(
        {
            "dumps": {"dumps_present": True},
            "lto_shape": {
                "all": {"shape_holds": True, "overwide": False, "undercovered": False},
                "construction": {"shape_holds": True},
                "validation": {"shape_holds": False},
            },
            "comparison": {
                "official_WB107_Cin_reused": False,
                "overcoverage_disappeared": True,
                "overcoverage_materially_reduced": True,
                "lto_still_overwide": False,
                "lto_still_undercovered": False,
            },
            "focus": {"retained": True, "momentum_anomaly_remains": False},
            "denominator": {"frozen_denominator_holds": True},
        }
    )
    assert result["decision"] == DECISION_MIXED
    assert result["verdict"] == "MIXED"
    assert result["b15_authorized"] is False
