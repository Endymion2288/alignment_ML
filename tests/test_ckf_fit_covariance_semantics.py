"""Task B11 CKF covariance semantics.  No Cin repair, no V2."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.ckf_fit_covariance_semantics import (
    DECISION,
    PRIMARY,
    CovarianceSemanticsError,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_cin_repair,
    refuse_focus_drop,
    refuse_measurement_model_v2,
    refuse_raw_ckf,
    refuse_truth_qoverp,
)


def test_config_inherits_wb105_and_wb103_eligibility():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_rescale_cin"] is True
    assert config["do_not_clip_eigenvalues"] is True
    assert config["do_not_rediagonalize_cin"] is True
    assert config["do_not_empirically_calibrate_cin"] is True
    assert inherited["workbook_105"]["decision"] == "mixed_or_inconclusive"
    assert inherited["workbook_105"]["decision_sha256"].startswith("e29fd93b")
    assert "ckf_fit_covariance_semantics_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(CovarianceSemanticsError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(CovarianceSemanticsError, match="rescaled"):
        refuse_cin_repair()
    with pytest.raises(CovarianceSemanticsError, match="raw CKF"):
        refuse_raw_ckf()
    with pytest.raises(CovarianceSemanticsError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(CovarianceSemanticsError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_decide_case_does_not_repair_cin():
    result = decide_case(
        {
            "source_cin_shape_holds": False,
            "overcoverage_present_before_propagation": True,
            "cin_rescaled": False,
        }
    )
    assert result["decision"] == DECISION
    assert result["primary_case"] == PRIMARY
    assert result["cin_modified"] is False
    assert result["suitable_independent_propagation_seed"] is False
