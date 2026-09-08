"""Task B14P prior admissibility.  No V2, no B15, no prior introduction."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.target_independent_weak_prior_contract import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    CASE_E,
    FROZEN_DIRECTION_CLASS,
    TargetIndependentPriorError,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_arbitrary_gaussian,
    refuse_b14m,
    refuse_b15,
    refuse_cin_injection,
    refuse_covariance_rescale,
    refuse_focus_drop,
    refuse_full_track_prior,
    refuse_measurement_model_v2,
    refuse_prior_introduction,
    refuse_qoverp_deletion,
    refuse_qoverp_fix,
    refuse_seed_scale_choice,
    refuse_truth_prior,
)


def test_config_inherits_wb112_case_b_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_introduce_target_independent_prior"] is True
    assert config["do_not_add_prior_into_wb109_cin"] is True
    assert config["do_not_delete_qoverp"] is True
    assert config["do_not_fix_qoverp"] is True
    assert config["weak_directions"] == ["y", "tx", "q_over_p"]
    assert config["measurement_supported_directions"] == ["x", "ty"]
    assert inherited["workbook_112"]["decision"] == (
        "reduced_measurement_supported_state_with_weak_nuisance"
    )
    assert inherited["workbook_112"]["typical_direction_class"] == FROZEN_DIRECTION_CLASS
    assert inherited["workbook_112"]["prior_introduced"] is False
    assert inherited["workbook_112"]["b14p_authorized"] is True
    assert inherited["workbook_112"]["b15_authorized"] is False
    assert inherited["workbook_111"]["decision"] == "mixed_or_inconclusive"
    assert inherited["workbook_110"]["decision"] == "lto_input_covariance_shape_not_validated"
    assert inherited["workbook_109"]["decision"] == (
        "leave_target_out_state_materialization_established"
    )
    assert "target_independent_weak_prior_contract_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(TargetIndependentPriorError, match="truth"):
        refuse_truth_prior()
    with pytest.raises(TargetIndependentPriorError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(TargetIndependentPriorError, match="seed scale"):
        refuse_seed_scale_choice()
    with pytest.raises(TargetIndependentPriorError, match="does not introduce"):
        refuse_prior_introduction()
    with pytest.raises(TargetIndependentPriorError, match="nuisance"):
        refuse_qoverp_deletion()
    with pytest.raises(TargetIndependentPriorError, match="fixed"):
        refuse_qoverp_fix()
    with pytest.raises(TargetIndependentPriorError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(TargetIndependentPriorError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(TargetIndependentPriorError, match="B15"):
        refuse_b15()
    with pytest.raises(TargetIndependentPriorError, match="B14M"):
        refuse_b14m()
    with pytest.raises(TargetIndependentPriorError, match="WB107"):
        refuse_full_track_prior()
    with pytest.raises(TargetIndependentPriorError, match="Gaussian"):
        refuse_arbitrary_gaussian()
    with pytest.raises(TargetIndependentPriorError, match="WB109 Cin"):
        refuse_cin_injection()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_case_a_all_weak_directions_admissible():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "admissible_by_weak_direction": {"y": True, "tx": True, "q_over_p": True},
            "independence_unresolved": False,
        }
    )
    assert result["decision"] == CASE_A
    assert result["prior_contract_established"] is True
    assert result["prior_introduced"] is False
    assert result["lto_cin_contract_established"] is False
    assert result["b15_authorized"] is False
    assert result["b14m_authorized"] is False


def test_case_b_no_admissible_prior():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "admissible_by_weak_direction": {"y": False, "tx": False, "q_over_p": False},
            "independence_unresolved": False,
        }
    )
    assert result["decision"] == CASE_B
    assert result["verdict"] == "FAIL"
    assert result["b14m_authorized"] is True
    assert result["b15_authorized"] is False
    assert result["prior_contract_established"] is False
    assert result["prior_introduced"] is False
    assert result["next_step"] == "b14m_profiled_marginalized_weak_nuisance_likelihood"
    assert result["do_not_force_5d_lto_covariance"] is True


def test_case_c_subset_only():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "admissible_by_weak_direction": {"y": True, "tx": False, "q_over_p": False},
            "independence_unresolved": False,
        }
    )
    assert result["decision"] == CASE_C
    assert result["b14m_authorized"] is False
    assert result["b15_authorized"] is False
    assert result["prior_contract_established"] is False


def test_case_d_independence_unresolved():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "admissible_by_weak_direction": {"y": True, "tx": False, "q_over_p": True},
            "independence_unresolved": True,
        }
    )
    assert result["decision"] == CASE_D
    assert result["b15_authorized"] is False


def test_mixed_when_flags_inconsistent():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "admissible_by_weak_direction": {},
            "independence_unresolved": False,
        }
    )
    assert result["decision"] == CASE_B
    assert result["b15_authorized"] is False
