"""Task B14S supported-state contract.  No V2, no B15, no prior introduction."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.lto_supported_state_contract import (
    CASE_A,
    CASE_B,
    CASE_BLOCKED,
    CASE_C,
    CASE_E,
    CLASS_MEASUREMENT,
    CLASS_PRIOR,
    CLASS_WEAK,
    LtoSupportedStateError,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_b15,
    refuse_covariance_rescale,
    refuse_focus_drop,
    refuse_full_track_prior,
    refuse_measurement_model_v2,
    refuse_prior_introduction,
    refuse_qoverp_deletion,
    refuse_seed_scale_choice,
    refuse_truth_qoverp,
)


def test_config_inherits_wb111_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14p"] is True
    assert config["do_not_introduce_target_independent_prior"] is True
    assert config["do_not_delete_qoverp"] is True
    assert config["do_not_choose_seed_scale_from_closure"] is True
    assert config["seed_sensitivity"]["scales"] == [0.1, 1.0, 10.0]
    assert config["seed_sensitivity"]["directions"] == ["x", "y", "tx", "ty", "q_over_p"]
    assert inherited["workbook_111"]["decision"] == "mixed_or_inconclusive"
    assert inherited["workbook_111"]["b15_authorized"] is False
    assert inherited["workbook_110"]["decision"] == "lto_input_covariance_shape_not_validated"
    assert inherited["workbook_109"]["decision"] == (
        "leave_target_out_state_materialization_established"
    )
    assert "lto_supported_state_contract_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(LtoSupportedStateError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(LtoSupportedStateError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(LtoSupportedStateError, match="seed scale"):
        refuse_seed_scale_choice()
    with pytest.raises(LtoSupportedStateError, match="does not introduce"):
        refuse_prior_introduction()
    with pytest.raises(LtoSupportedStateError, match="nuisance"):
        refuse_qoverp_deletion()
    with pytest.raises(LtoSupportedStateError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(LtoSupportedStateError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(LtoSupportedStateError, match="B15"):
        refuse_b15()
    with pytest.raises(LtoSupportedStateError, match="WB107"):
        refuse_full_track_prior()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_blocked_when_directional_smoke_absent():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "directional": {"present": False},
            "contract": {},
            "catastrophic": {},
        }
    )
    assert result["decision"] == CASE_BLOCKED
    assert result["verdict"] == "BLOCKED"
    assert result["b15_authorized"] is False
    assert result["b14p_authorized"] is False
    assert result["prior_introduced"] is False


def test_case_a_all_measurement_dominated():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "directional": {"present": True},
            "contract": {
                "typical_direction_class": {
                    name: CLASS_MEASUREMENT for name in ("x", "y", "tx", "ty", "q_over_p")
                }
            },
            "catastrophic": {"qoverp_gain_near_zero_coincides_with_seedlike": True},
        }
    )
    assert result["decision"] == CASE_A
    assert result["b14p_authorized"] is False
    assert result["b15_authorized"] is False


def test_case_b_weak_qoverp_nuisance():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "directional": {"present": True},
            "contract": {
                "typical_direction_class": {
                    "x": CLASS_MEASUREMENT,
                    "y": CLASS_MEASUREMENT,
                    "tx": CLASS_WEAK,
                    "ty": CLASS_WEAK,
                    "q_over_p": CLASS_PRIOR,
                }
            },
            "catastrophic": {"qoverp_gain_near_zero_coincides_with_seedlike": True},
        }
    )
    assert result["decision"] == CASE_B
    assert result["b14p_authorized"] is True
    assert result["b15_authorized"] is False
    assert result["next_step"] == "b14p_target_independent_weak_parameter_prior_contract"


def test_case_c_no_measurement_supported_direction():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "directional": {"present": True},
            "contract": {
                "typical_direction_class": {
                    name: CLASS_PRIOR for name in ("x", "y", "tx", "ty", "q_over_p")
                }
            },
            "catastrophic": {"qoverp_gain_near_zero_coincides_with_seedlike": True},
        }
    )
    assert result["decision"] == CASE_C
    assert result["b14p_authorized"] is False


def test_all_weakly_measured_is_case_b():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "directional": {"present": True},
            "contract": {
                "typical_direction_class": {
                    name: CLASS_WEAK for name in ("x", "y", "tx", "ty", "q_over_p")
                }
            },
            "catastrophic": {"qoverp_gain_near_zero_coincides_with_seedlike": True},
        }
    )
    assert result["decision"] == CASE_B
    assert result["b14p_authorized"] is True
    assert result["b15_authorized"] is False


def test_mixed_when_supported_body_and_unexplained_tail():
    result = decide_case(
        {
            "seed_scale_selected": False,
            "prior_introduced": False,
            "denominator": {"frozen_denominator_holds": True},
            "directional": {"present": True},
            "contract": {
                "typical_direction_class": {
                    "x": CLASS_MEASUREMENT,
                    "y": CLASS_MEASUREMENT,
                    "tx": CLASS_MEASUREMENT,
                    "ty": CLASS_MEASUREMENT,
                    "q_over_p": CLASS_WEAK,
                }
            },
            "catastrophic": {"qoverp_gain_near_zero_coincides_with_seedlike": False},
        }
    )
    assert result["decision"] == CASE_E
    assert result["b15_authorized"] is False
