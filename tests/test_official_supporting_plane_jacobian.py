"""Task B14U official supporting-plane Jacobian contract.  No B15, no V2, no B14M."""

from __future__ import annotations

import pytest

from alignment.profiled_measurement_likelihood import PINV_RELATIVE
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.official_supporting_plane_jacobian import (
    CASE_ESTABLISHED,
    CASE_INCONSISTENT,
    CASE_UNAVAILABLE,
    FOCUS_EVENT,
    RUNG_FACTORS,
    OfficialSupportingPlaneJacobianError,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    refuse_5d_cin_repair,
    refuse_b14m,
    refuse_b15,
    refuse_best_step_selection,
    refuse_best_tolerance,
    refuse_change_statistical_model,
    refuse_dummy_cov_bounded,
    refuse_full_sample,
    refuse_measurement_model_v2,
    refuse_measurement_update,
    refuse_prior,
    refuse_replace_likelihood,
    refuse_restart,
    refuse_ridge_information,
    refuse_switch_to_direct,
    smoke_gate,
)
from datasets.acts_fd_derivative_contract import CASE_CHAIN_BROKEN as WB119_DECISION


def test_config_inherits_wb119_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14l"] is True
    assert config["do_not_enter_b14u"] is False
    assert config["do_not_select_best_fd_step"] is True
    assert config["do_not_change_statistical_model"] is True
    assert float(config["profile_numerical"]["pinv_relative"]) == PINV_RELATIVE
    assert list(config["official_supporting_plane_jacobian"]["rung_factors"]) == list(
        RUNG_FACTORS
    )
    assert float(config["official_supporting_plane_jacobian"]["official_step_tolerance"]) == 1.0e-4
    assert tuple(config["focus_event"]) == FOCUS_EVENT
    assert inherited["workbook_119"]["decision"] == WB119_DECISION
    assert inherited["workbook_119"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_119"]["jacobian_contract_established"] is False
    assert inherited["workbook_119"]["took_wrong_jacobian"] is True
    assert inherited["workbook_118"]["decision"] == "mixed_or_inconclusive"
    assert "official_supporting_plane_jacobian_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="prior"):
        refuse_prior()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="ridge"):
        refuse_ridge_information()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="measurement update"):
        refuse_measurement_update()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="best finite-difference"):
        refuse_best_step_selection()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="stepTolerance"):
        refuse_best_tolerance()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="direct-from-source"):
        refuse_switch_to_direct()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="official likelihood"):
        refuse_replace_likelihood()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="dummy-cov"):
        refuse_dummy_cov_bounded()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="statistical model"):
        refuse_change_statistical_model()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="B14M"):
        refuse_b14m()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="restart"):
        refuse_restart()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="B15"):
        refuse_b15()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="1989"):
        refuse_full_sample()
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="5D Cin"):
        refuse_5d_cin_repair()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_contract_keeps_wb114_model():
    contract = likelihood_contract()
    assert contract["do_not_switch_to_direct"] is True
    assert contract["do_not_select_best_fd_step"] is True
    assert contract["do_not_replace_official_likelihood"] is True
    assert contract["do_not_use_dummy_cov_bounded_transportJacobian"] is True
    assert contract["measurement_update_in_evaluator"] is False
    assert contract["not_the_objective"] == "repair 5D Cin"
    assert contract["R_i"] == "(0.08 mm)^2 / 12"
    assert contract["wb119_took_wrong_jacobian"] is True


def _agree() -> dict:
    return {"agree": True, "relative_error": 0.01, "sign_consistent": True}


def _disagree() -> dict:
    return {"agree": False, "relative_error": 1.5, "sign_consistent": False}


def _control(**bits: object) -> dict:
    base = {
        "control_fd_all_converged": True,
        "control_rk_agrees_fd": True,
        "focus_fd_all_converged": True,
        "focus_rk_agrees_fd": True,
        "loc0_matches_official": True,
        "free_jacobian_available": True,
        "chain_complete": True,
        "branch_fallback": False,
        "dummy_cov_bounded_used": False,
        "comparisons": [{"parameter": name, "rk_free_vs_fd": _agree()} for name in
                        ("loc0", "loc1", "phi", "theta", "q_over_p")],
    }
    base.update(bits)
    return base


def _focus(**bits: object) -> dict:
    return _control(**bits)


def test_decide_case_paths():
    base = {
        "prior_introduced": False,
        "ridge_as_information": False,
        "statistical_model_changed": False,
        "switched_to_direct": False,
        "replaced_likelihood": False,
        "selected_best_step": False,
        "selected_best_tolerance": False,
        "restart_executed": False,
        "b14m_reopened": False,
        "smoke_present": True,
        "exclusion": {"target_exclusion_holds": True},
        "controls": [_control(), _control(), _control()],
        "focus_86": [_focus(), _focus(), _focus()],
    }
    established = decide_case(base)
    assert established["decision"] == CASE_ESTABLISHED
    assert established["jacobian_contract_established"] is True
    assert established["b14m_reopen_authorized"] is True
    assert established["restart_invariance_authorized"] is False
    assert established["full_sample_authorized"] is False
    missing = decide_case({**base, "controls": [_control(free_jacobian_available=False)] * 3})
    assert missing["decision"] == CASE_UNAVAILABLE
    assert missing["b14m_reopen_authorized"] is False
    broken = decide_case({**base, "controls": [_control(control_rk_agrees_fd=False)] * 3})
    assert broken["decision"] == CASE_INCONSISTENT
    assert broken["b14m_reopen_authorized"] is False
    focus_fail = decide_case({**base, "focus_86": [_focus(focus_rk_agrees_fd=False)] * 3})
    assert focus_fail["decision"] == CASE_INCONSISTENT
    empty = decide_case({**base, "smoke_present": False, "controls": [], "focus_86": []})
    assert empty["decision"] == CASE_UNAVAILABLE
    assert empty["b14m_reopen_authorized"] is False
    with pytest.raises(OfficialSupportingPlaneJacobianError, match="dummy-cov"):
        decide_case({**base, "targets": [{"dummy_cov_bounded_used": True}]})


def test_smoke_gate_never_authorizes_full_sample():
    gate = smoke_gate(
        {
            "smoke_present": True,
            "targets": [
                {
                    "event": event,
                    "comparisons": [1],
                    "fd_columns": [1],
                    "chain": {"present": True},
                    "dummy_cov_bounded_used": False,
                }
                for event in ("100043/0", "100043/1", "100043/37", "100048/86")
            ],
            "exclusion": {"target_exclusion_holds": True},
            "prior_introduced": False,
            "ridge_as_information": False,
            "statistical_model_changed": False,
        }
    )
    assert gate["passed"] is True
    assert gate["b14m_reopen_authorized"] is False
    assert gate["full_sample_authorized"] is False
