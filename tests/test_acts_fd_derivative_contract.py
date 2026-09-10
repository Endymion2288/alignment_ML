"""Task B14L ACTS-vs-FD derivative contract.  No B15, no V2, no B14M."""

from __future__ import annotations

import pytest

from alignment.profiled_measurement_likelihood import PINV_RELATIVE
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.acts_fd_derivative_contract import (
    ACCURACY_TOLERANCES,
    CASE_ACTS_NOT_CONV,
    CASE_CHAIN_BROKEN,
    CASE_FD_UNRELIABLE,
    CASE_MIXED,
    FOCUS_EVENT,
    RUNG_FACTORS,
    ActsFdDerivativeContractError,
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
from datasets.source_to_measurement_map_smoothness import (
    CASE_MIXED as WB118_DECISION,
)


def test_config_inherits_wb118_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14k"] is True
    assert config["do_not_enter_b14l"] is False
    assert config["do_not_select_best_fd_step"] is True
    assert config["do_not_change_statistical_model"] is True
    assert float(config["profile_numerical"]["pinv_relative"]) == PINV_RELATIVE
    assert list(config["derivative_contract"]["rung_factors"]) == list(RUNG_FACTORS)
    tols = [item["step_tolerance"] for item in config["derivative_contract"]["accuracy_rungs"]]
    assert tols == list(ACCURACY_TOLERANCES)
    assert tuple(config["focus_event"]) == FOCUS_EVENT
    assert inherited["workbook_118"]["decision"] == WB118_DECISION
    assert inherited["workbook_118"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_118"]["jacobian_contract_established"] is False
    assert inherited["workbook_117"]["decision"] == "source_to_measurement_transport_not_smooth"
    assert "acts_fd_derivative_contract_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(ActsFdDerivativeContractError, match="prior"):
        refuse_prior()
    with pytest.raises(ActsFdDerivativeContractError, match="ridge"):
        refuse_ridge_information()
    with pytest.raises(ActsFdDerivativeContractError, match="measurement update"):
        refuse_measurement_update()
    with pytest.raises(ActsFdDerivativeContractError, match="best finite-difference"):
        refuse_best_step_selection()
    with pytest.raises(ActsFdDerivativeContractError, match="stepTolerance"):
        refuse_best_tolerance()
    with pytest.raises(ActsFdDerivativeContractError, match="direct-from-source"):
        refuse_switch_to_direct()
    with pytest.raises(ActsFdDerivativeContractError, match="official likelihood"):
        refuse_replace_likelihood()
    with pytest.raises(ActsFdDerivativeContractError, match="statistical model"):
        refuse_change_statistical_model()
    with pytest.raises(ActsFdDerivativeContractError, match="B14M"):
        refuse_b14m()
    with pytest.raises(ActsFdDerivativeContractError, match="restart"):
        refuse_restart()
    with pytest.raises(ActsFdDerivativeContractError, match="B15"):
        refuse_b15()
    with pytest.raises(ActsFdDerivativeContractError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(ActsFdDerivativeContractError, match="1989"):
        refuse_full_sample()
    with pytest.raises(ActsFdDerivativeContractError, match="5D Cin"):
        refuse_5d_cin_repair()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_contract_keeps_wb114_model():
    contract = likelihood_contract()
    assert contract["do_not_switch_to_direct"] is True
    assert contract["do_not_select_best_fd_step"] is True
    assert contract["do_not_replace_official_likelihood"] is True
    assert contract["measurement_update_in_evaluator"] is False
    assert contract["not_the_objective"] == "repair 5D Cin"
    assert contract["R_i"] == "(0.08 mm)^2 / 12"
    assert contract["wb118_not_a_physical_conclusion"] is True


def _control(**bits: object) -> dict:
    base = {
        "control_fd_all_converged": True,
        "control_projection_agrees_fd": True,
        "acts_end_loc0_matches_official": True,
        "chain_complete": True,
        "focus_acts_stable": True,
        "focus_fd_focus_converged": True,
        "branch_fallback": False,
        "comparisons": [],
    }
    base.update(bits)
    return base


def _focus(**bits: object) -> dict:
    base = {
        "control_fd_all_converged": False,
        "control_projection_agrees_fd": False,
        "acts_end_loc0_matches_official": True,
        "chain_complete": True,
        "focus_acts_stable": True,
        "focus_fd_focus_converged": False,
        "branch_fallback": False,
        "comparisons": [
            {"parameter": "loc1", "projection_vs_fd": {"agree": False}},
            {"parameter": "phi", "projection_vs_fd": {"agree": False}},
            {"parameter": "q_over_p", "projection_vs_fd": {"agree": False}},
        ],
    }
    base.update(bits)
    return base


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
    fd_unreliable = decide_case(base)
    assert fd_unreliable["decision"] == CASE_FD_UNRELIABLE
    assert fd_unreliable["jacobian_contract_established"] is True
    assert fd_unreliable["b14m_reopen_authorized"] is True
    assert fd_unreliable["restart_invariance_authorized"] is False
    assert fd_unreliable["full_sample_authorized"] is False
    broken = decide_case(
        {**base, "controls": [_control(control_projection_agrees_fd=False)] * 3}
    )
    assert broken["decision"] == CASE_CHAIN_BROKEN
    assert broken["b14m_reopen_authorized"] is False
    drifted = decide_case({**base, "focus_86": [_focus(focus_acts_stable=False)] * 3})
    assert drifted["decision"] == CASE_ACTS_NOT_CONV
    assert drifted["b14m_reopen_authorized"] is False
    empty = decide_case({**base, "smoke_present": False, "controls": [], "focus_86": []})
    assert empty["decision"] == CASE_MIXED
    assert empty["b14m_reopen_authorized"] is False


def test_smoke_gate_never_authorizes_full_sample():
    gate = smoke_gate(
        {
            "smoke_present": True,
            "targets": [
                {
                    "event": event,
                    "comparisons": [1],
                    "fd_columns": [1],
                    "accuracy": [1],
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
    assert gate["restart_invariance_authorized"] is False
    assert gate["full_sample_authorized"] is False
