"""Task B14J supporting-plane Jacobian continuity.  No B15, no V2, no B14M."""

from __future__ import annotations

import pytest

from alignment.profiled_measurement_likelihood import PINV_RELATIVE
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.profile_transport_contract import CASE_MIXED as WB116_DECISION
from datasets.supporting_plane_jacobian_continuity import (
    CASE_BRANCH,
    CASE_CONTINUATION,
    CASE_ESTABLISHED,
    CASE_FD_STEP,
    CASE_TRANSPORT,
    REL_MAX,
    RUNG_FACTORS,
    SupportingPlaneJacobianContinuityError,
    audit_column,
    classify_column_mechanism,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    refuse_5d_cin_repair,
    refuse_b14m,
    refuse_b15,
    refuse_best_step_selection,
    refuse_change_statistical_model,
    refuse_delete_focus,
    refuse_full_sample,
    refuse_measurement_model_v2,
    refuse_measurement_update,
    refuse_prior,
    refuse_restart_before_contract,
    refuse_rewrite_profile_math,
    refuse_ridge_information,
    refuse_switch_to_direct,
    smoke_gate,
)


def _rung(factor: float, ok: bool = True) -> dict:
    return {
        "step_factor": factor,
        "ok": ok,
        "plus": {
            "ok": ok,
            "destination_geometry_ids": [1, 2, 3],
            "continuation_state_construction": ["transform_free_to_bound"] * 3,
            "free_to_bound_fallback": [False, False, False],
            "inside_active_bounds": [True, True, False],
            "hops": [{"geometry_surface_sequence": [10, 20]}],
        },
        "minus": {
            "ok": ok,
            "destination_geometry_ids": [1, 2, 3],
            "continuation_state_construction": ["transform_free_to_bound"] * 3,
            "free_to_bound_fallback": [False, False, False],
            "inside_active_bounds": [True, True, False],
            "hops": [{"geometry_surface_sequence": [10, 20]}],
        },
    }


def _column(rels, *, branch: bool = False, sign: bool = True) -> dict:
    rungs = [_rung(factor) for factor in RUNG_FACTORS]
    if branch:
        rungs[0]["plus"]["destination_geometry_ids"] = [9, 9, 9]
    pairs = []
    for index, rel in enumerate(rels):
        pairs.append(
            {
                "from_rung": index,
                "to_rung": index + 1,
                "relative_error": rel,
                "sign_consistent": sign,
            }
        )
    return {
        "parameter": "loc1",
        "selected_best_step": False,
        "sign_consistency": sign,
        "failed_finite_difference_evaluations": 0,
        "rungs": rungs,
        "consecutive_relative_errors": pairs,
    }


def test_config_inherits_wb116_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14t"] is True
    assert config["do_not_enter_b14j"] is False
    assert config["do_not_select_best_fd_step"] is True
    assert config["do_not_switch_official_likelihood_to_direct"] is True
    assert config["do_not_change_statistical_model"] is True
    assert config["do_not_use_ridge"] is True
    assert config["do_not_force_5d_lto_covariance"] is True
    assert config["do_not_claim_5d_cin_repaired"] is True
    assert float(config["profile_numerical"]["pinv_relative"]) == PINV_RELATIVE
    assert list(config["jacobian_continuity"]["rung_factors"]) == [1.0, 0.5, 0.25, 0.125]
    assert inherited["workbook_116"]["decision"] == WB116_DECISION
    assert inherited["workbook_116"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_116"]["jacobian_contract_established"] is False
    assert inherited["workbook_115"]["decision"] == "profile_transport_contract_broken"
    assert inherited["workbook_114"]["synthetic_profile_passed"] is True
    assert inherited["workbook_109"]["decision"] == (
        "leave_target_out_state_materialization_established"
    )
    assert "supporting_plane_jacobian_continuity_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="prior"):
        refuse_prior()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="ridge"):
        refuse_ridge_information()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="measurement update"):
        refuse_measurement_update()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="best finite-difference"):
        refuse_best_step_selection()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="direct-from-source"):
        refuse_switch_to_direct()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="100043/37"):
        refuse_delete_focus()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="rewritten"):
        refuse_rewrite_profile_math()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="statistical model"):
        refuse_change_statistical_model()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="B14M"):
        refuse_b14m()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="B15"):
        refuse_b15()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="1989"):
        refuse_full_sample()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="5D Cin"):
        refuse_5d_cin_repair()
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="restart"):
        refuse_restart_before_contract()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_contract_keeps_sequential_official():
    contract = likelihood_contract()
    assert contract["do_not_switch_to_direct"] is True
    assert contract["do_not_select_best_fd_step"] is True
    assert contract["measurement_update_in_evaluator"] is False
    assert contract["not_the_objective"] == "repair 5D Cin"
    assert contract["R_i"] == "(0.08 mm)^2 / 12"


def test_audit_column_does_not_pick_best_step():
    with pytest.raises(SupportingPlaneJacobianContinuityError, match="best"):
        audit_column({"selected_best_step": True, "rungs": [], "consecutive_relative_errors": []})
    passing = audit_column(_column([0.04, 0.02, 0.01]))
    assert passing["ladder_converged"] is True
    assert passing["selected_best_step"] is False
    assert passing["last_pair_relative_error"] <= REL_MAX
    coarse = audit_column(_column([0.80, 0.40, 0.04]))
    assert coarse["ladder_converged"] is True
    assert coarse["official_fd_in_asymptotic_region"] is False
    assert coarse["relative_errors_monotone_decreasing"] is True


def test_classify_direct_pass_sequential_fail_is_continuation():
    sequential = {
        "ladder_converged": False,
        "surface_path_branch_consistent": True,
        "relative_errors_monotone_decreasing": False,
        "official_fd_in_asymptotic_region": False,
    }
    direct = {"ladder_converged": True, "surface_path_branch_consistent": True}
    assert classify_column_mechanism(sequential, direct) == CASE_CONTINUATION
    both = classify_column_mechanism(
        {**sequential, "relative_errors_monotone_decreasing": False},
        {**direct, "ladder_converged": False},
    )
    assert both == CASE_TRANSPORT
    branch = classify_column_mechanism(
        {**sequential, "surface_path_branch_consistent": False},
        direct,
    )
    assert branch == CASE_BRANCH
    step = classify_column_mechanism(
        {
            "ladder_converged": False,
            "surface_path_branch_consistent": True,
            "relative_errors_monotone_decreasing": True,
            "official_fd_in_asymptotic_region": False,
        },
        {"ladder_converged": False, "surface_path_branch_consistent": True},
    )
    assert step == CASE_FD_STEP


def test_decide_case_does_not_reopen_b14m_or_claim_cin():
    base = {
        "prior_introduced": False,
        "ridge_as_information": False,
        "statistical_model_changed": False,
        "switched_to_direct": False,
        "selected_best_step": False,
        "restart_executed": False,
        "smoke_present": True,
        "smoke_gate": {"passed": False},
        "jacobian": {
            "jacobian_contract_established": False,
            "focus_86": [{"mechanism": CASE_TRANSPORT}],
        },
        "sequential_vs_direct": {
            "n_direct_pass_sequential_fail": 0,
            "n_both_fail": 3,
        },
    }
    transport = decide_case(base)
    assert transport["decision"] == CASE_TRANSPORT
    assert transport["b14m_reopen_authorized"] is False
    assert transport["restart_invariance_authorized"] is False
    assert transport["full_sample_authorized"] is False
    assert transport["five_d_cin_not_the_objective"] is True
    continuation = decide_case(
        {
            **base,
            "jacobian": {"jacobian_contract_established": False, "focus_86": [{"mechanism": CASE_CONTINUATION}]},
            "sequential_vs_direct": {"n_direct_pass_sequential_fail": 2, "n_both_fail": 1},
        }
    )
    assert continuation["decision"] == CASE_CONTINUATION
    passed = decide_case(
        {
            **base,
            "smoke_gate": {"passed": True},
            "jacobian": {"jacobian_contract_established": True, "focus_86": []},
        }
    )
    assert passed["decision"] == CASE_ESTABLISHED
    assert passed["b14m_reopen_authorized"] is True
    assert passed["restart_invariance_authorized"] is True
    assert passed["full_sample_authorized"] is False


def test_smoke_gate_blocks_b14m_without_jacobian_contract():
    gate = smoke_gate(
        {
            "smoke_present": True,
            "evaluations": {
                "all_required_evaluable": True,
                "measurement_update_in_evaluator": False,
                "focus_retained": True,
            },
            "jacobian": {
                "jacobian_contract_established": False,
                "n_targets": 12,
                "do_not_select_best_step": True,
            },
            "sequential_vs_direct": {"rows": [{"x": 1}]},
            "exclusion": {"target_exclusion_holds": True},
            "prior_introduced": False,
            "ridge_as_information": False,
            "statistical_model_changed": False,
        }
    )
    assert gate["passed"] is False
    assert gate["b14m_reopen_authorized"] is False
    assert gate["full_sample_authorized"] is False
    assert gate["restart_invariance_authorized"] is False
