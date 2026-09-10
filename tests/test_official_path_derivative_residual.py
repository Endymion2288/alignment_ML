"""Task B14V official-path derivative residual contract.  No B15, no V2, no B14M."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.profiled_measurement_likelihood import PINV_RELATIVE
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.official_path_derivative_residual import (
    CASE_ESTABLISHED,
    CASE_FD_REF,
    CASE_INCONSISTENT,
    CASE_MIXED,
    CASE_PATHOLOGY,
    FOCUS_EVENT,
    RUNG_FACTORS,
    OfficialPathDerivativeResidualError,
    audit_envelope_column,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    refuse_5d_cin_repair,
    refuse_b14m,
    refuse_b15,
    refuse_best_step_selection,
    refuse_best_tolerance,
    refuse_change_fd_ladder,
    refuse_change_jacobian,
    refuse_change_statistical_model,
    refuse_dummy_cov_bounded,
    refuse_full_sample,
    refuse_measurement_model_v2,
    refuse_measurement_update,
    refuse_prior,
    refuse_relax_gate,
    refuse_replace_likelihood,
    refuse_restart,
    refuse_ridge_information,
    refuse_switch_to_direct,
    smoke_gate,
)
from datasets.official_supporting_plane_jacobian import (
    CASE_INCONSISTENT as WB120_DECISION,
)


def test_config_inherits_wb120_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14u"] is True
    assert config["do_not_enter_b14v"] is False
    assert config["do_not_select_best_fd_step"] is True
    assert config["do_not_change_statistical_model"] is True
    assert float(config["profile_numerical"]["pinv_relative"]) == PINV_RELATIVE
    spec = config["official_path_derivative_residual"]
    assert list(spec["rung_factors"]) == list(RUNG_FACTORS)
    assert float(spec["official_step_tolerance"]) == 1.0e-4
    assert spec["do_not_relax_five_percent_gate"] is True
    assert spec["do_not_change_jacobian_implementation"] is True
    assert tuple(config["focus_event"]) == FOCUS_EVENT
    assert inherited["workbook_120"]["decision"] == WB120_DECISION
    assert inherited["workbook_120"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_120"]["jacobian_contract_established"] is False
    assert inherited["workbook_120"]["same_path_jacobian_available"] is True
    assert inherited["workbook_119"]["took_wrong_jacobian"] is True
    assert "official_path_derivative_residual_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(OfficialPathDerivativeResidualError, match="prior"):
        refuse_prior()
    with pytest.raises(OfficialPathDerivativeResidualError, match="ridge"):
        refuse_ridge_information()
    with pytest.raises(OfficialPathDerivativeResidualError, match="measurement update"):
        refuse_measurement_update()
    with pytest.raises(OfficialPathDerivativeResidualError, match="best finite-difference"):
        refuse_best_step_selection()
    with pytest.raises(OfficialPathDerivativeResidualError, match="stepTolerance"):
        refuse_best_tolerance()
    with pytest.raises(OfficialPathDerivativeResidualError, match="direct-from-source"):
        refuse_switch_to_direct()
    with pytest.raises(OfficialPathDerivativeResidualError, match="official likelihood"):
        refuse_replace_likelihood()
    with pytest.raises(OfficialPathDerivativeResidualError, match="dummy-cov"):
        refuse_dummy_cov_bounded()
    with pytest.raises(OfficialPathDerivativeResidualError, match="5%"):
        refuse_relax_gate()
    with pytest.raises(OfficialPathDerivativeResidualError, match="FD ladder"):
        refuse_change_fd_ladder()
    with pytest.raises(OfficialPathDerivativeResidualError, match="Jacobian implementation"):
        refuse_change_jacobian()
    with pytest.raises(OfficialPathDerivativeResidualError, match="statistical model"):
        refuse_change_statistical_model()
    with pytest.raises(OfficialPathDerivativeResidualError, match="B14M"):
        refuse_b14m()
    with pytest.raises(OfficialPathDerivativeResidualError, match="restart"):
        refuse_restart()
    with pytest.raises(OfficialPathDerivativeResidualError, match="B15"):
        refuse_b15()
    with pytest.raises(OfficialPathDerivativeResidualError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(OfficialPathDerivativeResidualError, match="1989"):
        refuse_full_sample()
    with pytest.raises(OfficialPathDerivativeResidualError, match="5D Cin"):
        refuse_5d_cin_repair()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_contract_keeps_wb114_model():
    contract = likelihood_contract()
    assert contract["do_not_switch_to_direct"] is True
    assert contract["do_not_select_best_fd_step"] is True
    assert contract["do_not_replace_official_likelihood"] is True
    assert contract["do_not_use_dummy_cov_bounded_transportJacobian"] is True
    assert contract["do_not_relax_five_percent_gate"] is True
    assert contract["do_not_change_jacobian_implementation"] is True
    assert contract["measurement_update_in_evaluator"] is False
    assert contract["not_the_objective"] == "repair 5D Cin"
    assert contract["R_i"] == "(0.08 mm)^2 / 12"
    assert contract["wb120_same_path_jacobian_available"] is True


def _fd_raw(parameter: str, rungs: list[np.ndarray], last_rel: float, ok: bool = True) -> dict:
    return {
        "parameter": parameter,
        "ok": ok,
        "rungs": [
            {
                "step_factor": factor,
                "residual_column": column.tolist(),
                "column_norm": float(np.linalg.norm(column)),
            }
            for factor, column in zip((1.0, 0.5, 0.25, 0.125), rungs)
        ],
        "consecutive_relative_errors": [
            {"relative_error": 0.0, "sign_consistent": True},
            {"relative_error": 0.0, "sign_consistent": True},
            {"relative_error": last_rel, "sign_consistent": True},
        ],
    }


def test_envelope_classifies_pathology_inconsistency_and_fd_reference():
    analytic = np.array([0.02, -0.02, 0.02])
    close = [
        analytic + 1.0e-6,
        analytic + 8.0e-7,
        analytic + 6.0e-7,
        analytic + 4.0e-7,
    ]
    path = audit_envelope_column(_fd_raw("loc1", close, 0.001), analytic * 1.0001)
    assert path["frozen_five_percent_pass"] is True
    assert path["residual_kind"] == "frozen_five_percent_pass"
    assert path["five_percent_gate_unchanged"] is True

    drifted = np.array([0.018, -0.022, 0.018])
    stable_fd = [drifted, drifted + 1.0e-6, drifted + 2.0e-6, drifted + 1.5e-6]
    inconsistent = audit_envelope_column(_fd_raw("loc1", stable_fd, 0.0007), analytic)
    assert inconsistent["fd_ladder_converged"] is True
    assert inconsistent["frozen_five_percent_pass"] is False
    assert inconsistent["far_below_measurement_sensitivity"] is True
    assert inconsistent["far_below_fd_self_difference"] is False
    assert inconsistent["small_column_relative_metric_pathology"] is False
    assert inconsistent["residual_kind"] == CASE_INCONSISTENT

    # Official h is an outlier; the last two FD rungs sit on analytic.
    pathology_fd = [analytic * 1.08, analytic * 1.001, analytic.copy(), analytic.copy()]
    pathology = audit_envelope_column(_fd_raw("loc1", pathology_fd, 0.001), analytic)
    assert pathology["frozen_five_percent_pass"] is False
    assert pathology["fd_ladder_converged"] is True
    assert pathology["inside_last_pair_envelope"] is True
    assert pathology["far_below_measurement_sensitivity"] is True
    assert pathology["small_column_relative_metric_pathology"] is True
    assert pathology["residual_kind"] == CASE_PATHOLOGY

    chaotic = [
        np.array([2.0, -1.0, 3.0]),
        np.array([0.2, 4.0, -2.0]),
        np.array([8.0, -6.0, 1.0]),
        np.array([-3.0, 5.0, 7.0]),
    ]
    fd_ref = audit_envelope_column(_fd_raw("phi", chaotic, 1.4), analytic * 100.0)
    assert fd_ref["fd_ladder_converged"] is False
    assert fd_ref["residual_kind"] == CASE_FD_REF


def _target(**bits: object) -> dict:
    base = {
        "control_fd_all_converged": True,
        "control_five_percent_all_pass": True,
        "focus_fd_all_converged": True,
        "focus_five_percent_all_pass": True,
        "loc0_matches_official": True,
        "free_jacobian_available": True,
        "chain_complete": True,
        "branch_fallback": False,
        "dummy_cov_bounded_used": False,
        "segment_composition_closed": True,
        "any_pathology": False,
        "any_fd_reference": False,
        "any_true_inconsistency": False,
        "envelopes": [{"parameter": "loc0", "frozen_five_percent_pass": True}],
        "composition": {"segment_composition_closed": True},
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
        "relaxed_five_percent_gate": False,
        "changed_fd_ladder": False,
        "changed_jacobian_implementation": False,
        "restart_executed": False,
        "b14m_reopened": False,
        "smoke_present": True,
        "exclusion": {"target_exclusion_holds": True},
        "controls": [_target(), _target(), _target()],
        "focus_86": [_target(), _target(), _target()],
    }
    established = decide_case(base)
    assert established["decision"] == CASE_ESTABLISHED
    assert established["jacobian_contract_established"] is True
    assert established["b14m_reopen_authorized"] is True
    assert established["restart_invariance_authorized"] is False
    assert established["full_sample_authorized"] is False
    assert established["five_percent_gate_unchanged"] is True

    fd_only = decide_case(
        {
            **base,
            "focus_86": [
                _target(
                    focus_five_percent_all_pass=False,
                    focus_fd_all_converged=False,
                    any_fd_reference=True,
                )
            ]
            * 3,
        }
    )
    assert fd_only["decision"] == CASE_FD_REF
    assert fd_only["b14m_reopen_authorized"] is False
    assert fd_only["next_step"] == "establish_independent_derivative_reference"

    path_only = decide_case(
        {
            **base,
            "controls": [
                _target(control_five_percent_all_pass=False, any_pathology=True)
            ]
            * 3,
        }
    )
    assert path_only["decision"] == CASE_PATHOLOGY
    assert path_only["jacobian_contract_established"] is False

    broken = decide_case(
        {
            **base,
            "controls": [
                _target(
                    control_five_percent_all_pass=False,
                    any_true_inconsistency=True,
                )
            ]
            * 3,
        }
    )
    assert broken["decision"] == CASE_INCONSISTENT
    assert broken["b14m_reopen_authorized"] is False

    mixed = decide_case(
        {
            **base,
            "controls": [
                _target(
                    control_five_percent_all_pass=False,
                    any_true_inconsistency=True,
                )
            ]
            * 3,
            "focus_86": [
                _target(
                    focus_five_percent_all_pass=False,
                    focus_fd_all_converged=False,
                    any_fd_reference=True,
                )
            ]
            * 3,
        }
    )
    assert mixed["decision"] == CASE_MIXED
    assert mixed["b14m_reopen_authorized"] is False
    assert mixed["next_step"] == "keep_residual_diagnosis_without_shrinking_fd"

    empty = decide_case({**base, "smoke_present": False, "controls": [], "focus_86": []})
    assert empty["decision"] == CASE_MIXED
    assert empty["b14m_reopen_authorized"] is False
    with pytest.raises(OfficialPathDerivativeResidualError, match="dummy-cov"):
        decide_case({**base, "targets": [{"dummy_cov_bounded_used": True}]})
    with pytest.raises(OfficialPathDerivativeResidualError, match="5%"):
        decide_case({**base, "relaxed_five_percent_gate": True})


def test_smoke_gate_never_authorizes_full_sample():
    gate = smoke_gate(
        {
            "smoke_present": True,
            "wb120_smoke_hash_match": True,
            "targets": [
                {
                    "event": event,
                    "envelopes": [1],
                    "composition": {"segment_composition_closed": True},
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
