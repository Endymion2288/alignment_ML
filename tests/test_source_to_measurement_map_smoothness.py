"""Task B14K source-to-measurement map smoothness.  No B15, no V2, no B14M."""

from __future__ import annotations

import pytest

from alignment.profiled_measurement_likelihood import PINV_RELATIVE
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.source_to_measurement_map_smoothness import (
    ACCURACY_TOLERANCES,
    CASE_ACTS_JAC,
    CASE_GENUINE,
    CASE_INTEGRATION,
    CASE_MIXED,
    CASE_PROJECTION,
    CASE_SCALE,
    CONTROL_EVENT,
    FOCUS_EVENT,
    RUNG_FACTORS,
    SourceToMeasurementMapSmoothnessError,
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
    refuse_restart,
    refuse_ridge_information,
    refuse_switch_to_direct,
    smoke_gate,
)
from datasets.supporting_plane_jacobian_continuity import (
    CASE_TRANSPORT as WB117_DECISION,
)


def test_config_inherits_wb117_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14j"] is True
    assert config["do_not_enter_b14k"] is False
    assert config["do_not_select_best_fd_step"] is True
    assert config["do_not_change_statistical_model"] is True
    assert config["do_not_use_ridge"] is True
    assert float(config["profile_numerical"]["pinv_relative"]) == PINV_RELATIVE
    assert list(config["map_smoothness"]["rung_factors"]) == list(RUNG_FACTORS)
    tols = [item["step_tolerance"] for item in config["map_smoothness"]["accuracy_rungs"]]
    assert tols == list(ACCURACY_TOLERANCES)
    assert tuple(config["map_smoothness"]["focus_event"]) == FOCUS_EVENT
    assert int(config["control_selection"]["selected_event_id"]) == CONTROL_EVENT[1]
    assert config["control_selection"]["jacobian_not_used"] is True
    assert inherited["workbook_117"]["decision"] == WB117_DECISION
    assert inherited["workbook_117"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_117"]["jacobian_contract_established"] is False
    assert inherited["workbook_116"]["decision"] == "mixed_or_inconclusive"
    assert "source_to_measurement_map_smoothness_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="prior"):
        refuse_prior()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="ridge"):
        refuse_ridge_information()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="measurement update"):
        refuse_measurement_update()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="best finite-difference"):
        refuse_best_step_selection()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="stepTolerance"):
        refuse_best_tolerance()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="direct-from-source"):
        refuse_switch_to_direct()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="statistical model"):
        refuse_change_statistical_model()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="B14M"):
        refuse_b14m()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="restart"):
        refuse_restart()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="B15"):
        refuse_b15()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="1989"):
        refuse_full_sample()
    with pytest.raises(SourceToMeasurementMapSmoothnessError, match="5D Cin"):
        refuse_5d_cin_repair()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_contract_keeps_wb114_model():
    contract = likelihood_contract()
    assert contract["do_not_switch_to_direct"] is True
    assert contract["do_not_select_best_fd_step"] is True
    assert contract["do_not_select_best_tolerance"] is True
    assert contract["measurement_update_in_evaluator"] is False
    assert contract["not_the_objective"] == "repair 5D Cin"
    assert contract["R_i"] == "(0.08 mm)^2 / 12"
    assert contract["wb117_not_a_physical_conclusion"] is True


def _focus(mechanism_bits: dict) -> dict:
    base = {
        "repeatability": {"deterministic": True},
        "projection": {"ill_conditioned": False, "projection_map_smooth": True},
        "accuracy": {"integration_resolution_insufficient": False},
        "resolution": {"official_h_signal_at_or_below_noise": False},
        "fd_scale": {"regime": "unclassified"},
        "first_failing_stages": ["free_near_plane"],
        "acts_derivative": {"any_diagnostic_jacobian_present": False},
    }
    base.update(mechanism_bits)
    return base


def test_decide_case_never_reopens_b14m():
    base = {
        "prior_introduced": False,
        "ridge_as_information": False,
        "statistical_model_changed": False,
        "switched_to_direct": False,
        "selected_best_step": False,
        "selected_best_tolerance": False,
        "restart_executed": False,
        "b14m_reopened": False,
        "smoke_present": True,
        "focus_86": [_focus({})],
    }
    genuine = decide_case(base)
    assert genuine["decision"] == CASE_GENUINE
    assert genuine["b14m_reopen_authorized"] is False
    assert genuine["restart_invariance_authorized"] is False
    assert genuine["jacobian_contract_established"] is False
    assert genuine["full_sample_authorized"] is False
    assert genuine["wb117_not_treated_as_physical_nonsmoothness"] is True
    integration = decide_case(
        {**base, "focus_86": [_focus({"accuracy": {"integration_resolution_insufficient": True}})]}
    )
    assert integration["decision"] == CASE_INTEGRATION
    projection = decide_case(
        {**base, "focus_86": [_focus({"projection": {"ill_conditioned": True, "projection_map_smooth": False}})]}
    )
    assert projection["decision"] == CASE_PROJECTION
    scale = decide_case(
        {**base, "focus_86": [_focus({"resolution": {"official_h_signal_at_or_below_noise": True}})]}
    )
    assert scale["decision"] == CASE_SCALE
    empty = decide_case({**base, "smoke_present": False, "focus_86": []})
    assert empty["decision"] == CASE_MIXED
    assert empty["b14m_reopen_authorized"] is False
    acts_present_only = decide_case(
        {
            **base,
            "focus_86": [
                _focus(
                    {
                        "accuracy": {
                            "integration_resolution_insufficient": False,
                            "integration_improves_but_not_converged": True,
                        },
                        "acts_derivative": {
                            "any_diagnostic_jacobian_present": True,
                            "acts_inconsistent_with_fd": False,
                        },
                    }
                )
            ],
        }
    )
    assert acts_present_only["decision"] == CASE_MIXED
    assert acts_present_only["decision"] != CASE_ACTS_JAC
    assert acts_present_only["next_step"] == (
        "open_wb119_analytic_vs_fd_derivative_contract"
    )
    assert acts_present_only["b14m_reopen_authorized"] is False
    acts_inconsistent = decide_case(
        {
            **base,
            "focus_86": [
                _focus(
                    {
                        "acts_derivative": {
                            "any_diagnostic_jacobian_present": True,
                            "acts_inconsistent_with_fd": True,
                        }
                    }
                )
            ],
        }
    )
    assert acts_inconsistent["decision"] == CASE_ACTS_JAC
    assert acts_inconsistent["b14m_reopen_authorized"] is False


def test_smoke_gate_never_authorizes_b14m():
    gate = smoke_gate(
        {
            "smoke_present": True,
            "targets": [
                {
                    "event": "100048/86",
                    "stagewise": [1],
                    "repeatability": {},
                    "resolution": {},
                    "accuracy": {},
                    "projection": {},
                    "fd_scale": {},
                    "acts_derivative": {},
                },
                {
                    "event": "100048/44",
                    "stagewise": [1],
                    "repeatability": {},
                    "resolution": {},
                    "accuracy": {},
                    "projection": {},
                    "fd_scale": {},
                    "acts_derivative": {},
                },
            ],
            "focus_vs_control": {"rows": [{"x": 1}]},
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
