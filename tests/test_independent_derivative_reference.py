"""Task B14W independent derivative reference.  No B15, no V2, no B14M."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.independent_derivative_reference import (
    CASE_COUPLING,
    CASE_FOCUS_NO_REF,
    CASE_FOCUS_REF,
    CASE_MIXED,
    RUNG_FACTORS,
    IndependentDerivativeReferenceError,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    loc1_chain_stages,
    polynomial_odd_reference,
    refuse_add_fd_rung,
    refuse_b14m,
    refuse_b15,
    refuse_best_step_selection,
    refuse_change_jacobian,
    refuse_change_statistical_model,
    refuse_full_sample,
    refuse_measurement_model_v2,
    refuse_prior,
    refuse_relax_gate,
    refuse_restart,
    refuse_ridge_information,
    richardson_reference,
    smoke_gate,
)
from datasets.official_path_derivative_residual import CASE_MIXED as WB121_DECISION


def test_config_inherits_wb121_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14v"] is True
    assert config["do_not_enter_b14w"] is False
    spec = config["independent_derivative_reference"]
    assert list(spec["rung_factors"]) == list(RUNG_FACTORS)
    assert spec["do_not_change_jacobian_implementation"] is True
    assert spec["do_not_add_fd_rung"] is True
    assert spec["do_not_relax_five_percent_gate"] is True
    assert inherited["workbook_121"]["decision"] == WB121_DECISION
    assert inherited["workbook_121"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_120"]["same_path_jacobian_available"] is True
    assert "independent_derivative_reference_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(IndependentDerivativeReferenceError, match="prior"):
        refuse_prior()
    with pytest.raises(IndependentDerivativeReferenceError, match="ridge"):
        refuse_ridge_information()
    with pytest.raises(IndependentDerivativeReferenceError, match="best finite-difference"):
        refuse_best_step_selection()
    with pytest.raises(IndependentDerivativeReferenceError, match="FD rung"):
        refuse_add_fd_rung()
    with pytest.raises(IndependentDerivativeReferenceError, match="5%"):
        refuse_relax_gate()
    with pytest.raises(IndependentDerivativeReferenceError, match="Jacobian implementation"):
        refuse_change_jacobian()
    with pytest.raises(IndependentDerivativeReferenceError, match="statistical model"):
        refuse_change_statistical_model()
    with pytest.raises(IndependentDerivativeReferenceError, match="B14M"):
        refuse_b14m()
    with pytest.raises(IndependentDerivativeReferenceError, match="restart"):
        refuse_restart()
    with pytest.raises(IndependentDerivativeReferenceError, match="B15"):
        refuse_b15()
    with pytest.raises(IndependentDerivativeReferenceError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(IndependentDerivativeReferenceError, match="1989"):
        refuse_full_sample()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_keeps_gate_and_jacobian():
    contract = likelihood_contract()
    assert contract["do_not_change_jacobian_implementation"] is True
    assert contract["do_not_relax_five_percent_gate"] is True
    assert contract["do_not_add_fd_rung"] is True
    assert contract["small_physical_effect_does_not_pass_contract"] is True
    assert contract["R_i"] == "(0.08 mm)^2 / 12"


def test_richardson_marks_oscillatory_not_applicable():
    failed = richardson_reference([0.046, 0.152, 0.104, 0.034])
    assert failed["applicable"] is False
    assert failed["status"] == "not_applicable"
    assert failed["do_not_force_extrapolation"] is True
    ok = richardson_reference([1.08, 1.02, 1.005, 1.001])
    assert ok["applicable"] is True
    assert ok["do_not_replace_official_fd"] is True


def test_polynomial_reports_condition_and_does_not_replace_fd():
    poly = polynomial_odd_reference([1.08, 1.02, 1.005, 1.001], [1.0, 0.5, 0.25, 0.125])
    assert poly["do_not_replace_official_fd"] is True
    assert "condition_number" in poly


def test_decide_case_paths():
    base = {
        "prior_introduced": False,
        "ridge_as_information": False,
        "statistical_model_changed": False,
        "selected_best_step": False,
        "added_fd_rung": False,
        "relaxed_five_percent_gate": False,
        "changed_jacobian_implementation": False,
        "restart_executed": False,
        "b14m_reopened": False,
        "smoke_present": True,
        "control1": [
            {
                "coupling_identified": True,
                "missing_pos_to_dir_in_rk_product": True,
                "station0_agrees": True,
            }
        ]
        * 3,
        "focus_references": [{"independent_reference_established": False}] * 3,
    }
    identified = decide_case(base)
    assert identified["decision"] == CASE_COUPLING
    assert identified["jacobian_contract_established"] is False
    assert identified["b14m_reopen_authorized"] is False
    assert identified["five_percent_gate_unchanged"] is True
    assert identified["small_physical_effect_does_not_pass_contract"] is True

    focus_only = decide_case(
        {
            **base,
            "control1": [
                {
                    "coupling_identified": False,
                    "missing_pos_to_dir_in_rk_product": False,
                    "station0_agrees": True,
                }
            ]
            * 3,
            "focus_references": [{"independent_reference_established": True}] * 3,
        }
    )
    assert focus_only["decision"] == CASE_FOCUS_REF
    assert focus_only["b14m_reopen_authorized"] is False

    none = decide_case(
        {
            **base,
            "control1": [
                {
                    "coupling_identified": False,
                    "missing_pos_to_dir_in_rk_product": False,
                    "station0_agrees": False,
                }
            ]
            * 3,
        }
    )
    assert none["decision"] == CASE_FOCUS_NO_REF
    with pytest.raises(IndependentDerivativeReferenceError, match="FD rung"):
        decide_case({**base, "added_fd_rung": True})


def test_loc1_chain_stages_record_zero_pos_to_dir():
    hop = {
        "source_bound_to_free_jacobian": [
            [1.0, 0.0, 0, 0, 0, 0],
            [0.0, 1.0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0],
            [0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
        "rk_free_transport_jacobian_product": [
            [1, 0, 0, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0, 0, 0],
            [0, 0, 1, 0, 0, 0, 0, 0],
            [0, 0, 0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 1, 0, 0, 0],
            [0, 0, 0, 0, 0, 1, 0, 0],
            [0, 0, 0, 0, 0, 0, 1, 0],
            [0, 0, 0, 0, 0, 0, 0, 1],
        ],
        "hop_dloc0_d_start_rk_free": [0.02, -0.02, 0, 0, 0],
        "chained_dloc0_d_source_rk_free": [0.02, -0.02, 0, 0, 0],
        "continuation_jacobian": [
            [1, 0, 0, 0, 0, 0],
            [0, 1, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
            [0, 0, 0, 0, 0, 0],
        ],
    }
    stages = loc1_chain_stages(hop)
    assert stages["rk_variational_pos_to_dir_norm"] == 0.0
    assert stages["source_bound_to_free_loc1_direction_norm"] == 0.0
    assert stages["continuation_loc1_angular_norm"] == 0.0
    assert stages["supporting_plane_hop_dloc0_d_start_loc1"] == -0.02
    assert stages["covariance_and_process_noise_excluded"] is True


def test_smoke_gate_never_authorizes_full_sample():
    gate = smoke_gate(
        {
            "smoke_present": True,
            "wb120_smoke_hash_match": True,
            "control1": [{"hits": [1]}],
            "focus_86": [1],
            "material_audits": [1],
            "segment_audits": [1],
            "focus_references": [1],
            "focus_per_hits": [1],
            "exclusion": {"target_exclusion_holds": True},
            "prior_introduced": False,
            "ridge_as_information": False,
        }
    )
    assert gate["passed"] is True
    assert gate["b14m_reopen_authorized"] is False
    assert gate["full_sample_authorized"] is False
    assert CASE_MIXED == "mixed_or_inconclusive"
