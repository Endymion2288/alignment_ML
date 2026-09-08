"""Task B14N profile-likelihood numerics.  No B15, no V2, no prior."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.profiled_measurement_likelihood import PINV_RELATIVE
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.profile_likelihood_numerics import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    CASE_F,
    ProfileLikelihoodNumericsError,
    audit_surface_ordering,
    decide_case,
    inherit_frozen_stage,
    load_config,
    prove_physical_chi2_invariant,
    refuse_b15,
    refuse_change_statistical_model,
    refuse_chi2_penalty,
    refuse_focus_drop,
    refuse_full_sample_without_gate,
    refuse_hessian_scale,
    refuse_measurement_model_v2,
    refuse_prior,
    refuse_propagation_as_nonidentifiability,
    refuse_rewrite_profile_math,
    refuse_ridge_information,
    refuse_scaling_as_prior,
    refuse_seed_campaign_scale,
    refuse_truth_scale,
    smoke_gate,
)
from datasets.profiled_weak_nuisance_likelihood import CASE_D as WB114_DECISION


def test_config_inherits_wb114_case_d_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14n"] is False
    assert config["do_not_rewrite_profile_math"] is True
    assert config["do_not_change_statistical_model"] is True
    assert config["do_not_use_ridge"] is True
    assert config["do_not_force_5d_lto_covariance"] is True
    assert config["do_not_interpret_scaling_as_prior"] is True
    assert config["do_not_submit_full_sample_without_smoke_gate"] is True
    assert float(config["profile_numerical"]["pinv_relative"]) == PINV_RELATIVE
    assert config["profile_parameter_scaling"]["not_a_prior"] is True
    assert inherited["workbook_114"]["decision"] == WB114_DECISION
    assert inherited["workbook_114"]["synthetic_profile_passed"] is True
    assert inherited["workbook_113"]["decision"] == "target_independent_prior_not_available"
    assert inherited["workbook_112"]["five_d_state_physically_supported"] is False
    assert inherited["workbook_109"]["decision"] == (
        "leave_target_out_state_materialization_established"
    )
    assert "profile_likelihood_numerics_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(ProfileLikelihoodNumericsError, match="prior"):
        refuse_prior()
    with pytest.raises(ProfileLikelihoodNumericsError, match="ridge"):
        refuse_ridge_information()
    with pytest.raises(ProfileLikelihoodNumericsError, match="chi2 penalty"):
        refuse_chi2_penalty()
    with pytest.raises(ProfileLikelihoodNumericsError, match="prior covariance"):
        refuse_scaling_as_prior()
    with pytest.raises(ProfileLikelihoodNumericsError, match="truth"):
        refuse_truth_scale()
    with pytest.raises(ProfileLikelihoodNumericsError, match="Hessian"):
        refuse_hessian_scale()
    with pytest.raises(ProfileLikelihoodNumericsError, match="seed campaign"):
        refuse_seed_campaign_scale()
    with pytest.raises(ProfileLikelihoodNumericsError, match="rewritten"):
        refuse_rewrite_profile_math()
    with pytest.raises(ProfileLikelihoodNumericsError, match="statistical model"):
        refuse_change_statistical_model()
    with pytest.raises(ProfileLikelihoodNumericsError, match="B15"):
        refuse_b15()
    with pytest.raises(ProfileLikelihoodNumericsError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(ProfileLikelihoodNumericsError, match="1989"):
        refuse_full_sample_without_gate()
    with pytest.raises(ProfileLikelihoodNumericsError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(ProfileLikelihoodNumericsError, match="propagation failure"):
        refuse_propagation_as_nonidentifiability()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_scaling_leaves_physical_chi2_unchanged():
    result = prove_physical_chi2_invariant(
        lambda theta: float(np.sum((theta - np.array([0.2, -1.0, 0.01, 0.03, 2e-3])) ** 2)),
        np.array([1.0, 0.0, 0.0, 0.02, 1.0e-3]),
        np.array([1.0, 1.0, 1.0e-3, 1.0e-3, 1.0e-3]),
        np.array([0.4, -1.5, 3.0, -2.0, 1.25]),
    )
    assert result["physical_chi2_invariant"] is True
    assert result["scaling_is_prior"] is False


def test_surface_ordering_flags_return_upstream():
    rows = [
        {
            "evaluate_only": True,
            "init_kind": "wb114_official_seed_mean",
            "source_id": "mc24_100043_00400_00499",
            "run_id": 100043,
            "event_id": 0,
            "target_station": 1,
            "hop_mode": "sequential_z_order",
            "surface_order": {
                "source_z_mm": -1860.15,
                "input_order_is_z_sorted": False,
                "returns_upstream_after_downstream": True,
                "z_sorted_path": [
                    {"role": "source", "z_mm": -1860.15},
                    {"station": 1, "z_mm": 47.4},
                    {"station": 0, "z_mm": -1860.15},
                ],
            },
        }
    ]
    report = audit_surface_ordering(rows)
    assert report["ordering_is_physical_transport_bug"] is True
    assert report["n_returns_upstream_after_downstream"] == 1


def test_decide_case_layers_without_claiming_physics():
    base = {
        "prior_introduced": False,
        "ridge_as_information": False,
        "chi2_penalty_used": False,
        "statistical_model_changed": False,
        "smoke_present": True,
        "smoke_gate": {"passed": False},
        "propagation": {"nominal_propagation_pass": False},
        "ordering": {"ordering_is_physical_transport_bug": False},
        "jacobian": {"jacobian_contract_established": True},
        "restart": {"restart_invariance_holds": False, "loc1_restart_equivalent": False},
        "focus": {"focus_not_step_limit_only": False},
        "optimizer": {"hidden_ridge_or_prior": False},
    }
    transport = decide_case(base)
    assert transport["decision"] == CASE_B
    assert transport["b15_authorized"] is False
    assert transport["full_sample_authorized"] is False
    deriv = decide_case(
        {
            **base,
            "propagation": {"nominal_propagation_pass": True},
            "focus": {"focus_not_step_limit_only": True},
            "jacobian": {"jacobian_contract_established": False},
        }
    )
    assert deriv["decision"] == CASE_C
    optim = decide_case(
        {
            **base,
            "propagation": {"nominal_propagation_pass": True},
            "focus": {"focus_not_step_limit_only": True},
            "jacobian": {"jacobian_contract_established": True},
        }
    )
    assert optim["decision"] == CASE_D
    passed = decide_case(
        {
            **base,
            "smoke_gate": {"passed": True},
            "propagation": {"nominal_propagation_pass": True},
            "focus": {"focus_not_step_limit_only": True},
            "jacobian": {"jacobian_contract_established": True},
            "restart": {"restart_invariance_holds": True, "loc1_restart_equivalent": True},
        }
    )
    assert passed["decision"] == CASE_A
    missing = decide_case({**base, "smoke_present": False})
    assert missing["decision"] == CASE_F


def test_smoke_gate_blocks_full_sample():
    gate = smoke_gate(
        {
            "smoke_present": True,
            "restart": {"restart_invariance_holds": True, "loc1_restart_equivalent": False},
            "jacobian": {"jacobian_contract_established": True},
            "propagation": {"nominal_propagation_pass": True},
            "focus": {"focus_not_step_limit_only": True},
            "optimizer": {"hidden_ridge_or_prior": False},
            "exclusion": {"target_exclusion_holds": True},
        }
    )
    assert gate["passed"] is False
    assert gate["full_sample_authorized"] is False
