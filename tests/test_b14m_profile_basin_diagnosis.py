"""Task B14M-S basin-diagnosis contract.  No optimizer change, no 1989."""

from __future__ import annotations

import pytest

from datasets.b14m_profile_basin_diagnosis import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    CASE_E,
    CASE_F,
    CASE_G,
    FAILURE_IDENTITIES,
    B14MProfileBasinError,
    classify_identity,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    refuse_b15,
    refuse_drop_37,
    refuse_full_sample,
    refuse_held_out_selection,
    refuse_min_restarts,
    refuse_optimizer_change,
    refuse_random_restarts,
    refuse_retune,
)
from datasets.shadow_mean_transport_contract import FROZEN_FIELD_GRADIENT_SHA
from datasets.surface_energy_loss_mean_semantics import PRODUCTION_ELOSS


def test_config_inherits_wb129_and_freezes_optimizer():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    spec = config["b14m_profile_basin_diagnosis"]
    assert config["do_not_change_optimizer"] is True
    assert config["do_not_change_termination_rule"] is True
    assert config["do_not_submit_1989"] is True
    assert config["do_not_enter_b14ms"] is False
    assert config["profile_smoke_root"] == (
        "outputs/leave_target_out_dump_v1/b14ms_basin_smoke"
    )
    assert "b14m_reopen_smoke" not in config["profile_smoke_root"]
    assert [tuple(item) for item in spec["failure_identities"]] == list(
        FAILURE_IDENTITIES
    )
    assert inherited["workbook_129"]["decision"] == (
        "profile_optimizer_restart_sensitive"
    )
    assert inherited["workbook_129"]["restart_invariance_established"] is False
    assert inherited["workbook_128"]["jacobian_contract_established"] is True
    assert inherited["workbook_103"]["contracted_denominator"] == 1989
    assert inherited["current"]["full_sample_authorized"] is False
    assert inherited["workbook_128"]["production_eloss_quantity"] == PRODUCTION_ELOSS
    assert FROZEN_FIELD_GRADIENT_SHA.startswith("ca5e4f0e")
    assert spec["pinv_relative"] == 1.0e-8
    assert spec["max_iterations"] == 50
    assert spec["line_search"][0] == 1.0
    assert spec["line_search"][-1] == 0.0078125


def test_forbidden_repairs():
    with pytest.raises(B14MProfileBasinError, match="optimizer"):
        refuse_optimizer_change()
    with pytest.raises(B14MProfileBasinError, match="Gauss-Newton"):
        refuse_retune()
    with pytest.raises(B14MProfileBasinError, match="1989"):
        refuse_full_sample()
    with pytest.raises(B14MProfileBasinError, match="B15"):
        refuse_b15()
    with pytest.raises(B14MProfileBasinError, match="37"):
        refuse_drop_37()
    with pytest.raises(B14MProfileBasinError, match="min of four"):
        refuse_min_restarts()
    with pytest.raises(B14MProfileBasinError, match="held-out"):
        refuse_held_out_selection()
    with pytest.raises(B14MProfileBasinError, match="random"):
        refuse_random_restarts()


def test_likelihood_contract_unchanged():
    contract = likelihood_contract()
    assert contract["prior"] is None
    assert contract["ridge"] is None
    assert contract["q_over_p_is_explicit_nuisance"] is True
    assert contract["do_not_change_optimizer"] is True
    assert contract["do_not_submit_1989"] is True


def test_case_a_beats_basin_language():
    stationarity = {
        "high_chi2_not_stationary": True,
        "linesearch_logic_broken": True,
        "any_not_stationary": True,
        "chi2_rel_spread": 0.5,
    }
    cross = {"nuisance_profile_multibasin": True}
    continuation = {"profile_hysteresis": True, "profile_objective_multimodality": True}
    assert (
        classify_identity(stationarity, cross, continuation, is_control=False) == CASE_A
    )


def test_case_c_only_after_stationary():
    stationarity = {
        "high_chi2_not_stationary": False,
        "linesearch_logic_broken": False,
        "any_not_stationary": False,
    }
    cross = {"nuisance_profile_multibasin": True, "alpha_globalization_sensitive": False}
    continuation = {
        "high_branch_not_stationary": False,
        "profile_hysteresis": False,
        "profile_objective_multimodality": False,
    }
    assert (
        classify_identity(stationarity, cross, continuation, is_control=False) == CASE_C
    )


def test_case_d_and_b_and_e():
    stationary = {
        "high_chi2_not_stationary": False,
        "linesearch_logic_broken": False,
        "any_not_stationary": False,
    }
    assert (
        classify_identity(
            stationary,
            {"nuisance_profile_multibasin": False, "alpha_globalization_sensitive": False},
            {
                "high_branch_not_stationary": False,
                "profile_hysteresis": False,
                "profile_objective_multimodality": True,
            },
            is_control=False,
        )
        == CASE_D
    )
    assert (
        classify_identity(
            stationary,
            {"nuisance_profile_multibasin": False, "alpha_globalization_sensitive": True},
            {
                "high_branch_not_stationary": False,
                "profile_hysteresis": False,
                "profile_objective_multimodality": False,
            },
            is_control=False,
        )
        == CASE_B
    )
    assert (
        classify_identity(
            stationary,
            {"nuisance_profile_multibasin": False, "alpha_globalization_sensitive": False},
            {
                "high_branch_not_stationary": False,
                "profile_hysteresis": True,
                "profile_objective_multimodality": True,
            },
            is_control=False,
        )
        == CASE_E
    )


def test_negative_control_stays_case_f():
    assert (
        classify_identity(
            {
                "any_not_stationary": True,
                "chi2_rel_spread": 0.001,
                "high_chi2_not_stationary": True,
            },
            {},
            {},
            is_control=True,
        )
        == CASE_F
    )


def test_primary_is_a_if_any_failure_is_a():
    inventory = {
        "per_identity": {
            "100043/37 T1": {"role": "failure", "case": CASE_A},
            "100043/37 T2": {"role": "failure", "case": CASE_C},
            "100048/86 T1": {"role": "failure", "case": CASE_B},
            "100043/0 T1": {"role": "negative_control", "case": CASE_F},
        }
    }
    decision = decide_case(inventory)
    assert decision["decision"] == CASE_A
    assert decision["restart_invariance_established"] is False
    assert decision["full_sample_authorized"] is False
    assert decision["failure_cases"]["100043/37 T2"] == CASE_C


def test_case_g_when_failures_mix_without_a():
    inventory = {
        "per_identity": {
            "100043/37 T1": {"role": "failure", "case": CASE_C},
            "100043/37 T2": {"role": "failure", "case": CASE_D},
            "100048/86 T1": {"role": "failure", "case": CASE_B},
        }
    }
    assert decide_case(inventory)["decision"] == CASE_G
