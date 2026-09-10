"""Task B14M-T globalization-repair contract.  No 1989, no ridge, no retune."""

from __future__ import annotations

import pytest

from datasets.b14m_profile_globalization_repair import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    CASE_E,
    CASE_F,
    CASE_G,
    GATE_IDENTITIES,
    REGRESSION_CONTROLS,
    SMOKE_EVENTS,
    B14MProfileGlobalizationError,
    audit_identity,
    classify_overall,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    preregistered_trust_region,
    refuse_b15,
    refuse_full_sample,
    refuse_held_out,
    refuse_jacobian_reopen,
    refuse_min_restarts,
    refuse_prior,
    refuse_ridge,
    refuse_smaller_lambda,
)
from datasets.b14m_restart_invariance import recover_optimizer_contract
from datasets.shadow_mean_transport_contract import FROZEN_FIELD_GRADIENT_SHA
from datasets.surface_energy_loss_mean_semantics import PRODUCTION_ELOSS


def test_config_preregisters_trust_region_and_inherits_wb130():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    contract = recover_optimizer_contract(config)
    tr = preregistered_trust_region(config)
    spec = config["b14m_profile_globalization_repair"]
    assert config["do_not_enter_b14mt"] is False
    assert config["do_not_change_optimizer"] is False
    assert config["do_not_change_statistical_model"] is True
    assert config["do_not_submit_1989"] is True
    assert config["profile_smoke_root"] == (
        "outputs/leave_target_out_dump_v1/b14mt_repair_smoke"
    )
    assert "b14ms_basin_smoke" not in config["profile_smoke_root"]
    assert [tuple(item) for item in spec["smoke_events"]] == list(SMOKE_EVENTS)
    assert [tuple(item) for item in spec["gate_identities"]] == list(GATE_IDENTITIES)
    assert [tuple(item) for item in spec["regression_controls"]] == list(
        REGRESSION_CONTROLS
    )
    assert spec["Delta_max"] == 10.0
    assert spec["Delta_min"] == 1.0e-8
    assert spec["rho_accept"] == 0.10
    assert spec["rho_expand"] == 0.75
    assert spec["shrink_factor"] == 0.25
    assert spec["expand_factor"] == 2.0
    assert spec["max_trust_region_trials"] == 20
    assert spec["max_outer_iterations"] == 50
    assert spec["reused_existing_project_trust_region"] is False
    assert tr["lambda_meaning"].startswith("numerical globalization")
    assert inherited["workbook_130"]["decision"] == (
        "flat_direction_termination_not_stationary"
    )
    assert inherited["workbook_130"]["nuisance_profile_multibasin"] is False
    assert inherited["workbook_129"]["restart_invariance_established"] is False
    assert inherited["workbook_128"]["jacobian_contract_established"] is True
    assert inherited["workbook_103"]["contracted_denominator"] == 1989
    assert inherited["current"]["full_sample_authorized"] is False
    assert inherited["workbook_128"]["production_eloss_quantity"] == PRODUCTION_ELOSS
    assert FROZEN_FIELD_GRADIENT_SHA.startswith("ca5e4f0e")
    assert contract["recovered"] is True
    assert contract["chi2_rel_tolerance"] == 0.01
    assert contract["prediction_abs_tolerance_mm"] == 0.1
    assert contract["supported_abs_tolerance"]["loc0_mm"] == 0.05
    assert contract["supported_abs_tolerance"]["theta"] == 1.0e-4


def test_forbidden_repairs():
    with pytest.raises(B14MProfileGlobalizationError, match="prior"):
        refuse_prior()
    with pytest.raises(B14MProfileGlobalizationError, match="ridge"):
        refuse_ridge()
    with pytest.raises(B14MProfileGlobalizationError, match="line search"):
        refuse_smaller_lambda()
    with pytest.raises(B14MProfileGlobalizationError, match="min of four"):
        refuse_min_restarts()
    with pytest.raises(B14MProfileGlobalizationError, match="held-out"):
        refuse_held_out()
    with pytest.raises(B14MProfileGlobalizationError, match="1989"):
        refuse_full_sample()
    with pytest.raises(B14MProfileGlobalizationError, match="B15"):
        refuse_b15()
    with pytest.raises(B14MProfileGlobalizationError, match="Jacobian"):
        refuse_jacobian_reopen()


def test_likelihood_contract_unchanged():
    contract = likelihood_contract()
    assert contract["prior"] is None
    assert contract["ridge"] is None
    assert contract["q_over_p_is_explicit_nuisance"] is True
    assert contract["lm_lambda_is_not_ridge"] is True
    assert contract["statistical_model_unchanged"] is True
    assert contract["do_not_submit_1989"] is True


def _row(**overrides):
    row = {
        "run_id": 100043,
        "event_id": 37,
        "target_station": 1,
        "profile_init_variant": "nominal",
        "ok": True,
        "propagation_success": True,
        "valid_solution": True,
        "valid_stationary": True,
        "inner_nuisance_stationary": True,
        "profile_alpha_stationary": True,
        "termination_reason": "supported_stationary",
        "inner_termination": "supported_stationary",
        "outer_termination": "supported_stationary",
        "chi2_prof": 90.0,
        "chi2": 90.0,
        "norm_g_n_R": 1.0e-8,
        "norm_g_alpha_prof_z": 1.0e-8,
        "norm_g_R": 1.0e-8,
        "predicted_target_loc0": 0.12,
        "surviving_predicted_measurement_loc0": [
            {"measurement_index": 0, "predicted_loc0": 1.0}
        ],
        "profiled_native_bound": {
            "loc0": 0.01,
            "loc1": 0.2,
            "phi": 0.001,
            "theta": 1.57,
            "q_over_p_per_mev": 0.005,
        },
        "transport_branch_identity": [
            {
                "measurement_index": 0,
                "identifier": 1,
                "station": 0,
                "layer": 0,
                "side": 0,
                "projection_kind": "supporting_plane",
                "continuation_state_construction": "sequential",
                "ok": True,
                "measurement_z_mm": 0.0,
            }
        ],
        "target_station_measurements_used": 0,
        "held_out_used_for_solver": False,
        "ridge_added": False,
        "prior_term_present": False,
    }
    row.update(overrides)
    return row


def _four(base=None, **per):
    variants = ("nominal", "loc1_plus_1mm", "phi_plus_1e-3", "qoverp_times_1p1")
    rows = []
    for name in variants:
        payload = dict(base or {})
        payload.update(per)
        payload["profile_init_variant"] = name
        rows.append(_row(**payload))
    return rows


def test_stationarity_required_before_invariance():
    rows = _four(chi2_prof=70000.0, chi2=70000.0)
    for row in rows:
        row["valid_stationary"] = False
        row["valid_solution"] = False
        row["norm_g_alpha_prof_z"] = 1.0e5
        row["termination_reason"] = "globalization_failure"
    contract = {
        "chi2_rel_tolerance": 0.01,
        "prediction_abs_tolerance_mm": 0.1,
        "supported_abs_tolerance": {"loc0_mm": 0.05, "theta": 1.0e-4},
    }
    audit = audit_identity((100043, 37, 1), rows, contract, {"gradient_norm_z": 1e-6})
    assert audit["all_stationary"] is False
    assert audit["objective_invariance"] is False
    assert audit["case"] == CASE_B
    assert audit["stall_not_escaped"] is True


def test_inner_failure_is_case_c():
    rows = _four()
    for row in rows:
        row["inner_nuisance_stationary"] = False
        row["valid_stationary"] = False
        row["valid_solution"] = False
        row["termination_reason"] = "inner_nuisance_profile_not_converged"
        row["norm_g_n_R"] = 12.0
    contract = {
        "chi2_rel_tolerance": 0.01,
        "prediction_abs_tolerance_mm": 0.1,
        "supported_abs_tolerance": {"loc0_mm": 0.05, "theta": 1.0e-4},
    }
    assert (
        audit_identity((100043, 0, 1), rows, contract, {"gradient_norm_z": 1e-6})["case"]
        == CASE_C
    )


def test_restart_sensitive_after_stationarity_is_case_d():
    rows = _four()
    rows[0]["chi2_prof"] = 90.0
    rows[1]["chi2_prof"] = 200.0
    rows[2]["chi2_prof"] = 90.0
    rows[3]["chi2_prof"] = 90.0
    contract = {
        "chi2_rel_tolerance": 0.01,
        "prediction_abs_tolerance_mm": 0.1,
        "supported_abs_tolerance": {"loc0_mm": 0.05, "theta": 1.0e-4},
    }
    assert (
        audit_identity((100043, 0, 1), rows, contract, {"gradient_norm_z": 1e-6})["case"]
        == CASE_D
    )


def test_case_e_is_pass_compatible():
    rows = _four()
    rows[1]["profiled_native_bound"]["loc1"] = 1.7
    contract = {
        "chi2_rel_tolerance": 0.01,
        "prediction_abs_tolerance_mm": 0.1,
        "supported_abs_tolerance": {"loc0_mm": 0.05, "theta": 1.0e-4},
    }
    audit = audit_identity((100043, 0, 1), rows, contract, {"gradient_norm_z": 1e-6})
    assert audit["case"] == CASE_E
    assert audit["all_stationary"] is True
    assert audit["objective_invariance"] is True
    assert audit["prediction_invariance"] is True


def test_overall_regression_only_if_gates_pass():
    def item(case, role="recontract"):
        return {"case": case, "present": True, "target_leakage": False, "role": role}

    per = {
        "100043/0 T1": item(CASE_D, "regression_control"),
        "100043/0 T2": item(CASE_A),
        "100043/0 T3": item(CASE_A),
        "100043/1 T1": item(CASE_A, "regression_control"),
        "100043/1 T2": item(CASE_A),
        "100043/1 T3": item(CASE_A),
        "100043/37 T1": item(CASE_A, "gate"),
        "100043/37 T2": item(CASE_A, "gate"),
        "100043/37 T3": item(CASE_A),
        "100048/86 T1": item(CASE_A, "gate"),
        "100048/86 T2": item(CASE_A, "regression_control"),
        "100048/86 T3": item(CASE_A),
    }
    decision = classify_overall(per, complete=True, leaked=False)
    assert decision["decision"] == CASE_F
    per["100043/0 T1"] = item(CASE_A, "regression_control")
    per["100043/37 T1"] = item(CASE_B, "gate")
    per["100043/37 T2"] = item(CASE_B, "gate")
    per["100048/86 T1"] = item(CASE_B, "gate")
    decision = classify_overall(per, complete=True, leaked=False)
    assert decision["decision"] == CASE_B
    per["100043/37 T3"] = item(CASE_E)
    per["100043/37 T1"] = item(CASE_A, "gate")
    per["100043/37 T2"] = item(CASE_A, "gate")
    per["100048/86 T1"] = item(CASE_A, "gate")
    decision = classify_overall(per, complete=True, leaked=False)
    assert decision["decision"] == CASE_A
    assert decision["verdict"] == "PASS"
    assert decision["pass_compatible_nonidentifiability"] is True
    for key in per:
        per[key] = item(CASE_C, per[key]["role"])
    decision = classify_overall(per, complete=True, leaked=False)
    assert decision["decision"] == CASE_C
