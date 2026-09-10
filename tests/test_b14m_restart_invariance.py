"""Task B14M-R restart-invariance contract.  No 1989, no B15, no retune."""

from __future__ import annotations

import pytest

from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.certified_mean_common_grid_fd import CASE_ESTABLISHED as WB128
from datasets.surface_energy_loss_mean_semantics import (
    CASE_ESTABLISHED as WB127,
    PRODUCTION_ELOSS,
)
from datasets.shadow_mean_transport_contract import FROZEN_FIELD_GRADIENT_SHA
from datasets.b14m_restart_invariance import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    CASE_E,
    CASE_F,
    REQUIRED_EVENTS,
    B14MRestartInvarianceError,
    audit_identity,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    recover_optimizer_contract,
    refuse_best_seed,
    refuse_blame_86_fd,
    refuse_b15,
    refuse_drop_37,
    refuse_drop_86,
    refuse_full_sample,
    refuse_hop6,
    refuse_prior,
    refuse_replace_44,
    refuse_retune,
    refuse_ridge,
    refuse_wb130,
)


def test_config_inherits_wb128_and_recovers_optimizer():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    contract = recover_optimizer_contract(config)
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is False
    assert config["do_not_enter_b14mr"] is False
    assert config["do_not_submit_1989"] is True
    assert config["do_not_invent_optimizer_tolerance"] is True
    assert config["do_not_drop_event_37"] is True
    assert config["do_not_drop_event_86"] is True
    assert config["profile_smoke_root"] == (
        "outputs/leave_target_out_dump_v1/b14m_reopen_smoke"
    )
    assert "b14m_smoke" in str(config["legacy_b14m_smoke_root"])
    spec = config["b14m_restart_invariance"]
    assert [tuple(item) for item in spec["smoke_events"]] == list(REQUIRED_EVENTS)
    assert inherited["workbook_128"]["decision"] == WB128
    assert inherited["workbook_127"]["decision"] == WB127
    assert inherited["workbook_128"]["jacobian_contract_established"] is True
    assert inherited["workbook_128"]["b14m_reopen_authorized"] is True
    assert inherited["workbook_128"]["full_sample_authorized"] is False
    assert inherited["workbook_103"]["contracted_denominator"] == 1989
    assert inherited["current"]["full_sample_authorized"] is False
    assert inherited["workbook_128"]["production_eloss_quantity"] == PRODUCTION_ELOSS
    assert FROZEN_FIELD_GRADIENT_SHA.startswith("ca5e4f0e")
    assert contract["recovered"] is True
    assert contract["optimizer_contract_not_recoverable"] is False
    assert contract["chi2_rel_tolerance"] == 0.01
    assert contract["prediction_abs_tolerance_mm"] == 0.1
    assert contract["supported_abs_tolerance"]["loc0_mm"] == 0.05
    assert contract["supported_abs_tolerance"]["theta"] == 0.0001
    assert contract["max_iterations"] == 50
    assert contract["pinv_relative"] == 1.0e-8
    assert contract["line_search"] == [
        1.0,
        0.5,
        0.25,
        0.125,
        0.0625,
        0.03125,
        0.015625,
        0.0078125,
    ]


def test_forbidden_repairs():
    with pytest.raises(B14MRestartInvarianceError, match="prior"):
        refuse_prior()
    with pytest.raises(B14MRestartInvarianceError, match="ridge"):
        refuse_ridge()
    with pytest.raises(B14MRestartInvarianceError, match="B15"):
        refuse_b15()
    with pytest.raises(B14MRestartInvarianceError, match="1989"):
        refuse_full_sample()
    with pytest.raises(B14MRestartInvarianceError, match="WB130"):
        refuse_wb130()
    with pytest.raises(B14MRestartInvarianceError, match="37"):
        refuse_drop_37()
    with pytest.raises(B14MRestartInvarianceError, match="86"):
        refuse_drop_86()
    with pytest.raises(B14MRestartInvarianceError, match="44"):
        refuse_replace_44()
    with pytest.raises(B14MRestartInvarianceError, match="hit 6"):
        refuse_hop6()
    with pytest.raises(B14MRestartInvarianceError, match="best restart"):
        refuse_best_seed()
    with pytest.raises(B14MRestartInvarianceError, match="Gauss-Newton"):
        refuse_retune()
    with pytest.raises(B14MRestartInvarianceError, match="WB128"):
        refuse_blame_86_fd()


def test_likelihood_contract_unchanged():
    contract = likelihood_contract()
    assert contract["prior"] is None
    assert contract["ridge"] is None
    assert contract["q_over_p_is_explicit_nuisance"] is True
    assert contract["validated_jacobian_is_not_a_new_statistical_model"] is True
    assert contract["do_not_submit_1989"] is True


def _row(variant: str, chi2: float, loc0: float, theta: float, loc1: float, pred: float):
    return {
        "profile_init_variant": variant,
        "chi2_prof": chi2,
        "predicted_target_loc0": pred,
        "profiled_native_state": [loc0, loc1, 0.1, theta, 0.002],
        "surviving_predicted_measurement_loc0": [
            {"measurement_index": 0, "predicted_loc0": pred + 0.01, "ok": True}
        ],
        "transport_branch_identity": [
            {
                "measurement_index": 0,
                "identifier": "a",
                "station": 0,
                "layer": 0,
                "side": 0,
                "projection_kind": "supporting_plane",
                "continuation_state_construction": "transform_free_to_bound",
                "ok": True,
                "measurement_z_mm": 0.0,
            }
        ],
        "propagation_success": True,
        "termination_reason": "converged",
        "target_station_measurements_used": 0,
        "target_exclusion_proven": True,
        "hessian_rank": 3,
    }


def test_case_c_nuisance_nonunique_prediction_stable():
    contract = {
        "chi2_rel_tolerance": 0.01,
        "prediction_abs_tolerance_mm": 0.1,
        "supported_abs_tolerance": {"loc0_mm": 0.05, "theta": 0.0001},
    }
    rows = [
        _row("nominal", 12.0, 0.1, 0.02, 1.0, 3.0),
        _row("loc1_plus_1mm", 12.0, 0.1, 0.02, 4.0, 3.0),
        _row("phi_plus_1e-3", 12.01, 0.1, 0.02, -2.0, 3.02),
        _row("qoverp_times_1p1", 12.0, 0.1, 0.02, 8.0, 3.01),
    ]
    audited = audit_identity(rows, contract)
    assert audited["objective_invariance"] is True
    assert audited["prediction_invariance"] is True
    assert audited["nuisance_nonunique"] is True
    assert audited["parameter_invariance"] is False
    inventory = {
        "smoke_present": True,
        "identities": [
            {
                **audited,
                "event": "100043/0",
                "transport_branch_invariance": True,
                "optimization_valid": True,
            }
        ]
        * 12,
        "schur": [
            {"event": "100043/0", "present": True, "passed": True},
            {"event": "100048/86", "present": True, "passed": True},
        ],
        "exclusion": {"target_exclusion_holds": True},
        "forbidden_jacobian_used": False,
        "prior_introduced": False,
        "ridge_added": False,
        "full_sample_submitted": False,
    }
    decision = decide_case(inventory, {"optimizer_contract_not_recoverable": False})
    assert decision["decision"] == CASE_C
    assert decision["restart_invariance_established"] is True
    assert decision["full_sample_authorized"] is False


def test_case_d_prediction_moves_with_nuisance():
    contract = {
        "chi2_rel_tolerance": 0.01,
        "prediction_abs_tolerance_mm": 0.1,
        "supported_abs_tolerance": {"loc0_mm": 0.05, "theta": 0.0001},
    }
    rows = [
        _row("nominal", 12.0, 0.1, 0.02, 1.0, 3.0),
        _row("loc1_plus_1mm", 12.0, 0.1, 0.02, 4.0, 3.5),
        _row("phi_plus_1e-3", 12.0, 0.1, 0.02, -2.0, 2.4),
        _row("qoverp_times_1p1", 12.0, 0.1, 0.02, 8.0, 3.0),
    ]
    audited = audit_identity(rows, contract)
    assert audited["objective_invariance"] is True
    assert audited["prediction_invariance"] is False
    inventory = {
        "smoke_present": True,
        "identities": [
            {
                **audited,
                "event": "100043/0",
                "transport_branch_invariance": True,
                "optimization_valid": True,
            }
        ]
        * 12,
        "schur": [
            {"event": "100043/0", "present": True, "passed": True},
            {"event": "100048/86", "present": True, "passed": True},
        ],
        "exclusion": {"target_exclusion_holds": True},
        "forbidden_jacobian_used": False,
        "prior_introduced": False,
        "ridge_added": False,
        "full_sample_submitted": False,
    }
    decision = decide_case(inventory, {"optimizer_contract_not_recoverable": False})
    assert decision["decision"] == CASE_D
    assert decision["restart_invariance_established"] is False


def test_case_b_and_e_and_f():
    empty = {
        "objective_invariance": False,
        "prediction_invariance": False,
        "transport_branch_invariance": True,
        "optimization_valid": True,
        "parameter_invariance": False,
        "nuisance_nonunique": False,
        "event": "100043/0",
    }
    base = {
        "smoke_present": True,
        "identities": [empty] * 12,
        "schur": [
            {"event": "100043/0", "present": True, "passed": True},
            {"event": "100048/86", "present": True, "passed": True},
        ],
        "exclusion": {"target_exclusion_holds": True},
        "forbidden_jacobian_used": False,
        "prior_introduced": False,
        "ridge_added": False,
        "full_sample_submitted": False,
    }
    assert decide_case(base, {"optimizer_contract_not_recoverable": False})["decision"] == CASE_B
    empty_e = {**empty, "objective_invariance": True, "transport_branch_invariance": False}
    base_e = {**base, "identities": [empty_e] * 12}
    assert decide_case(base_e, {"optimizer_contract_not_recoverable": False})["decision"] == CASE_E
    empty_ok = {
        **empty,
        "objective_invariance": True,
        "prediction_invariance": True,
        "parameter_invariance": True,
    }
    base_f = {
        **base,
        "identities": [empty_ok] * 12,
        "schur": [
            {"event": "100043/0", "present": True, "passed": False},
            {"event": "100048/86", "present": True, "passed": True},
        ],
    }
    assert decide_case(base_f, {"optimizer_contract_not_recoverable": False})["decision"] == CASE_F
    base_a = {**base, "identities": [empty_ok] * 12}
    assert decide_case(base_a, {"optimizer_contract_not_recoverable": False})["decision"] == CASE_A
