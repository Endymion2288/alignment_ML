"""Task B14ZB surface energy-loss semantics.  No B15, no V2, no B14M, no Jacobian."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.field_gradient_variational_repair import CASE_FOCUS_UNRESOLVED as WB123
from datasets.focus86_segment_derivative_reference import CASE_REF_NOT_EST as WB124
from datasets.focus86_common_grid_shadow import CASE_MEAN_NOT_EST as WB125
from datasets.shadow_mean_transport_contract import CASE_ELOSS as WB126
from datasets.surface_energy_loss_mean_semantics import (
    CASE_ESTABLISHED,
    CASE_FORMULA,
    CASE_GATING,
    CASE_PATH,
    CASE_QOP,
    FOCUS_FIRST_UNSTABLE,
    FROZEN_FIELD_GRADIENT_SHA,
    MEAN_LOC0,
    PRODUCTION_ELOSS,
    SurfaceEnergyLossSemanticsError,
    audit_required_segment,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    refuse_add_fd_rung,
    refuse_b14m,
    refuse_b15,
    refuse_change_repair,
    refuse_evaluate_jacobian,
    refuse_fit_eloss,
    refuse_full_sample,
    refuse_infer_loc0,
    refuse_prior,
    refuse_relax_gate,
    refuse_substitute_hit6,
)


def test_config_inherits_wb126_and_forbids_jacobian():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14za"] is True
    assert config["do_not_enter_b14zb"] is False
    spec = config["surface_energy_loss_mean_semantics"]
    assert {int(k): int(v) for k, v in spec["required_first_unstable"].items()} == {
        1: 6,
        2: 11,
        3: 11,
    }
    assert spec["required_first_unstable"] or FOCUS_FIRST_UNSTABLE
    assert spec["production_eloss_quantity"] == PRODUCTION_ELOSS
    assert spec["derivative_not_evaluated"] is True
    assert spec["do_not_fit_eloss_to_endpoint"] is True
    assert spec["do_not_infer_eloss_from_loc0"] is True
    assert float(spec["mean_gates"]["loc0_abs_mm"]) == MEAN_LOC0
    assert float(spec["official_step_tolerance"]) == 1.0e-4
    assert inherited["workbook_126"]["decision"] == WB126
    assert inherited["workbook_125"]["decision"] == WB125
    assert inherited["workbook_124"]["decision"] == WB124
    assert inherited["workbook_123"]["decision"] == WB123
    assert inherited["workbook_126"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_126"]["jacobian_contract_established"] is False
    assert FROZEN_FIELD_GRADIENT_SHA.startswith("ca5e4f0e")


def test_forbidden_repairs():
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="prior"):
        refuse_prior()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="FD rung"):
        refuse_add_fd_rung()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="5%"):
        refuse_relax_gate()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="field-gradient"):
        refuse_change_repair()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="Jacobian"):
        refuse_evaluate_jacobian()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="Eloss"):
        refuse_fit_eloss()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="loc0"):
        refuse_infer_loc0()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="hit 6"):
        refuse_substitute_hit6()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="B14M"):
        refuse_b14m()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="B15"):
        refuse_b15()
    with pytest.raises(SurfaceEnergyLossSemanticsError, match="1989"):
        refuse_full_sample()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_keeps_mean_and_forbids_jacobian():
    contract = likelihood_contract()
    assert contract["do_not_change_official_mean_path"] is True
    assert contract["do_not_evaluate_jacobian"] is True
    assert contract["do_not_fit_eloss_to_endpoint"] is True
    assert contract["production_eloss_quantity"] == PRODUCTION_ELOSS
    assert contract["R_i"] == "(0.08 mm)^2 / 12"


def _seg(
    closed: bool,
    taxonomy: str,
    node_closed: bool | None = None,
) -> dict:
    return {
        "mean_contract_closed": closed,
        "node_qop_all_closed": closed if node_closed is None else node_closed,
        "taxonomy": taxonomy,
        "shadow_p_residual": {"closed": closed, "delta_loc0": 0.0 if closed else 0.015},
    }


def test_decide_case_paths():
    path_fail = decide_case(
        {
            "focus_segments": [_seg(False, CASE_PATH, node_closed=True)] * 3,
            "mean_path_unchanged": True,
        }
    )
    assert path_fail["decision"] == CASE_PATH
    assert path_fail["shadow_mean_contract_established"] is False
    assert path_fail["focus_independent_reference_established"] is False
    assert path_fail["jacobian_contract_established"] is False
    assert path_fail["b14m_reopen_authorized"] is False

    mixed = decide_case(
        {
            "focus_segments": [
                _seg(False, CASE_GATING),
                _seg(False, CASE_FORMULA),
                _seg(True, CASE_ESTABLISHED),
            ],
            "mean_path_unchanged": True,
        }
    )
    assert mixed["decision"] == "mixed_or_inconclusive"
    assert mixed["b14m_reopen_authorized"] is False

    cert = decide_case(
        {
            "focus_segments": [_seg(True, CASE_ESTABLISHED)] * 3,
            "mean_path_unchanged": True,
        }
    )
    assert cert["decision"] == CASE_ESTABLISHED
    assert cert["verdict"] == "PASS"
    assert cert["shadow_mean_contract_established"] is True
    assert cert["focus_independent_reference_established"] is False
    assert cert["jacobian_contract_established"] is False
    assert cert["b14m_reopen_authorized"] is False
    assert cert["full_sample_authorized"] is False


def test_required_hop_is_not_earlier_hit6():
    hop6 = {
        "measurement_index": 6,
        "shadow_p_residual": {"closed": True, "delta_loc0": 0.0},
        "node_qop_all_closed": True,
        "hop_start_state": {"bound": [0, 0, 0, 0, 0]},
        "derivative_not_evaluated": True,
    }
    hop11 = {
        "measurement_index": 11,
        "shadow_p_residual": {
            "closed": False,
            "delta_loc0": 0.00131,
            "delta_qop_rel": 0.00454,
        },
        "node_qop_all_closed": False,
        "hop_start_state": {"bound": [0, 0, 0, 0, 0]},
        "first_divergence": {"found": True, "event_kind": "surface_material"},
        "surface_energy_loss_mean_contract": {
            "n_surfaces": 4,
            "n_match_computeEnergyLossBethe": 0,
            "n_match_computeEnergyLossMean": 0,
            "surfaces": [
                {
                    "update_state_called": True,
                    "has_surface_material_pointer": True,
                    "delta_qop_production": -1.0e-5,
                    "production_matches_bethe_formula": False,
                    "production_matches_evaluatePointwise": False,
                    "production_matches_mean_formula": True,
                    "production_matches_mode_formula": False,
                    "node_qop_closed": False,
                }
            ],
        },
        "derivative_not_evaluated": True,
    }
    row = {
        "run_id": 100048,
        "event_id": 86,
        "target_station": 2,
        "surface_energy_loss_mean_semantics": {"hops": [hop6, hop11]},
    }
    audited = audit_required_segment(row)
    assert audited["required_index"] == 11
    assert audited["measurement_index"] == 11
    assert audited["mean_contract_closed"] is False
    assert audited["do_not_substitute_earlier_long_hop"] is True
    assert audited["taxonomy"] == CASE_FORMULA


def test_path_after_node_closure():
    hop = {
        "measurement_index": 6,
        "shadow_p_residual": {
            "closed": False,
            "delta_loc0": 0.0169,
            "delta_path": -6.113,
        },
        "node_qop_all_closed": True,
        "node_qop_contract": {"all_closed": True},
        "surface_energy_loss_mean_contract": {
            "surfaces": [
                {
                    "update_state_called": True,
                    "has_surface_material_pointer": True,
                    "delta_qop_production": -6.5e-5,
                    "production_matches_bethe_formula": True,
                    "production_matches_evaluatePointwise": True,
                    "node_qop_closed": True,
                }
            ]
        },
        "hop_start_state": {"bound": [0, 0, 0, 0, 0]},
        "derivative_not_evaluated": True,
    }
    row = {
        "run_id": 100048,
        "event_id": 86,
        "target_station": 1,
        "surface_energy_loss_mean_semantics": {"hops": [hop]},
    }
    audited = audit_required_segment(row)
    assert audited["taxonomy"] == CASE_PATH
    assert audited["node_qop_all_closed"] is True
    assert CASE_QOP != audited["taxonomy"]
