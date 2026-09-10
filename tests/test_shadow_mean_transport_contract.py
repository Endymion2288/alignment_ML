"""Task B14ZA mean transport contract.  No B15, no V2, no B14M, no Jacobian."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.field_gradient_variational_repair import CASE_FOCUS_UNRESOLVED as WB123
from datasets.focus86_segment_derivative_reference import CASE_REF_NOT_EST as WB124
from datasets.focus86_common_grid_shadow import CASE_MEAN_NOT_EST as WB125
from datasets.shadow_mean_transport_contract import (
    CASE_ESTABLISHED,
    CASE_PHYSICS,
    CASE_ELOSS,
    CASE_QOP,
    CASE_RESOLVE,
    FOCUS_FIRST_UNSTABLE,
    FROZEN_FIELD_GRADIENT_SHA,
    MEAN_LOC0,
    SHADOW_D_STEPS,
    ShadowMeanTransportError,
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
    refuse_full_sample,
    refuse_prior,
    refuse_relax_gate,
    refuse_substitute_hit6,
    refuse_tune_dopri5,
    refuse_tune_grid,
)


def test_config_inherits_wb125_and_mean_only_grid():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14z"] is True
    assert config["do_not_enter_b14za"] is False
    spec = config["shadow_mean_transport_contract"]
    assert {int(k): int(v) for k, v in spec["required_first_unstable"].items()} == {
        1: 6,
        2: 11,
        3: 11,
    }
    assert spec["required_first_unstable"] or FOCUS_FIRST_UNSTABLE
    assert tuple(spec["shadow_d"]["sequence_mm"]) == SHADOW_D_STEPS
    assert spec["derivative_not_evaluated"] is True
    assert spec["jacobian_agreement_not_read"] is True
    assert spec["grid_selected_from_mean_contract_only"] is True
    assert float(spec["mean_gates"]["loc0_abs_mm"]) == MEAN_LOC0
    assert float(spec["official_step_tolerance"]) == 1.0e-4
    assert inherited["workbook_125"]["decision"] == WB125
    assert inherited["workbook_124"]["decision"] == WB124
    assert inherited["workbook_123"]["decision"] == WB123
    assert inherited["workbook_125"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_125"]["jacobian_contract_established"] is False
    assert FROZEN_FIELD_GRADIENT_SHA.startswith("ca5e4f0e")


def test_forbidden_repairs():
    with pytest.raises(ShadowMeanTransportError, match="prior"):
        refuse_prior()
    with pytest.raises(ShadowMeanTransportError, match="FD rung"):
        refuse_add_fd_rung()
    with pytest.raises(ShadowMeanTransportError, match="5%"):
        refuse_relax_gate()
    with pytest.raises(ShadowMeanTransportError, match="Jacobian agreement"):
        refuse_tune_grid()
    with pytest.raises(ShadowMeanTransportError, match="DOPRI5"):
        refuse_tune_dopri5()
    with pytest.raises(ShadowMeanTransportError, match="field-gradient"):
        refuse_change_repair()
    with pytest.raises(ShadowMeanTransportError, match="Jacobian"):
        refuse_evaluate_jacobian()
    with pytest.raises(ShadowMeanTransportError, match="hit 6"):
        refuse_substitute_hit6()
    with pytest.raises(ShadowMeanTransportError, match="B14M"):
        refuse_b14m()
    with pytest.raises(ShadowMeanTransportError, match="B15"):
        refuse_b15()
    with pytest.raises(ShadowMeanTransportError, match="1989"):
        refuse_full_sample()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_keeps_mean_and_forbids_jacobian():
    contract = likelihood_contract()
    assert contract["do_not_change_official_mean_path"] is True
    assert contract["do_not_relax_five_percent_gate"] is True
    assert contract["do_not_evaluate_jacobian"] is True
    assert contract["do_not_self_certify_analytic"] is True
    assert contract["do_not_substitute_earlier_long_hop"] is True
    assert contract["R_i"] == "(0.08 mm)^2 / 12"


def _seg(closed: bool, taxonomy: str = CASE_PHYSICS) -> dict:
    return {
        "mean_contract_closed": closed,
        "taxonomy": taxonomy,
        "shadow_p_residual": {"closed": closed, "delta_loc0": 0.0 if closed else 0.015},
    }


def test_decide_case_paths():
    mean_fail = decide_case(
        {
            "focus_segments": [_seg(False, CASE_PHYSICS)] * 3,
            "mean_path_unchanged": True,
        }
    )
    assert mean_fail["decision"] == CASE_PHYSICS
    assert mean_fail["focus_independent_reference_established"] is False
    assert mean_fail["jacobian_contract_established"] is False
    assert mean_fail["b14m_reopen_authorized"] is False

    mixed = decide_case(
        {
            "focus_segments": [
                _seg(False, CASE_QOP),
                _seg(False, CASE_RESOLVE),
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
    assert cert["focus_independent_reference_established"] is False
    assert cert["jacobian_contract_established"] is False
    assert cert["b14m_reopen_authorized"] is False
    assert cert["full_sample_authorized"] is False


def test_required_hop_is_not_earlier_hit6():
    hop6 = {
        "measurement_index": 6,
        "shadow_p_residual": {"closed": True, "delta_loc0": 0.0},
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
        "hop_start_state": {"bound": [0, 0, 0, 0, 0]},
        "first_divergence": {"found": True, "event_kind": "surface_material"},
        "surface_energy_loss_mean_contract": {
            "n_surfaces": 4,
            "n_match_computeEnergyLossMean": 0,
        },
        "qop_energy_loss_unit_contract": {"mass_hypothesis_gev": 0.1057},
        "derivative_not_evaluated": True,
    }
    row = {
        "run_id": 100048,
        "event_id": 86,
        "target_station": 2,
        "shadow_mean_transport_contract": {"hops": [hop6, hop11]},
    }
    audited = audit_required_segment(row)
    assert audited["required_index"] == 11
    assert audited["measurement_index"] == 11
    assert audited["mean_contract_closed"] is False
    assert audited["do_not_substitute_earlier_long_hop"] is True
    assert audited["taxonomy"] == CASE_ELOSS
