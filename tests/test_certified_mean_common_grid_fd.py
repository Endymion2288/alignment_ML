"""Task B14ZC certified-mean common-grid FD.  No B15, no 1989, no mean rewrite."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.field_gradient_variational_repair import CASE_FOCUS_UNRESOLVED as WB123
from datasets.focus86_common_grid_shadow import CASE_MEAN_NOT_EST as WB125
from datasets.shadow_mean_transport_contract import CASE_ELOSS as WB126
from datasets.surface_energy_loss_mean_semantics import CASE_ESTABLISHED as WB127
from datasets.certified_mean_common_grid_fd import (
    CASE_ESTABLISHED,
    CASE_FIELD,
    CASE_INCONSISTENT,
    CASE_MATERIAL,
    CASE_NUMERICAL,
    FOCUS_FIRST_UNSTABLE,
    FROZEN_FIELD_GRADIENT_SHA,
    PRODUCTION_ELOSS,
    CertifiedMeanCommonGridFdError,
    audit_required_segment,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    refuse_add_fd_rung,
    refuse_b14m,
    refuse_b15,
    refuse_change_mean,
    refuse_change_repair,
    refuse_full_sample,
    refuse_prior,
    refuse_relax_gate,
    refuse_substitute_hit6,
    refuse_tune_dopri5,
    refuse_tune_grid,
)


def test_config_inherits_wb127_and_keeps_mean():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14zb"] is True
    assert config["do_not_enter_b14zc"] is False
    spec = config["certified_mean_common_grid_fd"]
    assert {int(k): int(v) for k, v in spec["required_first_unstable"].items()} == {
        1: 6,
        2: 11,
        3: 11,
    }
    assert spec["production_eloss_quantity"] == PRODUCTION_ELOSS
    assert spec["do_not_change_mean_semantics"] is True
    assert spec["do_not_reuse_wb124_adaptive_dopri5"] is True
    assert spec["do_not_submit_1989"] is True
    assert float(spec["official_step_tolerance"]) == 1.0e-4
    assert inherited["workbook_127"]["decision"] == WB127
    assert inherited["workbook_126"]["decision"] == WB126
    assert inherited["workbook_125"]["decision"] == WB125
    assert inherited["workbook_123"]["decision"] == WB123
    assert inherited["workbook_127"]["shadow_mean_contract_established"] is True
    assert inherited["workbook_127"]["jacobian_contract_established"] is False
    assert inherited["workbook_127"]["b14m_reopen_authorized"] is False
    assert FROZEN_FIELD_GRADIENT_SHA.startswith("ca5e4f0e")


def test_forbidden_repairs():
    with pytest.raises(CertifiedMeanCommonGridFdError, match="prior"):
        refuse_prior()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="FD rung"):
        refuse_add_fd_rung()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="5%"):
        refuse_relax_gate()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="common grid"):
        refuse_tune_grid()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="DOPRI5"):
        refuse_tune_dopri5()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="field-gradient"):
        refuse_change_repair()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="mean semantics"):
        refuse_change_mean()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="hit 6"):
        refuse_substitute_hit6()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="B14M"):
        refuse_b14m()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="B15"):
        refuse_b15()
    with pytest.raises(CertifiedMeanCommonGridFdError, match="1989"):
        refuse_full_sample()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_keeps_mean_and_forbids_1989():
    contract = likelihood_contract()
    assert contract["do_not_change_mean_semantics"] is True
    assert contract["do_not_submit_1989"] is True
    assert contract["do_not_reuse_wb124_adaptive_dopri5"] is True
    assert contract["production_eloss_quantity"] == PRODUCTION_ELOSS
    assert contract["R_i"] == "(0.08 mm)^2 / 12"


def _col(converged: bool, finest: float = 1.0) -> dict:
    return {
        "parameter": "x",
        "converged": converged,
        "sign_consistent": converged,
        "last_pair_rel": 0.01 if converged else 0.2,
        "signed_h": finest,
        "signed_h2": finest,
        "signed_h4": finest,
        "signed_h8": finest,
        "rungs": [
            {"dloc0_d_start": finest, "step_factor": f}
            for f in (1.0, 0.5, 0.25, 0.125)
        ],
    }


def _seg(closed: bool, c_ok: bool, taxonomy: str, agrees: bool | None = None) -> dict:
    return {
        "mean_contract_closed": closed,
        "c_required_converged": c_ok,
        "independent_reference_established": closed and c_ok and bool(agrees),
        "repaired_agrees_independent": bool(agrees),
        "repaired_inconsistent": c_ok and agrees is False,
        "taxonomy": taxonomy,
    }


def test_decide_case_paths():
    field_fail = decide_case(
        {
            "focus_segments": [_seg(True, False, CASE_FIELD, agrees=False)] * 3,
            "controls": [
                {"event": "100043/0", "five_percent_pass": True},
                {"event": "100043/1", "five_percent_pass": True, "loc1_five_percent_pass": True},
                {"event": "100043/37", "five_percent_pass": True},
            ],
            "invariance": [{"mean_path_unchanged": True}] * 3,
        }
    )
    assert field_fail["decision"] == CASE_FIELD
    assert field_fail["shadow_mean_contract_established"] is True
    assert field_fail["focus_independent_reference_established"] is False
    assert field_fail["jacobian_contract_established"] is False
    assert field_fail["b14m_reopen_authorized"] is False

    inconsistent = decide_case(
        {
            "focus_segments": [_seg(True, True, CASE_INCONSISTENT, agrees=False)] * 3,
            "controls": [
                {"event": "100043/0", "five_percent_pass": True},
                {"event": "100043/1", "five_percent_pass": True, "loc1_five_percent_pass": True},
                {"event": "100043/37", "five_percent_pass": True},
            ],
            "invariance": [{"mean_path_unchanged": True}] * 3,
        }
    )
    assert inconsistent["decision"] == CASE_INCONSISTENT
    assert inconsistent["b14m_reopen_authorized"] is False

    cert = decide_case(
        {
            "focus_segments": [_seg(True, True, CASE_ESTABLISHED, agrees=True)] * 3,
            "controls": [
                {"event": "100043/0", "five_percent_pass": True},
                {"event": "100043/1", "five_percent_pass": True, "loc1_five_percent_pass": True},
                {"event": "100043/37", "five_percent_pass": True},
            ],
            "invariance": [{"mean_path_unchanged": True}] * 3,
        }
    )
    assert cert["decision"] == CASE_ESTABLISHED
    assert cert["verdict"] == "PASS"
    assert cert["focus_independent_reference_established"] is True
    assert cert["jacobian_contract_established"] is True
    assert cert["b14m_reopen_authorized"] is True
    assert cert["full_sample_authorized"] is False
    assert cert["restart_invariance_authorized"] is False


def test_required_hop_is_not_earlier_hit6():
    hop6 = {
        "measurement_index": 6,
        "shadow_p_residual": {"closed": True},
        "certified_mean_common_grid_fd": {
            "all_arms_same_mean_contract": True,
            "all_branch_identity_same": True,
            "all_material_node_identity_same": True,
            "shadow_fd": {"columns": []},
        },
        "hop_start_state": {"bound": [0, 0, 0, 0, 0]},
    }
    hop11 = {
        "measurement_index": 11,
        "shadow_p_residual": {"closed": True},
        "certified_mean_common_grid_fd": {
            "all_arms_same_mean_contract": True,
            "all_branch_identity_same": True,
            "all_material_node_identity_same": False,
            "any_projection_fail": False,
            "n_dot_direction": 0.9,
            "shadow_fd": {
                "columns": [
                    {**_col(False, 1.0), "parameter": name}
                    for name in ("loc0", "loc1", "phi", "theta", "q_over_p")
                ]
            },
        },
        "hop_start_state": {"bound": [0, 0, 0, 0, 0]},
    }
    row = {
        "run_id": 100048,
        "event_id": 86,
        "target_station": 2,
        "certified_mean_common_grid_fd": {"hops": [hop6, hop11]},
        "official_supporting_plane_jacobian": {
            "official_path_jacobian": {
                "hops": [
                    {
                        "measurement_index": 11,
                        "field_gradient_repair": {
                            "hop_dloc0_d_start": [1.0, 1.0, 1.0, 1.0, 1.0]
                        },
                    }
                ]
            }
        },
    }
    audited = audit_required_segment(row)
    assert audited["required_index"] == 11
    assert audited["measurement_index"] == 11
    assert audited["do_not_substitute_earlier_long_hop"] is True
    assert FOCUS_FIRST_UNSTABLE[2] == 11
    assert audited["taxonomy"] in {CASE_MATERIAL, CASE_FIELD, CASE_NUMERICAL}
    assert audited["independent_reference_established"] is False
