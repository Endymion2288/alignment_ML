"""Task B14Z 86 common-grid shadow reference.  No B15, no V2, no B14M, no B14Y reopen."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.field_gradient_variational_repair import CASE_FOCUS_UNRESOLVED as WB123
from datasets.focus86_segment_derivative_reference import CASE_REF_NOT_EST as WB124
from datasets.focus86_common_grid_shadow import (
    CASE_ESTABLISHED,
    CASE_FIELD,
    CASE_INCONSISTENT,
    CASE_MEAN_NOT_EST,
    FOCUS_FIRST_UNSTABLE,
    FROZEN_FIELD_GRADIENT_SHA,
    MEAN_LOC0,
    SHADOW_NOM_STEP,
    Focus86CommonGridShadowError,
    audit_required_segment,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    refuse_add_fd_rung,
    refuse_b14m,
    refuse_b15,
    refuse_change_repair,
    refuse_full_sample,
    refuse_prior,
    refuse_relax_gate,
    refuse_substitute_hit6,
    refuse_tune_dopri5,
    refuse_tune_grid,
)


def test_config_inherits_wb124_and_required_hops():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14x"] is True
    assert config["do_not_enter_b14y"] is True
    assert config["do_not_enter_b14z"] is False
    spec = config["focus86_common_grid_shadow"]
    assert {int(k): int(v) for k, v in spec["required_first_unstable"].items()} == {
        1: 6,
        2: 11,
        3: 11,
    }
    assert spec["required_first_unstable"] or FOCUS_FIRST_UNSTABLE
    assert float(spec["shadow_integrator"]["nominal_max_step_mm"]) == SHADOW_NOM_STEP
    assert float(spec["shadow_integrator"]["mean_loc0_abs_mm"]) == MEAN_LOC0
    assert float(spec["official_step_tolerance"]) == 1.0e-4
    assert inherited["workbook_124"]["decision"] == WB124
    assert inherited["workbook_123"]["decision"] == WB123
    assert inherited["workbook_124"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_124"]["jacobian_contract_established"] is False
    assert FROZEN_FIELD_GRADIENT_SHA.startswith("ca5e4f0e")


def test_forbidden_repairs():
    with pytest.raises(Focus86CommonGridShadowError, match="prior"):
        refuse_prior()
    with pytest.raises(Focus86CommonGridShadowError, match="FD rung"):
        refuse_add_fd_rung()
    with pytest.raises(Focus86CommonGridShadowError, match="5%"):
        refuse_relax_gate()
    with pytest.raises(Focus86CommonGridShadowError, match="common grid"):
        refuse_tune_grid()
    with pytest.raises(Focus86CommonGridShadowError, match="DOPRI5"):
        refuse_tune_dopri5()
    with pytest.raises(Focus86CommonGridShadowError, match="field-gradient"):
        refuse_change_repair()
    with pytest.raises(Focus86CommonGridShadowError, match="hit 6"):
        refuse_substitute_hit6()
    with pytest.raises(Focus86CommonGridShadowError, match="B14M"):
        refuse_b14m()
    with pytest.raises(Focus86CommonGridShadowError, match="B15"):
        refuse_b15()
    with pytest.raises(Focus86CommonGridShadowError, match="1989"):
        refuse_full_sample()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_keeps_mean_and_gate():
    contract = likelihood_contract()
    assert contract["do_not_change_official_mean_path"] is True
    assert contract["do_not_relax_five_percent_gate"] is True
    assert contract["do_not_self_certify_analytic"] is True
    assert contract["do_not_substitute_earlier_long_hop"] is True
    assert contract["do_not_call_vacuum_a_production_map_reference"] is True
    assert contract["do_not_tune_wb124_dopri5"] is True
    assert contract["do_not_tune_grid_from_jacobian"] is True
    assert contract["R_i"] == "(0.08 mm)^2 / 12"


def _seg(
    established: bool,
    inconsistent: bool = False,
    mean_closed: bool = True,
    n_mat: int = 0,
    n_dot: float = 0.5,
    loc1_sign_change: bool = False,
    grid_mean_stable: bool = True,
) -> dict:
    return {
        "independent_reference_established": established,
        "repaired_inconsistent": inconsistent,
        "mean_contract_closed": mean_closed,
        "n_surface_material_crossings": n_mat,
        "n_dot_direction": n_dot,
        "grid_mean_stable": grid_mean_stable,
        "parameters": [
            {"parameter": "loc0", "shadow_fd_nominal": {"sign_change": False}},
            {
                "parameter": "loc1",
                "shadow_fd_nominal": {"sign_change": loc1_sign_change},
            },
        ],
    }


def _control(event: str, passed: bool) -> dict:
    return {"event": event, "five_percent_pass": passed, "loc1_five_percent_pass": passed}


def _base() -> dict:
    return {
        "prior_introduced": False,
        "ridge_as_information": False,
        "added_fd_rung": False,
        "relaxed_five_percent_gate": False,
        "tuned_grid": False,
        "tuned_dopri5": False,
        "changed_field_gradient_repair": False,
        "b14m_reopened": False,
        "substituted_hit6": False,
        "invariance": [{"mean_path_unchanged": True}] * 12,
        "controls": [
            _control("100043/0", True),
            _control("100043/0", True),
            _control("100043/0", True),
            _control("100043/1", True),
            _control("100043/1", True),
            _control("100043/1", True),
            _control("100043/37", True),
            _control("100043/37", True),
            _control("100043/37", True),
        ],
        "focus_segments": [_seg(False, mean_closed=False)] * 3,
    }


def test_decide_case_paths():
    mean_fail = decide_case(_base())
    assert mean_fail["decision"] == CASE_MEAN_NOT_EST
    assert mean_fail["jacobian_contract_established"] is False
    assert mean_fail["b14m_reopen_authorized"] is False

    unresolved = decide_case(
        {**_base(), "focus_segments": [_seg(False, mean_closed=True, n_mat=0)] * 3}
    )
    assert unresolved["decision"] == CASE_FIELD
    assert unresolved["b14m_reopen_authorized"] is False

    cert = decide_case({**_base(), "focus_segments": [_seg(True)] * 3})
    assert cert["decision"] == CASE_ESTABLISHED
    assert cert["focus_independent_reference_established"] is True
    assert cert["jacobian_contract_established"] is True
    assert cert["b14m_reopen_authorized"] is True
    assert cert["restart_invariance_authorized"] is False
    assert cert["full_sample_authorized"] is False

    inconsistent = decide_case(
        {**_base(), "focus_segments": [_seg(False, inconsistent=True)] * 3}
    )
    assert inconsistent["decision"] == CASE_INCONSISTENT
    assert inconsistent["b14m_reopen_authorized"] is False


def test_required_hop_is_not_earlier_hit6():
    hop6 = {
        "measurement_index": 6,
        "number_of_propagation_steps": 84,
        "common_grid_shadow_reference": {"mean_contract": {"closed": True}},
        "field_gradient_repair": {
            "hop_dloc0_d_start": [1.0, -0.04, 1.0, 1.0, 1.0],
            "hop_start_segment_fd": {
                "columns": [
                    {
                        "parameter": "loc1",
                        "rungs": [
                            {"dloc0_d_start": -0.04},
                            {"dloc0_d_start": -0.0401},
                            {"dloc0_d_start": -0.04005},
                            {"dloc0_d_start": -0.04004},
                        ],
                    }
                ]
            },
        },
    }
    hop11 = {
        "measurement_index": 11,
        "number_of_propagation_steps": 80,
        "hop_start_state": {"bound": [0, 0, 0, 0, 0]},
        "field_gradient_repair": {"hop_dloc0_d_start": [1.0, -0.3, 20.0, -1000.0, -200.0]},
    }
    row = {
        "run_id": 100048,
        "event_id": 86,
        "target_station": 2,
        "official_supporting_plane_jacobian": {
            "official_path_jacobian": {"hops": [hop6, hop11]}
        },
    }
    audited = audit_required_segment(row)
    assert audited["required_index"] == 11
    assert audited["measurement_index"] == 11
    assert audited["independent_reference_established"] is False
    assert audited["do_not_substitute_earlier_long_hop"] is True
    assert audited["do_not_pass_on_target3_production_loc1_alone"] is True
