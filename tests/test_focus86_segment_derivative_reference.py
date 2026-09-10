"""Task B14Y 86 segment reference.  No B15, no V2, no B14M, no B14X reopen."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.field_gradient_variational_repair import CASE_FOCUS_UNRESOLVED as WB123
from datasets.focus86_segment_derivative_reference import (
    CASE_FD_NOT_CERTIFYING,
    CASE_INCONSISTENT,
    CASE_REF_NOT_EST,
    FOCUS_FIRST_UNSTABLE,
    INDEP_ABS_TOL_POS,
    Focus86SegmentReferenceError,
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
    refuse_tune_independent,
)


def test_config_inherits_wb123_and_required_hops():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14x"] is True
    assert config["do_not_enter_b14y"] is False
    spec = config["focus86_segment_derivative_reference"]
    assert {int(k): int(v) for k, v in spec["required_first_unstable"].items()} == {
        1: 6,
        2: 11,
        3: 11,
    }
    assert spec["required_first_unstable"] or FOCUS_FIRST_UNSTABLE
    assert float(spec["independent_integrator"]["abs_tol_pos_mm"]) == INDEP_ABS_TOL_POS
    assert float(spec["official_step_tolerance"]) == 1.0e-4
    assert config["wb124_smoke_dumps"][0]["sha256"] == (
        "1888e9019fa5e4ce43b72bc6f6a51f9680251253ac199fb17bd01c763a5fb1f1"
    )
    assert inherited["workbook_123"]["decision"] == WB123
    assert inherited["workbook_123"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_122"]["decision"] == "missing_deterministic_transport_coupling_identified"


def test_forbidden_repairs():
    with pytest.raises(Focus86SegmentReferenceError, match="prior"):
        refuse_prior()
    with pytest.raises(Focus86SegmentReferenceError, match="FD rung"):
        refuse_add_fd_rung()
    with pytest.raises(Focus86SegmentReferenceError, match="5%"):
        refuse_relax_gate()
    with pytest.raises(Focus86SegmentReferenceError, match="independent"):
        refuse_tune_independent()
    with pytest.raises(Focus86SegmentReferenceError, match="field-gradient"):
        refuse_change_repair()
    with pytest.raises(Focus86SegmentReferenceError, match="hit 6"):
        refuse_substitute_hit6()
    with pytest.raises(Focus86SegmentReferenceError, match="B14M"):
        refuse_b14m()
    with pytest.raises(Focus86SegmentReferenceError, match="B15"):
        refuse_b15()
    with pytest.raises(Focus86SegmentReferenceError, match="1989"):
        refuse_full_sample()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_keeps_mean_and_gate():
    contract = likelihood_contract()
    assert contract["do_not_change_official_mean_path"] is True
    assert contract["do_not_relax_five_percent_gate"] is True
    assert contract["do_not_self_certify_analytic"] is True
    assert contract["do_not_substitute_earlier_long_hop"] is True
    assert contract["R_i"] == "(0.08 mm)^2 / 12"


def _seg(established: bool, inconsistent: bool = False, prod_conv: bool = False) -> dict:
    return {
        "independent_reference_established": established,
        "repaired_inconsistent": inconsistent,
        "production_fd_loc1_converged": prod_conv,
    }


def _control(event: str, passed: bool) -> dict:
    return {"event": event, "five_percent_pass": passed, "loc1_five_percent_pass": passed}


def test_decide_case_paths():
    base = {
        "prior_introduced": False,
        "ridge_as_information": False,
        "added_fd_rung": False,
        "relaxed_five_percent_gate": False,
        "changed_official_mean": False,
        "tuned_independent": False,
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
        "focus_segments": [_seg(False)] * 3,
    }
    unresolved = decide_case(base)
    assert unresolved["decision"] == CASE_REF_NOT_EST
    assert unresolved["jacobian_contract_established"] is False
    assert unresolved["b14m_reopen_authorized"] is False

    cert = decide_case({**base, "focus_segments": [_seg(True, prod_conv=False)] * 3})
    assert cert["decision"] == CASE_FD_NOT_CERTIFYING
    assert cert["focus_independent_reference_established"] is True
    assert cert["b14m_reopen_authorized"] is True

    inconsistent = decide_case(
        {**base, "focus_segments": [_seg(False, inconsistent=True)] * 3}
    )
    assert inconsistent["decision"] == CASE_INCONSISTENT
    assert inconsistent["b14m_reopen_authorized"] is False


def test_required_hop_is_not_earlier_hit6():
    hop6 = {
        "measurement_index": 6,
        "number_of_propagation_steps": 84,
        "independent_segment_reference": {"mean": {"ok": True}},
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
