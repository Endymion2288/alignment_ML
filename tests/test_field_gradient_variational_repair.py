"""Task B14X field-gradient variational repair.  No B15, no V2, no B14M."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.field_gradient_variational_repair import (
    CASE_FOCUS_UNRESOLVED,
    CASE_HYPOTHESIS,
    CASE_REPAIRED,
    FIELD_FD_STEP_MM,
    FOCUS_FIRST_UNSTABLE,
    FieldGradientVariationalRepairError,
    audit_acts_source,
    audit_equation_contract,
    audit_segment_and_focus,
    decide_case,
    inherit_frozen_stage,
    likelihood_contract,
    load_config,
    refuse_add_fd_rung,
    refuse_b14m,
    refuse_b15,
    refuse_full_sample,
    refuse_prior,
    refuse_relax_gate,
    refuse_ridge_information,
    refuse_tune_field_step,
)
from datasets.independent_derivative_reference import CASE_COUPLING as WB122_DECISION


def test_config_inherits_wb122_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14w"] is True
    assert config["do_not_enter_b14x"] is False
    spec = config["field_gradient_variational_repair"]
    assert float(spec["field_gradient_fd_step_mm"]) == FIELD_FD_STEP_MM
    assert float(spec["official_step_tolerance"]) == 1.0e-4
    assert spec["do_not_tune_field_step_from_track_jacobian"] is True
    assert inherited["workbook_122"]["decision"] == WB122_DECISION
    assert inherited["workbook_122"]["b14m_reopen_authorized"] is False
    assert inherited["workbook_121"]["decision"] == "mixed_or_inconclusive"


def test_source_audit_proves_missing_eq18():
    source = audit_acts_source()
    assert source["missing_term_source_proven"] is True
    assert source["transport_matrix_fills_dGdx"] is False
    assert source["official_gradient_api_present"] is True
    assert source["eq18_terms_currently_zero_in_acts_comment"] is True
    equation = audit_equation_contract()
    assert equation["g_equals_zero_reduces_to_acts_d"] is True
    assert equation["derived_from_pinned_acts_mean_map"] is True
    assert equation["field_gradient_fd_step_mm"] == FIELD_FD_STEP_MM


def test_forbidden_repairs():
    with pytest.raises(FieldGradientVariationalRepairError, match="prior"):
        refuse_prior()
    with pytest.raises(FieldGradientVariationalRepairError, match="ridge"):
        refuse_ridge_information()
    with pytest.raises(FieldGradientVariationalRepairError, match="FD rung"):
        refuse_add_fd_rung()
    with pytest.raises(FieldGradientVariationalRepairError, match="5%"):
        refuse_relax_gate()
    with pytest.raises(FieldGradientVariationalRepairError, match="field-gradient FD"):
        refuse_tune_field_step()
    with pytest.raises(FieldGradientVariationalRepairError, match="B14M"):
        refuse_b14m()
    with pytest.raises(FieldGradientVariationalRepairError, match="B15"):
        refuse_b15()
    with pytest.raises(FieldGradientVariationalRepairError, match="1989"):
        refuse_full_sample()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_keeps_mean_and_gate():
    contract = likelihood_contract()
    assert contract["do_not_change_official_mean_path"] is True
    assert contract["do_not_relax_five_percent_gate"] is True
    assert contract["do_not_tune_field_step_from_track_jacobian"] is True
    assert contract["R_i"] == "(0.08 mm)^2 / 12"


def _control(event: str, passed: bool, new_nz: bool = True) -> dict:
    return {
        "event": event,
        "five_percent_pass": passed,
        "new_pos_to_dir_nonzero": new_nz,
        "old_pos_to_dir_zero": True,
    }


def test_decide_case_paths():
    base = {
        "prior_introduced": False,
        "ridge_as_information": False,
        "added_fd_rung": False,
        "relaxed_five_percent_gate": False,
        "changed_official_mean": False,
        "tuned_field_step": False,
        "b14m_reopened": False,
        "source_audit": {"missing_term_source_proven": True},
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
        "focus_segments": [{"independent_reference_established": False}] * 3,
    }
    unresolved = decide_case(base)
    assert unresolved["decision"] == CASE_FOCUS_UNRESOLVED
    assert unresolved["jacobian_contract_established"] is False
    assert unresolved["b14m_reopen_authorized"] is False

    repaired = decide_case(
        {
            **base,
            "focus_segments": [{"independent_reference_established": True}] * 3,
        }
    )
    assert repaired["decision"] == CASE_REPAIRED
    assert repaired["b14m_reopen_authorized"] is True

    hypo = decide_case({**base, "source_audit": {"missing_term_source_proven": False}})
    assert hypo["decision"] == CASE_HYPOTHESIS
    with pytest.raises(FieldGradientVariationalRepairError, match="FD rung"):
        decide_case({**base, "added_fd_rung": True})


def test_focus_requires_first_unstable_hop_not_earlier_long_hop():
    assert FOCUS_FIRST_UNSTABLE == {1: 6, 2: 11, 3: 11}
    hop6 = {
        "measurement_index": 6,
        "number_of_propagation_steps": 84,
        "field_gradient_repair": {
            "requested": True,
            "old_rk_pos_to_dir_norm": 0.0,
            "new_rk_pos_to_dir_norm": 1.0e-5,
            "hop_dloc0_d_start": [1.0, -0.04, 20.0, -1000.0, -200.0],
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
        "field_gradient_repair": {
            "requested": True,
            "old_rk_pos_to_dir_norm": 0.0,
            "new_rk_pos_to_dir_norm": 4.0e-4,
        },
    }
    row = {
        "run_id": 100048,
        "event_id": 86,
        "target_station": 2,
        "official_supporting_plane_jacobian": {
            "official_path_jacobian": {"hops": [hop6, hop11]}
        },
    }
    audited = audit_segment_and_focus(row)
    assert audited["required_first_unstable_index"] == 11
    assert audited["segment_fd_present_on_required_hop"] is False
    assert audited["independent_reference_established"] is False
    assert audited["acts_variational_self_reference_forbidden"] is True
