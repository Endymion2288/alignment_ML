"""Task B14T standalone measurement transport.  No B15, no V2, no prior."""

from __future__ import annotations

import pytest

from alignment.profiled_measurement_likelihood import PINV_RELATIVE
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.profile_likelihood_numerics import CASE_B as WB115_DECISION
from datasets.profile_transport_contract import (
    CASE_ESTABLISHED,
    CASE_MIXED,
    CASE_NAVIGATION,
    CASE_PROJECTION,
    CASE_STATE,
    CLASS_M,
    CLASS_S,
    ProfileTransportContractError,
    classify_failure_class,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_b14m,
    refuse_b15,
    refuse_change_statistical_model,
    refuse_delete_focus,
    refuse_full_sample,
    refuse_max_steps_hack,
    refuse_measurement_model_v2,
    refuse_measurement_update,
    refuse_merge_classes,
    refuse_physical_nonidentifiability,
    refuse_prior,
    refuse_rewrite_profile_math,
    refuse_ridge_information,
    refuse_tolerance_hunt,
    refuse_truth_qoverp,
    smoke_gate,
    standalone_likelihood_transport_contract,
)


def test_config_inherits_wb115_case_b_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is True
    assert config["do_not_enter_b14n"] is True
    assert config["do_not_enter_b14t"] is False
    assert config["do_not_rewrite_profile_math"] is True
    assert config["do_not_change_statistical_model"] is True
    assert config["do_not_use_ridge"] is True
    assert config["do_not_force_5d_lto_covariance"] is True
    assert config["do_not_add_measurement_update_in_evaluator"] is True
    assert config["do_not_use_arbitrary_max_steps_as_official_fix"] is True
    assert config["do_not_merge_class_m_and_class_s"] is True
    assert float(config["profile_numerical"]["pinv_relative"]) == PINV_RELATIVE
    assert inherited["workbook_115"]["decision"] == WB115_DECISION
    assert inherited["workbook_114"]["synthetic_profile_passed"] is True
    assert inherited["workbook_113"]["decision"] == "target_independent_prior_not_available"
    assert inherited["workbook_112"]["five_d_state_physically_supported"] is False
    assert inherited["workbook_109"]["decision"] == (
        "leave_target_out_state_materialization_established"
    )
    assert "profile_transport_contract_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(ProfileTransportContractError, match="prior"):
        refuse_prior()
    with pytest.raises(ProfileTransportContractError, match="ridge"):
        refuse_ridge_information()
    with pytest.raises(ProfileTransportContractError, match="measurement update"):
        refuse_measurement_update()
    with pytest.raises(ProfileTransportContractError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(ProfileTransportContractError, match="maxSteps"):
        refuse_max_steps_hack()
    with pytest.raises(ProfileTransportContractError, match="tolerance"):
        refuse_tolerance_hunt()
    with pytest.raises(ProfileTransportContractError, match="Class M"):
        refuse_merge_classes()
    with pytest.raises(ProfileTransportContractError, match="100043/37"):
        refuse_delete_focus()
    with pytest.raises(ProfileTransportContractError, match="rewritten"):
        refuse_rewrite_profile_math()
    with pytest.raises(ProfileTransportContractError, match="statistical model"):
        refuse_change_statistical_model()
    with pytest.raises(ProfileTransportContractError, match="B14M"):
        refuse_b14m()
    with pytest.raises(ProfileTransportContractError, match="B15"):
        refuse_b15()
    with pytest.raises(ProfileTransportContractError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(ProfileTransportContractError, match="1989"):
        refuse_full_sample()
    with pytest.raises(ProfileTransportContractError, match="physical_nonidentifiability"):
        refuse_physical_nonidentifiability()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_likelihood_contract_forbids_measurement_update():
    contract = standalone_likelihood_transport_contract()
    assert contract["measurement_update_in_evaluator"] is False
    assert "measurement update" in " ".join(contract["forbidden_repairs"])


def test_taxonomy_keeps_class_m_and_class_s_separate():
    class_m = classify_failure_class(
        {
            "run_id": 100043,
            "event_id": 37,
            "all_surfaces_reached": False,
            "propagation_hits": [
                {
                    "ok": False,
                    "start_z_mm": -1826.22,
                    "destination_surface_z_mm": 17.45,
                    "abort_reason": "Propagation reached the configured maximum number of steps",
                }
            ],
        }
    )
    class_s = classify_failure_class(
        {
            "run_id": 100048,
            "event_id": 86,
            "all_surfaces_reached": False,
            "propagation_hits": [
                {
                    "ok": False,
                    "start_z_mm": 1234.97,
                    "destination_surface_z_mm": 1235.86,
                    "abort_reason": "Global to local transformation failed: position not on surface.",
                }
            ],
        }
    )
    assert class_m == CLASS_M
    assert class_s == CLASS_S
    assert class_m != class_s


def test_decide_case_does_not_claim_physics():
    base = {
        "prior_introduced": False,
        "ridge_as_information": False,
        "statistical_model_changed": False,
        "physical_nonidentifiability_claimed": False,
        "smoke_present": True,
        "smoke_gate": {"passed": False},
        "evaluations": {
            "event_37_all_surviving_evaluable": False,
            "event_86_all_projections_defined": True,
        },
        "magnet": {
            "official_mode_b_all_targets_reached": False,
            "mode_a_all_targets_reached": False,
        },
        "units": {"n_dumped": 3},
        "stereo": {"pairs": []},
    }
    nav = decide_case(base)
    assert nav["decision"] in {CASE_NAVIGATION, CASE_STATE}
    assert nav["b15_authorized"] is False
    assert nav["physical_nonidentifiability_not_claimed"] is True
    proj = decide_case(
        {
            **base,
            "evaluations": {
                "event_37_all_surviving_evaluable": True,
                "event_86_all_projections_defined": False,
            },
        }
    )
    assert proj["decision"] == CASE_PROJECTION
    passed = decide_case({**base, "smoke_gate": {"passed": True}})
    assert passed["decision"] == CASE_ESTABLISHED
    assert passed["b14m_reopen_authorized"] is True
    missing = decide_case({**base, "smoke_present": False})
    assert missing["decision"] == CASE_MIXED


def test_smoke_gate_blocks_b14m_reopen():
    gate = smoke_gate(
        {
            "smoke_present": True,
            "evaluations": {
                "event_0_1_all_targets": True,
                "event_37_all_surviving_evaluable": False,
                "event_86_all_projections_defined": True,
                "measurement_update_in_evaluator": False,
            },
            "jacobian": {"jacobian_contract_established": True},
            "wb115_consistency": {"consistent": True},
            "exclusion": {"target_exclusion_holds": True},
            "magnet": {"diagnostic_max_steps_is_not_official_fix": True},
            "prior_introduced": False,
            "ridge_as_information": False,
            "statistical_model_changed": False,
        }
    )
    assert gate["passed"] is False
    assert gate["b14m_reopen_authorized"] is False
    assert gate["full_sample_authorized"] is False
