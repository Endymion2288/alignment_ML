"""Task B ACTS process-noise contract.  No sealed test, no Q/chi2 tuning."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.acts_process_noise_contract import (
    DECISION_ESTABLISHED,
    DECISION_NOT_ESTABLISHED,
    FAILURE_CONFIGURATION,
    FAILURE_MEASUREMENT_MODEL,
    MECHANISM_CLOSURE_FAILED,
    MECHANISM_NOT_MATERIALIZED,
    MECHANISM_NOT_WRITTEN,
    MODEL0,
    MODEL1,
    ProcessNoiseContractError,
    attach_highland_model,
    audit_process_noise_configuration,
    decide,
    highland_on_model0,
    inherit_frozen_stage,
    load_config,
    refuse_covariance_rescale,
    refuse_dummy_segmentfit,
    refuse_process_noise_tuning,
    refuse_truth_qoverp,
)


def test_config_inherits_frozen_stage_and_gates():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_87"]["decision"] == "faseracts_transport_covariance_not_validated"
    assert inherited["workbook_95"]["decision"] == "physical_qoverp_semantics_not_established"
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert config["closure_gates"]["whitened_chi2_per_ndof_max"] == 4.0
    assert config["models"][2]["diagnostic_only"] is True
    assert config["do_not_tune_process_noise"] is True


def test_b1_configuration_is_explicit_and_production_q_is_off():
    configuration = audit_process_noise_configuration(load_config())
    assert configuration["acts_version"] == "32.0.2"
    assert configuration["process_noise_flags"]["material_interactor_in_action_list"] is True
    assert configuration["process_noise_flags"]["production_process_noise_written_to_covariance"] is False
    assert configuration["production_transport_has_process_noise"] is False
    assert configuration["material_map"]["exists"] is True
    assert configuration["material_map"]["file_sha256"]
    assert configuration["geometry_hash"]
    assert configuration["material_map_hash"]
    assert configuration["field_hash"]
    assert configuration["conditions_hash"]
    assert configuration["contract_clear"] is True
    assert configuration["measurement_model_v2_entered"] is False


def test_forbidden_repairs_and_sealed_test():
    with pytest.raises(ProcessNoiseContractError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(ProcessNoiseContractError, match="dummy"):
        refuse_dummy_segmentfit()
    with pytest.raises(ProcessNoiseContractError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(ProcessNoiseContractError, match="process noise"):
        refuse_process_noise_tuning()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_highland_is_spd_and_not_a_pass_model():
    c0 = np.eye(4) * 1.0e-4
    c2 = highland_on_model0(c0, 1906.0, 5.0e5, 0.05)
    assert c2.shape == (4, 4)
    assert float(c2[2, 2]) > float(c0[2, 2])
    config = load_config()
    assert config["models"][1]["pass_model"] is True
    assert config["models"][2]["diagnostic_only"] is True


def _calibrated_cell() -> dict:
    return {
        "n_pairs": 25,
        "sufficient": True,
        "chi2_per_ndof": 1.1,
        "gate_verdict": {"calibrated": True},
        "pencil": {"variance_ratio_prop_over_emp": 1.0, "direction": [0.0, 1.0, 0.0, 0.0]},
    }


def _uncalibrated_cell() -> dict:
    return {
        "n_pairs": 25,
        "sufficient": True,
        "chi2_per_ndof": 800.0,
        "gate_verdict": {"calibrated": False, "reason": "gate_failure"},
        "pencil": {"variance_ratio_prop_over_emp": 12.0, "direction": [0.0, 1.0, 0.0, 0.0]},
    }


def _inventory(model1, *, q_added=25, success=25, materialized=True):
    model = {
        "n_success": success,
        "n_truth_rejected": 0,
        "n_dummy_rejected": 0,
        "n_q_added": q_added,
        "all_calibrated": all(
            cell["gate_verdict"]["calibrated"] for cell in model1.values()
        ),
        "per_pair": model1,
        "qoverp_bins": {},
    }
    empty = {
        "n_success": success,
        "n_truth_rejected": 0,
        "n_dummy_rejected": 0,
        "n_q_added": 0,
        "all_calibrated": False,
        "per_pair": {key: _uncalibrated_cell() for key in model1},
        "qoverp_bins": {},
    }
    split = {
        "n_sources": 2,
        "present_sources": ["a", "b"] if materialized else [],
        "missing_sources": [] if materialized else ["a", "b"],
        "n_records": success,
        "models": {MODEL0: empty, MODEL1: model, "model2_highland_diagnostic": empty},
    }
    return {"dumps_materialized": materialized, "splits": {"construction": split, "validation": split}}


def test_complete_acts_export_would_pass():
    cells = {pair: _calibrated_cell() for pair in ("(0,1)", "(0,2)", "(0,3)")}
    decision = decide(
        {"contract_clear": True, "acts_version": "32.0.2"},
        _inventory(cells),
    )
    assert decision["verdict"] == "PASS"
    assert decision["decision"] == DECISION_ESTABLISHED
    assert decision["measurement_model_v2_entered"] is False
    assert decision["contract_established_for_measurement_model_v2"] is True


def test_missing_dump_and_q_not_written_and_closure_fail():
    cells = {pair: _calibrated_cell() for pair in ("(0,1)", "(0,2)", "(0,3)")}
    missing = decide({"contract_clear": True}, _inventory(cells, materialized=False, success=0, q_added=0))
    assert missing["verdict"] == "FAIL"
    assert missing["decision"] == DECISION_NOT_ESTABLISHED
    assert missing["mechanism"] == MECHANISM_NOT_MATERIALIZED
    assert missing["failure_type"] == FAILURE_CONFIGURATION

    no_q = decide({"contract_clear": True}, _inventory(cells, q_added=0))
    assert no_q["mechanism"] == MECHANISM_NOT_WRITTEN
    assert no_q["failure_type"] == FAILURE_CONFIGURATION

    failed = {pair: _uncalibrated_cell() for pair in cells}
    closed = decide({"contract_clear": True}, _inventory(failed, q_added=25))
    assert closed["mechanism"] == MECHANISM_CLOSURE_FAILED
    assert closed["failure_type"] == FAILURE_MEASUREMENT_MODEL
    assert closed["highland_promoted"] is False


def test_attach_highland_does_not_replace_acts():
    row = {
        MODEL0: {
            "success": True,
            "state_xy_tx_ty": [0.0, 0.0, 0.01, 0.0],
            "covariance_4x4": np.eye(4).tolist(),
        },
        "p_mev": 4.0e5,
        "source_z_mm": -1860.15,
        "target_z_mm": 47.4,
    }
    attached = attach_highland_model([row], load_config())[0]
    assert attached["model2_highland_diagnostic"]["diagnostic_only"] is True
    assert attached["model2_highland_diagnostic"]["success"] is True
