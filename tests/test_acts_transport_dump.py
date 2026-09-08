"""Task B0 C++ ACTS transport dump.  No sealed test, no Q/chi2 tuning."""

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
)
from datasets.acts_transport_dump import (
    REQUIRED_ROW_KEYS,
    SITUATION_A,
    SITUATION_B,
    TransportDumpError,
    analyze_process_noise,
    decide,
    dump_path_for_source,
    inherit_frozen_stage,
    load_config,
    refuse_covariance_rescale,
    refuse_dummy_segmentfit,
    refuse_process_noise_tuning,
    refuse_truth_qoverp,
)


def test_config_inherits_frozen_stage_and_does_not_reopen_wb97():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_87"]["decision"] == "faseracts_transport_covariance_not_validated"
    assert inherited["workbook_95"]["decision"] == "physical_qoverp_semantics_not_established"
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert inherited["workbook_97"]["decision"] == "acts_process_noise_contract_not_established"
    assert inherited["workbook_97"]["mechanism"] == "acts_process_noise_not_materialized"
    assert inherited["workbook_97"]["failure_type"] == "acts_process_noise_configuration"
    assert config["closure_gates"]["whitened_chi2_per_ndof_max"] == 4.0
    assert config["models"][2]["diagnostic_only"] is True
    assert config["do_not_construct_acts_objects_in_python"] is True
    assert config["do_not_enter_measurement_model_v2"] is True
    assert config["dump_filename"] != "ckf_acts_process_noise.jsonl"
    assert "acts_transport_dump_v1" in config["dump_root"]


def test_dump_path_is_isolated_from_wb97():
    path = dump_path_for_source(load_config(), "mc24_100043_00200_00299")
    assert path.as_posix().endswith(
        "outputs/acts_transport_dump_v1/dumps/mc24_100043_00200_00299/ckf_acts_transport.jsonl"
    )
    assert "acts_process_noise_contract_v1" not in path.as_posix()


def test_forbidden_repairs_and_sealed_test():
    with pytest.raises(TransportDumpError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(TransportDumpError, match="dummy"):
        refuse_dummy_segmentfit()
    with pytest.raises(TransportDumpError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(TransportDumpError, match="process noise"):
        refuse_process_noise_tuning()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


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


def _inventory(model1, *, q_added=25, success=25, materialized=True, q_visible=True):
    model = {
        "n_success": success,
        "n_truth_rejected": 0,
        "n_dummy_rejected": 0,
        "n_q_added": q_added,
        "all_calibrated": all(cell["gate_verdict"]["calibrated"] for cell in model1.values()),
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
        "process_noise": {
            "process_noise_visible": q_visible,
            "process_noise_artificial_scale": False,
        },
        "models": {MODEL0: empty, MODEL1: model, "model2_highland_diagnostic": empty},
    }
    return {"dumps_materialized": materialized, "splits": {"construction": split, "validation": split}}


def test_complete_acts_dump_would_pass_without_entering_v2():
    cells = {pair: _calibrated_cell() for pair in ("(0,1)", "(0,2)", "(0,3)")}
    decision = decide({"contract_clear": True, "acts_version": "32.0.2"}, _inventory(cells))
    assert decision["verdict"] == "PASS"
    assert decision["decision"] == DECISION_ESTABLISHED
    assert decision["situation"] is None
    assert decision["measurement_model_v2_entered"] is False
    assert decision["inherited_wb97_decision"] == DECISION_NOT_ESTABLISHED


def test_situation_a_and_b_classification():
    cells = {pair: _calibrated_cell() for pair in ("(0,1)", "(0,2)", "(0,3)")}
    missing = decide(
        {"contract_clear": True},
        _inventory(cells, materialized=False, success=0, q_added=0, q_visible=False),
    )
    assert missing["situation"] == SITUATION_A
    assert missing["mechanism"] == MECHANISM_NOT_MATERIALIZED
    assert missing["failure_type"] == FAILURE_CONFIGURATION

    no_q = decide({"contract_clear": True}, _inventory(cells, q_added=0, q_visible=False))
    assert no_q["situation"] == SITUATION_A
    assert no_q["mechanism"] == MECHANISM_NOT_WRITTEN

    failed = {pair: _uncalibrated_cell() for pair in cells}
    closed = decide({"contract_clear": True}, _inventory(failed, q_added=25, q_visible=True))
    assert closed["situation"] == SITUATION_B
    assert closed["mechanism"] == MECHANISM_CLOSURE_FAILED
    assert closed["failure_type"] == FAILURE_MEASUREMENT_MODEL
    assert closed["highland_promoted"] is False


def test_analyze_process_noise_rejects_artificial_scale():
    config = load_config()
    q = (np.eye(5) * 1.0e-6).tolist()
    c0 = np.eye(5).tolist()
    visible = analyze_process_noise(
        [
            {
                **{key: "x" for key in REQUIRED_ROW_KEYS},
                "process_noise": q,
                "output_covariance_no_material": c0,
                "process_noise_artificial_scale": False,
            }
        ],
        config,
    )
    assert visible["process_noise_visible"] is True
    scaled = analyze_process_noise(
        [
            {
                **{key: "x" for key in REQUIRED_ROW_KEYS},
                "process_noise": q,
                "output_covariance_no_material": c0,
                "process_noise_artificial_scale": True,
            }
        ],
        config,
    )
    assert scaled["process_noise_visible"] is False
    assert scaled["process_noise_artificial_scale"] is True
