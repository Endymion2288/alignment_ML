"""Task A2 CKF 5x5 export contract.  No sealed test, no dummy/truth pass."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.numerical_contract import CovarianceNotSPDError
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.qoverp_covariance_export import (
    DECISION_ESTABLISHED,
    DECISION_NOT_ESTABLISHED,
    MECHANISM_DUMMY,
    MECHANISM_NOT_MATERIALIZED,
    MECHANISM_NO_CROSS,
    QoverPExportError,
    decide,
    derived_jacobian_5d,
    evaluate_records,
    inherit_frozen_stage,
    load_config,
    refuse_dummy_segmentfit,
    refuse_synthetic_injection,
    refuse_truth_qoverp,
    state_definition,
    transport_native_covariance,
)


def test_config_inherits_wb87_and_wb95():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_87"]["decision"] == "faseracts_transport_covariance_not_validated"
    assert inherited["workbook_95"]["decision"] == "physical_qoverp_semantics_not_established"
    assert config["do_not_enter_task_b_unless_pass"] is True
    assert config["source_collection"] == "CKFTrackCollection"
    assert config["covariance_dimension"] == 5


def test_state_units_and_forbidden_repairs():
    definition = state_definition()
    assert definition["q_over_p_native_unit"] == "per_MeV"
    assert definition["q_over_p_signed"] is True
    assert definition["covariance_dimension"] == 5
    with pytest.raises(QoverPExportError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(QoverPExportError, match="dummy"):
        refuse_dummy_segmentfit()
    with pytest.raises(QoverPExportError, match="synthetic"):
        refuse_synthetic_injection()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_derived_jacobian_keeps_qoverp_and_cross_terms():
    direction = np.array([0.02, -0.01, 1.0])
    ref = np.array([1.0, -2.0, 100.0])
    phi = float(np.arctan2(direction[1], direction[0]))
    theta = float(np.arctan(np.hypot(direction[0], direction[1]) / direction[2]))
    jac = derived_jacobian_5d(phi, theta, ref, direction)
    assert jac.shape == (5, 5)
    assert jac[4, 4] == pytest.approx(1.0)
    assert np.allclose(jac[4, :4], 0.0)
    assert np.allclose(jac[:4, 4], 0.0)
    native = np.diag([1.0e-2, 1.0e-2, 1.0e-6, 1.0e-6, 4.0e-10])
    native[0, 4] = native[4, 0] = 1.0e-7
    native[2, 4] = native[4, 2] = -5.0e-9
    derived = transport_native_covariance(native, phi, theta, ref, direction)
    assert derived.shape == (5, 5)
    assert derived[4, 4] == pytest.approx(4.0e-10)
    assert abs(derived[0, 4]) > 0.0 or abs(derived[1, 4]) > 0.0


def test_non_spd_native_is_rejected():
    bad = np.array(
        [
            [1.0, 0.0, 0.0, 0.0, 2.0],
            [0.0, 1.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0],
            [2.0, 0.0, 0.0, 0.0, 1.0],
        ]
    )
    with pytest.raises(CovarianceNotSPDError):
        transport_native_covariance(
            bad, 0.0, 0.1, np.zeros(3), np.array([0.0, 0.0, 1.0])
        )


def _physical_record(qoverp: float, seed: int) -> dict:
    rng = np.random.default_rng(seed)
    cov = np.eye(5) * 1.0e-4
    cov[4, 4] = 3.0e-10
    cov[0, 4] = cov[4, 0] = 1.0e-8
    cov[2, 4] = cov[4, 2] = -2.0e-9
    cov = 0.5 * (cov + cov.T)
    return {
        "native_state": [0.1, -0.2, 0.01, 0.02, qoverp],
        "native_covariance": cov.tolist(),
        "has_covariance": True,
        "is_truth": False,
    }


def test_complete_ckf_export_would_pass():
    config = load_config()
    construction = evaluate_records(
        [_physical_record(-2.0e-6, i) for i in range(25)], config
    )
    validation = evaluate_records(
        [_physical_record(-3.0e-6, 100 + i) for i in range(25)], config
    )
    decision = decide(construction, validation, dumps_materialized=True)
    assert decision["verdict"] == "PASS"
    assert decision["decision"] == DECISION_ESTABLISHED
    assert decision["measurement_model_v2_entered"] is False


def test_missing_dump_and_dummy_fail():
    empty = evaluate_records([], load_config())
    missing = decide(empty, empty, dumps_materialized=False)
    assert missing["verdict"] == "FAIL"
    assert missing["decision"] == DECISION_NOT_ESTABLISHED
    assert missing["mechanism"] == MECHANISM_NOT_MATERIALIZED

    dummy = {
        "native_state": [0.0, 0.0, 0.0, 0.0, 1.0e-5],
        "native_covariance": np.diag([1.0, 1.0, 1.0, 1.0, 5.0e-6]).tolist(),
        "has_covariance": True,
        "is_truth": False,
    }
    config = load_config()
    summary = evaluate_records([dummy] * 25, config)
    failed = decide(summary, summary, dumps_materialized=True)
    assert failed["verdict"] == "FAIL"
    assert failed["mechanism"] == MECHANISM_DUMMY

    diagonal = {
        "native_state": [0.0, 0.0, 0.0, 0.0, -2.0e-6],
        "native_covariance": np.diag([1.0e-4, 1.0e-4, 1.0e-6, 1.0e-6, 3.0e-10]).tolist(),
        "has_covariance": True,
        "is_truth": False,
    }
    no_cross = evaluate_records([diagonal] * 25, config)
    failed_cross = decide(no_cross, no_cross, dumps_materialized=True)
    assert failed_cross["mechanism"] == MECHANISM_NO_CROSS
