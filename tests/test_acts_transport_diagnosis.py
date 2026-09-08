"""Task B3 ACTS transport covariance diagnosis.  No sealed test, no Q/chi2 tuning."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.acts_transport_diagnosis import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    DECISION_DIAGNOSED,
    TransportDiagnosisError,
    classify_failure,
    compare_c0_to_jacobian,
    dump_path_for_source,
    inherit_frozen_stage,
    jsonable,
    load_config,
    refuse_covariance_rescale,
    refuse_dummy_segmentfit,
    refuse_measurement_model_v2,
    refuse_outlier_rejection,
    refuse_process_noise_tuning,
    refuse_truth_qoverp,
    residual_decomposition,
    tail_provenance,
)


def test_config_inherits_frozen_wb87_through_wb98():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_87"]["decision"] == "faseracts_transport_covariance_not_validated"
    assert inherited["workbook_95"]["decision"] == "physical_qoverp_semantics_not_established"
    assert inherited["workbook_96"]["decision"] == "ckf_qoverp_covariance_export_established"
    assert inherited["workbook_97"]["decision"] == "acts_process_noise_contract_not_established"
    assert inherited["workbook_97"]["failure_type"] == "acts_process_noise_configuration"
    assert inherited["workbook_98"]["decision"] == "acts_process_noise_contract_not_established"
    assert inherited["workbook_98"]["mechanism"] == "acts_process_noise_closure_failed"
    assert inherited["workbook_98"]["situation"] == "acts_q_materialized_closure_failed"
    assert inherited["workbook_98"]["failure_type"] == "measurement_track_model_insufficient"
    assert config["do_not_enter_measurement_model_v2"] is True
    assert config["do_not_reject_outliers"] is True
    assert config["do_not_clip_chi2_tails"] is True
    assert config["qoverp_bins_abs_per_mev"]["low_momentum"] == [5.0e-6, 1.0]
    assert "acts_transport_diagnosis_v1" in config["output_root"]
    assert config["dump_root"] != config["output_root"]


def test_dump_path_reads_wb98_dumps_without_overwrite():
    path = dump_path_for_source(load_config(), "mc24_100043_00200_00299")
    assert path.as_posix().endswith(
        "outputs/acts_transport_dump_v1/dumps/mc24_100043_00200_00299/ckf_acts_transport.jsonl"
    )
    assert "acts_transport_diagnosis_v1" not in path.as_posix()


def test_forbidden_repairs_and_sealed_test():
    with pytest.raises(TransportDiagnosisError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(TransportDiagnosisError, match="dummy"):
        refuse_dummy_segmentfit()
    with pytest.raises(TransportDiagnosisError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(TransportDiagnosisError, match="process noise"):
        refuse_process_noise_tuning()
    with pytest.raises(TransportDiagnosisError, match="outlier"):
        refuse_outlier_rejection()
    with pytest.raises(TransportDiagnosisError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_compare_c0_does_not_infer_f():
    identity = np.eye(5)
    report = compare_c0_to_jacobian(identity, identity, identity)
    assert report["inferred_f_from_covariance"] is False
    assert report["frobenius_relative_error"] == pytest.approx(0.0)
    assert report["per_state_diagonal"][0]["state"] == "x"
    scaled = 2.0 * identity
    off = compare_c0_to_jacobian(identity, identity, scaled)
    assert off["frobenius_relative_error"] == pytest.approx(0.5)


def test_tail_provenance_keeps_every_event():
    events = []
    for index in range(100):
        events.append(
            {
                "chi2_per_ndof": float(index + 1),
                "residual": {"x": 0.0, "y": 0.0, "tx": 0.0, "ty": 0.0, "q_over_p": 0.0},
                "pull": {"x": 0.0, "y": 0.0, "tx": 0.0, "ty": 0.0, "q_over_p": 0.0},
                "chi2_contribution": {"x": 0.0, "y": 0.0, "tx": 0.0, "ty": 0.0},
            }
        )
    report = tail_provenance(events, {"tail_quantiles": [0.99, 0.999]})
    assert report["n_dropped"] == 0
    assert report["outlier_rejection"] is False
    assert report["tails"]["0.99"]["rejected"] == 0
    assert report["tails"]["0.99"]["n"] >= 1
    assert report["tails"]["0.99"]["n_retained"] == report["tails"]["0.99"]["n"]


def test_residual_decomposition_reports_x_charge_correlation():
    events = [
        {
            "residual": {"x": -1.0, "y": 0.1, "tx": 0.0, "ty": 0.0, "q_over_p": 0.0},
            "pull": {"x": -1.0, "y": 0.1, "tx": 0.0, "ty": 0.0, "q_over_p": 0.0},
            "chi2_contribution": {"x": 1.0, "y": 0.01, "tx": 0.0, "ty": 0.0},
            "chi2_per_ndof": 0.25,
            "charge": -1.0,
            "abs_q_over_p": 1.0e-6,
            "q_frobenius": 1.0,
        },
        {
            "residual": {"x": 1.0, "y": 0.1, "tx": 0.0, "ty": 0.0, "q_over_p": 0.0},
            "pull": {"x": 1.0, "y": 0.1, "tx": 0.0, "ty": 0.0, "q_over_p": 0.0},
            "chi2_contribution": {"x": 1.0, "y": 0.01, "tx": 0.0, "ty": 0.0},
            "chi2_per_ndof": 0.25,
            "charge": 1.0,
            "abs_q_over_p": 2.0e-6,
            "q_frobenius": 2.0,
        },
        {
            "residual": {"x": 0.5, "y": 0.0, "tx": 0.0, "ty": 0.0, "q_over_p": 0.0},
            "pull": {"x": 0.5, "y": 0.0, "tx": 0.0, "ty": 0.0, "q_over_p": 0.0},
            "chi2_contribution": {"x": 0.25, "y": 0.0, "tx": 0.0, "ty": 0.0},
            "chi2_per_ndof": 0.1,
            "charge": 1.0,
            "abs_q_over_p": 3.0e-6,
            "q_frobenius": 3.0,
        },
    ]
    report = residual_decomposition(events)
    assert report["n"] == 3
    assert report["x_vs_charge_correlation"] is not None
    assert report["components"]["x"]["residual"]["mean"] is not None


def _base_classify_inputs(*, jacobian_ok=True, frame_ok=True, tail_share=0.1, mean_over=2.0):
    contract = {
        "frame_contract_holds": frame_ok,
        "transformation_jacobian": {"bending_axis_consistent_with_Bx": True},
    }
    jacobian = {
        "jacobian_contract_holds": jacobian_ok,
        "n_compared": 100,
        "frobenius_relative_error": {"median": 1.0e-4 if jacobian_ok else 0.2, "p95": 2.0e-4 if jacobian_ok else 0.3},
        "q_as_c1_minus_c0": {"n_not_psd": 5},
    }
    residuals = {
        "overall": {
            "x_vs_charge_correlation": 0.01,
            "x_abs_vs_q_frobenius_correlation": 0.05,
            "chi2_per_ndof": {"median": 0.2},
        }
    }
    tails = {
        "mean_over_median": mean_over,
        "tails": {"0.99": {"chi2_share": tail_share}},
    }
    return contract, jacobian, residuals, tails, load_config()


def test_classify_case_a_when_jacobian_fails():
    contract, jacobian, residuals, tails, config = _base_classify_inputs(jacobian_ok=False)
    result = classify_failure(contract, jacobian, residuals, tails, config)
    assert result["primary_case"] == CASE_A
    assert result["next_step"] == "fix_transport_contract"
    assert result["measurement_model_v2_entered"] is False


def test_classify_does_not_treat_jacobian_p95_tail_as_case_a():
    contract, jacobian, residuals, tails, config = _base_classify_inputs(
        jacobian_ok=True, tail_share=0.8, mean_over=40.0
    )
    jacobian["frobenius_relative_error"]["p95"] = 0.32
    jacobian["jacobian_tail_present"] = True
    result = classify_failure(contract, jacobian, residuals, tails, config)
    assert result["primary_case"] == CASE_C
    assert result["evidence"]["jacobian_contract_holds"] is True
    assert result["evidence"]["jacobian_tail_present"] is True


def test_classify_case_c_when_tails_dominate_mean():
    contract, jacobian, residuals, tails, config = _base_classify_inputs(
        tail_share=0.8, mean_over=40.0
    )
    result = classify_failure(contract, jacobian, residuals, tails, config)
    assert result["primary_case"] == CASE_C
    assert result["next_step"] == "uncertainty_model_analysis_no_covariance_tuning"
    assert CASE_D in result["secondary_cases"]


def test_classify_case_d_when_contract_holds_without_tail_or_material():
    contract, jacobian, residuals, tails, config = _base_classify_inputs(
        tail_share=0.1, mean_over=2.0
    )
    result = classify_failure(contract, jacobian, residuals, tails, config)
    assert result["primary_case"] == CASE_D
    assert result["next_step"] == "reassess_measurement_model_v2_input_model"


def test_classify_case_b_when_q_not_psd_and_no_tail():
    contract, jacobian, residuals, tails, config = _base_classify_inputs(
        tail_share=0.1, mean_over=2.0
    )
    jacobian["q_as_c1_minus_c0"] = {"n_not_psd": 80}
    residuals["overall"]["x_abs_vs_q_frobenius_correlation"] = 0.4
    result = classify_failure(contract, jacobian, residuals, tails, config)
    assert result["primary_case"] == CASE_B
    assert result["next_step"] == "acts_material_diagnosis"


def test_jsonable_strips_nan_and_decision_token():
    cleaned = jsonable({"value": float("nan"), "ok": 1.0})
    assert cleaned["value"] is None
    assert cleaned["ok"] == 1.0
    assert DECISION_DIAGNOSED == "acts_transport_covariance_failure_diagnosed"
