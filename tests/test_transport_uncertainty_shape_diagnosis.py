"""Task B9 shape diagnosis.  Contracted input only, no C/Q tune, no V2."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.transport_uncertainty_shape_diagnosis import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    CASE_E,
    ShapeDiagnosisError,
    decide_case,
    inherit_frozen_stage,
    load_config,
    refuse_covariance_rescale,
    refuse_empirical_cross_covariance,
    refuse_focus_drop,
    refuse_measurement_model_v2,
    refuse_process_noise_tuning,
    refuse_q_psd_projection,
    refuse_raw_ckf,
    refuse_tighten_contract,
    refuse_truth_qoverp,
)
from datasets.acts_transport_diagnosis import dump_path_for_source as dump_from_diagnosis


def test_config_inherits_wb104_and_wb103_eligibility():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["official_input_scope"] == "contract_eligible"
    assert config["do_not_tighten_reconstruction_contract"] is True
    assert config["do_not_invent_empirical_cross_covariance"] is True
    assert config["closure_gates"]["pencil_variance_ratio_max"] == 4.0
    assert inherited["workbook_103"]["decision"] == "ckf_reconstruction_contract_validated"
    assert inherited["workbook_104"]["decision"] == "transport_covariance_shape_not_validated"
    assert inherited["workbook_104"]["primary_case"] == "transport_covariance_shape_not_validated"
    assert inherited["workbook_104"]["decision_sha256"].startswith("8e620b50")
    assert "transport_uncertainty_shape_diagnosis_v1" in config["output_root"]


def test_dump_path_stays_on_wb98():
    path = dump_from_diagnosis(load_config(), "mc24_100043_00200_00299")
    assert path.as_posix().endswith(
        "outputs/acts_transport_dump_v1/dumps/mc24_100043_00200_00299/ckf_acts_transport.jsonl"
    )


def test_forbidden_repairs():
    with pytest.raises(ShapeDiagnosisError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(ShapeDiagnosisError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(ShapeDiagnosisError, match="process noise"):
        refuse_process_noise_tuning()
    with pytest.raises(ShapeDiagnosisError, match="PSD-projected"):
        refuse_q_psd_projection()
    with pytest.raises(ShapeDiagnosisError, match="raw CKF"):
        refuse_raw_ckf()
    with pytest.raises(ShapeDiagnosisError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(ShapeDiagnosisError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(ShapeDiagnosisError, match="Cov"):
        refuse_empirical_cross_covariance()
    with pytest.raises(ShapeDiagnosisError, match="tightened"):
        refuse_tighten_contract()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def _focus(*, loo_holds: bool = False, share: float = 0.8) -> dict:
    return {"retained": True, "loo_shape_holds": loo_holds, "chi2_share": share}


def test_decide_case_a_shared_fit_and_c0_overwide():
    result = decide_case(
        {"official_adds_c_target": False, "official_residual_reference": "truth"},
        {"target_station_in_same_ckf_fit_fraction": 1.0},
        {"source_cin_shape_holds": True},
        {"c0_shape_holds": False, "c1_shape_holds": False, "material_is_primary_mismatch": False},
        {"systematic_transport_failure": False},
        _focus(),
    )
    assert result["primary_case"] == CASE_A
    assert result["next_step"] == "leave_target_out_prediction_contract"
    assert result["measurement_model_v2_authorized"] is False
    assert result["empirical_cross_covariance_invented"] is False


def test_decide_case_b_cin_overwide_without_shared_or_c0():
    result = decide_case(
        {"official_adds_c_target": False, "official_residual_reference": "truth"},
        {"target_station_in_same_ckf_fit_fraction": 0.1},
        {"source_cin_shape_holds": False},
        {"c0_shape_holds": True, "c1_shape_holds": True, "material_is_primary_mismatch": False},
        {"systematic_transport_failure": False},
        _focus(),
    )
    assert result["primary_case"] == CASE_B
    assert result["next_step"] == "ckf_fit_covariance_semantics_audit"


def test_decide_case_c_only_material_on_fails():
    result = decide_case(
        {"official_adds_c_target": False, "official_residual_reference": "truth"},
        {"target_station_in_same_ckf_fit_fraction": 0.1},
        {"source_cin_shape_holds": True},
        {"c0_shape_holds": True, "c1_shape_holds": False, "material_is_primary_mismatch": True},
        {"systematic_transport_failure": False},
        _focus(),
    )
    assert result["primary_case"] == CASE_C


def test_decide_case_d_jacobian_systematic():
    result = decide_case(
        {"official_adds_c_target": False, "official_residual_reference": "truth"},
        {"target_station_in_same_ckf_fit_fraction": 0.1},
        {"source_cin_shape_holds": True},
        {"c0_shape_holds": True, "c1_shape_holds": True, "material_is_primary_mismatch": False},
        {"systematic_transport_failure": True},
        _focus(),
    )
    assert result["primary_case"] == CASE_D


def test_decide_case_e_when_shared_fit_and_cin_both_fail():
    result = decide_case(
        {"official_adds_c_target": False, "official_residual_reference": "truth"},
        {"target_station_in_same_ckf_fit_fraction": 1.0},
        {"source_cin_shape_holds": False},
        {"c0_shape_holds": False, "c1_shape_holds": False, "material_is_primary_mismatch": False},
        {"systematic_transport_failure": False},
        _focus(),
    )
    assert result["primary_case"] == CASE_E
    assert CASE_A in result["secondary_cases"] or CASE_A in result["active_mechanisms"]
    assert CASE_B in result["active_mechanisms"]
    assert result["next_step"] == "no_forced_single_root_cause"


def test_decide_case_refuses_dropped_focus():
    with pytest.raises(ShapeDiagnosisError, match="100043/37"):
        decide_case(
            {"official_adds_c_target": False},
            {"target_station_in_same_ckf_fit_fraction": 1.0},
            {"source_cin_shape_holds": False},
            {"c0_shape_holds": False, "c1_shape_holds": False, "material_is_primary_mismatch": False},
            {"systematic_transport_failure": False},
            {"retained": False, "loo_shape_holds": False, "chi2_share": 0.8},
        )
