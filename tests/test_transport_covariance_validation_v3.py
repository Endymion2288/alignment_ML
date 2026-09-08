"""Task B8 Transport Covariance V3.  Contracted input only, no C/Q tune."""

from __future__ import annotations

import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.transport_covariance_validation_v3 import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    TransportV3Error,
    decide_case,
    dump_path_for_source,
    inherit_frozen_stage,
    load_config,
    refuse_covariance_rescale,
    refuse_focus_drop,
    refuse_measurement_model_v2,
    refuse_process_noise_tuning,
    refuse_q_psd_projection,
    refuse_raw_ckf,
    refuse_truth_qoverp,
)


def test_config_inherits_wb103_and_frozen_gates():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["official_input_scope"] == "contract_eligible"
    assert config["do_not_use_raw_ckf_as_official_input"] is True
    assert config["do_not_drop_focus_identity"] is True
    assert config["do_not_project_q_to_psd"] is True
    assert config["closure_gates"]["pencil_variance_ratio_max"] == 4.0
    assert inherited["workbook_103"]["decision"] == "ckf_reconstruction_contract_validated"
    assert inherited["workbook_103"]["primary_case"] == "input_scope_mismatch_confirmed"
    assert inherited["workbook_103"]["contract_sha256"].startswith("e8c1f927")
    assert "transport_covariance_validation_v3" in config["output_root"]


def test_dump_path_stays_on_wb98():
    path = dump_path_for_source(load_config(), "mc24_100043_00200_00299")
    assert path.as_posix().endswith(
        "outputs/acts_transport_dump_v1/dumps/mc24_100043_00200_00299/ckf_acts_transport.jsonl"
    )


def test_forbidden_repairs():
    with pytest.raises(TransportV3Error, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(TransportV3Error, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(TransportV3Error, match="process noise"):
        refuse_process_noise_tuning()
    with pytest.raises(TransportV3Error, match="PSD-projected"):
        refuse_q_psd_projection()
    with pytest.raises(TransportV3Error, match="raw CKF"):
        refuse_raw_ckf()
    with pytest.raises(TransportV3Error, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(TransportV3Error, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def _replay(*, calibrated: bool, chi2: bool, shape: bool) -> dict:
    gates = {
        "whitened_chi2_per_ndof": chi2,
        "generalized_eigenvalue": shape,
        "pencil_variance_ratio": shape,
    }
    pair = {"calibrated": calibrated, "gates": gates}
    return {
        "all_calibrated": calibrated,
        "per_pair": {"(0,1)": pair, "(0,2)": pair, "(0,3)": pair},
    }


def _official(con_cal: bool, val_cal: bool, con_shape: bool, val_shape: bool, val_chi2: bool):
    return {
        "splits": {
            "construction": {
                "model1": _replay(calibrated=con_cal, chi2=con_cal, shape=con_shape)
            },
            "validation": {
                "model1": _replay(calibrated=val_cal, chi2=val_chi2, shape=val_shape)
            },
        }
    }


def _eigen(con_shape: bool, val_shape: bool, val_chi2: bool) -> dict:
    return {
        "splits": {
            "construction": {"shape_holds": con_shape, "chi2_holds": con_shape},
            "validation": {"shape_holds": val_shape, "chi2_holds": val_chi2},
        }
    }


def _focus(share: float) -> dict:
    return {
        "retained": True,
        "events": [{"aggregate_chi2_share": {"chi2_share": share}}],
    }


def test_decide_case_a_requires_both_splits():
    result = decide_case(
        _official(True, True, True, True, True),
        _eigen(True, True, True),
        _focus(0.1),
        {"chi2_per_ndof": {"mean": 1.1, "median": 0.9, "finite_count": 1974}},
    )
    assert result["primary_case"] == CASE_A
    assert result["measurement_model_v2_authorized"] is True
    assert result["measurement_model_v2_entered"] is False


def test_decide_case_c_when_validation_shape_fails():
    result = decide_case(
        _official(False, False, False, False, True),
        _eigen(False, False, True),
        _focus(0.4),
        {"chi2_per_ndof": {"mean": 25.0, "median": 0.11, "finite_count": 1974}},
    )
    assert result["primary_case"] == CASE_C
    assert CASE_B in result["secondary_cases"]
    assert result["measurement_model_v2_authorized"] is False
    assert result["focus_retained"] is True


def test_decide_case_b_when_typical_holds_and_tail_blocks():
    result = decide_case(
        _official(False, True, False, True, True),
        _eigen(False, True, True),
        _focus(0.5),
        {"chi2_per_ndof": {"mean": 20.0, "median": 0.2, "finite_count": 1974}},
    )
    assert result["primary_case"] == CASE_B
    assert result["next_step"] == "ckf_momentum_state_audit"
    assert CASE_D != result["primary_case"]


def test_decide_case_refuses_dropped_focus():
    with pytest.raises(TransportV3Error, match="100043/37"):
        decide_case(
            _official(False, False, False, False, False),
            _eigen(False, False, False),
            {"retained": False, "events": []},
            {"chi2_per_ndof": {"mean": 25.0, "median": 0.11, "finite_count": 1974}},
        )


def test_decide_case_d_when_sample_too_small():
    result = decide_case(
        _official(False, False, False, False, False),
        _eigen(False, False, False),
        _focus(0.1),
        {"chi2_per_ndof": {"mean": 2.0, "median": 0.2, "finite_count": 12}},
    )
    assert result["primary_case"] == CASE_D
    assert result["next_step"] == "no_new_source_campaign"
    assert result["measurement_model_v2_authorized"] is False
