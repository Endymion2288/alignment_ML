"""Task B14M profiled weak-nuisance likelihood.  No V2, no B15, no prior."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.profiled_measurement_likelihood import (
    PINV_RELATIVE,
    ProfiledMeasurementLikelihoodError,
    classify_nuisance,
    prediction_uncertainty_from_hessian,
    refuse_alignment_rank_tolerance,
    refuse_ridge,
    schur_profile_hessian,
    symmetric_pinv,
    validate_linear_profile_agreement,
)
from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.ckf_contract_validation import load_config as load_wb103
from datasets.profiled_weak_nuisance_likelihood import (
    CASE_A,
    CASE_B,
    CASE_C,
    CASE_D,
    CASE_E,
    ProfiledWeakNuisanceError,
    decide_case,
    inherit_frozen_stage,
    load_config,
    measurement_likelihood_contract,
    profiled_state_partition,
    refuse_b15,
    refuse_best_seed,
    refuse_cin_likelihood,
    refuse_covariance_rescale,
    refuse_focus_drop,
    refuse_marginalization,
    refuse_measurement_model_v2,
    refuse_prior_introduction,
    refuse_qoverp_deletion,
    refuse_qoverp_fix,
    refuse_seed_scale_choice,
    refuse_truth_qoverp,
)
from datasets.target_independent_weak_prior_contract import FROZEN_DIRECTION_CLASS


def test_config_inherits_wb113_case_b_and_wb103():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    wb103 = load_wb103()
    assert config["eligibility"] == wb103["eligibility"]
    assert config["do_not_enter_b15"] is True
    assert config["do_not_enter_b14m"] is False
    assert config["do_not_execute_marginalization"] is True
    assert config["do_not_use_wb109_cin_as_likelihood"] is True
    assert config["do_not_use_ridge"] is True
    assert config["do_not_force_5d_lto_covariance"] is True
    assert config["do_not_delete_qoverp"] is True
    assert config["do_not_fix_qoverp"] is True
    assert config["do_not_select_best_seed"] is True
    assert config["profiling_is_primary_contract"] is True
    assert config["marginalization_executed"] is False
    assert float(config["profile_numerical"]["pinv_relative"]) == PINV_RELATIVE
    assert inherited["workbook_113"]["decision"] == "target_independent_prior_not_available"
    assert inherited["workbook_113"]["b14m_authorized"] is True
    assert inherited["workbook_113"]["prior_introduced"] is False
    assert inherited["workbook_112"]["typical_direction_class"] == FROZEN_DIRECTION_CLASS
    assert inherited["workbook_112"]["five_d_state_physically_supported"] is False
    assert inherited["workbook_111"]["decision"] == "mixed_or_inconclusive"
    assert inherited["workbook_110"]["decision"] == "lto_input_covariance_shape_not_validated"
    assert inherited["workbook_109"]["decision"] == (
        "leave_target_out_state_materialization_established"
    )
    assert "profiled_weak_nuisance_likelihood_v1" in config["output_root"]


def test_forbidden_repairs():
    with pytest.raises(ProfiledWeakNuisanceError, match="truth"):
        refuse_truth_qoverp()
    with pytest.raises(ProfiledWeakNuisanceError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(ProfiledWeakNuisanceError, match="seed scale"):
        refuse_seed_scale_choice()
    with pytest.raises(ProfiledWeakNuisanceError, match="prior"):
        refuse_prior_introduction()
    with pytest.raises(ProfiledWeakNuisanceError, match="nuisance"):
        refuse_qoverp_deletion()
    with pytest.raises(ProfiledWeakNuisanceError, match="fixed"):
        refuse_qoverp_fix()
    with pytest.raises(ProfiledWeakNuisanceError, match="100043/37"):
        refuse_focus_drop()
    with pytest.raises(ProfiledWeakNuisanceError, match="Measurement Model V2"):
        refuse_measurement_model_v2()
    with pytest.raises(ProfiledWeakNuisanceError, match="B15"):
        refuse_b15()
    with pytest.raises(ProfiledWeakNuisanceError, match="Cin"):
        refuse_cin_likelihood()
    with pytest.raises(ProfiledWeakNuisanceError, match="marginalization"):
        refuse_marginalization()
    with pytest.raises(ProfiledWeakNuisanceError, match="best seed"):
        refuse_best_seed()
    with pytest.raises(ProfiledMeasurementLikelihoodError, match="ridge"):
        refuse_ridge()
    with pytest.raises(ProfiledMeasurementLikelihoodError, match="0.01"):
        refuse_alignment_rank_tolerance()
    with pytest.raises(ProfiledMeasurementLikelihoodError, match="ridge"):
        symmetric_pinv(np.eye(2), ridge=1.0e-6)
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_synthetic_joint_profile_schur_agree():
    result = validate_linear_profile_agreement()
    assert result["passed"] is True
    assert result["ridge_added"] is False
    assert result["alignment_rank_tolerance_used"] is False
    assert result["full_rank"]["alpha_agrees"] is True
    assert result["full_rank"]["nuisance_agrees"] is True
    assert result["full_rank"]["chi2_agrees"] is True
    assert result["rank_deficient_nuisance"]["diagnosed"] is True
    assert result["rank_deficient_nuisance"]["nuisance_rank"] < 3
    assert result["rank_deficient_nuisance"]["forced_invertible"] is False


def test_rank_deficient_hnn_is_diagnosed_not_regularized():
    hessian = np.diag([10.0, 4.0, 1.0e-16, 8.0, 0.0])
    schur = schur_profile_hessian(hessian)
    assert schur["ridge_added"] is False
    assert schur["nuisance_rank"] < 3
    assert schur["nuisance_null_basis"].shape[1] >= 1


def test_null_direction_is_not_zero_uncertainty():
    hessian = np.diag([10.0, 0.0, 0.0, 8.0, 0.0])
    jac = np.array([[1.0, 0.0, 0.0, 0.0, 1.0], [0.0, 0.0, 0.0, 1.0, 0.0]])
    result = prediction_uncertainty_from_hessian(hessian, jac)
    assert result["ridge_added"] is False
    assert result["seed_covariance_filled"] is False
    assert result["pinv_null_not_zero_uncertainty"] is True
    assert result["observable_finite"][0] is False
    assert result["observable_finite"][1] is True
    assert result["prediction_not_identified"] is True
    assert not np.isfinite(result["prediction_covariance_finite_mask"][0, 0])


def test_partition_does_not_truncate():
    config = load_config()
    part = profiled_state_partition(config)
    assert part["truncated"] is False
    assert part["nuisance_fixed_to_seed"] is False
    assert part["q_over_p_deleted"] is False
    assert part["alpha"]["native"] == ["loc0", "theta"]
    assert part["nu"]["native"] == ["loc1", "phi", "q_over_p"]
    contract = measurement_likelihood_contract(config)
    assert contract["wb109_cin_used_as_likelihood"] is False
    assert contract["wb107_cin_used_as_likelihood"] is False


def test_classify_nuisance():
    assert classify_nuisance([10.0, 4.0, 2.0]) == "identified"
    assert classify_nuisance([10.0, 4.0, 1.0e-16]) == "flat"
    assert classify_nuisance([10.0, 4.0, 1.0e-6], weak_relative=1.0e-4) == "weakly_identified"
    assert classify_nuisance([10.0], fit_failed=True) == "fit_failed"
    assert classify_nuisance([10.0], multimodal=True) == "multimodal"


def _base_inventory(**overrides: object) -> dict:
    payload = {
        "prior_introduced": False,
        "ridge_added": False,
        "q_over_p_deleted": False,
        "q_over_p_fixed": False,
        "best_seed_selected": False,
        "marginalization_executed": False,
        "denominator": {"frozen_denominator_holds": True},
        "numerical_validation": {"passed": True},
        "acts_profile_materialized": True,
        "numerically_unstable": False,
        "seed_invariance": {
            "n_groups_with_multiple_inits": 3,
            "prediction_seed_invariant": True,
            "optimizer_underconverged": False,
        },
        "predictions": {
            "n_prediction_identified": 6,
            "n_prediction_not_identified": 0,
        },
        "identifiability": {"qoverp_flat_success_rows": 4},
        "calibration": {"passed": False},
    }
    payload.update(overrides)
    return payload


def test_case_b_identifiable_prediction_unresolved_nuisance():
    result = decide_case(_base_inventory())
    assert result["decision"] == CASE_B
    assert result["b15_authorized"] is False
    assert result["prior_introduced"] is False
    assert result["do_not_force_5d_lto_covariance"] is True
    assert result["marginalization_not_defined_without_prior"] is True
    assert result["next_step"] == "observable_space_measurement_model_not_5d_cin"


def test_case_a_requires_calibration():
    result = decide_case(
        _base_inventory(
            identifiability={"qoverp_flat_success_rows": 0},
            calibration={"passed": True},
        )
    )
    assert result["decision"] == CASE_A
    assert result["verdict"] == "PASS"
    assert result["b15_authorized"] is False
    assert result["route_decision_after_pass_required"] is True


def test_case_c_seed_dependent_prediction():
    result = decide_case(
        _base_inventory(
            seed_invariance={
                "n_groups_with_multiple_inits": 3,
                "prediction_seed_invariant": False,
                "optimizer_underconverged": False,
            }
        )
    )
    assert result["decision"] == CASE_C
    assert result["verdict"] == "FAIL"


def test_case_d_underconverged_inits():
    result = decide_case(
        _base_inventory(
            seed_invariance={
                "n_groups_with_multiple_inits": 3,
                "prediction_seed_invariant": False,
                "optimizer_underconverged": True,
            }
        )
    )
    assert result["decision"] == CASE_D
    assert result["verdict"] == "FAIL"


def test_case_d_synthetic_failure():
    result = decide_case(_base_inventory(numerical_validation={"passed": False}))
    assert result["decision"] == CASE_D
    assert result["verdict"] == "FAIL"


def test_case_e_when_acts_profile_absent():
    result = decide_case(_base_inventory(acts_profile_materialized=False))
    assert result["decision"] == CASE_E
    assert result["b15_authorized"] is False
    assert result["measurement_model_v2_authorized"] is False
