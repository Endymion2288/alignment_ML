"""Task A q/p semantics.  No sealed test, no covariance rescale, no truth seed."""

from __future__ import annotations

import numpy as np
import pytest

from datasets.access_policy import AccessPolicyError, AccessScope, authorize_path
from datasets.qoverp_semantics import (
    DECISION_ESTABLISHED,
    DECISION_NOT_ESTABLISHED,
    MECHANISM_NOT_EXPORTED,
    SEGMENTFIT_DUMMY_QOVERP_PER_MEV,
    QoverPSemanticsError,
    classify_candidate,
    decide,
    dummy_segmentfit_variance_per_mev2,
    inherit_frozen_stage,
    load_config,
    q_over_p_per_mev_from_charge_and_p,
    refuse_covariance_rescale,
    refuse_delete_qoverp_column,
    refuse_dummy_as_physical,
    refuse_truth_as_real_data_solution,
    state_definition,
)


def test_config_inherits_frozen_wb87_93_94():
    config = load_config()
    inherited = inherit_frozen_stage(config)
    assert inherited["workbook_87"]["decision"] == "faseracts_transport_covariance_not_validated"
    assert inherited["workbook_93"]["contract_established_for_t12"] is False
    assert inherited["workbook_94"]["conditional_go"] is False
    assert inherited["workbook_94"]["unconstrained_tracker_only_stopped"] is True
    assert config["do_not_use_truth_q_over_p_as_real_data_solution"] is True
    assert config["do_not_start_new_source_campaign"] is True


def test_dummy_segmentfit_formula_and_units():
    assert SEGMENTFIT_DUMMY_QOVERP_PER_MEV == pytest.approx(1.0 / 1.0e5)
    assert dummy_segmentfit_variance_per_mev2() == pytest.approx(5.0e-6)
    assert dummy_segmentfit_variance_per_mev2(1.0e-5) == pytest.approx(50000.0 * 1.0e-10)
    assert q_over_p_per_mev_from_charge_and_p(-1.0, 1.0e5) == pytest.approx(-1.0e-5)
    assert q_over_p_per_mev_from_charge_and_p(1.0, 2.0e5) > 0.0
    definition = state_definition()
    assert definition["q_over_p_native_unit"] == "per_MeV"
    assert definition["q_over_p_signed"] is True
    assert definition["native_athena"][-1] == "q_over_p"


def test_forbidden_repairs_are_hard_errors():
    with pytest.raises(QoverPSemanticsError, match="truth"):
        refuse_truth_as_real_data_solution()
    with pytest.raises(QoverPSemanticsError, match="dummy"):
        refuse_dummy_as_physical()
    with pytest.raises(QoverPSemanticsError, match="column"):
        refuse_delete_qoverp_column()
    with pytest.raises(QoverPSemanticsError, match="rescale"):
        refuse_covariance_rescale()
    with pytest.raises(AccessPolicyError):
        authorize_path("outputs/sealed_test/tracklets.root", AccessScope.DEVELOPMENT_VALIDATION)


def test_dummy_and_truth_cannot_be_physical():
    dummy = classify_candidate(
        {
            "reconstruction_chain": True,
            "is_dummy_segmentfit": True,
            "is_truth": False,
            "has_estimate": True,
            "has_uncertainty": True,
            "has_correlation": False,
            "provenance_complete": True,
        }
    )
    assert dummy["role"] == "rejected"
    assert dummy["complete_physical_prior"] is False
    truth = classify_candidate(
        {
            "reconstruction_chain": False,
            "is_dummy_segmentfit": False,
            "is_truth": True,
            "has_estimate": True,
            "has_uncertainty": True,
            "has_correlation": True,
            "provenance_complete": True,
        }
    )
    assert truth["role"] == "rejected"
    assert "truth" in truth["reason"]


def test_ckf_mean_only_is_incomplete():
    mean_only = classify_candidate(
        {
            "reconstruction_chain": True,
            "is_dummy_segmentfit": False,
            "is_truth": False,
            "has_estimate": True,
            "has_uncertainty": False,
            "has_correlation": False,
            "provenance_complete": True,
        }
    )
    assert mean_only["role"] == "incomplete"
    assert mean_only["reason"] == "mean_state_only_uncertainty_not_exported"
    diagonal = classify_candidate(
        {
            "reconstruction_chain": True,
            "is_dummy_segmentfit": False,
            "is_truth": False,
            "has_estimate": True,
            "has_uncertainty": True,
            "has_correlation": False,
            "provenance_complete": True,
        }
    )
    assert diagonal["role"] == "incomplete"
    assert "correlation" in diagonal["reason"]


def test_complete_physical_candidate_would_pass():
    candidate = {
        "name": "synthetic_complete_ckf_5x5",
        "reconstruction_chain": True,
        "is_dummy_segmentfit": False,
        "is_truth": False,
        "has_estimate": True,
        "has_uncertainty": True,
        "has_correlation": True,
        "provenance_complete": True,
    }
    assert classify_candidate(candidate)["complete_physical_prior"] is True
    decision = decide([candidate])
    assert decision["verdict"] == "PASS"
    assert decision["decision"] == DECISION_ESTABLISHED
    assert decision["measurement_model_v2_entered"] is False
    assert decision["truth_qoverp_used_as_real_data_solution"] is False


def test_current_export_candidates_fail():
    decision = decide(
        [
            {
                "name": "segmentfit_dummy_qoverp",
                "reconstruction_chain": True,
                "is_dummy_segmentfit": True,
                "is_truth": False,
                "has_estimate": True,
                "has_uncertainty": True,
                "has_correlation": False,
                "provenance_complete": True,
            },
            {
                "name": "ntuple_mode_1_or_2_truth_qoverp",
                "reconstruction_chain": False,
                "is_dummy_segmentfit": False,
                "is_truth": True,
                "has_estimate": True,
                "has_uncertainty": False,
                "has_correlation": False,
                "provenance_complete": True,
            },
            {
                "name": "ckf_track_mean_momentum",
                "reconstruction_chain": True,
                "is_dummy_segmentfit": False,
                "is_truth": False,
                "has_estimate": True,
                "has_uncertainty": False,
                "has_correlation": False,
                "provenance_complete": True,
            },
        ]
    )
    assert decision["verdict"] == "FAIL"
    assert decision["decision"] == DECISION_NOT_ESTABLISHED
    assert decision["mechanism"] == MECHANISM_NOT_EXPORTED
    assert decision["n_complete_physical_priors"] == 0
    assert decision["contract_established_for_process_noise"] is False
    assert decision["unconstrained_tracker_only_stopped"] is True
    assert decision["state_definition"]["q_over_p_native_unit"] == "per_MeV"
    assert decision["units"]["q_over_p"] == "per_MeV"
    assert decision["covariance_source"]["physical_prior"] is None
    assert decision["uncertainty_propagation"]["rescale_allowed"] is False
    assert decision["validation_split"]["file_level_disjoint"] is True


def test_zero_momentum_is_illegal():
    with pytest.raises(QoverPSemanticsError, match="momentum"):
        q_over_p_per_mev_from_charge_and_p(1.0, 0.0)
    assert np.isfinite(dummy_segmentfit_variance_per_mev2())
