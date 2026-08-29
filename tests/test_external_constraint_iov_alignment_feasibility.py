from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from alignment.external_constraint_iov_feasibility import (
    DECISION_SURVEY_AND_IOV,
    GO_NO_GO_QUESTIONS,
    SCHEMA_VERSION,
    augment_with_priors,
    collinear_toy_jacobian,
    combined_identifiability,
    decide_next_stage,
    external_constraint_inventory,
    iov_parameterization_report,
    jacobian_in_scan_units,
    load_framework_config,
    scan_priors,
    unlocked,
)
from alignment.operating_protocol_v1_final_closure import (
    FROZEN_V2_CHECKPOINT_SHA256,
    OPERATING_MODE,
)
from alignment.true_cluster_local_residual import assert_no_alignment_payload


def _config():
    return load_framework_config(
        Path("configs/external_constraint_iov_alignment_feasibility_v1.yaml")
    )


def test_config_freezes_protocol_and_forbids_invented_survey():
    config = _config()
    assert config["schema_version"] == SCHEMA_VERSION
    assert config["real_data_operating_mode"] == OPERATING_MODE
    assert config["do_not_invent_survey_numbers"] is True
    assert config["do_not_select_prior_from_residual"] is True
    assert config["do_not_enter_full_module_identifiability_map"] is True
    assert config["do_not_mix_cross_year_residuals_or_alignment_constants"] is True
    assert config["frozen_v2_checkpoint_sha256"] == FROZEN_V2_CHECKPOINT_SHA256
    physics = config["physics_scales"]
    assert physics["measurement_sigma_mm"] == pytest.approx(physics["strip_pitch_mm"])
    assert physics["measurement_sigma_source"] == "strip_pitch_not_residual_rms"
    assert 0.208 in config["prior_scan"]["sigma_ry_mrad"]
    assert 0.080 in config["prior_scan"]["sigma_cdx_mm"]
    assert config["parameterization"]["do_not_share_one_constant_set_across_years"] is True


def test_collinear_toy_plus_ry_prior_restores_rank3():
    config = _config()
    jacobian = collinear_toy_jacobian()
    none = combined_identifiability(jacobian, sigma_r_mm=0.080)
    assert none["rank"] == 2
    assert none["spectrum_kind"] == "column_normalized_scan_units"
    assert abs(float(none["ry_cdx_correlation"])) > 0.95
    tight = combined_identifiability(jacobian, sigma_r_mm=0.080, sigma_ry_mrad=0.05)
    assert tight["rank"] == 3
    assert abs(float(tight["ry_cdx_correlation"])) < 0.90
    assert unlocked(tight, config["unlock"]) is True
    weak = combined_identifiability(jacobian, sigma_r_mm=0.080, sigma_ry_mrad=100.0)
    assert unlocked(weak, config["unlock"]) is False


def test_prior_rows_are_constraint_matrix_not_residual_cut():
    jacobian = np.ones((5, 3), dtype=np.float64)
    scan = jacobian_in_scan_units(jacobian)
    np.testing.assert_allclose(scan[:, 1], 0.001 * np.ones(5))
    augmented = augment_with_priors(scan, sigma_r_mm=0.080, sigma_ry_mrad=1.0, sigma_cdx_mm=0.01)
    assert augmented.shape[0] == 7
    np.testing.assert_allclose(augmented[-2], [0.0, 1.0, 0.0])
    np.testing.assert_allclose(augmented[-1], [0.0, 0.0, 100.0])


def test_inventory_does_not_invent_survey_and_iov_rejects_shared_constants():
    config = _config()
    inventory = external_constraint_inventory(config)
    assert inventory["as_built_survey_available"] is False
    assert inventory["n_as_built_survey_numbers"] == 0
    assert all(not row.get("is_as_built_survey") for row in inventory["sources"])
    optical = next(row for row in inventory["sources"] if row["id"] == "independent_optical_or_laser_survey")
    assert optical["found"] is False
    assert optical["numerical_value"] is None
    layerpitch = next(row for row in inventory["sources"] if row["id"] == "geomdb_layerpitch")
    assert layerpitch["numerical_value"] == pytest.approx(31.5)
    assert layerpitch["is_as_built_survey"] is False
    assert_no_alignment_payload(inventory)
    parameterization = iov_parameterization_report(config)
    assert parameterization["do_not_share_one_constant_set_across_years"] is True
    assert parameterization["form"].startswith("theta^(y)")
    station_ry = next(row for row in parameterization["modes"] if row["mode"] == "station_ry")
    assert station_ry["year_specific"] is True
    cdx = next(row for row in parameterization["modes"] if row["mode"] == "C_dx_internal_deformation")
    assert cdx["survey_fixed_if_available"] is True


def test_decision_answers_both_questions_without_payload():
    config = _config()
    jacobian = collinear_toy_jacobian()
    scan = scan_priors(jacobian, config, sample_id="toy")
    decision = decide_next_stage(
        inventory=external_constraint_inventory(config),
        collinear_scan=scan,
        mixed_scan={"track_only_rank": 3},
        parameterization=iov_parameterization_report(config),
    )
    assert decision["decision"] == DECISION_SURVEY_AND_IOV
    assert decision["go_to_full_module_identifiability_map"] is False
    assert decision["emits_alignment_payload"] is False
    assert decision["survey_numbers_were_invented"] is False
    assert decision["answers"]["q2_year_iov_parameterization"]["share_one_constant_set"] is False
    assert decision["answers"]["q2_year_iov_parameterization"]["use_common_static_plus_year_delta"] is True
    assert GO_NO_GO_QUESTIONS[0] == decision["answers"]["q1_type_and_precision"]["question"]
    assert scan["weakest_ry_only_unlock_mrad"] is not None
    assert scan["ry_only_leftover_sigma_cdx_mm_at_tightest"] is not None
    assert_no_alignment_payload(decision)
