from __future__ import annotations

from pathlib import Path

import numpy as np

from alignment.five_dof_sampling import FREE_PARAMETERS, SURVEY_PARAMETER
from alignment.hierarchical_v1 import STATION_SOLVE_PARAMETERS
from alignment.real_data_reduced_station_mode import (
    DECISION_MONITORING_ONLY,
    DECISION_V2_CANDIDATE,
    TRACK_DRIVEN_PARAMETERS,
    a_subspace_projection,
    assert_track_driven_names,
    attach_leakage,
    campaign_allows_cdx_mode,
    campaign_allows_geometry_write,
    classify_reduced_campaign,
    evaluate_reduced_mode,
    hard_reject_dx_ry_contamination,
    implied_cdx_from_dx_ry,
    load_reduced_mode_config,
    predeclared_modes,
    run_to_run_parameter_consistency,
    solve_reduced_self_nulling,
)


A_NATIVE = {
    "ift_dx_mm": -59.21304180539863,
    "ift_dy_mm": 1.702908121992806,
    "ift_dz_mm": 10.118594792888189,
    "ift_rx_mrad": -2.029981711910054,
    "ift_ry_mrad": -31.90780811244653,
    "ift_rz_mrad": -0.36665518323206925,
}


def _config():
    return load_reduced_mode_config(
        Path("configs/operating_protocol_v1_real_data_reduced_station_mode_feasibility_v1.yaml")
    )


def _identity_covariance(n_obs: int) -> np.ndarray:
    return np.repeat(np.eye(4)[None, :, :], n_obs, axis=0)


def _isolated_js(true_delta: dict[str, float], n_per_param: int = 4) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    names = list(STATION_SOLVE_PARAMETERS)
    n_obs = n_per_param * len(names)
    jacobian = np.zeros((n_obs, 4, 6), dtype=np.float64)
    residual = np.zeros((n_obs, 4), dtype=np.float64)
    for index, name in enumerate(names):
        for copy in range(n_per_param):
            row = index * n_per_param + copy
            jacobian[row, 0, index] = 1.0
            residual[row, 0] = -float(true_delta.get(name, 0.0))
    return jacobian, _identity_covariance(n_obs), residual


def test_config_freezes_write_cdx_v2_and_survey_dz():
    config = _config()
    assert config["geometry_write_allowed"] is False
    assert config["official_conditions_db_write"] is False
    assert config["cdx_mode_blocked"] is True
    assert config["track_driven_dz"] is False
    assert float(config["survey_dz_mm"]) == 0.0
    assert config["do_not_retrain_v2"] is True
    assert config["schur_projection_production"] is False
    assert campaign_allows_cdx_mode(DECISION_V2_CANDIDATE) is False
    assert campaign_allows_geometry_write(DECISION_MONITORING_ONLY) is False
    modes = predeclared_modes(config)
    assert "three_dof_dy_rx_rz" in modes
    assert SURVEY_PARAMETER not in modes["three_dof_dy_rx_rz"]
    assert SURVEY_PARAMETER not in modes["four_dof_dx_fixed"]


def test_dz_cannot_enter_track_driven_solve():
    try:
        assert_track_driven_names(["ift_dy_mm", SURVEY_PARAMETER])
    except ValueError as exc:
        assert "dz" in str(exc)
    else:
        raise AssertionError("expected dz rejection")


def test_three_dof_is_orthogonal_and_four_dof_with_ry_is_not():
    three = a_subspace_projection(("ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"), A_NATIVE)
    four = a_subspace_projection(("ift_dy_mm", "ift_rx_mrad", "ift_ry_mrad", "ift_rz_mrad"), A_NATIVE)
    five = a_subspace_projection(TRACK_DRIVEN_PARAMETERS, A_NATIVE)
    assert three < 0.05
    assert four > 0.15
    assert abs(five - 1.0) < 1.0e-12


def test_dx_ry_contamination_is_hard_reject():
    implied = implied_cdx_from_dx_ry({"ift_dx_mm": -1.3, "ift_ry_mrad": 20.0}, A_NATIVE)
    reject = hard_reject_dx_ry_contamination(("ift_dx_mm", "ift_dy_mm"), implied)
    assert reject["hard_reject"] is True
    assert reject["cannot_rescue_by_more_events_or_looser_thresholds"] is True
    safe = hard_reject_dx_ry_contamination(("ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"), {"max_abs_implied_C_dx_um": None})
    assert safe["hard_reject"] is False
    assert safe["contains_dx_or_ry"] is False


def test_reduced_solve_recovers_isolated_three_dof():
    truth = {"ift_dy_mm": 0.02, "ift_rx_mrad": -0.05, "ift_rz_mrad": 0.03}
    jacobian, covariance, residual = _isolated_js(truth)
    fit = solve_reduced_self_nulling(
        jacobian,
        covariance,
        residual,
        all_parameter_names=STATION_SOLVE_PARAMETERS,
        floated=("ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"),
    )
    assert fit["full_rank"] is True
    assert fit["survey_dz_mm"] == 0.0
    assert SURVEY_PARAMETER not in fit["parameter_names"]
    assert abs(fit["delta"]["ift_dy_mm"] - 0.02) < 1.0e-9
    assert fit["residual_reduction_is_not_alignment_success"] is True


def test_evaluate_admits_consistent_orthogonal_mode():
    jacobian, covariance, residual = _isolated_js(
        {"ift_dy_mm": 0.01, "ift_rx_mrad": 0.02, "ift_rz_mrad": -0.01}
    )
    fit = solve_reduced_self_nulling(
        jacobian,
        covariance,
        residual,
        all_parameter_names=STATION_SOLVE_PARAMETERS,
        floated=("ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"),
    )
    leakage = attach_leakage(fit, A_NATIVE)
    consistency = run_to_run_parameter_consistency(
        fit["delta"], fit["delta"], fit["sigma"], fit["sigma"], fit["parameter_names"]
    )
    result = evaluate_reduced_mode(
        mode_name="three_dof_dy_rx_rz",
        floated=fit["parameter_names"],
        per_run_solve={14973: fit, 14974: fit},
        per_run_leakage={14973: leakage, 14974: leakage},
        consistency=consistency,
        transfers=[{"chi2_ratio": 1.0}, {"chi2_ratio": 0.99}],
        blind_transfer_ok=True,
    )
    assert result["admitted_as_v2_candidate"] is True
    assert result["geometry_write_allowed"] is False


def test_evaluate_rejects_inconsistent_or_contaminated_mode():
    jacobian, covariance, residual = _isolated_js({"ift_dx_mm": -3.0, "ift_dy_mm": 0.01})
    fit = solve_reduced_self_nulling(
        jacobian,
        covariance,
        residual,
        all_parameter_names=STATION_SOLVE_PARAMETERS,
        floated=("ift_dx_mm", "ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"),
    )
    leakage = attach_leakage(fit, A_NATIVE)
    assert leakage["hard_reject"] is True
    other = dict(fit)
    other["delta"] = {name: -10.0 * float(fit["delta"][name]) for name in fit["parameter_names"]}
    consistency = run_to_run_parameter_consistency(
        fit["delta"], other["delta"], fit["sigma"], fit["sigma"], fit["parameter_names"]
    )
    result = evaluate_reduced_mode(
        mode_name="four_dof_ry_fixed",
        floated=fit["parameter_names"],
        per_run_solve={14973: fit, 14974: fit},
        per_run_leakage={14973: leakage, 14974: leakage},
        consistency=consistency,
        transfers=[{"chi2_ratio": 2.0}],
        blind_transfer_ok=True,
    )
    assert result["admitted_as_v2_candidate"] is False
    assert result["hard_isolation_reject"] is True
    assert result["cannot_rescue_by_more_events_or_looser_thresholds"] is True


def test_classify_monitoring_only_when_no_mode_admitted():
    result = classify_reduced_campaign(
        [
            {
                "mode_name": "four_dof_dx_fixed",
                "admitted_as_v2_candidate": False,
                "hard_isolation_reject": True,
                "floated_parameters": ["ift_dy_mm", "ift_rx_mrad", "ift_ry_mrad", "ift_rz_mrad"],
                "survey_fixed_parameters": [SURVEY_PARAMETER, "ift_dx_mm"],
            }
        ]
    )
    assert result["decision"] == DECISION_MONITORING_ONLY
    assert result["geometry_write_allowed"] is False
    assert result["v2_candidate_mode"] is None
    assert SURVEY_PARAMETER in result["survey_or_external_parameters"]
    assert set(FREE_PARAMETERS).issubset(set(result["survey_or_external_parameters"]))


def test_classify_v2_candidate_still_forbids_write():
    result = classify_reduced_campaign(
        [
            {
                "mode_name": "three_dof_dy_rx_rz",
                "admitted_as_v2_candidate": True,
                "hard_isolation_reject": False,
                "floated_parameters": ["ift_dy_mm", "ift_rx_mrad", "ift_rz_mrad"],
                "survey_fixed_parameters": [SURVEY_PARAMETER, "ift_dx_mm", "ift_ry_mrad"],
            }
        ]
    )
    assert result["decision"] == DECISION_V2_CANDIDATE
    assert result["geometry_write_allowed"] is False
    assert result["cdx_mode_allowed"] is False
    assert result["do_not_emit_cdx_payload"] is True


def test_residual_cannot_be_used_as_success():
    try:
        classify_reduced_campaign([], residual_used_as_success=True)
    except ValueError as exc:
        assert "residual" in str(exc)
    else:
        raise AssertionError("expected residual-success rejection")
