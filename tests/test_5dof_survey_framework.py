from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pytest
import yaml

from alignment.capture_criteria import (
    DEFAULT_ENGINEERING,
    SCHEMA_VERSION,
    evaluate_parameter_capture,
    overall_framework_capture,
    prior_contributions,
)
from alignment.five_dof_sampling import (
    FORBIDDEN_SINGLE_STEP_SEVERITY,
    SURVEY_PARAMETER,
    draw_joint_five_dof,
    five_dof_severity,
    sample_linear_regime_points,
)
from alignment.physical_jacobian import solve_physical_finite_difference
from scripts.prepare_multidof_alignment_iteration import compile_iteration


def _criteria(registered: dict[str, float]) -> dict:
    per = {}
    for name, sigma in registered.items():
        role = "survey_constrained" if name == "ift_dz_mm" else "free"
        per[name] = {
            "role": role,
            "engineering_tolerance": None if role != "free" else DEFAULT_ENGINEERING[name],
            "registered_sigma": None if role != "free" else sigma,
            "statistical_k": 3.0,
            "coverage_k": 3.0,
        }
    return {"per_parameter": per, "statistical_k": 3.0, "coverage_k": 3.0}


def test_joint_five_dof_sampling_is_seed_stable_and_zeros_dz():
    first = sample_linear_regime_points(
        seed=20260819,
        start_severity=0.12,
        linear_held_out=(0.10, 0.14),
        stress_held_out=(0.20,),
    )
    second = sample_linear_regime_points(
        seed=20260819,
        start_severity=0.12,
        linear_held_out=(0.10, 0.14),
        stress_held_out=(0.20,),
    )
    assert first["start"] == second["start"]
    assert first["start"][SURVEY_PARAMETER] == 0.0
    assert math.isclose(five_dof_severity(first["start"]), 0.12, rel_tol=0.0, abs_tol=1.0e-12)
    for item in first["linear_held_out"] + first["stress_held_out"]:
        assert item["alignment_parameter_values"][SURVEY_PARAMETER] == 0.0
        assert math.isclose(
            five_dof_severity(item["alignment_parameter_values"]),
            item["severity"],
            rel_tol=0.0,
            abs_tol=1.0e-12,
        )
    start_components = [abs(first["start"][name]) for name in ("ift_dx_mm", "ift_dy_mm", "ift_rx_mrad", "ift_ry_mrad", "ift_rz_mrad")]
    assert sum(value > 1.0e-12 for value in start_components) >= 2


def test_sampling_rejects_nonlinear_single_step_severity():
    rng = np.random.default_rng(0)
    with pytest.raises(ValueError, match="forbidden"):
        draw_joint_five_dof(rng, severity=FORBIDDEN_SINGLE_STEP_SEVERITY)
    with pytest.raises(ValueError, match="start severity"):
        sample_linear_regime_points(seed=1, start_severity=0.5, linear_held_out=(), stress_held_out=())


def test_dual_capture_passes_pilot_dy_statistical_and_fails_engineering():
    criteria = _criteria(
        {
            "ift_dx_mm": 0.0289,
            "ift_dy_mm": 0.3133,
            "ift_rx_mrad": 0.3841,
            "ift_ry_mrad": 0.1114,
            "ift_rz_mrad": 0.5018,
            "ift_dz_mm": 4.25,
        }
    )
    dy = evaluate_parameter_capture(name="ift_dy_mm", error=0.1022, fit_sigma=0.3133, criteria=criteria)
    assert dy["engineering_capture"] is False
    assert dy["statistical_capture"] is True
    assert dy["coverage_capture"] is True
    assert dy["track_capture"] is True
    dx = evaluate_parameter_capture(name="ift_dx_mm", error=-0.0022, fit_sigma=0.0289, criteria=criteria)
    rx = evaluate_parameter_capture(name="ift_rx_mrad", error=-0.0897, fit_sigma=0.3841, criteria=criteria)
    ry = evaluate_parameter_capture(name="ift_ry_mrad", error=-0.0132, fit_sigma=0.1114, criteria=criteria)
    rz = evaluate_parameter_capture(name="ift_rz_mrad", error=0.2264, fit_sigma=0.5018, criteria=criteria)
    dz = evaluate_parameter_capture(name="ift_dz_mm", error=-0.4867, fit_sigma=4.25, criteria=criteria)
    aggregate = overall_framework_capture([dx, dy, rx, ry, rz, dz])
    assert aggregate["engineering_capture_success"] is False
    assert aggregate["framework_capture_success"] is True
    assert dz["track_capture"] is None


def test_coverage_rejects_large_error_even_when_registered_sigma_is_loose():
    criteria = _criteria(
        {
            "ift_dx_mm": 0.03,
            "ift_dy_mm": 0.3,
            "ift_rx_mrad": 0.4,
            "ift_ry_mrad": 0.1,
            "ift_rz_mrad": 0.5,
        }
    )
    dy = evaluate_parameter_capture(name="ift_dy_mm", error=0.4, fit_sigma=0.05, criteria=criteria)
    assert dy["statistical_capture"] is True
    assert dy["coverage_capture"] is False
    assert dy["track_capture"] is False


def test_survey_prior_dominates_weak_dz_column():
    derivative = np.zeros((4, 4, 2))
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 0.01
    nominal = np.zeros((4, 4))
    positive = np.asarray([nominal + derivative[:, :, parameter] for parameter in range(2)])
    negative = np.asarray([nominal - derivative[:, :, parameter] for parameter in range(2)])
    observed = nominal + np.einsum("nrp,p->nr", derivative, np.asarray([0.2, 0.4]))
    covariance = np.tile(np.eye(4), (4, 1, 1))
    prior = np.asarray([math.nan, 5.0])
    fit = solve_physical_finite_difference(
        nominal,
        positive,
        negative,
        observed,
        covariance,
        parameter_names=("ift_dx_mm", "ift_dz_mm"),
        positive_values=(1.0, 2.0),
        negative_values=(-1.0, -2.0),
        parameter_scales=(5.0, 5.0),
        prior_sigma_native=prior,
    )
    rows = prior_contributions(
        fit.parameter_names,
        normal_matrix_native=fit.normal_matrix_native,
        covariance_native=fit.covariance_native,
        prior_sigma_native=fit.prior_sigma_native,
    )
    by_name = {row["name"]: row for row in rows}
    assert by_name["ift_dx_mm"]["prior_sigma"] is None
    assert by_name["ift_dx_mm"]["track_dominated"] is True
    assert by_name["ift_dz_mm"]["prior_dominated"] is True
    assert by_name["ift_dz_mm"]["prior_information_fraction"] > 0.5


def test_compile_iteration_accepts_sampled_held_out_points(tmp_path):
    sampled = sample_linear_regime_points(
        seed=20260819,
        start_severity=0.12,
        linear_held_out=(0.10,),
        stress_held_out=(0.20,),
    )
    template = yaml.safe_load(Path("configs/physical_refit_5dof_survey_dz_iteration.yaml").read_text(encoding="utf-8"))
    template["physical_refit_capture_scan"]["held_out_closure_points"] = [
        {
            "name": "closure_linear_00",
            "alignment_parameter_values": sampled["linear_held_out"][0]["alignment_parameter_values"],
        }
    ]
    compiled, contract = compile_iteration(template, iteration=0, current_values=sampled["start"])
    names = {point["name"] for point in compiled["physical_refit_capture_scan"]["rigid_points"]}
    assert "iteration_00_anchor" in names
    assert "iteration_00_fd_ift_dz_mm_p" in names
    assert "iteration_00_closure_linear_00" in names
    assert contract["anchor_parameter_values"]["ift_dz_mm"] == 0.0
    assert len(compiled["physical_refit_capture_scan"]["rigid_points"]) == 1 + 1 + 12 + 1


def test_register_script_uses_train_pilot_and_excludes_validation(tmp_path):
    closure = {
        "target_point": "iteration_02_closure_6dof_c",
        "observation_statistics": {"unique_physical_edges": 193},
        "parameters": [
            {"name": name, "recovered_sigma": sigma, "local_delta_error": error}
            for name, sigma, error in (
                ("ift_dx_mm", 0.0289, -0.0022),
                ("ift_dy_mm", 0.3133, 0.1022),
                ("ift_dz_mm", 4.25, -0.49),
                ("ift_rx_mrad", 0.3841, -0.09),
                ("ift_ry_mrad", 0.1114, -0.013),
                ("ift_rz_mrad", 0.5018, 0.226),
            )
        ],
    }
    source = tmp_path / "train_closure.json"
    source.write_text(json.dumps(closure), encoding="utf-8")
    output = tmp_path / "criteria.json"
    from scripts.register_5dof_capture_criteria import main as register_main
    import sys

    previous = sys.argv
    sys.argv = ["register_5dof_capture_criteria.py", "--train-closure", str(source), "--output", str(output)]
    try:
        register_main()
    finally:
        sys.argv = previous
    payload = json.loads(output.read_text(encoding="utf-8"))
    assert payload["schema_version"] == SCHEMA_VERSION
    assert payload["validation_used_in_registration"] is False
    assert payload["test_data_accessed"] is False
    assert payload["registration_self_score"]["engineering_capture_success"] is False
    assert payload["registration_self_score"]["framework_capture_success"] is True
    assert payload["per_parameter"]["ift_dz_mm"]["role"] == "survey_constrained"
    assert payload["per_parameter"]["ift_dy_mm"]["registered_sigma"] == 0.3133
