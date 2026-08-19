from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from scripts.compare_route_selected_updates import (
    _load_update,
    _per_source_solve,
    frozen_decision,
)

_NAMES = ["ift_dx_mm", "ift_dy_mm", "ift_ry_mrad"]


def _fake_update(root: Path, errors=(0.01, -0.02, 0.1)) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    parameters = []
    for name, anchor, error in zip(_NAMES, (-0.14, 0.11, 0.74), errors):
        parameters.append(
            {
                "name": name,
                "station_id": 0,
                "component": "dx",
                "unit": "mm",
                "anchor_value": anchor,
                "target_value": 0.0,
                "expected_delta_to_target": -anchor,
                "recovered_local_delta": -anchor + error,
                "local_delta_error": error,
                "proposed_next_value": error,
                "recovered_sigma": 0.01,
                "observability_fraction": 1.0,
                "capture_tolerance": None,
                "capture_success": None,
            }
        )
    summary = {
        "parameters": parameters,
        "probe_points": {name: {"positive": f"{name}_p", "negative": f"{name}_m"} for name in _NAMES},
        "normal_matrix_rank": 3,
        "normal_matrix_condition_number": 80.0,
        "parameter_correlation": np.eye(3).tolist(),
        "response_chi2": 10.0,
        "response_ndof": 100,
        "capture_success": None,
        "association_contract": {
            "target": {"anchor_edge_audit": {"anchor_edge_availability": 0.97}},
            "positive": {name: {"anchor_edge_audit": {"anchor_edge_availability": 0.9}} for name in _NAMES},
            "negative": {name: {"anchor_edge_audit": {"anchor_edge_availability": 0.9}} for name in _NAMES},
            "selected_route_overlap": {
                "common_selected_observations": 100,
                "anchor_common_fraction": 0.9,
            },
        },
    }
    (root / "route_selected_update.json").write_text(json.dumps(summary), encoding="utf-8")

    rng = np.random.default_rng(7)
    n_obs = 6
    derivative = rng.normal(size=(n_obs, 4, 3))
    true_delta = np.asarray([0.14, -0.11, -0.74])
    response = derivative @ true_delta
    covariance = np.repeat(np.eye(4)[None, :, :], n_obs, axis=0)
    keys = []
    for row in range(n_obs):
        source_index = row % 2
        origin_run = 9_000_000_000 + source_index * 100
        signature = json.dumps([{"origin_run_id": origin_run}])
        keys.append(json.dumps(["pooled", 996000, row, signature, 0, 1]))
    np.savez_compressed(
        root / "route_selected_update_arrays.npz",
        parameter_names=np.asarray(_NAMES),
        observation_keys=np.asarray(keys),
        response=response,
        derivative_native=derivative,
        covariance=covariance,
    )
    return root


def test_load_update_applies_frozen_tolerances(tmp_path: Path):
    item = _load_update(_fake_update(tmp_path / "u"), "validation")
    assert item["capture_tolerances"] == {"ift_dx_mm": 0.1, "ift_dy_mm": 0.1, "ift_ry_mrad": 1.0}
    parameters = {row["name"]: row for row in item["summary"]["parameters"]}
    assert parameters["ift_dx_mm"]["local_delta_error"] == 0.01


def test_per_source_solve_recovers_injected_delta(tmp_path: Path):
    item = _load_update(_fake_update(tmp_path / "u"), "validation")
    result = _per_source_solve(item, ["source_a", "source_b"])
    for source in ("source_a", "source_b"):
        delta = result["per_source"][source]["recovered_local_delta"]
        assert delta["ift_dx_mm"] == np.float64(0.14) or abs(delta["ift_dx_mm"] - 0.14) < 1e-6
        assert abs(delta["ift_dy_mm"] + 0.11) < 1e-6
        assert abs(delta["ift_ry_mrad"] + 0.74) < 1e-6
    assert result["spread"]["ift_dx_mm"]["std"] < 1e-6


def test_frozen_decision_pass_and_fail(tmp_path: Path):
    passing = _load_update(_fake_update(tmp_path / "pass"), "validation")
    passing["parameters"] = {
        row["name"]: {
            "local_delta_error": row["local_delta_error"],
            "recovered_local_delta": row["recovered_local_delta"],
            "recovered_sigma": row["recovered_sigma"],
            "anchor_value": row["anchor_value"],
            "capture_tolerance": 0.1 if "d" in row["name"] else 1.0,
            "capture_success": True,
        }
        for row in passing["summary"]["parameters"]
    }
    decision, _ = frozen_decision([passing])
    assert decision == "validation_closure_confirmed"

    failing = _load_update(_fake_update(tmp_path / "fail", errors=(0.3, -0.02, 0.1)), "validation")
    failing["parameters"] = {
        row["name"]: {
            "local_delta_error": row["local_delta_error"],
            "recovered_local_delta": row["recovered_local_delta"],
            "recovered_sigma": row["recovered_sigma"],
            "anchor_value": row["anchor_value"],
            "capture_tolerance": 0.1 if "d" in row["name"] else 1.0,
            "capture_success": False,
        }
        for row in failing["summary"]["parameters"]
    }
    decision, reason = frozen_decision([failing])
    assert decision == "hold_and_diagnose"
    assert "ift_dx_mm" in reason
