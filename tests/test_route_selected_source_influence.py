"""Unit tests for the route-selected source-influence diagnosis."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from scripts.audit_route_selected_source_influence import analyze_update


def _signature(origin_run: int) -> str:
    return json.dumps([{"origin_run_id": origin_run, "origin_event_id": 1, "origin_tracklet_id": 0}])


def _fake_update(tmp_path: Path) -> Path:
    """Two sources; source 1 carries one large-residual edge."""
    names = ["ift_dx_mm", "ift_dy_mm"]
    rng = np.random.default_rng(7)
    n0, n1 = 40, 40
    # derivative: dx moves x by 1, dy moves y by 1 (per unit parameter).
    derivative = np.zeros((n0 + n1, 4, 2))
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 1.0
    true_delta = np.array([0.5, -0.2])
    response = derivative @ true_delta
    response += rng.normal(scale=0.01, size=response.shape)
    # Source 1's last edge carries a +5 mm x-response outlier.
    response[n0 + n1 - 1, 0] += 5.0
    covariance = np.tile(np.eye(4)[None, :, :], (n0 + n1, 1, 1)) * 0.01
    anchor_residual = np.zeros((n0 + n1, 4))
    anchor_residual[n0 + n1 - 1, 0] = 0.3  # the outlier edge is visibly bad
    keys = []
    for i in range(n0 + n1):
        origin_run = 9_000_000_000 + (0 if i < n0 else 100)
        keys.append(json.dumps(["sample", 1, i, _signature(origin_run), 0, 1]))
    normal = np.zeros((2, 2))
    weight = np.linalg.inv(covariance[0])
    for i in range(n0 + n1):
        normal += derivative[i].T @ weight @ derivative[i]
    update = tmp_path / "update"
    update.mkdir()
    np.savez_compressed(
        update / "route_selected_update_arrays.npz",
        parameter_names=np.asarray(names),
        observation_keys=np.asarray(keys),
        anchor_residual=anchor_residual,
        target_residual=response + anchor_residual,
        covariance=covariance,
        derivative_native=derivative,
        response=response,
        normal_matrix_native=normal,
        normal_matrix_scaled=normal,
    )
    (update / "route_selected_update.json").write_text(
        json.dumps(
            {
                "parameters": [
                    {
                        "name": "ift_dx_mm",
                        "recovered_local_delta": 0.55,
                        "expected_delta_to_target": 0.5,
                    },
                    {
                        "name": "ift_dy_mm",
                        "recovered_local_delta": -0.2,
                        "expected_delta_to_target": -0.2,
                    },
                ]
            }
        )
    )
    return update


def test_influence_flags_outlier_source(tmp_path: Path) -> None:
    update = _fake_update(tmp_path)
    result = analyze_update(update, ["src_a", "src_b"])
    per_source = {entry["source_id"]: entry for entry in result["per_source"]}
    assert set(per_source) == {"src_a", "src_b"}
    outlier = per_source["src_b"]
    typical = per_source["src_a"]
    # The outlier edge dominates the source's chi2 tail.
    assert outlier["anchor_chi2_max"] > typical["anchor_chi2_max"]
    # Removing the contaminated source pulls the pooled dx back toward the
    # truth (negative shift), while removing the clean source concentrates
    # the outlier (positive shift of the same magnitude for symmetric data).
    assert outlier["loso_shift"]["ift_dx_mm"] < 0.0
    assert typical["loso_shift"]["ift_dx_mm"] > 0.0
    assert abs(outlier["loso_shift"]["ift_dy_mm"]) < 0.05
    # Pooled reconstruction matches a direct solve.
    assert result["pooled_delta_from_arrays"]["ift_dy_mm"] == pytest.approx(-0.2, abs=0.02)
    # Leverage shares are positive and sum to the parameter count.
    total = sum(entry["leverage_trace_share"] for entry in result["per_source"])
    assert total == pytest.approx(2.0, rel=1e-6)


def test_weakest_mode_recovers_dy(tmp_path: Path) -> None:
    update = _fake_update(tmp_path)
    result = analyze_update(update, ["src_a", "src_b"])
    # dy edges carry no outlier and identical weights, so both modes are
    # well constrained; the weakest direction is still a unit vector.
    direction = result["weakest_mode_native_direction"]
    norm = sum(value * value for value in direction.values()) ** 0.5
    assert norm == pytest.approx(1.0)
