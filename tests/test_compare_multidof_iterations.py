from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest

from scripts.compare_multidof_iterations import (
    DECISION_NO_ITERATION2,
    DECISION_ROUTE_UPDATE,
    _load_iteration,
    _workbook_fragment,
    frozen_decision,
)


def _fake_closure(root: Path, *, dx_error: float, dy_error: float, ry_error: float) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    parameters = [
        {
            "name": "ift_dx_mm",
            "component": "dx_mm",
            "station_id": 0,
            "unit": "mm",
            "anchor_value": 2.0,
            "target_value": 0.0,
            "expected_delta_to_target": -2.0,
            "fit_split_recovered_local_delta": -2.1,
            "fit_split_local_delta_error": dx_error,
            "held_out_independent_recovered_local_delta": -2.2,
            "held_out_independent_local_delta_error": dx_error,
            "held_out_independent_capture_success": abs(dx_error) <= 0.1,
            "capture_success": abs(dx_error) <= 0.1,
            "capture_tolerance": 0.1,
            "proposed_next_value": -0.1,
            "recovered_sigma": 0.1,
            "observability_fraction": 1.0,
        },
        {
            "name": "ift_dy_mm",
            "component": "dy_mm",
            "station_id": 0,
            "unit": "mm",
            "anchor_value": -1.5,
            "target_value": 0.0,
            "expected_delta_to_target": 1.5,
            "fit_split_recovered_local_delta": 1.4,
            "fit_split_local_delta_error": dy_error,
            "held_out_independent_recovered_local_delta": 1.4,
            "held_out_independent_local_delta_error": dy_error,
            "held_out_independent_capture_success": abs(dy_error) <= 0.1,
            "capture_success": abs(dy_error) <= 0.1,
            "capture_tolerance": 0.1,
            "proposed_next_value": 0.1,
            "recovered_sigma": 0.07,
            "observability_fraction": 1.0,
        },
        {
            "name": "ift_ry_mrad",
            "component": "ry_mrad",
            "station_id": 0,
            "unit": "mrad",
            "anchor_value": 35.0,
            "target_value": 0.0,
            "expected_delta_to_target": -35.0,
            "fit_split_recovered_local_delta": -35.1,
            "fit_split_local_delta_error": ry_error,
            "held_out_independent_recovered_local_delta": -35.2,
            "held_out_independent_local_delta_error": ry_error,
            "held_out_independent_capture_success": abs(ry_error) <= 1.0,
            "capture_success": abs(ry_error) <= 1.0,
            "capture_tolerance": 1.0,
            "proposed_next_value": 0.1,
            "recovered_sigma": 0.05,
            "observability_fraction": 1.0,
        },
    ]
    step = {
        "parameters": parameters,
        "parameter_correlation": [
            [1.0, 0.0, 0.85],
            [0.0, 1.0, 0.0],
            [0.85, 0.0, 1.0],
        ],
        "fit_split": {
            "fit": {
                "recovered_local_delta": {
                    "ift_dx_mm": -2.1,
                    "ift_dy_mm": 1.4,
                    "ift_ry_mrad": -35.1,
                },
                "normal_matrix_rank": 3,
                "normal_matrix_condition_number": 500.0,
            }
        },
        "held_out_split": {
            "independent_fit_diagnostic_only": {
                "recovered_local_delta": {
                    "ift_dx_mm": -2.2,
                    "ift_dy_mm": 1.4,
                    "ift_ry_mrad": -35.2,
                }
            },
            "frozen_fit_split_update_application": {
                "baseline_response_chi2": 1000.0,
                "post_update_response_chi2": 100.0,
                "response_chi2_reduction": 900.0,
                "response_chi2_reduction_fraction": 0.9,
            },
        },
        "candidate_metrics": [
            {
                "point_role": "anchor",
                "candidate_complete_truth_chain_recall": 0.44,
                "0->1_candidate_truth_edge_recall": 0.59,
                "1->2_candidate_truth_edge_recall": 0.72,
                "2->3_candidate_truth_edge_recall": 0.69,
            }
        ],
        "proposed_next_parameter_values": {
            "ift_dx_mm": -0.1,
            "ift_dy_mm": 0.1,
            "ift_ry_mrad": 0.1,
        },
    }
    (root / "multisource_local_step.json").write_text(json.dumps(step), encoding="utf-8")
    with (root / "source_fit_diagnostics.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=["source_id", "split", "status", "recovered_local_delta"],
        )
        writer.writeheader()
        for index, delta in enumerate((-2.0, -2.2, -2.1)):
            writer.writerow(
                {
                    "source_id": f"mc24_100043_0000{index}",
                    "split": "train",
                    "status": "ok",
                    "recovered_local_delta": repr(
                        {"ift_dx_mm": delta, "ift_dy_mm": 1.4, "ift_ry_mrad": -35.0}
                    ),
                }
            )
    return root


def test_load_iteration_extracts_frozen_fields(tmp_path: Path):
    closure = _fake_closure(tmp_path / "iter0", dx_error=-0.14, dy_error=0.11, ry_error=-0.2)
    item = _load_iteration(closure, "iteration-0")
    assert item["held_out_capture_pass"] == {
        "ift_dx_mm": False,
        "ift_dy_mm": False,
        "ift_ry_mrad": True,
    }
    assert item["held_out_capture_pass_all"] is False
    assert item["dx_ry_parameter_correlation"] == pytest.approx(0.85)
    assert item["held_out_chi2_reduction_fraction"] == pytest.approx(0.9)
    assert item["anchor_chain_recall"] == pytest.approx(0.44)
    assert item["source_to_source_spread"]["ift_dx_mm"] == pytest.approx(0.1, abs=1e-9)


def test_frozen_decision_holds_when_last_iteration_fails(tmp_path: Path):
    failing = _load_iteration(
        _fake_closure(tmp_path / "iter0", dx_error=-0.14, dy_error=0.11, ry_error=-0.2),
        "iteration-0",
    )
    decision = frozen_decision([failing])
    assert decision["decision"] == DECISION_NO_ITERATION2
    assert set(decision["failing_parameters"]) == {"ift_dx_mm", "ift_dy_mm"}


def test_none_capture_tolerance_falls_back_to_frozen(tmp_path: Path):
    closure = _fake_closure(tmp_path / "iter1", dx_error=0.05, dy_error=-0.04, ry_error=0.3)
    step_path = closure / "multisource_local_step.json"
    step = json.loads(step_path.read_text(encoding="utf-8"))
    for parameter in step["parameters"]:
        parameter["capture_tolerance"] = None
        parameter["held_out_independent_capture_success"] = None
    step_path.write_text(json.dumps(step), encoding="utf-8")
    item = _load_iteration(closure, "iteration-1")
    assert item["capture_tolerances"] == {"ift_dx_mm": 0.1, "ift_dy_mm": 0.1, "ift_ry_mrad": 1.0}
    assert item["held_out_capture_pass_all"] is True


def test_frozen_decision_proceeds_when_last_iteration_passes(tmp_path: Path):
    failing = _load_iteration(
        _fake_closure(tmp_path / "iter0", dx_error=-0.14, dy_error=0.11, ry_error=-0.2),
        "iteration-0",
    )
    passing = _load_iteration(
        _fake_closure(tmp_path / "iter1", dx_error=0.05, dy_error=-0.04, ry_error=0.3),
        "iteration-1",
    )
    decision = frozen_decision([failing, passing])
    assert decision["decision"] == DECISION_ROUTE_UPDATE
    fragment = _workbook_fragment([failing, passing], decision)
    assert "iteration-1" in fragment and "通过" in fragment
