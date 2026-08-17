from __future__ import annotations

import json

from evaluation.physical_capture_scan import (
    collect_physical_capture_scan,
    write_capture_scan_artifacts,
)
from scripts.run_physical_refit_capture_scan import _build_plan, _has_ntuple_tree


def _diagnostics() -> dict[str, object]:
    state = {name: {"mean": 0.1, "rms": 0.2} for name in ("x_mm", "y_mm", "tx", "ty")}
    return {
        "0->1": {
            "records": 3,
            "field_aware": {
                "residual": state,
                "pull": state,
                "chi2": {"mean": 1.5, "median": 1.0, "rms": 2.0},
            },
        }
    }


def test_physical_capture_collection_records_real_refit_metrics(tmp_path):
    point_dir = tmp_path / "points" / "mag_1_mixed_a" / "closure"
    point_dir.mkdir(parents=True)
    closure = {
        "physical_geometry_repropagation": True,
        "accepted_truth_matched_pairs": 27,
        "active_truth_matched_pairs": 25,
        "normal_matrix_rank": 6,
        "normal_matrix_condition_number": 123.0,
        "movable_station_max_norm_error_mm": 0.005,
        "movable_station_rms_error_mm": 0.002,
        "physical_response_increment_rms_error_mm": 0.001,
        "absolute_error_norm_mm_by_station": {"0": 0.0, "1": 0.001, "2": 0.002, "3": 0.005},
        "displaced_field_aware_by_station_pair": _diagnostics(),
    }
    (point_dir / "closure.json").write_text(json.dumps(closure), encoding="utf-8")
    plan = {
        "points": [
            {
                "name": "mag_1_mixed_a",
                "relative_point_dir": "points/mag_1_mixed_a",
                "magnitude_mm": 1.0,
                "direction_trial": "mixed_a",
                "injected_offsets_xy_mm": {
                    "0": [0.0, 0.0],
                    "1": [1.0, 0.0],
                    "2": [0.0, 1.0],
                    "3": [0.7, -0.7],
                },
            }
        ]
    }

    points, diagnostics = collect_physical_capture_scan(tmp_path, plan, 0.01)
    summary = write_capture_scan_artifacts(tmp_path, points, diagnostics, 0.01)

    assert points[0]["capture_success"] is True
    assert points[0]["normal_matrix_condition_number"] == 123.0
    assert diagnostics[0]["r_x_mm_mean"] == 0.1
    assert summary["by_magnitude"][0]["capture_fraction"] == 1.0
    assert (tmp_path / "capture_scan.png").is_file()
    assert (tmp_path / "capture_scan_station_pair_diagnostics.csv").is_file()


def test_ift_ry_plan_requires_downstream_reference_and_explicit_global_transforms():
    identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config = {
        "scan_mode": "ift_ry_rotation",
        "station_ids": [0, 1, 2, 3],
        "reference_station_ids": [1, 2, 3],
        "movable_station_ids": [0],
        "condition_axis": "ift_ry_mrad",
        "q_over_p_mode": 0,
        "rotation_points": [
            {
                "name": "ry_0",
                "ry_mrad": 0.0,
                "station_transforms": {0: identity, 1: identity, 2: identity, 3: identity},
            },
            {
                "name": "ry_p60",
                "ry_mrad": 60.0,
                "station_transforms": {
                    0: [0.0, 0.0, 0.0, 0.0, 0.060, 0.0],
                    1: identity,
                    2: identity,
                    3: identity,
                },
            },
        ],
    }

    plan = _build_plan(config)

    assert plan["scan_mode"] == "ift_ry_rotation"
    assert plan["reference_station_ids"] == [1, 2, 3]
    assert plan["movable_station_ids"] == [0]
    assert plan["points"][1]["condition_value"] == 60.0
    assert plan["points"][1]["injected_station_transforms"]["0"][4] == 0.060


def test_refit_resume_rejects_partial_root_header(tmp_path):
    partial = tmp_path / "interrupted.root"
    partial.write_bytes(b"root\x00partial-output")

    assert _has_ntuple_tree(partial) is False
