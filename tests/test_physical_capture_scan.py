from __future__ import annotations

import json

from evaluation.physical_capture_scan import (
    collect_physical_capture_scan,
    write_capture_scan_artifacts,
)


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
