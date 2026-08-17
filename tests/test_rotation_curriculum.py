from __future__ import annotations

from scripts.build_physical_curriculum_corpus import _source_scan_config
from scripts.run_physical_refit_capture_scan import _build_plan


def test_rotation_curriculum_emits_explicit_ift_ry_physical_scan_without_legacy_offsets():
    identity = [0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    config = {
        "refit": {
            "nevents": 10,
            "station_ids": [0, 1, 2, 3],
            "q_over_p_mode": 0,
            "min_truth_match_fraction": 0.99,
            "chi2_gate": 25.0,
            "refinement_iterations": 1,
        },
        "rotation_curriculum": {
            "scan_mode": "ift_ry_rotation",
            "condition_axis": "ift_ry_mrad",
            "reference_station_ids": [1, 2, 3],
            "movable_station_ids": [0],
            "rotation_points": [
                {
                    "name": "ry_0",
                    "ry_mrad": 0.0,
                    "station_transforms": {0: identity, 1: identity, 2: identity, 3: identity},
                },
                {
                    "name": "ry_p10",
                    "ry_mrad": 10.0,
                    "station_transforms": {
                        0: [0.0, 0.0, 0.0, 0.0, 0.010, 0.0],
                        1: identity,
                        2: identity,
                        3: identity,
                    },
                },
            ],
        },
    }

    scan = _source_scan_config(
        config,
        {"source_id": "source", "split": "train", "input_xaod": "/tmp/source.root"},
        {},
    )["physical_refit_capture_scan"]
    plan = _build_plan(scan)

    assert scan["run_alignment_closure"] is False
    assert "magnitudes_mm" not in scan
    assert plan["scan_mode"] == "ift_ry_rotation"
    assert plan["points"][1]["injected_station_transforms"]["0"][4] == 0.010
