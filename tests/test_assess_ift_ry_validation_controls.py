import json

import pytest

from scripts.assess_ift_ry_validation_controls import _load_mlp, _validate_grid


def _write(path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def _route():
    return {
        "pooled_candidate_complete_truth_chain_recall": 0.98,
        "pooled_score_threshold_complete_truth_chain_recall": 0.95,
        "pooled_complete_track_efficiency": 0.80,
        "pooled_complete_track_purity": 0.97,
        "pooled_track_fake_rate": 0.03,
        "pooled_missing_station_recovery": 0.20,
    }


def _mlp_artifact(root, *, test_events_loaded=False):
    _write(
        root / "validation_control_contract.json",
        {
            "condition_axis": "ift_ry_mrad",
            "loaded_event_splits": ["validation"],
            "test_events_loaded": test_events_loaded,
            "test_artifacts_opened": False,
        },
    )
    _write(
        root / "validation_selected_operating_point.json",
        {
            "selection_split": "validation_only",
            "test_opened": False,
            "capture_success_criteria": {
                "minimum_complete_track_efficiency": 0.70,
                "minimum_complete_track_purity": 0.95,
                "maximum_track_fake_rate": 0.05,
            },
        },
    )
    _write(
        root / "validation_results.json",
        {
            "route_magnitude_summary": [
                {
                    "condition_axis": "ift_ry_mrad",
                    "condition_magnitude": 0.0,
                    **_route(),
                },
                {
                    "condition_axis": "ift_ry_mrad",
                    "condition_magnitude": 60.0,
                    **_route(),
                },
            ]
        },
    )


def test_ift_ry_mlp_assessment_reads_only_sealed_rotation_artifact(tmp_path):
    root = tmp_path / "mlp"
    _mlp_artifact(root)

    rows, metadata = _load_mlp(root)

    assert [row["condition_magnitude"] for row in rows] == [0.0, 60.0]
    assert all(row["condition_axis"] == "ift_ry_mrad" for row in rows)
    assert all(row["capture_success"] for row in rows)
    assert metadata["contract"]["test_events_loaded"] is False
    assert _validate_grid(rows) == (0.0, 60.0)


def test_ift_ry_mlp_assessment_rejects_unsealed_artifact(tmp_path):
    root = tmp_path / "mlp"
    _mlp_artifact(root, test_events_loaded=True)

    with pytest.raises(ValueError, match="sealed from test events"):
        _load_mlp(root)
