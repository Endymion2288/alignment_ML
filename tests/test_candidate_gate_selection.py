from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.run_global_assignment_mlp_baseline import (
    _assignment_scan_gate_keys,
    _apply_frozen_calibration_sets,
    _calibrate_score_sets,
    _choose_candidate_gate,
)


def test_assignment_scan_gate_mode_expands_only_when_explicitly_requested():
    views = {"chi2_le_250": object(), "ungated": object()}

    assert _assignment_scan_gate_keys(views, "ungated", False) == ["ungated"]
    assert _assignment_scan_gate_keys(views, "ungated", True) == ["chi2_le_250", "ungated"]


def test_candidate_gate_selection_uses_recall_floor_then_average_precision():
    rows = [
        {
            "magnitude_mm": 0.0,
            "candidate_chi2_gate": 25.0,
            "candidate_truth_recall": 0.98,
            "average_precision": 0.99,
        },
        {
            "magnitude_mm": 0.0,
            "candidate_chi2_gate": 250.0,
            "candidate_truth_recall": 0.995,
            "average_precision": 0.80,
        },
        {
            "magnitude_mm": 0.0,
            "candidate_chi2_gate": 1000.0,
            "candidate_truth_recall": 0.997,
            "average_precision": 0.85,
        },
        {
            "magnitude_mm": 0.0,
            "candidate_chi2_gate": None,
            "candidate_truth_recall": 1.0,
            "average_precision": 0.84,
        },
    ]

    selected = _choose_candidate_gate(rows, "nominal_only", minimum_truth_recall=0.99)

    assert selected["candidate_chi2_gate"] == 1000.0


def test_candidate_gate_selection_prefers_tighter_gate_for_equal_ap():
    rows = [
        {
            "magnitude_mm": 0.0,
            "candidate_chi2_gate": 250.0,
            "candidate_truth_recall": 1.0,
            "average_precision": 0.80,
        },
        {
            "magnitude_mm": 0.0,
            "candidate_chi2_gate": None,
            "candidate_truth_recall": 1.0,
            "average_precision": 0.80,
        },
    ]

    selected = _choose_candidate_gate(rows, "nominal_only", minimum_truth_recall=0.99)

    assert selected["candidate_chi2_gate"] == 250.0


def test_station_pair_calibration_fits_and_reuses_only_pair_specific_temperatures():
    sets = [
        SimpleNamespace(station_pair=(0, 1), labels=np.asarray([True, False, True, False])),
        SimpleNamespace(station_pair=(1, 2), labels=np.asarray([True, False, True, False])),
    ]
    raw_scores = [
        np.asarray([0.99, 0.70, 0.95, 0.60]),
        np.asarray([0.80, 0.20, 0.70, 0.10]),
    ]

    calibrated, report = _calibrate_score_sets(
        sets, raw_scores, calibration_bins=5, scope="station_pair"
    )

    assert report["scope"] == "station_pair"
    assert set(report["temperature_by_station_pair"]) == {"0->1", "1->2"}
    assert report["temperature"] is None
    replayed = _apply_frozen_calibration_sets(sets, raw_scores, report)
    for first, second in zip(calibrated, replayed):
        np.testing.assert_allclose(first, second)
