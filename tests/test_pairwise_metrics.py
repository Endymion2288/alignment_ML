from __future__ import annotations

import numpy as np

from evaluation.pairwise_metrics import (
    apply_temperature,
    binary_average_precision,
    binary_calibration,
    binary_roc_auc,
    fit_temperature,
)


def test_pairwise_scores_report_rank_and_calibration_metrics():
    scores = np.asarray([0.10, 0.20, 0.80, 0.90], dtype=np.float64)
    labels = np.asarray([False, False, True, True])

    assert binary_roc_auc(scores, labels) == 1.0
    assert binary_average_precision(scores, labels) == 1.0
    report = binary_calibration(scores, labels, bins=4)
    assert report["rows"] == 4
    assert report["positive_rows"] == 2
    assert report["expected_calibration_error"] is not None


def test_temperature_is_validation_only_scalar_and_preserves_score_order():
    scores = np.asarray([0.25, 0.35, 0.65, 0.75], dtype=np.float64)
    labels = np.asarray([False, False, True, True])
    temperature = fit_temperature(scores, labels)
    calibrated = apply_temperature(scores, temperature)

    assert temperature > 0.0
    assert np.all(np.diff(calibrated) > 0.0)
    assert np.all((calibrated > 0.0) & (calibrated < 1.0))
