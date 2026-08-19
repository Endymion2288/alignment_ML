"""Unit tests for the train-only mode-0 covariance calibration controls."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.covariance_calibration import (
    CovarianceCalibration,
    apply_to_records,
    derive_from_pull_rows,
)
from datasets.propagation_loader import PropagationRecords


def _records(covariance: np.ndarray, stations: tuple[int, int] = (0, 1)) -> PropagationRecords:
    size = covariance.shape[0]
    return PropagationRecords(
        run_id=np.ones(size, dtype=np.int64),
        event_id=np.arange(size, dtype=np.int64),
        source_tracklet_id=np.arange(size, dtype=np.int64),
        target_tracklet_id=np.arange(size, dtype=np.int64),
        source_station_id=np.full(size, stations[0], dtype=np.int64),
        target_station_id=np.full(size, stations[1], dtype=np.int64),
        truth_particle_id=np.ones(size, dtype=np.int64),
        target_z_mm=np.zeros(size, dtype=np.float64),
        prediction=np.zeros((size, 4), dtype=np.float64),
        covariance=covariance,
        success=np.ones(size, dtype=bool),
        has_covariance=np.ones(size, dtype=bool),
    )


def test_derive_freezes_squared_robust_widths() -> None:
    rows = [
        {"station_pair": "0->1", "component": "x_mm", "robust_sigma": "0.5"},
        {"station_pair": "0->1", "component": "y_mm", "robust_sigma": "0.01"},
        {"station_pair": "0->1", "component": "tx", "robust_sigma": "2.0"},
        {"station_pair": "0->1", "component": "ty", "robust_sigma": "1.0"},
    ]
    calibration = derive_from_pull_rows(rows, provenance={"split": "train"})
    factors = calibration.factors[(0, 1)]
    assert factors["x_mm"] == pytest.approx(0.25)
    assert factors["y_mm"] == pytest.approx(1.0e-4)
    assert factors["tx"] == pytest.approx(4.0)
    assert factors["ty"] == pytest.approx(1.0)
    assert calibration.provenance["split"] == "train"


def test_derive_rejects_nonpositive_widths() -> None:
    rows = [{"station_pair": "0->1", "component": "x_mm", "robust_sigma": "0.0"}]
    with pytest.raises(ValueError, match="positive"):
        derive_from_pull_rows(rows, provenance={})


def test_apply_rescales_only_matching_station_pair() -> None:
    covariance = np.tile(np.eye(4)[None, :, :], (2, 1, 1))
    covariance[0] = np.diag([4.0, 9.0, 16.0, 25.0])
    records = _records(covariance, stations=(0, 1))
    records.source_station_id[1] = 1
    records.target_station_id[1] = 2
    calibration = CovarianceCalibration(
        factors={(0, 1): {"x_mm": 0.25, "ty": 4.0}}, provenance={}
    )
    scaled = apply_to_records(records, calibration)
    expected = np.diag([1.0, 9.0, 16.0, 100.0])
    np.testing.assert_allclose(scaled.covariance[0], expected)
    np.testing.assert_allclose(scaled.covariance[1], np.eye(4))
    np.testing.assert_allclose(records.covariance[0], np.diag([4.0, 9.0, 16.0, 25.0]))


def test_apply_preserves_correlation_structure() -> None:
    base = np.array([[4.0, 1.0], [1.0, 9.0]])
    covariance = np.zeros((1, 4, 4))
    covariance[0, :2, :2] = base
    covariance[0, 2, 2] = 1.0
    covariance[0, 3, 3] = 1.0
    records = _records(covariance)
    calibration = CovarianceCalibration(
        factors={(0, 1): {"x_mm": 0.25, "y_mm": 4.0}}, provenance={}
    )
    scaled = apply_to_records(records, calibration)
    # C' = D^1/2 C D^1/2 with D = diag(0.25, 4): off-diagonal scales by sqrt(0.25*4)=1.
    assert scaled.covariance[0, 0, 0] == pytest.approx(1.0)
    assert scaled.covariance[0, 1, 1] == pytest.approx(36.0)
    assert scaled.covariance[0, 0, 1] == pytest.approx(1.0)
    correlation = scaled.covariance[0, 0, 1] / np.sqrt(1.0 * 36.0)
    original_correlation = 1.0 / np.sqrt(4.0 * 9.0)
    assert correlation == pytest.approx(original_correlation)


def test_json_roundtrip() -> None:
    calibration = CovarianceCalibration(
        factors={(1, 2): {"x_mm": 0.5}}, provenance={"split": "train"}
    )
    restored = CovarianceCalibration.from_json(calibration.to_json())
    assert restored.factors == calibration.factors


def test_json_rejects_bad_scale() -> None:
    payload = {
        "schema_version": "faser-mode0-covariance-calibration-v1",
        "factors": {"0->1": {"x_mm": -1.0}},
    }
    with pytest.raises(ValueError, match="positive"):
        CovarianceCalibration.from_json(payload)
