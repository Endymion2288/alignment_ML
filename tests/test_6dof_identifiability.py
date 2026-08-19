from __future__ import annotations

import numpy as np

from alignment.physical_jacobian import solve_physical_finite_difference
from scripts.audit_6dof_identifiability import (
    _column_cosines,
    _complete_route_mask,
    _gate,
    _masked_bank,
    _weighted_column_norms,
)


def _synthetic_fit(derivative: np.ndarray, *, scales=(1.0, 1.0, 1.0), step=1.0):
    """Build a fit from an explicit [pairs, 4, parameters] Jacobian."""
    pairs = derivative.shape[0]
    parameters = derivative.shape[2]
    covariance = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], pairs, axis=0)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    positive = np.zeros((parameters, pairs, 4), dtype=np.float64)
    negative = np.zeros((parameters, pairs, 4), dtype=np.float64)
    for p in range(parameters):
        positive[p] = step * derivative[:, :, p]
        negative[p] = -step * derivative[:, :, p]
    observed = np.zeros((pairs, 4), dtype=np.float64)
    return solve_physical_finite_difference(
        nominal,
        positive,
        negative,
        observed,
        covariance,
        parameter_names=tuple(f"p{index}" for index in range(parameters)),
        positive_values=[step] * parameters,
        negative_values=[-step] * parameters,
        parameter_scales=list(scales),
    )


def _bank(names, scales=(1.0, 1.0, 1.0)):
    return {
        "names": tuple(names),
        "scales": np.asarray(scales, dtype=np.float64),
    }


def test_weighted_column_norms_and_cosines_recover_known_jacobian():
    derivative = np.zeros((3, 4, 3), dtype=np.float64)
    derivative[:, 0, 0] = 2.0  # p0 -> rx only
    derivative[:, 1, 1] = 1.0  # p1 -> ry only
    derivative[:, 0, 2] = 3.0  # p2 = 1.5x p0 (exactly degenerate)
    fit = _synthetic_fit(derivative)
    norms = _weighted_column_norms(fit)
    assert np.isclose(norms[0], 2.0 * np.sqrt(3.0))
    assert np.isclose(norms[1], np.sqrt(3.0))
    cosines = _column_cosines(fit)
    assert np.isclose(cosines[0, 1], 0.0)
    assert np.isclose(cosines[0, 2], 1.0)
    assert np.isclose(cosines[1, 2], 0.0)


def test_gate_admits_stable_identifiable_and_rejects_degenerate():
    derivative = np.zeros((4, 4, 3), dtype=np.float64)
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 1.0
    derivative[:, 0, 2] = 2.0  # p2 exactly parallel to p0
    fit = _synthetic_fit(derivative)
    source_entry = {"source_id": "s0", "fit": fit}
    gate = _gate(
        ("p0", "p1", "p2"),
        np.asarray([10.0, 10.0, 10.0]),
        fit,
        [source_entry],
        max_condition_number=1.0e4,
        max_source_spread=0.5,
    )
    # p2 carries twice the information of the exactly parallel p0, so the
    # greedy admission keeps p2 and rejects p0 as rank-deficient against it.
    assert set(gate["admitted_parameters"]) == {"p1", "p2"}
    assert gate["per_parameter"]["p0"]["admitted"] is False
    assert "rank deficient" in gate["per_parameter"]["p0"]["exclusion_reason"]


def test_gate_rejects_unstable_source_response():
    derivative = np.zeros((4, 4, 2), dtype=np.float64)
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 1.0
    pooled_fit = _synthetic_fit(derivative, scales=(1.0, 1.0))
    weak = np.zeros((4, 4, 2), dtype=np.float64)
    weak[:, 0, 0] = 1.0
    weak[:, 1, 1] = 0.1  # p1 nearly absent in this source
    weak_fit = _synthetic_fit(weak, scales=(1.0, 1.0))
    gate = _gate(
        ("p0", "p1"),
        np.asarray([10.0, 10.0]),
        pooled_fit,
        [{"source_id": "strong", "fit": pooled_fit}, {"source_id": "weak", "fit": weak_fit}],
        max_condition_number=1.0e4,
        max_source_spread=0.5,
    )
    assert gate["per_parameter"]["p0"]["response_stable"] is True
    assert gate["per_parameter"]["p1"]["response_stable"] is False
    assert gate["per_parameter"]["p1"]["admitted"] is False
    assert "unstable source response" in gate["per_parameter"]["p1"]["exclusion_reason"]


def test_complete_route_mask_requires_all_observed_pairs_per_event():
    bank = {
        "run_id": np.asarray([1, 1, 1, 1, 1, 2, 2], dtype=np.int64),
        "event_id": np.asarray([10, 10, 10, 11, 11, 10, 10], dtype=np.int64),
        "truth_particle_id": np.asarray([7, 7, 7, 7, 7, 7, 7], dtype=np.int64),
        "source_station_id": np.asarray([0, 0, 0, 0, 0, 0, 0], dtype=np.int64),
        "target_station_id": np.asarray([1, 2, 3, 1, 2, 1, 2], dtype=np.int64),
    }
    mask = _complete_route_mask(bank)
    # Event 10 of run 1 covers 0->1/0->2/0->3; event 11 lacks 0->3; the same
    # particle in run 2 must not complete the run-1 event-11 route.
    assert mask.tolist() == [True, True, True, False, False, False, False]


def test_masked_bank_slices_parameter_axis():
    bank = {
        "anchor_residual": np.zeros((2, 4)),
        "reference_residual": np.zeros((2, 4)),
        "covariance": np.repeat(np.eye(4)[None], 2, axis=0),
        "run_id": np.asarray([1, 1]),
        "event_id": np.asarray([5, 6]),
        "truth_particle_id": np.asarray([1, 2]),
        "source_station_id": np.asarray([0, 1]),
        "target_station_id": np.asarray([1, 2]),
        "positive_residual": np.zeros((3, 2, 4)),
        "negative_residual": np.zeros((3, 2, 4)),
    }
    masked = _masked_bank(bank, np.asarray([True, False]))
    assert masked["anchor_residual"].shape == (1, 4)
    assert masked["positive_residual"].shape == (3, 1, 4)
