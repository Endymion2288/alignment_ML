"""T02: event bootstrap must keep with-replacement multiplicity."""

from __future__ import annotations

import numpy as np
import pytest

from alignment.identifiable_subspace import FROZEN_RANK_TOLERANCE
from alignment.resampling import (
    describe_event_draw,
    event_identity_keys,
    group_rows_by_event,
    index_physical_bank,
    rows_for_event_draw,
)
from alignment.tracker_only_identifiable_subspace import (
    bootstrap_subspaces_report,
    subspace_from_physical_bank,
)


def _synthetic_derivative():
    derivative = np.zeros((6, 4, 3), dtype=np.float64)
    derivative[:, 0, 0] = 1.0
    derivative[:, 1, 1] = 0.5
    derivative[:, 0, 2] = 1.0
    return derivative


def _synthetic_bank(derivative, *, source_id="s0", scales=(5.0, 60.0, 0.12)):
    pairs, _dim, parameters = derivative.shape
    covariance = np.repeat(np.eye(4, dtype=np.float64)[None, :, :], pairs, axis=0)
    nominal = np.zeros((pairs, 4), dtype=np.float64)
    step = 1.0
    positive = np.zeros((parameters, pairs, 4), dtype=np.float64)
    negative = np.zeros((parameters, pairs, 4), dtype=np.float64)
    for index in range(parameters):
        positive[index] = nominal + step * derivative[:, :, index]
        negative[index] = nominal - step * derivative[:, :, index]
    names = ("ift_dx_mm", "ift_ry_mrad", "C_dx")[:parameters]
    return {
        "source_id": source_id,
        "split": "train",
        "names": names,
        "scales": np.asarray(scales[:parameters], dtype=np.float64),
        "anchor_residual": nominal,
        "positive_residual": positive,
        "negative_residual": negative,
        "reference_residual": np.array(nominal, copy=True),
        "covariance": covariance,
        "run_id": np.arange(pairs, dtype=np.int64),
        "event_id": np.arange(pairs, dtype=np.int64),
        "positive_values": np.full(parameters, step, dtype=np.float64),
        "negative_values": np.full(parameters, -step, dtype=np.float64),
    }


def _two_event_bank():
    derivative = _synthetic_derivative()
    bank = _synthetic_bank(derivative, source_id="srcA")
    # Two events: first four pairs vs last two pairs.
    bank["run_id"] = np.array([1, 1, 1, 1, 2, 2], dtype=np.int64)
    bank["event_id"] = np.array([10, 10, 10, 10, 20, 20], dtype=np.int64)
    bank["source_id"] = "srcA"
    return bank


def test_forced_event0_event0_event1_normal_is_two_n0_plus_n1():
    bank = _two_event_bank()
    keys = event_identity_keys(bank)
    groups = group_rows_by_event(keys)
    event0 = ("srcA", 1, 10)
    event1 = ("srcA", 2, 20)
    assert len(groups[event0]) == 4
    assert len(groups[event1]) == 2
    drawn = (event0, event0, event1)
    rows = rows_for_event_draw(groups, drawn)
    assert rows.tolist() == groups[event0] + groups[event0] + groups[event1]

    names = tuple(bank["names"])
    scales = frozen_scales_for(names)
    fit0 = subspace_from_physical_bank(
        bank, pair_indices=np.asarray(groups[event0], dtype=np.int64)
    )[1]
    fit1 = subspace_from_physical_bank(
        bank, pair_indices=np.asarray(groups[event1], dtype=np.int64)
    )[1]
    fit_draw = subspace_from_physical_bank(bank, pair_indices=rows)[1]
    a0 = np.asarray(fit0["weighted_matrix"])
    a1 = np.asarray(fit1["weighted_matrix"])
    a_draw = np.asarray(fit_draw["weighted_matrix"])
    normal0 = a0.T @ a0
    normal1 = a1.T @ a1
    assert np.allclose(a_draw.T @ a_draw, 2.0 * normal0 + normal1, atol=1.0e-12)
    info = describe_event_draw(groups, drawn)
    assert info.n_draws == 3
    assert info.n_unique == 2
    assert info.effective_multiplicity == pytest.approx(1.5)


def test_same_run_event_from_different_sources_do_not_merge():
    derivative = _synthetic_derivative()
    left = _synthetic_bank(derivative, source_id="fileA")
    right = _synthetic_bank(derivative, source_id="fileB")
    left["run_id"] = np.full(left["run_id"].shape, 7, dtype=np.int64)
    right["run_id"] = np.full(right["run_id"].shape, 7, dtype=np.int64)
    left["event_id"] = np.full(left["event_id"].shape, 99, dtype=np.int64)
    right["event_id"] = np.full(right["event_id"].shape, 99, dtype=np.int64)
    left["source_uid"] = np.full(left["run_id"].shape, "fileA", dtype=object)
    right["source_uid"] = np.full(right["run_id"].shape, "fileB", dtype=object)
    merged = dict(left)
    n = int(left["anchor_residual"].shape[0])
    for key in ("anchor_residual", "reference_residual", "covariance", "run_id", "event_id"):
        merged[key] = np.concatenate([left[key], right[key]], axis=0)
    merged["source_uid"] = np.concatenate([left["source_uid"], right["source_uid"]])
    merged["positive_residual"] = np.concatenate(
        [left["positive_residual"], right["positive_residual"]], axis=1
    )
    merged["negative_residual"] = np.concatenate(
        [left["negative_residual"], right["negative_residual"]], axis=1
    )
    keys = event_identity_keys(merged)
    groups = group_rows_by_event(keys)
    assert set(groups) == {("fileA", 7, 99), ("fileB", 7, 99)}
    assert groups[("fileA", 7, 99)] == list(range(n))
    assert groups[("fileB", 7, 99)] == list(range(n, 2 * n))


def test_boolean_mask_is_not_used_by_new_bootstrap():
    bank = _two_event_bank()
    report = bootstrap_subspaces_report(
        bank,
        n_replicates=4,
        seed=11,
        rank_tolerance=FROZEN_RANK_TOLERANCE,
        rcond=1.0e-10,
        min_pairs=2,
    )
    assert report["boolean_mask_used"] is False
    assert report["historical_rank_not_reinterpreted"] is True
    assert report["n_valid"] >= 1
    assert all(draw["n_draws"] >= draw["n_unique"] for draw in report["draws"])


def test_index_bank_repeats_rows():
    bank = _two_event_bank()
    repeated = index_physical_bank(bank, [0, 0, 1])
    assert repeated["anchor_residual"].shape[0] == 3
    assert np.allclose(repeated["anchor_residual"][0], bank["anchor_residual"][0])
    assert np.allclose(repeated["anchor_residual"][1], bank["anchor_residual"][0])


def test_rank_tolerance_stays_frozen():
    assert FROZEN_RANK_TOLERANCE == pytest.approx(0.01)
