from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from baselines.field_chi2_matching import FieldCandidate
from training.curriculum_mlp import (
    CandidateSet,
    adaptive_threshold_grid,
    choose_operating_threshold,
    filter_candidate_scores,
    filter_candidate_sets,
    station_pair_feature_view,
)


def test_operating_point_uses_validation_constraint_then_efficiency():
    rows = [
        {
            "score_threshold": 0.30,
            "association_efficiency": 0.90,
            "inclusive_fake_rate": 0.10,
            "inclusive_association_purity": 0.90,
            "predicted_matches": 100,
        },
        {
            "score_threshold": 0.60,
            "association_efficiency": 0.80,
            "inclusive_fake_rate": 0.04,
            "inclusive_association_purity": 0.96,
            "predicted_matches": 80,
        },
        {
            "score_threshold": 0.70,
            "association_efficiency": 0.75,
            "inclusive_fake_rate": 0.03,
            "inclusive_association_purity": 0.97,
            "predicted_matches": 70,
        },
    ]

    selected = choose_operating_threshold(rows, "inclusive_fake_rate", 0.05)

    assert selected is not None
    assert selected["score_threshold"] == 0.60


def test_adaptive_threshold_grid_resolves_a_compressed_calibrated_tail():
    scores = [np.asarray([0.491, 0.493, 0.497, 0.501, 0.509], dtype=np.float64)]

    grid = adaptive_threshold_grid(scores, [0.01, 0.50, 0.99], quantiles=5)

    assert 0.01 in grid
    assert 0.50 in grid
    assert 0.99 in grid
    assert any(np.isclose(value, 0.509) for value in grid)
    assert any(value > 0.509 and value < 1.0 for value in grid)


def test_physical_candidate_gate_filters_sets_and_scores_with_one_mask():
    def candidate(chi2: float) -> FieldCandidate:
        return FieldCandidate(
            source_index=0,
            target_index=1,
            source_station=0,
            target_station=1,
            chi2=chi2,
            residual=np.zeros(4),
            pull=np.zeros(4),
            combined_covariance=np.eye(4),
        )

    original = CandidateSet(
        sample=SimpleNamespace(),
        event=SimpleNamespace(),
        station_pair=(0, 1),
        candidates=(candidate(4.0), candidate(25.0), candidate(250.0)),
        features=np.asarray([[4.0], [25.0], [250.0]]),
        labels=np.asarray([True, False, False]),
    )
    filtered = filter_candidate_sets([original], 25.0)
    scores = filter_candidate_scores([original], [np.asarray([0.9, 0.5, 0.1])], 25.0)

    assert [value.chi2 for value in filtered[0].candidates] == [4.0, 25.0]
    assert filtered[0].features[:, 0].tolist() == [4.0, 25.0]
    assert filtered[0].labels.tolist() == [True, False]
    assert scores[0].tolist() == [0.9, 0.5]


def test_station_pair_feature_view_keeps_common_features_and_own_indicator_only():
    pairs = ((0, 1), (0, 2), (1, 2))
    features = np.arange(36, dtype=np.float64).reshape(2, 18)
    candidate_set = CandidateSet(
        sample=SimpleNamespace(),
        event=SimpleNamespace(),
        station_pair=(0, 2),
        candidates=(),
        features=features,
        labels=np.asarray([True, False]),
    )

    view = station_pair_feature_view([candidate_set], (0, 2), pairs)

    assert len(view) == 1
    assert view[0].features.shape == (2, 16)
    np.testing.assert_array_equal(view[0].features[:, :15], features[:, :15])
    np.testing.assert_array_equal(view[0].features[:, 15], features[:, 16])
