from __future__ import annotations

import numpy as np

from baselines.chi2_matching import Match
from datasets.root_loader import EventTracklets
from evaluation.metrics import assess_event_matches, assess_event_unmatched_endpoints


def test_ambiguous_truth_labels_are_not_counted_as_fakes():
    event = EventTracklets(
        run_id=1,
        event_id=2,
        station_id=np.asarray([0, 0, 0, 0, 1, 1, 1, 1]),
        tracklet_id=np.arange(8, dtype=np.int32),
        z_mm=np.asarray([0.0, 0.0, 0.0, 0.0, 1.0, 1.0, 1.0, 1.0]),
        state=np.zeros((8, 4), dtype=np.float64),
        covariance=np.tile(np.eye(4, dtype=np.float64), (8, 1, 1)),
        chi2=np.ones(8, dtype=np.float64),
        ndof=np.ones(8, dtype=np.float64),
        n_hit=np.full(8, 3, dtype=np.int16),
        hit_pattern=np.full(8, 0b111, dtype=np.uint64),
        truth_particle_id=np.asarray([10, 10, 30, 50, 10, 20, 40, 50]),
        truth_pdg=np.full(8, 13, dtype=np.int32),
        truth_match_fraction=np.ones(8, dtype=np.float64),
    )
    matches = [
        Match(source_index=0, target_index=4, chi2=1.0),
        Match(source_index=2, target_index=5, chi2=1.0),
        Match(source_index=3, target_index=7, chi2=1.0),
    ]

    metrics, assessments = assess_event_matches(event, matches, 0, 1)

    assert [assessment.relation for assessment in assessments] == [
        "ambiguous",
        "incorrect",
        "correct",
    ]
    assert metrics.possible_matches == 1
    assert metrics.predicted_matches == 3
    assert metrics.scored_predicted_matches == 2
    assert metrics.unscorable_predicted_matches == 1
    assert metrics.correct_matches == 1
    assert metrics.incorrect_scored_matches == 1
    assert metrics.efficiency == 1.0
    assert metrics.purity == 0.5
    assert metrics.fake_rate == 0.5
    assert metrics.inclusive_purity == 1 / 3
    assert metrics.inclusive_fake_rate == 2 / 3


def test_unmatched_metrics_separate_missing_truth_from_synthetic_fakes():
    event = EventTracklets(
        run_id=1,
        event_id=2,
        station_id=np.asarray([0, 0, 0, 1, 1, 1]),
        tracklet_id=np.arange(6, dtype=np.int32),
        z_mm=np.asarray([0.0, 0.0, 0.0, 1.0, 1.0, 1.0]),
        state=np.zeros((6, 4), dtype=np.float64),
        covariance=np.tile(np.eye(4, dtype=np.float64), (6, 1, 1)),
        chi2=np.ones(6, dtype=np.float64),
        ndof=np.ones(6, dtype=np.float64),
        n_hit=np.full(6, 3, dtype=np.int16),
        hit_pattern=np.full(6, 0b111, dtype=np.uint64),
        # ID 10 is complete, ID 20 is source-missing in target, ID 30 is
        # target-missing in source, and -1 denotes synthetic fake rows.
        truth_particle_id=np.asarray([10, 20, -1, 10, 30, -1]),
        truth_pdg=np.full(6, 13, dtype=np.int32),
        truth_match_fraction=np.ones(6, dtype=np.float64),
    )
    metrics = assess_event_unmatched_endpoints(
        event,
        [Match(source_index=0, target_index=3, chi2=1.0)],
        0,
        1,
    )
    assert metrics.missing_truth_endpoints == 2
    assert metrics.correctly_unmatched_missing_truth == 2
    assert metrics.missing_truth_unmatched_recall == 1.0
    assert metrics.fake_endpoints == 2
    assert metrics.correctly_unmatched_fakes == 2
    assert metrics.fake_unmatched_recall == 1.0
