from __future__ import annotations

import numpy as np

from datasets.propagation_loader import PropagationRecords
from datasets.root_loader import EventTracklets
from evaluation.field_propagation import evaluate_field_propagation


def test_truth_matched_field_propagation_uses_combined_covariance():
    source_state = np.asarray([0.0, 0.0, 0.1, -0.1])
    target_state = np.asarray([1.0, -1.0, 0.1, -0.1])
    event = EventTracklets(
        run_id=4,
        event_id=9,
        station_id=np.asarray([0, 1], dtype=np.int16),
        tracklet_id=np.asarray([3, 8], dtype=np.int32),
        z_mm=np.asarray([0.0, 10.0]),
        state=np.asarray([source_state, target_state]),
        covariance=np.tile(np.diag([0.2, 0.3, 0.01, 0.02]), (2, 1, 1)),
        chi2=np.asarray([1.0, 1.0]),
        ndof=np.asarray([2.0, 2.0]),
        n_hit=np.asarray([3, 3], dtype=np.int16),
        hit_pattern=np.asarray([7, 7], dtype=np.uint64),
        truth_particle_id=np.asarray([42, 42], dtype=np.int64),
        truth_pdg=np.asarray([13, 13], dtype=np.int32),
        truth_match_fraction=np.asarray([1.0, 1.0]),
    )
    records = PropagationRecords(
        run_id=np.asarray([4], dtype=np.int64),
        event_id=np.asarray([9], dtype=np.int64),
        source_tracklet_id=np.asarray([3], dtype=np.int32),
        target_tracklet_id=np.asarray([8], dtype=np.int32),
        source_station_id=np.asarray([0], dtype=np.int16),
        target_station_id=np.asarray([1], dtype=np.int16),
        truth_particle_id=np.asarray([42], dtype=np.int64),
        target_z_mm=np.asarray([10.0]),
        prediction=np.asarray([target_state]),
        covariance=np.asarray([np.diag([0.4, 0.5, 0.03, 0.04])]),
        success=np.asarray([True]),
        has_covariance=np.asarray([True]),
    )

    evaluation = evaluate_field_propagation([event], records)

    assert evaluation.size == 1
    assert evaluation.rejected_counts["accepted"] == 1
    assert np.allclose(evaluation.residual, 0.0)
    assert np.allclose(evaluation.chi2, 0.0)
    assert np.allclose(evaluation.combined_covariance[0], event.covariance[1] + records.covariance[0])
