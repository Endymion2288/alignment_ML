from __future__ import annotations

from dataclasses import replace

import numpy as np

from datasets.propagation_loader import PropagationRecords
from datasets.root_loader import EventTracklets
from scripts.audit_field_candidate_coverage import _candidate_reason


def _event() -> EventTracklets:
    return EventTracklets(
        run_id=4,
        event_id=9,
        station_id=np.asarray([0, 1], dtype=np.int16),
        tracklet_id=np.asarray([3, 8], dtype=np.int32),
        z_mm=np.asarray([0.0, 10.0]),
        state=np.asarray([[0.0, 0.0, 0.1, -0.1], [1.0, -1.0, 0.1, -0.1]]),
        covariance=np.tile(np.diag([0.2, 0.3, 0.01, 0.02]), (2, 1, 1)),
        chi2=np.asarray([1.0, 1.0]),
        ndof=np.asarray([2.0, 2.0]),
        n_hit=np.asarray([3, 3], dtype=np.int16),
        hit_pattern=np.asarray([7, 7], dtype=np.uint64),
        truth_particle_id=np.asarray([42, 42], dtype=np.int64),
    )


def _records() -> PropagationRecords:
    return PropagationRecords(
        run_id=np.asarray([4], dtype=np.int64),
        event_id=np.asarray([9], dtype=np.int64),
        source_tracklet_id=np.asarray([3], dtype=np.int32),
        target_tracklet_id=np.asarray([8], dtype=np.int32),
        source_station_id=np.asarray([0], dtype=np.int16),
        target_station_id=np.asarray([1], dtype=np.int16),
        truth_particle_id=np.asarray([42], dtype=np.int64),
        target_z_mm=np.asarray([10.0]),
        prediction=np.asarray([[1.0, -1.0, 0.1, -0.1]]),
        covariance=np.asarray([np.diag([0.4, 0.5, 0.03, 0.04])]),
        success=np.asarray([True]),
        has_covariance=np.asarray([True]),
        q_over_p_mode=np.asarray([0], dtype=np.int8),
    )


def test_coverage_reason_matches_physical_candidate_eligibility_order():
    event = _event()
    records = _records()
    assert _candidate_reason(event, 0, 1, 0, 1, 0, records, 1.0e-6) == "candidate_eligible"
    assert _candidate_reason(event, 0, 1, 0, 1, None, records, 1.0e-6) == "no_exact_exported_record"

    failed = replace(records, success=np.asarray([False]))
    assert _candidate_reason(event, 0, 1, 0, 1, 0, failed, 1.0e-6) == "acts_propagation_failed"
