from __future__ import annotations

import numpy as np

from datasets.root_loader import EventTracklets
from scripts.audit_refitted_source_diversity import summarize_refitted_source


def _event(stations, pdg, fraction, state) -> EventTracklets:
    size = len(stations)
    return EventTracklets(
        run_id=1,
        event_id=2,
        station_id=np.asarray(stations, dtype=np.int16),
        tracklet_id=np.arange(size, dtype=np.int32),
        z_mm=np.asarray(stations, dtype=np.float64),
        state=np.asarray(state, dtype=np.float64),
        covariance=np.tile(np.eye(4, dtype=np.float64), (size, 1, 1)),
        chi2=np.ones(size, dtype=np.float64),
        ndof=np.ones(size, dtype=np.float64),
        n_hit=np.full(size, 6, dtype=np.int16),
        hit_pattern=np.full(size, 0b111, dtype=np.uint64),
        truth_particle_id=np.arange(10, 10 + size, dtype=np.int64),
        truth_pdg=np.asarray(pdg, dtype=np.int32),
        truth_match_fraction=np.asarray(fraction, dtype=np.float64),
    )


def test_refitted_source_diversity_reports_truth_matched_muon_state_by_station():
    events = [
        _event(
            [0, 1, 2, 3],
            [13, 13, -13, 11],
            [1.0, 0.99, 0.98, 1.0],
            [[-2.0, 1.0, -0.02, 0.01], [-1.0, 2.0, -0.01, 0.02], [5.0, 6.0, 0.03, 0.04], [9.0, 9.0, 0.0, 0.0]],
        )
    ]

    summary = summarize_refitted_source("source_train", events, min_truth_match_fraction=0.99)

    assert summary["events"] == 1
    assert summary["truth_matched_muon_tracklets"] == 2
    assert summary["truth_pdg_counts"] == {"-13": 1, "11": 1, "13": 2}
    state = summary["truth_matched_muon_state"]
    assert state["x_mm"]["min"] == -2.0
    assert state["x_mm"]["max"] == -1.0
    assert summary["truth_matched_muon_state_by_station"]["2"]["x_mm"]["count"] == 0
