from __future__ import annotations

from collections import Counter

import numpy as np
import pytest

from datasets.root_loader import load_events
from datasets.synthetic import make_synthetic_tracklet_root
from datasets.synthetic_overlay import (
    SYNTHETIC_ROLE_RANDOM_EASY_FAKE,
    SYNTHETIC_ROLE_TRUE,
    _passes_truth_relative_band,
    write_synthetic_multitrack_root,
)


def test_synthetic_overlay_namespaces_truth_and_marks_fakes(tmp_path):
    source = make_synthetic_tracklet_root(tmp_path / "single_tracks.root")
    output = tmp_path / "overlay.root"
    summary = write_synthetic_multitrack_root(
        load_events(source, require_mc_labels=True),
        output,
        output_events=6,
        tracks_per_event=2,
        station_ids=(0, 1),
        missing_tracklet_probability=0.0,
        fake_mean_per_station=1.0,
        seed=27,
    )
    events = load_events(output, require_mc_labels=True)

    assert summary.true_tracklets_written == 24
    assert summary.fake_tracklets_written > 0
    assert summary.random_easy_fake_tracklets_written == summary.fake_tracklets_written
    assert summary.field_hard_fake_tracklets_written == 0
    assert summary.optional_q_over_p_fields == ()
    assert len(events) == 6
    unknown_tracklets = 0
    for event in events:
        labels = event.truth_particle_id
        roles = event.synthetic_role
        assert labels is not None
        assert roles is not None
        known = labels[labels >= 0]
        counts = Counter(int(label) for label in known)
        assert len(counts) == 2
        assert set(counts.values()) == {2}
        unknown_tracklets += int((labels < 0).sum())
        assert np.all(roles[labels >= 0] == SYNTHETIC_ROLE_TRUE)
        assert np.all(roles[labels < 0] == SYNTHETIC_ROLE_RANDOM_EASY_FAKE)
    assert unknown_tracklets == summary.fake_tracklets_written


def test_geometry_conditioned_fake_requires_real_propagation_and_no_track_cloning(tmp_path):
    source = make_synthetic_tracklet_root(tmp_path / "single_tracks.root")
    events = load_events(source, require_mc_labels=True)
    with pytest.raises(ValueError, match="physical_propagations"):
        write_synthetic_multitrack_root(
            events,
            tmp_path / "missing_propagation.root",
            output_events=1,
            tracks_per_event=2,
            station_ids=(0, 1),
            hard_negative_mean_per_target_station=0.1,
        )
    with pytest.raises(ValueError, match="refusing to duplicate"):
        write_synthetic_multitrack_root(
            events,
            tmp_path / "duplicate_tracks.root",
            output_events=1,
            tracks_per_event=5,
            station_ids=(0, 1),
        )


def test_relative_hard_negative_band_is_near_but_not_truth_dominating():
    assert _passes_truth_relative_band(11.0, 10.0, 1.1, 10.0)
    assert not _passes_truth_relative_band(10.9, 10.0, 1.1, 10.0)
    assert not _passes_truth_relative_band(101.0, 10.0, 1.1, 10.0)
    assert not _passes_truth_relative_band(1.0, 0.0, 1.1, 10.0)
    assert _passes_truth_relative_band(1.0, float("nan"), None, None)
