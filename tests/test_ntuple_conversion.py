from __future__ import annotations

import awkward as ak
import numpy as np
import uproot

from datasets.root_loader import load_events
from scripts.convert_ntuple_tracklets import convert_ntuple_tracklets


def _write_mock_enhanced_ntuple(path):
    branches = {
        "run": np.asarray([7, 7], dtype=np.int32),
        "eventID": np.asarray([10, 11], dtype=np.int32),
        "Tracklet_station_id": ak.Array([[0, 1], [0]]),
        "Tracklet_id": ak.Array([[0, 1], [0]]),
        "Tracklet_x_mm": ak.Array([[1.0, 2.0], [3.0]]),
        "Tracklet_y_mm": ak.Array([[4.0, 5.0], [6.0]]),
        "Tracklet_z_mm": ak.Array([[7.0, 8.0], [9.0]]),
        "Tracklet_tx": ak.Array([[0.01, 0.02], [0.03]]),
        "Tracklet_ty": ak.Array([[0.04, 0.05], [0.06]]),
        "Tracklet_chi2": ak.Array([[1.0, 2.0], [3.0]]),
        "Tracklet_ndof": ak.Array([[3.0, 4.0], [5.0]]),
        "Tracklet_n_hit": ak.Array([[6, 6], [6]]),
        "Tracklet_hit_pattern": ak.Array([[7, 7], [7]]),
        "Tracklet_has_covariance": ak.Array([[True, False], [True]]),
        "Tracklet_truth_particle_id": ak.Array([[101, 102], [103]]),
        "Tracklet_truth_pdg": ak.Array([[11, -11], [11]]),
        "Tracklet_truth_match_fraction": ak.Array([[1.0, 0.5], [0.75]]),
        "Tracklet_q_over_p_per_MeV": ak.Array([[1.0e-5, -2.0e-5], [3.0e-5]]),
        "Tracklet_q_over_p_from_momentum_per_MeV": ak.Array(
            [[1.0e-5, -2.0e-5], [3.0e-5]]
        ),
        "Tracklet_q_over_p_variance_per_MeV2": ak.Array([[1.0e-9, 2.0e-9], [3.0e-9]]),
        "Tracklet_has_q_over_p_covariance": ak.Array([[True, False], [True]]),
    }
    for name in (
        "xx_mm2",
        "xy_mm2",
        "xtx_mm",
        "xty_mm",
        "yy_mm2",
        "ytx_mm",
        "yty_mm",
        "txtx",
        "txty",
        "tyty",
    ):
        branches[f"Tracklet_cov_{name}"] = ak.Array([[0.01, 0.01], [0.01]])
    with uproot.recreate(path) as root_file:
        root_file["nt"] = branches


def test_ntuple_converter_flattens_and_drops_missing_covariance(tmp_path):
    source = tmp_path / "enhanced.root"
    destination = tmp_path / "canonical.root"
    _write_mock_enhanced_ntuple(source)

    summary = convert_ntuple_tracklets(source, destination, include_truth=True)

    assert summary.events_read == 2
    assert summary.tracklets_read == 3
    assert summary.tracklets_written == 2
    assert summary.dropped_missing_covariance == 1
    assert summary.optional_fields == (
        "has_q_over_p_covariance",
        "q_over_p_from_momentum_per_mev",
        "q_over_p_per_mev",
        "q_over_p_variance_per_mev2",
    )

    events = load_events(destination, require_mc_labels=True)
    assert [(event.run_id, event.event_id, event.size) for event in events] == [
        (7, 10, 1),
        (7, 11, 1),
    ]
    assert events[0].truth_particle_id.tolist() == [101]
    assert events[0].q_over_p_per_mev.tolist() == [1.0e-5]
    assert events[0].has_q_over_p_covariance.tolist() == [True]
