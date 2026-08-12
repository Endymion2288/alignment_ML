from __future__ import annotations

from datasets.root_loader import load_events
from datasets.synthetic import make_synthetic_tracklet_root
from scripts.audit_tracklets import summarize_events


def test_content_audit_reports_observed_station_payload(tmp_path):
    root_path = make_synthetic_tracklet_root(tmp_path / "synthetic.root")
    events = load_events(root_path, require_mc_labels=True)

    summary = summarize_events(
        events,
        root_path,
        source_station=0,
        target_station=1,
        chi2_gate=50.0,
    )

    assert summary["events"] == 4
    assert summary["tracklets"] == 12
    assert summary["station_counts"] == {"0": 4, "1": 8}
    assert summary["covariance"]["positive_definite_rows"] == 12
    assert summary["candidate_diagnostics"]["candidate_pairs"] == 4
