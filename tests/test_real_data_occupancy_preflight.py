from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from alignment.real_data_occupancy_preflight import (
    BLIND_ROLES,
    assert_occupancy_tree_is_residual_blind,
    contiguous_windows,
    first_passing_window,
    load_window_rule,
    next_segment,
    occupancy_window_summary,
    select_window_for_segment,
    window_meets_minima,
    window_minima,
)
from alignment.real_data_operating_protocol import (
    BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY,
    ROLE_CALIBRATION,
    assign_block_status,
)
from scripts.run_physical_refit_capture_scan import _audit_has_positive_tracklets


def _table(*, n=250, populated_at=0, n_tracklets=4):
    clusters = np.zeros(n, dtype=np.int32)
    tracklets = np.zeros(n, dtype=np.int32)
    four = np.zeros(n, dtype=np.int32)
    station_c = np.zeros((n, 4), dtype=np.int32)
    station_t = np.zeros((n, 4), dtype=np.int32)
    start = populated_at
    stop = populated_at + 100
    clusters[start:stop] = 20
    tracklets[start:stop] = n_tracklets
    four[start:stop] = 1
    station_c[start:stop] = 5
    station_t[start:stop] = 1
    return {
        "n_sct_clusters": clusters,
        "n_refit_tracklets": tracklets,
        "four_station_refit_multiplicity": four,
        "n_clusters_station": station_c,
        "n_refit_station": station_t,
        "run": np.full(n, 14973, dtype=np.int32),
        "lumi_block": np.ones(n, dtype=np.uint32),
        "stable_beams": np.ones(n, dtype=np.int32),
    }


def test_frozen_window_rule_is_residual_blind_and_keeps_roles():
    rule = load_window_rule(Path("configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml"))
    assert rule["residual_blind"] is True
    assert rule["v2_scoring"] is False
    assert rule["alignment_perturbation"] is False
    minima = window_minima(rule)
    assert minima["total_tracklets"] == 32
    assert minima["n_events_with_four_station_tracklets"] == 8
    assert {int(run): role for run, role in rule["blind_roles"].items()} == BLIND_ROLES
    with pytest.raises(ValueError, match="residual-blind"):
        assert_occupancy_tree_is_residual_blind(["n_sct_clusters", "residual_dx"])
    with pytest.raises(ValueError, match="v2"):
        assert_occupancy_tree_is_residual_blind(["v2_score"])


def test_first_skip_window_that_meets_minima_is_selected_not_the_richest():
    rule = load_window_rule(Path("configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml"))
    minima = window_minima(rule)
    table = _table(n=300, populated_at=0, n_tracklets=4)
    table["n_refit_tracklets"][200:300] = 40
    windows = contiguous_windows(table, nevents=100, skip_start=0, skip_stride=100)
    chosen = first_passing_window(windows, minima)
    assert chosen is not None
    assert chosen["skip_events"] == 0
    assert chosen["total_tracklets"] == 400
    richer = [window for window in windows if window["skip_events"] == 200][0]
    assert richer["total_tracklets"] > chosen["total_tracklets"]


def test_empty_prefix_is_skipped_in_skip_events_order():
    rule = load_window_rule(Path("configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml"))
    windows = contiguous_windows(
        _table(n=300, populated_at=200),
        nevents=100,
        skip_start=0,
        skip_stride=100,
    )
    chosen = first_passing_window(windows, window_minima(rule))
    assert chosen is not None
    assert chosen["skip_events"] == 200
    assert window_meets_minima(windows[0], window_minima(rule)) is False


def test_exhausted_segment_requests_next_filename_index(tmp_path):
    rule = load_window_rule(Path("configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml"))
    rec = tmp_path / "014973"
    rec.mkdir()
    for segment in ("00000", "00001"):
        (rec / f"Faser-Physics-014973-{segment}-r0022-xAOD.root").write_text("x", encoding="utf-8")
    empty = contiguous_windows(
        _table(n=100, populated_at=100),
        nevents=100,
        skip_start=0,
        skip_stride=100,
    )
    selected = select_window_for_segment(
        run=14973,
        segment="00000",
        windows=empty,
        rule=rule,
        rec_root=tmp_path,
    )
    assert selected["window_accepted"] is False
    assert selected["status"] == "need_next_segment"
    assert selected["next_segment"] == next_segment("00000") == "00001"
    assert selected["role"] == ROLE_CALIBRATION


def test_no_later_segment_is_insufficient_occupancy(tmp_path):
    rule = load_window_rule(Path("configs/operating_protocol_v1_real_data_occupancy_window_rule.yaml"))
    rec = tmp_path / "014973"
    rec.mkdir()
    (rec / "Faser-Physics-014973-00000-r0022-xAOD.root").write_text("x", encoding="utf-8")
    selected = select_window_for_segment(
        run=14973,
        segment="00000",
        windows=[],
        rule=rule,
        rec_root=tmp_path,
    )
    assert selected["status"] == BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY
    assert selected["window_accepted"] is False


def test_occupancy_summary_counts_only():
    summary = occupancy_window_summary(_table(n=100, populated_at=0))
    assert summary["n_events_with_clusters"] == 100
    assert summary["n_events_with_tracklets"] == 100
    assert summary["total_tracklets"] == 400
    assert summary["n_events_with_four_station_tracklets"] == 100
    assert "chi2" not in summary
    assert "residual" not in json.dumps(summary)


def test_insufficient_occupancy_block_status():
    assert (
        assign_block_status(
            role=ROLE_CALIBRATION,
            dq_failed=False,
            cross_level_contaminated=False,
            station_mode_valid=True,
            cdx_mode_valid=False,
            holdout_dq_worsened=False,
            anomalous_run_drift=False,
            insufficient_real_data_occupancy=True,
        )
        == BLOCK_INSUFFICIENT_REAL_DATA_OCCUPANCY
    )


def test_empty_content_audit_is_not_success(tmp_path):
    audit = tmp_path / "content_audit.json"
    audit.write_text(json.dumps({"events": 0, "tracklets": 0}), encoding="utf-8")
    assert _audit_has_positive_tracklets(audit) is False
    audit.write_text(json.dumps({"events": 100, "tracklets": 0}), encoding="utf-8")
    assert _audit_has_positive_tracklets(audit) is False
    audit.write_text(json.dumps({"events": 12, "tracklets": 40}), encoding="utf-8")
    assert _audit_has_positive_tracklets(audit) is True
