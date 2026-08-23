from __future__ import annotations

from pathlib import Path

import numpy as np

from alignment.real_data_operating_protocol import ROLE_CALIBRATION, ROLE_HELD_OUT_DQ, ROLE_HOLDOUT
from alignment.real_data_v2_route_acceptance_scaling import (
    VERDICT_DOMAIN_SHIFT,
    VERDICT_PENDING,
    VERDICT_STATISTICS_LIMITED,
    VERDICT_TRACKLET_LIMITED,
    assign_scaling_verdict,
    campaign_allows_cdx_mode,
    campaign_allows_station_mode,
    load_scaling_config,
    occupancy_slice_summary,
    plan_campaign,
    plan_scale,
)


def _config():
    return load_scaling_config(
        Path("configs/operating_protocol_v1_real_data_v2_route_acceptance_scaling_v1.yaml")
    )


def _block(*, scale: str, run: int, role: str, **overrides) -> dict:
    row = {
        "scale": scale,
        "run": run,
        "role": role,
        "association_complete": True,
        "used_for_verdict": role == ROLE_CALIBRATION,
        "n_tracklets": 500,
        "n_all_pairs_candidates": 600,
        "selected_routes": 0,
        "complete_four_station_routes": 0,
        "event_concentration": {"max_event_share_of_selected_routes": 0.0},
    }
    row.update(overrides)
    return row


def test_frozen_config_forbids_alignment_and_residual():
    config = _config()
    assert config["do_not_enter_alignment"] is True
    assert config["geometry_write_allowed"] is False
    assert config["use_residual"] is False
    assert config["station_mode_blocked"] is True
    assert config["cdx_mode_blocked"] is True
    assert campaign_allows_station_mode(VERDICT_STATISTICS_LIMITED) is False
    assert campaign_allows_cdx_mode(VERDICT_DOMAIN_SHIFT) is False


def test_plan_caps_14977_tenk_to_full_and_keeps_skip():
    item = plan_scale(
        run=14977,
        role=ROLE_HELD_OUT_DQ,
        segment="00005",
        skip_events=135900,
        n_segment_events=137587,
        scale_name="n10000",
        requested_nevents=10000,
    )
    assert item["nevents"] == 1687
    assert item["capped_to_segment"] is True
    assert item["is_full_segment"] is True
    assert item["skip_events"] == 135900
    assert item["source_id"].endswith("_n01687")


def test_plan_campaign_deduplicates_identical_athena_jobs():
    config = _config()
    frozen = {
        "runs": {
            "14977": {
                "run": 14977,
                "role": ROLE_HELD_OUT_DQ,
                "segment": "00005",
                "skip_events": 135900,
                "n_events_scanned": 137587,
            }
        }
    }
    planned = [
        row
        for row in plan_campaign(frozen, config)
        if row["run"] == 14977 and row["scale"] in {"n10000", "full"}
    ]
    athena = [row for row in planned if row["needs_athena"]]
    assert len(athena) == 1
    assert athena[0]["nevents"] == 1687


def test_occupancy_slice_uses_same_skip():
    table = {
        "n_sct_clusters": np.arange(20, dtype=np.int32) + 1,
        "n_refit_tracklets": np.full(20, 2, dtype=np.int32),
        "four_station_refit_multiplicity": np.ones(20, dtype=np.int32),
        "n_clusters_station": np.ones((20, 4), dtype=np.int32),
        "n_refit_station": np.ones((20, 4), dtype=np.int32),
    }
    summary = occupancy_slice_summary(table, skip_events=10, nevents=5)
    assert summary["n_events"] == 5
    assert summary["total_tracklets"] == 10
    assert summary["skip_events"] == 10


def test_verdict_ignores_holdout_and_14977():
    config = _config()
    rows = [
        _block(scale="n100", run=14973, role=ROLE_CALIBRATION, n_tracklets=53, n_all_pairs_candidates=65),
        _block(scale="n100", run=14974, role=ROLE_CALIBRATION, n_tracklets=66, n_all_pairs_candidates=69),
        _block(scale="n1000", run=14973, role=ROLE_CALIBRATION, n_tracklets=500, n_all_pairs_candidates=600),
        _block(scale="n1000", run=14974, role=ROLE_CALIBRATION, n_tracklets=520, n_all_pairs_candidates=610),
        _block(
            scale="n1000",
            run=14975,
            role=ROLE_HOLDOUT,
            selected_routes=12,
            complete_four_station_routes=8,
            used_for_verdict=False,
        ),
        _block(
            scale="n1000",
            run=14977,
            role=ROLE_HELD_OUT_DQ,
            selected_routes=9,
            complete_four_station_routes=6,
            used_for_verdict=False,
        ),
    ]
    result = assign_scaling_verdict(rows, config)
    assert result["decision"] == VERDICT_DOMAIN_SHIFT
    assert ROLE_HELD_OUT_DQ in result["excluded_roles"]


def test_verdict_statistics_limited_when_selected_graph_recovers_without_complete_routes():
    config = _config()
    rows = [
        _block(scale="n100", run=14973, role=ROLE_CALIBRATION, n_tracklets=53, n_all_pairs_candidates=65),
        _block(scale="n100", run=14974, role=ROLE_CALIBRATION, n_tracklets=66, n_all_pairs_candidates=69),
        _block(
            scale="n10000",
            run=14973,
            role=ROLE_CALIBRATION,
            n_tracklets=7417,
            n_all_pairs_candidates=9000,
            selected_routes=17,
            complete_four_station_routes=0,
            event_concentration={"max_event_share_of_selected_routes": 0.12},
        ),
        _block(
            scale="n10000",
            run=14974,
            role=ROLE_CALIBRATION,
            n_tracklets=7942,
            n_all_pairs_candidates=9500,
            selected_routes=17,
            complete_four_station_routes=0,
            event_concentration={"max_event_share_of_selected_routes": 0.18},
        ),
    ]
    result = assign_scaling_verdict(rows, config)
    assert result["decision"] == VERDICT_STATISTICS_LIMITED
    assert result["complete_four_station_recovered_at_scale"] is None


def test_verdict_statistics_limited_when_larger_scale_recovers():
    config = _config()
    rows = [
        _block(scale="n100", run=14973, role=ROLE_CALIBRATION, n_tracklets=53, n_all_pairs_candidates=65),
        _block(scale="n100", run=14974, role=ROLE_CALIBRATION, n_tracklets=66, n_all_pairs_candidates=69),
        _block(
            scale="n1000",
            run=14973,
            role=ROLE_CALIBRATION,
            selected_routes=8,
            complete_four_station_routes=6,
            event_concentration={"max_event_share_of_selected_routes": 0.2},
        ),
        _block(
            scale="n1000",
            run=14974,
            role=ROLE_CALIBRATION,
            selected_routes=7,
            complete_four_station_routes=5,
            event_concentration={"max_event_share_of_selected_routes": 0.25},
        ),
    ]
    assert assign_scaling_verdict(rows, config)["decision"] == VERDICT_STATISTICS_LIMITED


def test_verdict_tracklet_limited_when_largest_still_sparse():
    config = _config()
    rows = [
        _block(scale="n100", run=14973, role=ROLE_CALIBRATION, n_tracklets=20, n_all_pairs_candidates=10),
        _block(scale="n100", run=14974, role=ROLE_CALIBRATION, n_tracklets=22, n_all_pairs_candidates=11),
        _block(scale="full", run=14973, role=ROLE_CALIBRATION, n_tracklets=80, n_all_pairs_candidates=90),
        _block(scale="full", run=14974, role=ROLE_CALIBRATION, n_tracklets=90, n_all_pairs_candidates=95),
    ]
    assert assign_scaling_verdict(rows, config)["decision"] == VERDICT_TRACKLET_LIMITED


def test_verdict_pending_until_two_calibration_blocks_exist():
    config = _config()
    rows = [_block(scale="n100", run=14973, role=ROLE_CALIBRATION)]
    assert assign_scaling_verdict(rows, config)["decision"] == VERDICT_PENDING


def test_verdict_pending_while_only_n100_is_complete():
    config = _config()
    rows = [
        _block(scale="n100", run=14973, role=ROLE_CALIBRATION, n_tracklets=53, n_all_pairs_candidates=65),
        _block(scale="n100", run=14974, role=ROLE_CALIBRATION, n_tracklets=66, n_all_pairs_candidates=69),
    ]
    result = assign_scaling_verdict(rows, config)
    assert result["decision"] == VERDICT_PENDING
    assert "expanded_scales_not_yet_associated" in result["reasons"]
