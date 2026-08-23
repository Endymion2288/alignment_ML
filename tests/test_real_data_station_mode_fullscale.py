from __future__ import annotations

from pathlib import Path

from alignment.real_data_operating_protocol import ROLE_CALIBRATION, ROLE_HELD_OUT_DQ, ROLE_HOLDOUT
from alignment.real_data_station_mode_fullscale import (
    DECISION_CROSS_LEVEL_CONTAMINATED,
    DECISION_CROSS_LEVEL_PRECONDITION_UNVERIFIED,
    DECISION_ROUTE_DQ_INSUFFICIENT,
    DECISION_SELF_NULLING_FD_SUBMITTED,
    DECISION_SELF_NULLING_PENDING,
    assign_next_decision,
    campaign_allows_cdx_mode,
    campaign_allows_geometry_write,
    evaluate_fullscale_route_dq,
    load_fullscale_config,
    summarize_residual_observables,
    summarize_selected_routes,
)


def _config():
    return load_fullscale_config(
        Path("configs/operating_protocol_v1_real_data_station_mode_self_nulling_fullscale_v1.yaml")
    )


def _route(run: int, event: int, stations: list[int], complete: bool = False) -> dict:
    return {
        "run_id": run,
        "event_id": event,
        "is_complete_four_station_route": complete,
        "endpoint_stations": stations,
        "utility": 0.2,
        "endpoint_provenance": [
            {
                "origin_run_id": run,
                "origin_event_id": event,
                "station_id": station,
                "origin_tracklet_id": station,
            }
            for station in stations
        ],
    }


def _calib_block(run: int, **overrides) -> dict:
    row = {
        "run": run,
        "role": ROLE_CALIBRATION,
        "used_for_verdict": True,
        "source_id": f"data24_r{run}",
        "n_all_pairs_candidates": 50000,
        "selected_routes": 100,
        "candidate_graph_nonempty": True,
        "values_finite": True,
        "event_concentration": {"max_event_share_of_selected_routes": 0.02},
        "residual_observables": {
            "selected_field_edge": {
                "residual_x_mm": {"p16": -2.0, "p84": 2.0},
                "pull_x": {"p16": -1.0, "p84": 1.0},
                "chi2": {"p16": 1.0, "p84": 20.0},
            }
        },
        "truth_free_complete_route_fraction": 0.01,
    }
    row.update(overrides)
    return row


def test_config_freezes_write_cdx_and_v2():
    config = _config()
    assert config["geometry_write_allowed"] is False
    assert config["official_conditions_db_write"] is False
    assert config["cdx_mode_blocked"] is True
    assert config["joint_station_cdx_newton"] is False
    assert config["do_not_retrain_v2"] is True
    assert config["residual_reduction_is_not_alignment_success"] is True
    assert campaign_allows_cdx_mode(DECISION_SELF_NULLING_PENDING) is False
    assert campaign_allows_geometry_write(DECISION_CROSS_LEVEL_PRECONDITION_UNVERIFIED) is False


def test_selected_route_summary_is_truth_free():
    summary = summarize_selected_routes(
        [
            _route(14973, 1, [0, 1], complete=False),
            _route(14973, 2, [0, 1, 2, 3], complete=True),
        ]
    )
    assert summary["selected_routes"] == 2
    assert summary["complete_four_station_routes"] == 1
    assert summary["truth_free_complete_route_fraction"] == 0.5
    assert "purity" not in summary
    assert summary["edge_reuse"]["reused_endpoints"] == 0
    assert summary["event_concentration"]["max_event_share_of_selected_routes"] == 0.5


def test_residual_summary_is_dq_only():
    report = summarize_residual_observables(
        [{"residual_x_mm": 1.0, "residual_y_mm": -0.5, "residual_tx": 0.0, "residual_ty": 0.0, "pull_x_mm": 0.2, "pull_y_mm": -0.1, "chi2": 4.0}],
        [{"residual_x_mm": 0.3, "residual_y_mm": 0.1, "residual_tx": 0.0, "residual_ty": 0.0, "chi2": 2.0}],
    )
    assert report["residual_reduction_is_not_alignment_success"] is True
    assert report["dq_observable_only"] is True
    assert report["selected_field_edge"]["n_edges"] == 1


def test_route_dq_uses_calibration_only():
    config = _config()
    rows = [
        _calib_block(14973),
        _calib_block(14974),
        {
            "run": 14977,
            "role": ROLE_HELD_OUT_DQ,
            "used_for_verdict": False,
            "selected_routes": 0,
            "n_all_pairs_candidates": 10,
            "candidate_graph_nonempty": True,
            "event_concentration": {"max_event_share_of_selected_routes": 1.0},
        },
        {
            "run": 14975,
            "role": ROLE_HOLDOUT,
            "used_for_verdict": False,
            "selected_routes": 1,
            "n_all_pairs_candidates": 90000,
            "candidate_graph_nonempty": True,
            "event_concentration": {"max_event_share_of_selected_routes": 0.01},
        },
    ]
    result = evaluate_fullscale_route_dq(rows, config)
    assert result["passed"] is True
    assert ROLE_HELD_OUT_DQ in result["excluded_roles"]


def test_route_dq_fails_when_selected_graph_empty():
    config = _config()
    rows = [
        _calib_block(14973, selected_routes=0),
        _calib_block(14974, selected_routes=0),
    ]
    result = evaluate_fullscale_route_dq(rows, config)
    assert result["passed"] is False
    assert "too_few_selected_routes" in result["reasons"]


def test_next_decision_pending_until_newton():
    pending = assign_next_decision(
        route_dq_passed=True,
        fd_submitted=False,
        newton_complete=False,
        capture_passed=None,
        implied_cdx_exceeds_operating_band=False,
        independent_cdx_evidence=False,
        holdout_dq_worsened=False,
    )
    assert pending["decision"] == DECISION_SELF_NULLING_PENDING
    assert pending["geometry_write_allowed"] is False
    submitted = assign_next_decision(
        route_dq_passed=True,
        fd_submitted=True,
        newton_complete=False,
        capture_passed=None,
        implied_cdx_exceeds_operating_band=False,
        independent_cdx_evidence=False,
        holdout_dq_worsened=False,
    )
    assert submitted["decision"] == DECISION_SELF_NULLING_FD_SUBMITTED
    assert submitted["cdx_mode_allowed"] is False


def test_next_decision_contaminated_when_implied_cdx_exceeds():
    result = assign_next_decision(
        route_dq_passed=True,
        fd_submitted=True,
        newton_complete=True,
        capture_passed=None,
        implied_cdx_exceeds_operating_band=True,
        independent_cdx_evidence=False,
        holdout_dq_worsened=False,
    )
    assert result["decision"] == DECISION_CROSS_LEVEL_CONTAMINATED
    assert result["geometry_write_allowed"] is False
    assert result["cdx_mode_allowed"] is False


def test_next_decision_insufficient_without_routes():
    result = assign_next_decision(
        route_dq_passed=False,
        fd_submitted=False,
        newton_complete=False,
        capture_passed=None,
        implied_cdx_exceeds_operating_band=False,
        independent_cdx_evidence=False,
        holdout_dq_worsened=False,
    )
    assert result["decision"] == DECISION_ROUTE_DQ_INSUFFICIENT
