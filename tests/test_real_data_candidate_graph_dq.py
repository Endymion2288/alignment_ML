from __future__ import annotations

from pathlib import Path

from alignment.real_data_candidate_graph_dq import (
    CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED,
    CAMPAIGN_CROSS_LEVEL_CONTAMINATED,
    CAMPAIGN_CROSS_LEVEL_PRECONDITION_UNVERIFIED,
    CAMPAIGN_STATION_FIT_FAILED,
    CAMPAIGN_STATION_MODE_DRYRUN_PASSED_BUT_NOT_WRITABLE,
    assign_campaign_decision,
    campaign_allows_cdx_mode,
    evaluate_block_candidate_graph_dq,
    evaluate_candidate_graph_dq_gate,
    load_candidate_graph_dq_gate,
)
from alignment.real_data_operating_protocol import ROLE_CALIBRATION, ROLE_HELD_OUT_DQ, ROLE_HOLDOUT


def _gate():
    return load_candidate_graph_dq_gate(
        Path("configs/operating_protocol_v1_real_data_candidate_graph_dq_gate.yaml")
    )


def _percentiles(low: float, high: float) -> dict[str, float]:
    mid = 0.5 * (low + high)
    return {"p05": low - 0.1, "p16": low, "p50": mid, "p84": high, "p95": high + 0.1}


def _passing_block(*, source_id: str, run: int, role: str = ROLE_CALIBRATION) -> dict:
    return {
        "source_id": source_id,
        "run": run,
        "role": role,
        "physical_all_pairs_candidate_graph_nonempty": True,
        "selected_edges": 12,
        "selected_routes": 4,
        "values_finite": True,
        "physical_event_concentration": {
            "max_event_share": 0.20,
            "top2_event_share": 0.35,
        },
        "per_event_route_distribution": {"max_event_share_of_selected_routes": 0.25},
        "residual_percentiles": {"x_mm": _percentiles(-1.0, 1.0), "y_mm": _percentiles(-1.0, 1.0)},
        "pull_percentiles": {"x": _percentiles(-1.5, 1.5), "y": _percentiles(-1.5, 1.5)},
        "chi2_percentiles": _percentiles(2.0, 20.0),
        "dz_percentiles": _percentiles(1100.0, 3200.0),
    }


def test_frozen_gate_rejects_empty_selected_graph():
    gate = _gate()
    empty = _passing_block(source_id="cal_a", run=14973)
    empty["selected_edges"] = 0
    empty["selected_routes"] = 0
    result = evaluate_block_candidate_graph_dq(empty, gate)
    assert result["passed"] is False
    assert "empty_selected_edges" in result["reasons"]
    assert "empty_selected_routes" in result["reasons"]


def test_frozen_gate_rejects_single_event_dominance():
    gate = _gate()
    dominated = _passing_block(source_id="cal_a", run=14973)
    dominated["physical_event_concentration"] = {
        "max_event_share": 0.61,
        "top2_event_share": 0.90,
    }
    result = evaluate_block_candidate_graph_dq(dominated, gate)
    assert result["passed"] is False
    assert "physical_single_event_dominated" in result["reasons"]
    assert "physical_high_multiplicity_dominated" in result["reasons"]


def test_frozen_gate_passes_populated_compatible_calibration():
    gate = _gate()
    left = _passing_block(source_id="data24_r14973", run=14973)
    right = _passing_block(source_id="data24_r14974", run=14974)
    holdout = _passing_block(source_id="data24_r14975", run=14975, role=ROLE_HOLDOUT)
    holdout["selected_edges"] = 0
    holdout["selected_routes"] = 0
    dq = _passing_block(source_id="data24_r14977", run=14977, role=ROLE_HELD_OUT_DQ)
    dq["selected_edges"] = 0
    dq["selected_routes"] = 0
    result = evaluate_candidate_graph_dq_gate([left, right, holdout, dq], gate)
    assert result["passed"] is True
    assert result["do_not_lower_gate"] is True
    assert result["calibration_compatibility"]["compatible"] is True


def test_frozen_gate_rejects_contradictory_calibration_residuals():
    gate = _gate()
    left = _passing_block(source_id="data24_r14973", run=14973)
    right = _passing_block(source_id="data24_r14974", run=14974)
    right["residual_percentiles"] = {"x_mm": _percentiles(50.0, 80.0), "y_mm": _percentiles(-1.0, 1.0)}
    result = evaluate_candidate_graph_dq_gate([left, right], gate)
    assert result["passed"] is False
    assert "percentile_iqr_disjoint:residual_x_mm" in result["calibration_compatibility"]["reasons"]


def test_campaign_decision_priority_and_cdx_admission():
    assert (
        assign_campaign_decision(
            candidate_graph_dq_failed=True,
            station_fit_failed=True,
            implied_cdx_exceeds_operating_band=True,
            independent_cdx_evidence=True,
            station_and_holdout_dq_ok=True,
        )
        == CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED
    )
    assert (
        assign_campaign_decision(
            candidate_graph_dq_failed=False,
            station_fit_failed=True,
            implied_cdx_exceeds_operating_band=True,
            independent_cdx_evidence=False,
            station_and_holdout_dq_ok=False,
        )
        == CAMPAIGN_STATION_FIT_FAILED
    )
    assert (
        assign_campaign_decision(
            candidate_graph_dq_failed=False,
            station_fit_failed=False,
            implied_cdx_exceeds_operating_band=True,
            independent_cdx_evidence=False,
            station_and_holdout_dq_ok=True,
        )
        == CAMPAIGN_CROSS_LEVEL_CONTAMINATED
    )
    assert (
        assign_campaign_decision(
            candidate_graph_dq_failed=False,
            station_fit_failed=False,
            implied_cdx_exceeds_operating_band=False,
            independent_cdx_evidence=False,
            station_and_holdout_dq_ok=True,
        )
        == CAMPAIGN_CROSS_LEVEL_PRECONDITION_UNVERIFIED
    )
    assert (
        assign_campaign_decision(
            candidate_graph_dq_failed=False,
            station_fit_failed=False,
            implied_cdx_exceeds_operating_band=False,
            independent_cdx_evidence=True,
            station_and_holdout_dq_ok=True,
        )
        == CAMPAIGN_STATION_MODE_DRYRUN_PASSED_BUT_NOT_WRITABLE
    )
    assert campaign_allows_cdx_mode(CAMPAIGN_STATION_MODE_DRYRUN_PASSED_BUT_NOT_WRITABLE) is True
    assert campaign_allows_cdx_mode(CAMPAIGN_CANDIDATE_GRAPH_DQ_FAILED) is False
    assert campaign_allows_cdx_mode(CAMPAIGN_CROSS_LEVEL_PRECONDITION_UNVERIFIED) is False
