from __future__ import annotations

import pytest

from training.route_reduction_audit import (
    SHORT_SPARSE_ROUTE_FRACTION,
    classify_margin_bin,
    logit_margin_gradient_norm,
    production_margin,
    recommend_next_objective,
    summarize_reduction_audit,
)


def test_frozen_margin_bins():
    assert production_margin(1.2, 0.4) == pytest.approx(0.8)
    assert production_margin(0.4, None) == pytest.approx(0.4)
    assert production_margin(-0.2, -0.8) == pytest.approx(-0.2)
    assert classify_margin_bin(-0.1) == "hard"
    assert classify_margin_bin(0.0) == "hard"
    assert classify_margin_bin(0.4) == "near_boundary"
    assert classify_margin_bin(1.0) == "easy"
    assert classify_margin_bin(1.5) == "easy"


def test_active_relu_logit_grad_norm_counts_exclusive_edges():
    assert logit_margin_gradient_norm(3, 1, shared_edges=0) == 2.0
    assert logit_margin_gradient_norm(3, 1, shared_edges=1) == (2.0) ** 0.5


def _row(
    *,
    payload: str,
    event_id: int,
    bin_name: str,
    loss: float,
    n_event: int,
    short_bin: str,
    competitor_n: int | None,
    is_max: bool | None = None,
):
    delta = {"hard": -0.2, "near_boundary": 0.4, "easy": 1.4}[bin_name]
    return {
        "payload_id": payload,
        "run_id": 1,
        "event_id": event_id,
        "u_truth": 1.0,
        "production_margin": delta,
        "margin_bin": bin_name,
        "short_station_bin": short_bin,
        "competitor_n_stations": competitor_n,
        "n_complete_truth_in_event": n_event,
        "dustbin_aware_margin_loss": loss,
        "workbook56_packing_loss": loss,
        "dustbin_mean_share": loss / n_event,
        "packing_mean_share": loss / n_event,
        "dustbin_mean_grad_share": (1.0 if loss > 0 else 0.0) / n_event,
        "production_fragment_winner": bin_name == "hard",
        "production_by_length": {
            "2": {"bin": short_bin if competitor_n == 2 else "absent", "present": competitor_n == 2, "delta": delta if competitor_n == 2 else None, "loss": loss if competitor_n == 2 else 0.0},
            "3": {"bin": short_bin if competitor_n == 3 else "absent", "present": competitor_n == 3, "delta": delta if competitor_n == 3 else None, "loss": loss if competitor_n == 3 else 0.0},
            "4": {"bin": bin_name if competitor_n == 4 else "easy", "present": True, "delta": 1.2 if competitor_n != 4 else delta, "loss": 0.0 if competitor_n != 4 else loss},
        },
        "origin_run_id": 9000000000,
        "payload_family": payload,
    }


def test_mean_reduction_dilutes_two_easy_routes_against_one_hard():
    rows = [
        _row(payload="iteration_00_draw_01", event_id=1, bin_name="hard", loss=1.2, n_event=3, short_bin="near_boundary", competitor_n=2),
        _row(payload="iteration_00_draw_01", event_id=1, bin_name="easy", loss=0.0, n_event=3, short_bin="easy", competitor_n=2),
        _row(payload="iteration_00_draw_01", event_id=1, bin_name="easy", loss=0.0, n_event=3, short_bin="absent", competitor_n=4),
    ]
    summary = summarize_reduction_audit(rows)
    assert summary["hard"] == 1
    assert summary["near_boundary"] == 0
    assert summary["short_station_near_boundary"] + summary["short_station_hard"] >= 1
    assert summary["mean_dilutes_boundary"] is True
    assert summary["boundary_share_of_mean_dustbin_loss"] == 1.0
    assert summary["hard_share_of_mean_dustbin_loss"] == 1.0
    assert summary["near_boundary_share_of_mean_dustbin_loss"] == 0.0
    assert summary["n_event_max_is_short_boundary"] == 1


def test_recommend_absent_short_is_domain_coverage():
    decision = recommend_next_objective(
        {
            "complete_truth_chains": 100,
            "short_station_hard": 0,
            "short_station_near_boundary": 0,
            "four_station_boundary": 40,
            "n_event_max_is_short_boundary": 0,
            "mean_dilutes_boundary": True,
        }
    )
    assert decision["pre_register_hard_aware_reduction_control"] is False
    assert decision["next_step"] == "stop_curriculum_domain_coverage_limitation"
    assert decision["transfer_used_to_pick_reduction_or_weight"] is False
    assert decision["continue_to_15d_relative_wls"] is False


def test_recommend_sparse_short_is_domain_coverage():
    n = 1000
    near = int(SHORT_SPARSE_ROUTE_FRACTION * n) - 1
    decision = recommend_next_objective(
        {
            "complete_truth_chains": n,
            "short_station_hard": 0,
            "short_station_near_boundary": near,
            "four_station_boundary": 200,
            "n_event_max_is_short_boundary": 1,
            "mean_dilutes_boundary": True,
        }
    )
    assert decision["train_short_boundary_is_sparse"] is True
    assert decision["pre_register_hard_aware_reduction_control"] is False


def test_recommend_short_never_event_max_is_domain_coverage():
    decision = recommend_next_objective(
        {
            "complete_truth_chains": 100,
            "short_station_hard": 0,
            "short_station_near_boundary": 20,
            "four_station_boundary": 20,
            "n_event_max_is_short_boundary": 0,
            "mean_dilutes_boundary": True,
        }
    )
    assert decision["next_step"] == "stop_curriculum_domain_coverage_limitation"
    assert "never_the_per_event_max" in decision["reason"]


def test_recommend_present_short_event_max_and_dilution_opens_max_reduction():
    decision = recommend_next_objective(
        {
            "complete_truth_chains": 100,
            "short_station_hard": 0,
            "short_station_near_boundary": 20,
            "four_station_boundary": 20,
            "n_event_max_is_short_boundary": 8,
            "mean_dilutes_boundary": True,
        }
    )
    assert decision["pre_register_hard_aware_reduction_control"] is True
    assert decision["reduction_if_opened"] == "max_over_complete_truth_routes_in_event"
    assert decision["do_not_increase_dustbin_weight"] is True
    assert decision["new_checkpoint_authorized"] is False
