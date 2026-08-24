from __future__ import annotations

from training.hard_aware_reduction_eval import (
    classify_stop_reason,
    classify_train_reduction_effect,
    match_control_candidate,
    route_identity,
)


def _row(*, payload: str, event_id: int, truth_id: int, bin_name: str, short: str, loss: float, n: int = 2):
    delta = {"hard": -0.2, "near_boundary": 0.4, "easy": 1.4}[bin_name]
    return {
        "payload_id": payload,
        "run_id": 1,
        "event_id": event_id,
        "truth_id": truth_id,
        "margin_bin": bin_name,
        "short_station_bin": short,
        "production_margin": delta,
        "u_truth": 1.0,
        "u_best_solver_fragment": 1.0 - delta,
        "competitor_n_stations": 3 if short != "absent" else 4,
        "dustbin_aware_margin_loss": loss,
        "n_complete_truth_in_event": n,
        "production_fragment_winner": bin_name == "hard",
    }


def test_route_identity_requires_truth_id():
    row = _row(payload="iteration_00_draw_01", event_id=1, truth_id=7, bin_name="near_boundary", short="near_boundary", loss=0.6)
    assert route_identity(row)[-1] == 7


def test_event_max_short_pushed_to_easy_is_domain_coverage_if_gates_fail():
    control = [
        _row(payload="iteration_00_draw_01", event_id=1, truth_id=1, bin_name="near_boundary", short="near_boundary", loss=0.8),
        _row(payload="iteration_00_draw_01", event_id=1, truth_id=2, bin_name="easy", short="easy", loss=0.0),
    ]
    candidate = [
        _row(payload="iteration_00_draw_01", event_id=1, truth_id=1, bin_name="easy", short="easy", loss=0.0),
        _row(payload="iteration_00_draw_01", event_id=1, truth_id=2, bin_name="easy", short="easy", loss=0.0),
    ]
    matched = match_control_candidate(control, candidate)
    assert matched["control_event_max_short"] == 1
    assert matched["event_max_short_now_easy"] == 1
    effect = classify_train_reduction_effect(matched)
    assert effect == "short_event_max_pushed_off_boundary"
    stop = classify_stop_reason(gates_passed=False, train_effect=effect)
    assert stop["failure_class"] == "training_domain_coverage_limitation"
    assert stop["continue_to_15d_relative_wls"] is False


def test_event_max_short_still_near_is_reduction_failure():
    control = [
        _row(payload="iteration_00_draw_01", event_id=1, truth_id=1, bin_name="near_boundary", short="near_boundary", loss=0.8),
        _row(payload="iteration_00_draw_01", event_id=1, truth_id=2, bin_name="easy", short="easy", loss=0.0),
    ]
    candidate = [
        _row(payload="iteration_00_draw_01", event_id=1, truth_id=1, bin_name="near_boundary", short="near_boundary", loss=0.5),
        _row(payload="iteration_00_draw_01", event_id=1, truth_id=2, bin_name="easy", short="easy", loss=0.0),
    ]
    matched = match_control_candidate(control, candidate)
    effect = classify_train_reduction_effect(matched)
    assert effect == "short_event_max_still_on_boundary"
    stop = classify_stop_reason(gates_passed=False, train_effect=effect)
    assert stop["failure_class"] == "objective_reduction_failure"


def test_all_gates_pass_authorizes_15d():
    stop = classify_stop_reason(gates_passed=True, train_effect="short_event_max_pushed_off_boundary")
    assert stop["continue_to_15d_relative_wls"] is True
    assert stop["failure_class"] is None
