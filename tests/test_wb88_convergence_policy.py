"""Hermetic checks for the WB88 stopping policy.  No Calypso."""

from alignment.wb88_convergence_policy import (
    ABSOLUTE_CHI2_FLOOR,
    evaluate_policies,
    historical_relative_change,
    objective_stability,
)


def _row(iteration, chi2, previous, step, recorded_rel=None):
    return {
        "iteration": iteration,
        "chi2": chi2,
        "previous_chi2": previous,
        "scaled_update_norm": step,
        "consecutive_convergence_count": 0,
        "validation_relative_change": recorded_rel,
    }


def test_exact_zero_is_defined_and_stops():
    history = [
        _row(1, 0.0, None, 0.0),
        _row(2, 0.0, 0.0, 0.0),
        _row(3, 0.0, 0.0, 0.0),
    ]
    assert historical_relative_change(0.0, 0.0) is None
    stable = objective_stability(0.0, 0.0)
    assert stable["objective_stability_metric"] == 0.0
    assert stable["objective_stability_regime"] == "absolute_floor"
    rows = evaluate_policies(history)
    assert rows[0]["objective_stability_metric"] is None
    assert rows[1]["wb88_consecutive_count"] == 1
    assert rows[2]["wb88_would_stop"] is True
    assert rows[2]["wb86_would_stop"] is False
    assert rows[2]["iteration"] == 3


def test_numerical_floor_does_not_use_the_ratio():
    tiny = 1.0e-20
    history = [
        _row(1, 1000.0, None, 0.1),
        _row(2, tiny, 1000.0, 1.0e-12),
        _row(3, 4.0 * tiny, tiny, 1.0e-12),
        _row(4, tiny, 4.0 * tiny, 1.0e-12),
    ]
    rows = evaluate_policies(history)
    assert rows[2]["historical_relative_change"] == 3.0
    assert rows[2]["objective_stability_metric"] == 4.0 * tiny
    assert rows[2]["objective_stability_metric"] < ABSOLUTE_CHI2_FLOOR
    assert rows[3]["wb88_would_stop"] is True
    assert rows[3]["wb86_would_stop"] is False


def test_ordinary_large_chi2_keeps_the_relative_rule():
    history = [
        _row(1, 1000.0, None, 0.1),
        _row(2, 1000.0, 1000.0, 1.0e-6),
        _row(3, 1000.0 * (1.0 - 1.0e-6), 1000.0, 1.0e-6),
    ]
    rows = evaluate_policies(history)
    assert rows[1]["objective_stability_regime"] == "relative"
    assert rows[1]["objective_stability_metric"] == 0.0
    assert rows[2]["wb86_would_stop"] is True
    assert rows[2]["wb88_would_stop"] is True


def test_large_update_blocks_a_small_chi2():
    history = [
        _row(1, 0.0, None, 10.0),
        _row(2, 0.0, 0.0, 10.0),
        _row(3, 0.0, 0.0, 10.0),
    ]
    rows = evaluate_policies(history)
    assert all(row["wb88_would_stop"] is False for row in rows)
    assert all(row["wb88_gate"] is False for row in rows)


def test_relative_jump_above_tolerance_does_not_stop():
    history = [
        _row(1, 10.0, None, 1.0e-6),
        _row(2, 12.0, 10.0, 1.0e-6),
        _row(3, 9.0, 12.0, 1.0e-6),
    ]
    rows = evaluate_policies(history)
    assert rows[-1]["wb86_would_stop"] is False
    assert rows[-1]["wb88_would_stop"] is False
    assert rows[-1]["objective_stability_regime"] == "relative"
