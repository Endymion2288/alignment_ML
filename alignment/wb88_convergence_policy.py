"""WB88 stopping policy.  Evaluates recorded iterations.  Does not solve.

The physical iteration stays the frozen WB86 loop.  This module only decides
whether an already computed step meets the historical rule or the prospective
absolute/relative rule.  It does not change the solver, the Jacobian, the
finite-difference step, the left-SE(3) update, or any WB86 artifact.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

from alignment.common_track_solver import (
    DEFAULT_DAMPING,
    SCALED_UPDATE_TOL,
    VALIDATION_REL_TOL,
    scaled_update_norm,
)
from alignment.wb85_physical_qualification_protocol import CONSECUTIVE_REQUIRED, MAX_ITERATIONS


# Frozen before WB88 replay.  Same constant the solver already uses as
# DEFAULT_DAMPING.  Not fit to a trajectory.
ABSOLUTE_CHI2_FLOOR = float(DEFAULT_DAMPING)
POLICY_WB86 = "wb86_historical_relative"
POLICY_WB88 = "wb88_absolute_or_relative"


def historical_relative_change(chi2: float, previous_chi2: float | None) -> float | None:
    """WB86 rule.  Undefined when the previous objective is missing or zero."""
    if previous_chi2 is None or abs(float(previous_chi2)) <= 0.0:
        return None
    return abs(float(chi2) - float(previous_chi2)) / abs(float(previous_chi2))


def objective_stability(chi2: float, previous_chi2: float | None) -> dict[str, Any]:
    """WB88 hybrid.  Defined at exact zero and at a numerical-floor chi2.

    Regime A.  When the previous chi2 is above the absolute floor, the metric
    is the frozen relative change.

    Regime B.  When the current absolute chi2 is on or below the floor, the
    metric is that absolute chi2.  The comparison is ``metric < VALIDATION_REL_TOL``.
    Because the floor itself is ``1e-8``, an objective on the floor satisfies
    the tolerance without dividing two noise-level numbers.

    Regime C.  ``chi2 = previous = 0`` returns metric 0.  It does not return null.

    Iteration 1 has no previous objective, so the metric stays null.  One
    undefined first step cannot by itself complete the consecutive pair.
    """
    relative = historical_relative_change(chi2, previous_chi2)
    if previous_chi2 is None:
        return {
            "objective_stability_metric": None,
            "objective_stability_regime": "no_previous",
            "historical_relative_change": None,
        }
    if abs(float(chi2)) <= ABSOLUTE_CHI2_FLOOR or abs(float(previous_chi2)) <= ABSOLUTE_CHI2_FLOOR:
        return {
            "objective_stability_metric": abs(float(chi2)),
            "objective_stability_regime": "absolute_floor",
            "historical_relative_change": relative,
        }
    return {
        "objective_stability_metric": relative,
        "objective_stability_regime": "relative",
        "historical_relative_change": relative,
    }


def _update_is_small(step: float) -> bool:
    return float(step) < float(SCALED_UPDATE_TOL)


def _objective_passes(metric: float | None) -> bool:
    return metric is not None and float(metric) < float(VALIDATION_REL_TOL)


def evaluate_policies(history: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Replay one recorded trajectory under both stopping rules.

    Consecutive counts are recomputed from the recorded chi2 and scaled update.
    The recorded WB86 count is kept beside them and is not trusted as the only
    copy.  Neither rule changes theta.
    """
    rows: list[dict[str, Any]] = []
    wb86_run = 0
    wb88_run = 0
    wb86_stopped = False
    wb88_stopped = False
    for raw in history:
        chi2 = float(raw["chi2"])
        previous = None if raw["previous_chi2"] is None else float(raw["previous_chi2"])
        step = float(raw["scaled_update_norm"])
        stability = objective_stability(chi2, previous)
        relative = stability["historical_relative_change"]
        wb86_gate = _update_is_small(step) and _objective_passes(relative)
        wb88_gate = _update_is_small(step) and _objective_passes(stability["objective_stability_metric"])
        if wb86_stopped:
            wb86_run = wb86_run
        elif wb86_gate:
            wb86_run += 1
        else:
            wb86_run = 0
        if wb88_stopped:
            wb88_run = wb88_run
        elif wb88_gate:
            wb88_run += 1
        else:
            wb88_run = 0
        if wb86_run >= int(CONSECUTIVE_REQUIRED):
            wb86_stopped = True
        if wb88_run >= int(CONSECUTIVE_REQUIRED):
            wb88_stopped = True
        rows.append(
            {
                "iteration": int(raw["iteration"]),
                "scaled_update_norm": step,
                "chi2": chi2,
                "previous_chi2": previous,
                "historical_relative_change": relative,
                "objective_stability_metric": stability["objective_stability_metric"],
                "objective_stability_regime": stability["objective_stability_regime"],
                "wb86_gate": bool(wb86_gate),
                "wb88_gate": bool(wb88_gate),
                "wb86_consecutive_count": int(wb86_run),
                "wb88_consecutive_count": int(wb88_run),
                "wb86_would_stop": bool(wb86_stopped),
                "wb88_would_stop": bool(wb88_stopped),
                "recorded_consecutive_convergence_count": int(raw["consecutive_convergence_count"]),
                "recorded_validation_relative_change": raw["validation_relative_change"],
            }
        )
    return rows


def stopping_iteration(rows: Sequence[Mapping[str, Any]], key: str) -> int | None:
    for row in rows:
        if row[key] and int(row["wb86_consecutive_count" if key.startswith("wb86") else "wb88_consecutive_count"]) >= int(CONSECUTIVE_REQUIRED):
            return int(row["iteration"])
    return None


def solver_scaled_truth_error(names: Sequence[str], theta: Sequence[float], theta_true: Sequence[float]) -> float:
    delta = [float(a) - float(b) for a, b in zip(theta, theta_true)]
    return scaled_update_norm(names, delta)


def raw_chart_error(theta: Sequence[float], theta_true: Sequence[float]) -> float:
    return math.sqrt(sum((float(a) - float(b)) ** 2 for a, b in zip(theta, theta_true)))


def policy_contract() -> dict[str, Any]:
    return {
        "policy": POLICY_WB88,
        "absolute_chi2_floor": ABSOLUTE_CHI2_FLOOR,
        "absolute_chi2_floor_source": "alignment.common_track_solver.DEFAULT_DAMPING",
        "scaled_update_tol": float(SCALED_UPDATE_TOL),
        "validation_rel_tol": float(VALIDATION_REL_TOL),
        "consecutive_required": int(CONSECUTIVE_REQUIRED),
        "max_iterations": int(MAX_ITERATIONS),
        "formula": (
            "gate if scaled_update_norm < 1e-3 and metric < 1e-3; "
            "metric = |chi2| when previous is missing? no: iteration 1 metric is null; "
            "metric = |chi2| when abs(chi2) or abs(previous_chi2) <= 1e-8; "
            "otherwise metric = |chi2 - previous| / |previous|; "
            "stop after 2 consecutive gates; MAX_ITERATIONS stays 10"
        ),
    }
