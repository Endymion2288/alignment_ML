"""Deterministic validation-only search for station-pair score thresholds."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from typing import Callable, Mapping, Sequence

import numpy as np


StationPair = tuple[int, int]
Evaluation = Mapping[str, object]
Evaluator = Callable[[Mapping[StationPair, float]], Evaluation]
PairThresholdEvaluator = Callable[[StationPair, float], Evaluation]


@dataclass(frozen=True)
class ThresholdSearchResult:
    """One selected threshold vector and its validation-only search trace."""

    thresholds: dict[StationPair, float]
    evaluation: Evaluation
    trace: tuple[dict[str, object], ...]


@dataclass(frozen=True)
class _ParetoState:
    """Additive association counts for one partial threshold vector."""

    correct_matches: int
    predicted_matches: int
    possible_matches: int
    thresholds: dict[StationPair, float]


def _association_counts(evaluation: Evaluation) -> tuple[int, int, int]:
    association = evaluation.get("association")
    if not isinstance(association, Mapping):
        raise ValueError("pair threshold evaluator must return an association mapping")
    try:
        possible = int(association["possible_matches"])
        predicted = int(association["predicted_matches"])
        correct = int(association["correct_matches"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError("association mapping lacks integer match counts") from error
    if possible < 0 or predicted < 0 or correct < 0 or correct > predicted:
        raise ValueError("association match counts are invalid")
    return possible, predicted, correct


def _association_evaluation(
    possible_matches: int,
    predicted_matches: int,
    correct_matches: int,
) -> dict[str, object]:
    """Build the additive metrics needed by the validation quality criterion."""
    if possible_matches < 0 or predicted_matches < 0 or correct_matches < 0:
        raise ValueError("association counts must be non-negative")
    if correct_matches > predicted_matches:
        raise ValueError("correct matches cannot exceed predicted matches")
    efficiency = None if not possible_matches else correct_matches / possible_matches
    purity = None if not predicted_matches else correct_matches / predicted_matches
    fake_rate = None if not predicted_matches else (predicted_matches - correct_matches) / predicted_matches
    return {
        "association": {
            "possible_matches": possible_matches,
            "predicted_matches": predicted_matches,
            "correct_matches": correct_matches,
            "association_efficiency": efficiency,
            "inclusive_association_purity": purity,
            "inclusive_fake_rate": fake_rate,
        }
    }


def _threshold_key(thresholds: Mapping[StationPair, float], pairs: Sequence[StationPair]) -> tuple[float, ...]:
    """Stable tie-breaker that prefers the more conservative threshold vector."""
    return tuple(float(thresholds[pair]) for pair in pairs)


def _pareto_prune(states: Sequence[_ParetoState], pairs: Sequence[StationPair]) -> list[_ParetoState]:
    """Keep only states not dominated in (predicted, correct) count space."""
    ordered = sorted(
        states,
        key=lambda state: (
            state.predicted_matches,
            -state.correct_matches,
            tuple(-value for value in _threshold_key(state.thresholds, pairs)),
        ),
    )
    result: list[_ParetoState] = []
    best_correct = -1
    for state in ordered:
        # A preceding state has no more predicted matches.  If it also has at
        # least as many correct matches, it cannot be worse for either the
        # purity/fake constraint or efficiency after later additive choices.
        if state.correct_matches <= best_correct:
            continue
        result.append(state)
        best_correct = state.correct_matches
    return result


def quality_rank(
    evaluation: Evaluation,
    maximum_inclusive_fake_rate: float,
    minimum_inclusive_purity: float,
) -> tuple[int, float, float, float, float]:
    """Rank feasible quality points before efficiency, then rank violations.

    The first tuple field makes a constraint-satisfying operating point always
    preferable to an infeasible one.  This avoids a high-efficiency candidate
    silently winning while exceeding the declared fake-rate limit.
    """
    if not 0.0 <= maximum_inclusive_fake_rate <= 1.0:
        raise ValueError("maximum_inclusive_fake_rate must be in [0, 1]")
    if not 0.0 <= minimum_inclusive_purity <= 1.0:
        raise ValueError("minimum_inclusive_purity must be in [0, 1]")
    association = evaluation.get("association")
    if not isinstance(association, Mapping):
        raise ValueError("threshold evaluator must return an association mapping")
    efficiency = association.get("association_efficiency")
    fake_rate = association.get("inclusive_fake_rate")
    purity = association.get("inclusive_association_purity")
    if efficiency is None or fake_rate is None or purity is None:
        return (0, -np.inf, -np.inf, -np.inf, -np.inf)
    efficiency_value = float(efficiency)
    fake_value = float(fake_rate)
    purity_value = float(purity)
    if not np.isfinite([efficiency_value, fake_value, purity_value]).all():
        return (0, -np.inf, -np.inf, -np.inf, -np.inf)
    violation = max(0.0, fake_value - maximum_inclusive_fake_rate) + max(
        0.0, minimum_inclusive_purity - purity_value
    )
    feasible = violation <= 1.0e-12
    return (
        int(feasible),
        -float(violation),
        efficiency_value,
        -fake_value,
        purity_value,
    )


def exact_pareto_thresholds(
    station_pairs: Sequence[StationPair],
    threshold_grid: Sequence[float],
    evaluator: PairThresholdEvaluator,
    maximum_inclusive_fake_rate: float,
    minimum_inclusive_purity: float,
) -> ThresholdSearchResult:
    """Exactly select a finite per-pair threshold vector on validation.

    Bipartite assignments for distinct station pairs are independent.  Their
    ``correct_matches`` and ``predicted_matches`` therefore add exactly, so a
    Pareto frontier in those two counts preserves every threshold vector that
    can improve the global inclusive-purity/fake constraint or efficiency.
    The solver receives no truth; only the already-evaluated validation metric
    payload is combined here.
    """
    pairs = tuple((int(source), int(target)) for source, target in station_pairs)
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("station_pairs must be non-empty and unique")
    values = tuple(sorted({float(value) for value in threshold_grid}))
    if not values or not np.isfinite(values).all() or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("threshold_grid must contain finite values in [0, 1]")

    options_by_pair: dict[StationPair, list[_ParetoState]] = {}
    trace: list[dict[str, object]] = []
    for pair in pairs:
        options: list[_ParetoState] = []
        expected_possible: int | None = None
        for threshold in values:
            possible, predicted, correct = _association_counts(evaluator(pair, threshold))
            if expected_possible is None:
                expected_possible = possible
            elif possible != expected_possible:
                raise ValueError("possible match denominator changed across score thresholds")
            options.append(
                _ParetoState(
                    correct_matches=correct,
                    predicted_matches=predicted,
                    possible_matches=possible,
                    thresholds={pair: threshold},
                )
            )
        options_by_pair[pair] = _pareto_prune(options, (pair,))
        trace.append(
            {
                "stage": "pair_options",
                "station_pair": f"{pair[0]}->{pair[1]}",
                "grid_options": len(options),
                "pareto_options": len(options_by_pair[pair]),
            }
        )

    frontier = [_ParetoState(0, 0, 0, {})]
    active_pairs: list[StationPair] = []
    for pair in pairs:
        active_pairs.append(pair)
        merged: list[_ParetoState] = []
        for state in frontier:
            for option in options_by_pair[pair]:
                merged.append(
                    _ParetoState(
                        correct_matches=state.correct_matches + option.correct_matches,
                        predicted_matches=state.predicted_matches + option.predicted_matches,
                        possible_matches=state.possible_matches + option.possible_matches,
                        thresholds={**state.thresholds, **option.thresholds},
                    )
                )
        frontier = _pareto_prune(merged, tuple(active_pairs))
        trace.append(
            {
                "stage": "combined_frontier",
                "through_station_pair": f"{pair[0]}->{pair[1]}",
                "raw_states": len(merged),
                "pareto_states": len(frontier),
            }
        )
    if not frontier:  # pragma: no cover - non-empty pair/grid invariant
        raise RuntimeError("exact threshold search produced no Pareto states")

    ranked: list[tuple[tuple[int, float, float, float, float], _ParetoState, Evaluation]] = []
    for state in frontier:
        evaluation = _association_evaluation(
            state.possible_matches,
            state.predicted_matches,
            state.correct_matches,
        )
        rank = quality_rank(
            evaluation,
            maximum_inclusive_fake_rate=maximum_inclusive_fake_rate,
            minimum_inclusive_purity=minimum_inclusive_purity,
        )
        ranked.append((rank, state, evaluation))
    rank, selected, evaluation = max(
        ranked,
        key=lambda item: (item[0], _threshold_key(item[1].thresholds, pairs)),
    )
    trace.append(
        {
            "stage": "selection",
            "raw_grid_cartesian_combinations": int(len(values) ** len(pairs)),
            "pareto_cartesian_combinations": int(prod(len(options_by_pair[pair]) for pair in pairs)),
            "final_pareto_states": len(frontier),
            "rank": list(rank),
            "thresholds": {
                f"{pair[0]}->{pair[1]}": selected.thresholds[pair] for pair in pairs
            },
        }
    )
    return ThresholdSearchResult(
        thresholds=dict(selected.thresholds),
        evaluation=evaluation,
        trace=tuple(trace),
    )


def coordinate_descent_thresholds(
    station_pairs: Sequence[StationPair],
    threshold_grid: Sequence[float],
    initial_thresholds: Mapping[StationPair, float],
    evaluator: Evaluator,
    maximum_inclusive_fake_rate: float,
    minimum_inclusive_purity: float,
    maximum_sweeps: int,
) -> ThresholdSearchResult:
    """Optimize a finite threshold vector with deterministic coordinate descent.

    Each candidate threshold is evaluated on the held-out validation split by
    ``evaluator``.  The routine has no truth access beyond the metric mapping
    returned after the truth-blind assignment has already run.
    """
    pairs = tuple((int(source), int(target)) for source, target in station_pairs)
    if not pairs or len(set(pairs)) != len(pairs):
        raise ValueError("station_pairs must be non-empty and unique")
    values = tuple(sorted({float(value) for value in threshold_grid}))
    if not values or not np.isfinite(values).all() or any(value < 0.0 or value > 1.0 for value in values):
        raise ValueError("threshold_grid must contain finite values in [0, 1]")
    if maximum_sweeps < 1:
        raise ValueError("maximum_sweeps must be positive")
    if set(initial_thresholds) != set(pairs):
        raise ValueError("initial_thresholds must specify exactly the station pairs")
    thresholds = {pair: float(initial_thresholds[pair]) for pair in pairs}
    if any(value not in values for value in thresholds.values()):
        raise ValueError("initial thresholds must be entries in threshold_grid")

    current = evaluator(thresholds)
    current_rank = quality_rank(
        current,
        maximum_inclusive_fake_rate=maximum_inclusive_fake_rate,
        minimum_inclusive_purity=minimum_inclusive_purity,
    )
    trace: list[dict[str, object]] = [
        {
            "sweep": 0,
            "updated_station_pair": None,
            "thresholds": {f"{pair[0]}->{pair[1]}": value for pair, value in thresholds.items()},
            "rank": list(current_rank),
        }
    ]
    for sweep in range(1, maximum_sweeps + 1):
        changed = False
        for pair in pairs:
            best_threshold = thresholds[pair]
            best_evaluation = current
            best_rank = current_rank
            for candidate in values:
                proposal = dict(thresholds)
                proposal[pair] = candidate
                evaluation = evaluator(proposal)
                rank = quality_rank(
                    evaluation,
                    maximum_inclusive_fake_rate=maximum_inclusive_fake_rate,
                    minimum_inclusive_purity=minimum_inclusive_purity,
                )
                if rank > best_rank or (rank == best_rank and candidate > best_threshold):
                    best_threshold = candidate
                    best_evaluation = evaluation
                    best_rank = rank
            if best_threshold != thresholds[pair]:
                thresholds[pair] = best_threshold
                current = best_evaluation
                current_rank = best_rank
                changed = True
            trace.append(
                {
                    "sweep": sweep,
                    "updated_station_pair": f"{pair[0]}->{pair[1]}",
                    "selected_threshold": thresholds[pair],
                    "thresholds": {
                        f"{item[0]}->{item[1]}": value for item, value in thresholds.items()
                    },
                    "rank": list(current_rank),
                }
            )
        if not changed:
            break
    return ThresholdSearchResult(
        thresholds=dict(thresholds),
        evaluation=current,
        trace=tuple(trace),
    )
