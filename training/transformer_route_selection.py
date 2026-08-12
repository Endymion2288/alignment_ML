"""Validation-only route operating-point selection for Transformer V1."""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from baselines.route_assignment import RouteAssignmentConfig, adjacent_station_pairs
from training.curriculum_mlp import CandidateSet
from training.route_assignment import (
    RouteAssignmentContext,
    evaluate_adjacent_route_assignment_sets,
    prepare_route_assignment_context,
)


@dataclass(frozen=True)
class RouteSelectionResult:
    """One frozen validation choice and its auditable finite search trace."""

    thresholds: Mapping[tuple[int, int], float]
    unmatched_penalty: float
    evaluation_by_magnitude: Mapping[float, Mapping[str, object]]
    capture_by_magnitude: Mapping[float, bool]
    rank: tuple[float, ...]
    search_rows: tuple[Mapping[str, object], ...]
    selection_event_counts_by_magnitude: Mapping[float, int]
    full_validation_event_counts_by_magnitude: Mapping[float, int]
    full_validation_rerank_candidates_requested: int
    full_validation_rerank_candidates_evaluated: int


class RouteHypothesisOverflow(RuntimeError):
    """A candidate operating point is too loose for the frozen route solver."""


def capture_success(route: Mapping[str, object], criteria: Mapping[str, object]) -> bool:
    """Apply the predeclared route-level primary criterion truth only for metrics."""
    requirements = (
        ("complete_track_efficiency", ">=", "minimum_complete_track_efficiency"),
        ("complete_track_purity", ">=", "minimum_complete_track_purity"),
        ("track_fake_rate", "<=", "maximum_track_fake_rate"),
    )
    for metric, relation, limit_name in requirements:
        value = route.get(metric)
        if value is None:
            return False
        limit = float(criteria[limit_name])
        if relation == ">=" and float(value) < limit:
            return False
        if relation == "<=" and float(value) > limit:
            return False
    return True


def _validate_controls(
    station_path: Sequence[int],
    threshold_grid: Sequence[float],
    unmatched_penalties: Sequence[float],
    initial_thresholds: Mapping[tuple[int, int], float],
    maximum_sweeps: int,
    criteria: Mapping[str, object],
) -> tuple[tuple[int, ...], tuple[tuple[int, int], ...], tuple[float, ...], tuple[float, ...]]:
    path = tuple(int(station) for station in station_path)
    pairs = adjacent_station_pairs(path)
    thresholds = tuple(sorted({float(value) for value in threshold_grid}))
    penalties = tuple(float(value) for value in unmatched_penalties)
    if not thresholds or any(not math.isfinite(value) or not 0.0 <= value <= 1.0 for value in thresholds):
        raise ValueError("route threshold grid must contain finite probabilities in [0, 1]")
    if not penalties or any(not math.isfinite(value) for value in penalties):
        raise ValueError("route unmatched-penalty grid must be finite and non-empty")
    if set(initial_thresholds) != set(pairs):
        raise ValueError("initial route thresholds must define exactly the adjacent station pairs")
    if any(float(value) not in thresholds for value in initial_thresholds.values()):
        raise ValueError("initial route thresholds must be declared grid values")
    if maximum_sweeps < 1:
        raise ValueError("route threshold selection requires at least one sweep")
    for key in (
        "minimum_complete_track_efficiency",
        "minimum_complete_track_purity",
        "maximum_track_fake_rate",
    ):
        value = float(criteria[key])
        if not math.isfinite(value) or not 0.0 <= value <= 1.0:
            raise ValueError(f"capture criterion '{key}' must be in [0, 1]")
    return path, pairs, thresholds, penalties


def _group_by_magnitude(
    candidate_sets: Sequence[CandidateSet], scores: Sequence[np.ndarray]
) -> dict[float, tuple[list[CandidateSet], list[np.ndarray]]]:
    if len(candidate_sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    grouped: dict[float, tuple[list[CandidateSet], list[np.ndarray]]] = {}
    for candidate_set, score in zip(candidate_sets, scores):
        magnitude = float(candidate_set.sample.magnitude_mm)
        sets, values = grouped.setdefault(magnitude, ([], []))
        sets.append(candidate_set)
        values.append(np.asarray(score, dtype=np.float64))
    if not grouped:
        raise ValueError("route selection has no validation candidate sets")
    return grouped


def _event_groups(
    candidate_sets: Sequence[CandidateSet], scores: Sequence[np.ndarray]
) -> dict[tuple[str, str, int, int], tuple[list[CandidateSet], list[np.ndarray]]]:
    """Keep all adjacent matrices of a synthetic event together for sampling."""
    grouped: dict[tuple[str, str, int, int], tuple[list[CandidateSet], list[np.ndarray]]] = {}
    for candidate_set, score in zip(candidate_sets, scores):
        key = (
            str(candidate_set.sample.source_id),
            str(candidate_set.sample.payload_id),
            int(candidate_set.event.run_id),
            int(candidate_set.event.event_id),
        )
        sets, values = grouped.setdefault(key, ([], []))
        sets.append(candidate_set)
        values.append(np.asarray(score, dtype=np.float64))
    return grouped


def _selection_subset(
    grouped: Mapping[float, tuple[list[CandidateSet], list[np.ndarray]]],
    events_per_magnitude: int | None,
) -> tuple[dict[float, tuple[list[CandidateSet], list[np.ndarray]]], dict[float, int]]:
    """Take a stable, label-free event subset for finite route-grid search."""
    if events_per_magnitude is not None and events_per_magnitude < 1:
        raise ValueError("selection_events_per_magnitude must be positive when supplied")
    result: dict[float, tuple[list[CandidateSet], list[np.ndarray]]] = {}
    counts: dict[float, int] = {}
    for magnitude, (sets, values) in grouped.items():
        event_rows = _event_groups(sets, values)
        ordered = sorted(
            event_rows,
            key=lambda key: hashlib.sha256("\x1f".join(str(value) for value in key).encode()).hexdigest(),
        )
        selected_keys = ordered if events_per_magnitude is None else ordered[:events_per_magnitude]
        selected_sets: list[CandidateSet] = []
        selected_scores: list[np.ndarray] = []
        for key in selected_keys:
            event_sets, event_scores = event_rows[key]
            selected_sets.extend(event_sets)
            selected_scores.extend(event_scores)
        if not selected_sets:
            raise ValueError("route selection subset contains no candidate sets")
        result[magnitude] = (selected_sets, selected_scores)
        counts[magnitude] = len(selected_keys)
    return result, counts


def _selection_rank(
    evaluations: Mapping[float, Mapping[str, object]],
    captures: Mapping[float, bool],
    thresholds: Mapping[tuple[int, int], float],
) -> tuple[float, ...]:
    """Predeclared lexicographic validation policy with a nominal guardrail."""
    magnitudes = tuple(sorted(evaluations))
    nominal = 0.0 if 0.0 in captures else min(magnitudes)
    route_values = []
    for magnitude in magnitudes:
        route = evaluations[magnitude].get("route")
        if not isinstance(route, Mapping):
            raise RuntimeError("route evaluation is malformed")
        route_values.append(route)

    def mean_metric(name: str) -> float:
        values = [float(route[name]) for route in route_values if route.get(name) is not None]
        return float(np.mean(values)) if values else -math.inf

    captured_magnitudes = [magnitude for magnitude in magnitudes if captures[magnitude]]
    maximum_capture = max(captured_magnitudes) if captured_magnitudes else -1.0
    # Priority order: nominal primary point, number of curriculum points that
    # meet the fixed criterion, largest recovered magnitude, then mean quality.
    # The final negative threshold sum is a deterministic conservative tie-break.
    return (
        float(bool(captures[nominal])),
        float(sum(bool(value) for value in captures.values())),
        float(maximum_capture),
        mean_metric("complete_track_efficiency"),
        mean_metric("complete_track_purity"),
        -mean_metric("track_fake_rate"),
        -float(sum(thresholds.values())),
    )


def select_route_operating_point(
    candidate_sets: Sequence[CandidateSet],
    scores: Sequence[np.ndarray],
    *,
    station_path: Sequence[int] = (0, 1, 2, 3),
    threshold_grid: Sequence[float],
    unmatched_penalties: Sequence[float],
    initial_thresholds: Mapping[tuple[int, int], float],
    maximum_sweeps: int,
    capture_criteria: Mapping[str, object],
    calibration_bins: int,
    maximum_hypotheses: int = 100_000,
    selection_events_per_magnitude: int | None = None,
    full_validation_rerank_candidates: int = 1,
) -> RouteSelectionResult:
    """Choose thresholds/penalty solely from validation physical payloads.

    A coordinate search makes the V1 route scan tractable while every tested
    vector is evaluated with the unchanged exact unit-capacity route solver.
    When the search uses a deterministic event subset, its top finite set of
    candidate controls can be re-ranked on every validation event before the
    final freeze.  No calibration, model update, threshold adaptation, or test
    data occurs in this function.
    """
    path, pairs, grid, penalties = _validate_controls(
        station_path,
        threshold_grid,
        unmatched_penalties,
        initial_thresholds,
        maximum_sweeps,
        capture_criteria,
    )
    if full_validation_rerank_candidates < 1:
        raise ValueError("full_validation_rerank_candidates must be positive")
    grouped = _group_by_magnitude(candidate_sets, scores)
    selection_grouped, selection_counts = _selection_subset(
        grouped, selection_events_per_magnitude
    )
    _, full_counts = _selection_subset(grouped, None)
    contexts: dict[float, RouteAssignmentContext] = {
        magnitude: prepare_route_assignment_context(sets, values, calibration_bins, path)
        for magnitude, (sets, values) in selection_grouped.items()
    }
    cache: dict[tuple[float, tuple[float, ...]], tuple[dict[float, Mapping[str, object]], dict[float, bool], tuple[float, ...]]] = {}
    search_rows: list[Mapping[str, object]] = []

    def evaluate(
        thresholds: Mapping[tuple[int, int], float], penalty: float, record: bool
    ) -> tuple[dict[float, Mapping[str, object]], dict[float, bool], tuple[float, ...]]:
        key = (float(penalty), tuple(float(thresholds[pair]) for pair in pairs))
        cached = cache.get(key)
        if cached is not None:
            return cached
        config = RouteAssignmentConfig(
            score_threshold_by_pair=dict(thresholds),
            unmatched_penalty=float(penalty),
            station_path=path,
            maximum_hypotheses=maximum_hypotheses,
        )
        evaluations: dict[float, Mapping[str, object]] = {}
        captures: dict[float, bool] = {}
        for magnitude, (sets, values) in sorted(selection_grouped.items()):
            try:
                evaluation = evaluate_adjacent_route_assignment_sets(
                    sets,
                    values,
                    config,
                    calibration_bins=calibration_bins,
                    context=contexts[magnitude],
                )
            except RuntimeError as error:
                if "route hypothesis count exceeds maximum_hypotheses" in str(error):
                    raise RouteHypothesisOverflow(str(error)) from error
                raise
            route = evaluation.get("route")
            if not isinstance(route, Mapping):
                raise RuntimeError("route selection evaluation is malformed")
            evaluations[magnitude] = evaluation
            captures[magnitude] = capture_success(route, capture_criteria)
        rank = _selection_rank(evaluations, captures, thresholds)
        cached = (evaluations, captures, rank)
        cache[key] = cached
        if record:
            search_rows.append(
                {
                    "unmatched_penalty": float(penalty),
                    "thresholds": {f"{left}->{right}": float(thresholds[(left, right)]) for left, right in pairs},
                    "rank": list(rank),
                    "capture_by_magnitude": {str(magnitude): bool(value) for magnitude, value in captures.items()},
                    "selection_event_counts_by_magnitude": {
                        str(magnitude): selection_counts[magnitude] for magnitude in sorted(selection_counts)
                    },
                    "complete_track_efficiency_by_magnitude": {
                        str(magnitude): evaluations[magnitude]["route"]["complete_track_efficiency"]
                        for magnitude in sorted(evaluations)
                    },
                    "complete_track_purity_by_magnitude": {
                        str(magnitude): evaluations[magnitude]["route"]["complete_track_purity"]
                        for magnitude in sorted(evaluations)
                    },
                    "track_fake_rate_by_magnitude": {
                        str(magnitude): evaluations[magnitude]["route"]["track_fake_rate"]
                        for magnitude in sorted(evaluations)
                    },
                }
            )
        return cached

    best: tuple[dict[tuple[int, int], float], float, dict[float, Mapping[str, object]], dict[float, bool], tuple[float, ...]] | None = None
    for penalty in penalties:
        current = {pair: float(initial_thresholds[pair]) for pair in pairs}
        try:
            evaluations, captures, rank = evaluate(current, penalty, record=True)
        except RouteHypothesisOverflow:
            continue
        for _ in range(maximum_sweeps):
            for pair in pairs:
                local_best: tuple[dict[tuple[int, int], float], dict[float, Mapping[str, object]], dict[float, bool], tuple[float, ...]] | None = None
                for threshold in grid:
                    proposal = dict(current)
                    proposal[pair] = float(threshold)
                    try:
                        proposed_evaluations, proposed_captures, proposed_rank = evaluate(
                            proposal, penalty, record=True
                        )
                    except RouteHypothesisOverflow:
                        search_rows.append(
                            {
                                "unmatched_penalty": float(penalty),
                                "thresholds": {
                                    f"{left}->{right}": float(proposal[(left, right)])
                                    for left, right in pairs
                                },
                                "invalid_reason": "route_hypothesis_limit_exceeded",
                            }
                        )
                        continue
                    candidate = (proposal, proposed_evaluations, proposed_captures, proposed_rank)
                    if local_best is None or candidate[3] > local_best[3]:
                        local_best = candidate
                if local_best is None:  # pragma: no cover - declared grid is non-empty
                    # Every proposed threshold for this coordinate exceeded the
                    # declared hypothesis bound, so this penalty cannot define
                    # a frozen operating point on the selection subset.
                    current = {}
                    break
                current, evaluations, captures, rank = local_best
            if not current:
                break
        if not current:
            continue
        candidate_global = (current, penalty, evaluations, captures, rank)
        if best is None or candidate_global[4] > best[4]:
            best = candidate_global
    if best is None:  # pragma: no cover - validated penalty grid is non-empty
        raise RuntimeError("route threshold search did not evaluate a point")

    full_contexts = {
        magnitude: prepare_route_assignment_context(sets, values, calibration_bins, path)
        for magnitude, (sets, values) in grouped.items()
    }

    def evaluate_full(
        thresholds: Mapping[tuple[int, int], float], penalty: float
    ) -> tuple[dict[float, Mapping[str, object]], dict[float, bool], tuple[float, ...]]:
        final_config = RouteAssignmentConfig(
            score_threshold_by_pair=dict(thresholds),
            unmatched_penalty=float(penalty),
            station_path=path,
            maximum_hypotheses=maximum_hypotheses,
        )
        evaluations: dict[float, Mapping[str, object]] = {}
        captures: dict[float, bool] = {}
        for magnitude, (sets, values) in sorted(grouped.items()):
            try:
                evaluation = evaluate_adjacent_route_assignment_sets(
                    sets,
                    values,
                    final_config,
                    calibration_bins=calibration_bins,
                    context=full_contexts[magnitude],
                )
            except RuntimeError as error:
                if "route hypothesis count exceeds maximum_hypotheses" in str(error):
                    raise RouteHypothesisOverflow(
                        "validation route operating point overflows the full graph; "
                        "raise the minimum declared threshold before test is opened"
                    ) from error
                raise
            route = evaluation.get("route")
            if not isinstance(route, Mapping):
                raise RuntimeError("full validation route evaluation is malformed")
            evaluations[magnitude] = evaluation
            captures[magnitude] = capture_success(route, capture_criteria)
        return evaluations, captures, _selection_rank(evaluations, captures, thresholds)

    # Every successful candidate lives in ``cache``.  The subset search gives
    # a tractable proposal set, while this bounded second pass protects against
    # a lucky 16-event subset defining the frozen primary operating point.
    ranked_candidates = sorted(
        (
            (
                tuple(float(thresholds[pair]) for pair in pairs),
                float(penalty),
                rank,
            )
            for (penalty, threshold_values), (_, _, rank) in cache.items()
            for thresholds in (
                {pair: float(value) for pair, value in zip(pairs, threshold_values)},
            )
        ),
        key=lambda value: (value[2], -value[1], tuple(-entry for entry in value[0])),
        reverse=True,
    )
    if not ranked_candidates:  # pragma: no cover - follows ``best`` above
        raise RuntimeError("route threshold search has no successful candidate settings")
    rerank_count = min(int(full_validation_rerank_candidates), len(ranked_candidates))
    reranked: list[
        tuple[
            dict[tuple[int, int], float],
            float,
            dict[float, Mapping[str, object]],
            dict[float, bool],
            tuple[float, ...],
        ]
    ] = []
    for threshold_values, penalty, subset_rank in ranked_candidates[:rerank_count]:
        thresholds = {pair: float(value) for pair, value in zip(pairs, threshold_values)}
        try:
            evaluations, captures, full_rank = evaluate_full(thresholds, penalty)
        except RouteHypothesisOverflow:
            search_rows.append(
                {
                    "unmatched_penalty": penalty,
                    "thresholds": {
                        f"{left}->{right}": thresholds[(left, right)] for left, right in pairs
                    },
                    "full_validation_rerank": True,
                    "invalid_reason": "route_hypothesis_limit_exceeded_on_full_validation",
                }
            )
            continue
        reranked.append((thresholds, penalty, evaluations, captures, full_rank))
        search_rows.append(
            {
                "unmatched_penalty": penalty,
                "thresholds": {
                    f"{left}->{right}": thresholds[(left, right)] for left, right in pairs
                },
                "full_validation_rerank": True,
                "subset_rank": list(subset_rank),
                "full_validation_rank": list(full_rank),
                "full_validation_capture_by_magnitude": {
                    str(magnitude): bool(value) for magnitude, value in captures.items()
                },
                "full_validation_complete_track_efficiency_by_magnitude": {
                    str(magnitude): evaluations[magnitude]["route"]["complete_track_efficiency"]
                    for magnitude in sorted(evaluations)
                },
                "full_validation_complete_track_purity_by_magnitude": {
                    str(magnitude): evaluations[magnitude]["route"]["complete_track_purity"]
                    for magnitude in sorted(evaluations)
                },
                "full_validation_track_fake_rate_by_magnitude": {
                    str(magnitude): evaluations[magnitude]["route"]["track_fake_rate"]
                    for magnitude in sorted(evaluations)
                },
            }
        )
    if not reranked:
        raise RouteHypothesisOverflow(
            "every subset-selected route setting overflows the full validation graph; "
            "raise the minimum declared threshold before test is opened"
        )
    thresholds, penalty, evaluations, captures, rank = max(
        reranked,
        key=lambda value: (
            value[4],
            -value[1],
            tuple(-float(value[0][pair]) for pair in pairs),
        ),
    )
    return RouteSelectionResult(
        thresholds=dict(thresholds),
        unmatched_penalty=float(penalty),
        evaluation_by_magnitude=dict(evaluations),
        capture_by_magnitude=dict(captures),
        rank=rank,
        search_rows=tuple(search_rows),
        selection_event_counts_by_magnitude=selection_counts,
        full_validation_event_counts_by_magnitude=full_counts,
        full_validation_rerank_candidates_requested=int(full_validation_rerank_candidates),
        full_validation_rerank_candidates_evaluated=len(reranked),
    )
