"""Batch evaluation helpers for frozen adjacent four-station route assignment."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from baselines.global_assignment import ScoreMatrix
from baselines.route_assignment import (
    RouteAssignmentConfig,
    adjacent_route_assignment,
    adjacent_station_pairs,
)
from evaluation.metrics import (
    AssociationMetrics,
    UnmatchedEndpointMetrics,
    assess_event_matches,
    assess_event_unmatched_endpoints,
)
from evaluation.route_metrics import RouteMetrics, assess_adjacent_route_assignment
from evaluation.route_metrics import assess_complete_route_score_retention
from training.global_assignment import GlobalAssignmentContext, prepare_global_assignment_context


@dataclass(frozen=True)
class _RouteEventGroup:
    sample: object
    event: object
    set_indices_by_pair: Mapping[tuple[int, int], int]
    station_matrices: Mapping[tuple[int, int], tuple[ScoreMatrix, Sequence[object]]]

    @property
    def key(self) -> tuple[str, str, int, int]:
        return (
            str(self.sample.source_id),
            str(self.sample.payload_id),
            int(self.event.run_id),
            int(self.event.event_id),
        )


@dataclass(frozen=True)
class RouteAssignmentContext:
    """Dense score matrices grouped by physical synthetic event."""

    global_context: GlobalAssignmentContext
    groups: tuple[_RouteEventGroup, ...]
    station_path: tuple[int, ...]


def prepare_route_assignment_context(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    calibration_bins: int,
    station_path: Sequence[int] = (0, 1, 2, 3),
) -> RouteAssignmentContext:
    """Build adjacent route event groups from already-calibrated MLP scores."""
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    stations = tuple(int(station) for station in station_path)
    pairs = adjacent_station_pairs(stations)
    global_context = prepare_global_assignment_context(sets, scores, calibration_bins)
    grouped: dict[tuple[str, str, int, int], dict[str, object]] = {}
    for index, (candidate_set, matrix) in enumerate(zip(sets, global_context.matrices)):
        sample = candidate_set.sample
        event = candidate_set.event
        key = (
            str(sample.source_id),
            str(sample.payload_id),
            int(event.run_id),
            int(event.event_id),
        )
        payload = grouped.setdefault(
            key,
            {"sample": sample, "event": event, "indices": {}, "matrices": {}},
        )
        if payload["sample"] is not sample or payload["event"] is not event:
            raise ValueError("one route event key resolves to distinct sample/event objects")
        pair = tuple(int(value) for value in candidate_set.station_pair)
        if pair not in pairs:
            continue
        indices = payload["indices"]
        matrices = payload["matrices"]
        if not isinstance(indices, dict) or not isinstance(matrices, dict):  # pragma: no cover
            raise RuntimeError("invalid route grouping payload")
        if pair in indices:
            raise ValueError("duplicate adjacent station-pair matrix in one route event")
        indices[pair] = index
        matrices[pair] = (matrix, candidate_set.candidates)
    groups: list[_RouteEventGroup] = []
    for key, payload in sorted(grouped.items()):
        indices = payload["indices"]
        matrices = payload["matrices"]
        if not isinstance(indices, dict) or not isinstance(matrices, dict):  # pragma: no cover
            raise RuntimeError("invalid route grouping payload")
        missing = set(pairs) - set(indices)
        if missing:
            raise ValueError(
                f"route event {key} is missing adjacent station-pair matrices: {sorted(missing)}"
            )
        groups.append(
            _RouteEventGroup(
                sample=payload["sample"],
                event=payload["event"],
                set_indices_by_pair=dict(indices),
                station_matrices=dict(matrices),
            )
        )
    return RouteAssignmentContext(
        global_context=global_context,
        groups=tuple(groups),
        station_path=stations,
    )


def evaluate_adjacent_route_assignment_sets(
    sets: Sequence[object],
    scores: Sequence[np.ndarray],
    config: RouteAssignmentConfig,
    calibration_bins: int,
    context: RouteAssignmentContext | None = None,
    complete_route_scores_by_event: Mapping[
        tuple[str, str, int, int], Mapping[tuple[int, ...], float]
    ]
    | None = None,
) -> dict[str, object]:
    """Evaluate a fixed truth-free route configuration over physical events."""
    if len(sets) != len(scores):
        raise ValueError("candidate sets and scores are not aligned")
    resolved = (
        prepare_route_assignment_context(sets, scores, calibration_bins, config.station_path)
        if context is None
        else context
    )
    if tuple(config.station_path) != resolved.station_path:
        raise ValueError("route context station path differs from assignment configuration")
    pairs = adjacent_station_pairs(resolved.station_path)
    aggregate_route = RouteMetrics()
    association_by_pair: dict[tuple[int, int], AssociationMetrics] = {
        pair: AssociationMetrics() for pair in pairs
    }
    unmatched_by_pair: dict[tuple[int, int], UnmatchedEndpointMetrics] = {
        pair: UnmatchedEndpointMetrics() for pair in pairs
    }
    candidate_edges = 0
    candidate_edges_by_pair: dict[tuple[int, int], int] = {pair: 0 for pair in pairs}
    hypotheses = 0
    selected_routes = 0
    route_query_totals: dict[str, int] | None = None
    if complete_route_scores_by_event is not None:
        expected_keys = {group.key for group in resolved.groups}
        supplied_keys = set(complete_route_scores_by_event)
        if supplied_keys != expected_keys:
            raise ValueError(
                "complete route-score event keys differ from the physical route context: "
                f"missing={len(expected_keys - supplied_keys)}, extra={len(supplied_keys - expected_keys)}"
            )
        route_query_totals = {
            "route_events": 0,
            "route_events_without_truth": 0,
            "complete_truth_chains": 0,
            "raw_route_candidate_retained_complete_truth_chains": 0,
            "route_score_retained_complete_truth_chains": 0,
        }

    for group in resolved.groups:
        complete_route_scores = (
            None
            if complete_route_scores_by_event is None
            else complete_route_scores_by_event[group.key]
        )
        result = adjacent_route_assignment(
            group.event,
            group.station_matrices,
            config,
            complete_route_scores=complete_route_scores,
        )
        route_metrics = assess_adjacent_route_assignment(
            group.event,
            result,
            group.station_matrices,
            resolved.station_path,
            config.score_threshold_by_pair,
        )
        aggregate_route.add(route_metrics)
        candidate_edges += result.candidate_edges_above_threshold
        hypotheses += result.hypotheses_considered
        selected_routes += result.selected_routes
        if complete_route_scores is not None and route_query_totals is not None:
            route_query = assess_complete_route_score_retention(
                group.event,
                resolved.station_path,
                complete_route_scores,
                config.complete_route_score_threshold,
            )
            for key in route_query_totals:
                route_query_totals[key] += int(route_query[key])
        for pair in pairs:
            candidate_edges_by_pair[pair] += result.candidate_edges_above_threshold_by_pair[pair]
            association, _ = assess_event_matches(
                group.event,
                result.matches_by_pair[pair],
                source_station=pair[0],
                target_station=pair[1],
            )
            unmatched = assess_event_unmatched_endpoints(
                group.event,
                result.matches_by_pair[pair],
                source_station=pair[0],
                target_station=pair[1],
            )
            association_by_pair[pair].add(association)
            unmatched_by_pair[pair].add(unmatched)

    candidate_by_pair = resolved.global_context.candidate_by_pair
    result: dict[str, object] = {
        "candidate": {
            f"{pair[0]}->{pair[1]}": candidate_by_pair[pair]
            for pair in pairs
        },
        "route": aggregate_route.as_dict(),
        "assignment": {
            "events": len(resolved.groups),
            "candidate_edges_above_threshold": candidate_edges,
            "candidate_edges_above_threshold_by_pair": {
                f"{pair[0]}->{pair[1]}": candidate_edges_by_pair[pair] for pair in pairs
            },
            "hypotheses_considered": hypotheses,
            "selected_routes": selected_routes,
        },
        "by_station_pair": {
            f"{pair[0]}->{pair[1]}": {
                "candidate": candidate_by_pair[pair],
                "association": association_by_pair[pair].as_dict(),
                "unmatched": unmatched_by_pair[pair].as_dict(),
                "score_threshold": float(config.score_threshold_by_pair[pair]),
            }
            for pair in pairs
        },
    }
    if route_query_totals is not None:
        denominator = route_query_totals["complete_truth_chains"]
        result["route_query"] = {
            **route_query_totals,
            "raw_route_candidate_complete_truth_chain_recall": (
                None
                if not denominator
                else float(
                    route_query_totals["raw_route_candidate_retained_complete_truth_chains"]
                    / denominator
                )
            ),
            "route_score_threshold_complete_truth_chain_recall": (
                None
                if not denominator
                else float(route_query_totals["route_score_retained_complete_truth_chains"] / denominator)
            ),
            "complete_route_score_threshold": config.complete_route_score_threshold,
        }
    return result
